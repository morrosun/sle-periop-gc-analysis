# -*- coding: utf-8 -*-
"""
167_adjudication_estimates.py
=============================
v7b 补充分析 —— 把人工（LLM 代理）双标核验的结果**量化回灌**到 v7 的分层结论上。

做三件事
  1. 分层加权反推**真实通道规模**：v7 名义 57 / 25 / 792 例，经双标校正后还剩多少？
     抽样是分层（oversample 稀疏层）的，故必须按层权重反推，并给 bootstrap CI。
  2. **触发词分解**：算法在 INACTIVE / affirmative 上"过判"是哪些词造成的、暴露面多大。
  3. **按金标准重估死亡率**：v7 的核心发现是"三通道死亡 3.5% / 4.0% / 9.7%"。
     若分层本身错了，这个死亡梯度还剩多少？用 ref_std 重新分层加权估计。

输入  out/adjud_merged.csv, out/_v7b_results.json, data/gc_inactivity.csv,
      data/cohort_rheum_icu.csv, mimiciv_note.discharge
输出  out/167_estimates.txt, out/table_v7b_truth.csv,
      out/table_v7b_triggers.csv, out/table_v7b_mortality.csv, out/_v7b_est.json
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import psycopg2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")

L = []


def P(s=""):
    print(s)
    sys.stdout.flush()
    L.append(str(s))


def load_mod(name, rel):
    p = os.path.join(ROOT, "scripts", rel)
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 复用 165（其 verbatim 复现了 163 的判定，自检 0/148 不一致）
M165 = load_mod("mod165", "165_make_blind_sheets.py")
M163 = M165.M163

CLS_A = ["INACTIVE", "INF_DEFER", "UNDERDOC"]


# ===================================================================== 1
def truth_by_layer():
    """分层加权反推真实通道构成（任务 A）。"""
    mg = pd.read_csv(os.path.join(OUT, "adjud_merged.csv"))
    A = mg[(mg.task == "A") & (mg.ref_std.notna())].copy()
    A["w"] = A.w.astype(float)
    A["ref_std"] = A.ref_std.astype(str)
    A["algo_label"] = A.algo_label.astype(str)

    est = {}
    for c in CLS_A:
        est[c] = float((A.w * (A.ref_std == c)).sum())
    tot = sum(est.values())

    # 名义（v7 报告值，抽样框 + 18 例纯否定假阳性）
    nom = {"INACTIVE": 57, "INF_DEFER": 25, "UNDERDOC": 792}

    # ---- bootstrap（层内有放回重抽，权重=层规模/层内例数）
    rs = np.random.RandomState(20260916)
    lay = {}
    for st, s in A.groupby("stratum"):
        lay[st] = (len(s), s.w.iloc[0])
    B = 2000
    boot = {c: np.zeros(B) for c in CLS_A}
    arr = {st: (A[A.stratum == st].ref_std.values,
                A[A.stratum == st].w.values[0])
           for st in lay}
    for b in range(B):
        for st, (vals, wgt) in arr.items():
            n = len(vals)
            idx = rs.randint(0, n, n)
            sv = vals[idx]
            for c in CLS_A:
                boot[c][b] += wgt * (sv == c).sum()
    # 同时 bootstrap 每个通道"占名义值的比例"
    ci = {}
    for c in CLS_A:
        ci[c] = (float(np.percentile(boot[c], 2.5)),
                 float(np.percentile(boot[c], 97.5)))

    P("=" * 96)
    P("1. 真实通道构成的反推估计（任务 A，n=%d，分层加权）" % len(A))
    P("=" * 96)
    P("  名义 v7 值以抽样框 856 例为基准；双标校正后的估计如下。")
    P("")
    P("  %-12s %10s %10s %14s %10s %9s"
      % ("通道", "v7 名义", "校正估计", "95% CI (boot)", "PPV", "确证 n/抽"))
    rows = []
    for c in CLS_A:
        s = A[A.algo_label == c]
        ppv = float((s.ref_std == c).mean()) if len(s) else np.nan
        P("  %-12s %10d %10.1f %6.1f - %6.1f %10.3f %9d/%d"
          % (c, nom[c], est[c], ci[c][0], ci[c][1], ppv,
             int((s.ref_std == c).sum()), len(s)))
        rows.append(dict(channel=c, nominal=nom[c], est=round(est[c], 1),
                         lo=round(ci[c][0], 1), hi=round(ci[c][1], 1),
                         ppv=round(ppv, 4),
                         confirmed=int((s.ref_std == c).sum()), n_sampled=len(s)))
    P("")
    P("  合计估计 %.0f 例（抽样框 856 例）" % tot)
    P("")
    P("  ★ 结论：v7 把「明确无活动」记为 57 例、「因感染暂缓」记为 25 例；")
    P("    双标校正后点估计分别为 %.0f 与 %.0f 例，" % (est["INACTIVE"], est["INF_DEFER"]))
    P("    即约 %.0f%% 与 %.0f%% 的名义病例不成立。"
      % (100 * (1 - est["INACTIVE"] / nom["INACTIVE"]),
         100 * (1 - est["INF_DEFER"] / nom["INF_DEFER"])))
    return A, rows, ci


# ===================================================================== 2
def triggers(A):
    """触发词分解。"""
    gi = pd.read_csv(os.path.join(DATA, "gc_inactivity.csv"))
    a = gi[gi.spec_raw_hits == 0]

    P("")
    P("=" * 96)
    P("2. 触发词分解 —— 算法为什么「过判」")
    P("=" * 96)
    P("")
    P("2a. 全队列（856 例）算法判 INACTIVE 的 %d 例，按触发词类别："
      % int((a.inact3 == "INACTIVE").sum()))
    kinds = (a[a.inact3 == "INACTIVE"].inact_kinds
             .fillna("(none)").value_counts())
    rows_t = []
    tot_i = int((a.inact3 == "INACTIVE").sum())
    for k, v in kinds.items():
        P("     %-24s %4d 例  (%5.1f%%)" % (k, v, 100 * v / tot_i))
        rows_t.append(dict(task="A", trigger=k, n=int(v),
                           pct=round(100 * v / tot_i, 1), scope="all 856"))
    P("     %-24s %4d" % ("合计", tot_i))
    P("")
    P("     ★ 裸词 `asymptomatic` 单独解释 %d/%d = %.1f%% 的 INACTIVE 判定。"
      % (kinds.get("asymptomatic", 0), tot_i,
         100 * kinds.get("asymptomatic", 0) / tot_i))
    P("       人工核验显示它命中的是**非风湿**的无症状问题"
      "（无症状房颤/菌尿/低血压/卵巢囊肿……）。")

    # ---- 抽样内：触发词 × 人工判定
    kk = dict(zip(a.stay_key, a.inact_kinds.fillna("(none)")))
    A2 = A[A.algo_label == "INACTIVE"].copy()
    A2["kind"] = A2.stay_key.map(kk).fillna("(none)")
    P("")
    P("2b. 抽样内算法判 INACTIVE 的 %d 例：触发词 × 人工判定" % len(A2))
    P("     %-18s %6s %10s %10s %10s" % ("触发词", "n", "确证INACT", "实为DEFER",
                                        "实为UDOC"))
    for k in list(kinds.index) + ["(none)"]:
        s = A2[A2.kind == k]
        if not len(s):
            continue
        P("     %-18s %6d %10d %10d %10d"
          % (k, len(s), int((s.ref_std == "INACTIVE").sum()),
             int((s.ref_std == "INF_DEFER").sum()),
             int((s.ref_std == "UNDERDOC").sum())))
        rows_t.append(dict(task="A", trigger=k + " [sampled]", n=len(s),
                           confirmed_INACTIVE=int((s.ref_std == "INACTIVE").sum()),
                           as_INF_DEFER=int((s.ref_std == "INF_DEFER").sum()),
                           as_UNDERDOC=int((s.ref_std == "UNDERDOC").sum()),
                           scope="sampled 35"))

    # ---- 任务 B：重算 SPEC 类别命中（全 394 例）
    P("")
    P("2c. 任务 B 抽样框（spec_raw>0，n=394）：各类风湿活动特异词表")
    P("    被**未被否定**命中（=算法判「有活动」）的例数")
    # stay_key 在本项目即 hadm_id（165 以同一映射读取住院记录，复现自检 0/148）
    b = gi[gi.spec_raw_hits > 0]
    mp = {int(k): int(k) for k in b.stay_key}
    keys = sorted(mp.values())
    cc = psycopg2.connect(host=os.environ.get("MIMIC_DB_HOST", "localhost"),
                          port=int(os.environ.get("MIMIC_DB_PORT", "5432")),
                          user=os.environ.get("MIMIC_DB_USER", "postgres"),
                          password=os.environ.get("MIMIC_DB_PASSWORD", ""),
                          dbname="mimiciv")
    parts = []
    for i in range(0, len(keys), 400):
        ch = keys[i:i + 400]
        q = ("SELECT hadm_id, note_seq, text FROM mimiciv_note.discharge "
             "WHERE hadm_id IN (%s)" % ",".join(map(str, ch)))
        parts.append(pd.read_sql(q, cc))
    cc.close()
    nt = pd.concat(parts, ignore_index=True)
    nt = (nt.sort_values(["hadm_id", "note_seq"])
            .drop_duplicates("hadm_id", keep="last"))
    txt = dict(zip(nt.hadm_id.astype(int), nt.text))
    P("     可读到出院记录 %d / %d 例" % (len(txt), len(keys)))

    def spec_by_class(blobN):
        hit = {}
        for k in M163.SPEC:
            h = 0
            for rx in M163.IND_RX[k]:
                for m in rx.finditer(blobN):
                    if not M163.negated(blobN, m.start()):
                        h = 1
                        break
                if h:
                    break
            hit[k] = h
        return hit

    cnt = {k: 0 for k in M163.SPEC}
    n_ok = 0
    for r in b.itertuples():
        hid = mp.get(int(r.stay_key))
        t = txt.get(hid) if hid else None
        if not t:
            continue
        n_ok += 1
        hc, dd, moa = M165.build_blob(t)
        blobN = M165.join_blob(hc, dd, moa)
        for k, v in spec_by_class(blobN).items():
            cnt[k] += v
    P("     %-14s %8s" % ("SPEC 类别", "命中例数"))
    for k in M163.SPEC:
        P("     %-14s %8d" % (k, cnt[k]))
        rows_t.append(dict(task="B", trigger="SPEC:" + k, n=int(cnt[k]),
                           scope="all 394"))
    P("     注：同一例可被多类命中，故列和 > 例数；flar 类含裸词")
    P("         (flare|exacerbation)，会命中 COPD / 心衰的 exacerbation。")
    return rows_t


# ===================================================================== 3
def mortality_by_truth(A):
    """按人工金标准重新分层，估计三通道的真实死亡率。"""
    P("")
    P("=" * 96)
    P("3. 按金标准重估死亡率 —— v7 的死亡梯度还剩多少？")
    P("=" * 96)
    A = A.copy()
    A["death"] = A.death_30d.astype(float)
    v7 = {"INACTIVE": 3.5, "INF_DEFER": 4.0, "UNDERDOC": 9.7}
    P("")
    P("  %-12s %12s %12s %14s %10s"
      % ("通道", "v7 死亡率%", "校正估计%", "95% CI (boot)", "事件/权重和"))
    rs = np.random.RandomState(7)
    rows = []
    for c in CLS_A:
        m = (A.ref_std == c).values
        w = A.w.values * m
        d = A.death.values * A.w.values * m      # 事件权重和（必须乘 w）
        if w.sum() == 0:
            continue
        rate = 100 * d.sum() / w.sum()
        bs = np.zeros(2000)
        for bb in range(2000):
            num = den = 0.0
            for st, s in A.groupby("stratum"):
                ii = rs.randint(0, len(s), len(s))
                sv = s.ref_std.values[ii]
                dv = s.death.values[ii]
                wv = s.w.values[0]
                k = (sv == c)
                num += wv * dv[k].sum()
                den += wv * k.sum()
            bs[bb] = 100 * num / den if den else np.nan
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        P("  %-12s %12.1f %12.1f %6.1f - %6.1f %10.1f/%.1f"
          % (c, v7[c], rate, lo, hi, d.sum(), w.sum()))
        rows.append(dict(channel=c, mort_v7=v7[c], mort_est=round(rate, 1),
                         lo=round(float(lo), 1), hi=round(float(hi), 1),
                         events=float(d.sum()), wsum=float(w.sum())))
    P("")
    P("  ★ 判读：v7 的死亡梯度（3.5 / 4.0 / 9.7%）在双标校正后被大幅压缩：")
    P("    「明确无活动」由 3.5% 变为 6.4%（真层仅约 24 例、1 个事件，")
    P("    CI 0–100%，不可解释）；「因感染暂缓」校正后仅约 6 例、0 事件，")
    P("    完全不可估计；只有「未记录」层稳定在 9.6%（2.3–18.9）。")
    P("    即可稳健报告的只有「未记录层死亡率约 1/10」；")
    P("    另两个通道的死亡率差异是**分层伪影**，不应写进结论。")
    return rows


def main():
    J = json.load(open(os.path.join(OUT, "_v7b_results.json"), encoding="utf-8"))
    A, trow, ci = truth_by_layer()
    grow = triggers(A)
    mrow = mortality_by_truth(A)

    pd.DataFrame(trow).to_csv(os.path.join(OUT, "table_v7b_truth.csv"),
                              index=False)
    pd.DataFrame(grow).to_csv(os.path.join(OUT, "table_v7b_triggers.csv"),
                              index=False)
    pd.DataFrame(mrow).to_csv(os.path.join(OUT, "table_v7b_mortality.csv"),
                              index=False)
    with open(os.path.join(OUT, "_v7b_est.json"), "w", encoding="utf-8") as f:
        json.dump(dict(truth=trow, triggers=grow, mortality=mrow,
                       frame_a=J["frame_a"], frame_b=J["frame_b"]),
                  f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(OUT, "167_estimates.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    P("")
    P("saved: 167_estimates.txt / table_v7b_{truth,triggers,mortality}.csv / "
      "_v7b_est.json")


if __name__ == "__main__":
    main()
