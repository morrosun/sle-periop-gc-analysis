# -*- coding: utf-8 -*-
"""
200_rule_ablation.py
====================
v10 ① —— **规则消融**：R1 / R2 / R3 里到底哪一条在起作用？

背景
  v8 把三条规则（R1 实体词限定 / R2 否定与假说反转 / R3 领域判断）**打包**上线，
  样本内 PPV 1.000 / 0.833 / 1.000，样本外崩到 0.393 / 0.000 / 0.343（v9）。
  打包上线意味着**不知道功劳或过失属于哪一条**：
    - 拿掉后 PPV 几乎不动 → 这条是装饰；
    - 拿掉后 PPV 反而升高 → 它在**过度否决**（把真阳性也杀了）；
    - 拿掉后 PPV 塌掉     → 它才是真正承重的那条。
  只有做消融才能回答，也才能判断「修订版引擎还有没有救」。

做法
  词表、阈值、优先级与 168 完全一致，只拨动 RULES 开关：
    FULL    R1+R2+R3   （＝ v8 原样，用于自洽校验）
    -R1 / -R2 / -R3    逐条留一
    R1only / R2only / R3only   单条独存
    NONE    三条全关（≈ v7 式裸词，但词表仍是 v8 的，只能算近似）
  在**两批**人类金标准上各测一遍：
    样本内 = v7b 的 148 例（三条规则就是照它的失败机制设计的）
    样本外 = v9 的 131 例（与 v7b 零重叠）—— 这一列才是判据
  逐类给出 PPV（未加权 + 设计加权）、灵敏度、算法判阳率、κ。

自洽校验（不过则一切结论作废）
  FULL 的标签必须与 data/gc_activity_density.csv 的 v8_label 逐例一致。

输出
  out/200_rule_ablation.txt
  out/table_v10_ablation.csv            长表（variant × sample × task × cls）
  out/table_v10_ablation_summary.csv    每个 variant 一行的汇总
  out/table_v10_ablation_blocks.csv     各规则在 FULL 下拦下的命中数
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import psycopg2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
SCR = os.path.join(ROOT, "scripts")

CLS_A = ["INACTIVE", "INF_DEFER", "UNDERDOC"]
LBL_A = CLS_A
LBL_B = ["ACTIVITY-POSITIVE", "PURELY-NEGATED"]

S_IN, S_OUT = "样本内(v7b)", "样本外(v9)"

#    显示名      R1     R2     R3
VARIANTS = [
    ("FULL",   True,  True,  True),
    ("-R1",    False, True,  True),
    ("-R2",    True,  False, True),
    ("-R3",    True,  True,  False),
    ("R1only", True,  False, False),
    ("R2only", False, True,  False),
    ("R3only", False, False, True),
    ("NONE",   False, False, False),
]

# 汇总表列顺序
SUM_COLS = [("INACTIVE", "A"), ("INF_DEFER", "A"), ("UNDERDOC", "A"),
            ("ACTIVITY-POSITIVE", "B")]

L = []


def P(s=""):
    print(s)
    sys.stdout.flush()
    L.append(str(s))


def load_mod(tag, path):
    spec = importlib.util.spec_from_file_location(tag, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[tag] = mod
    spec.loader.exec_module(mod)
    return mod


M168 = load_mod("m168_abl", os.path.join(SCR, "168_activity_rule_engine.py"))
M165 = load_mod("m165_abl", os.path.join(SCR, "165_make_blind_sheets.py"))


# ------------------------------------------------------------------ 统计工具
def cohen_kappa(a, b, labels):
    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=object)
    if len(a) == 0:
        return np.nan, np.nan, np.nan
    po = float((a == b).mean())
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in labels)
    return po, pe, ((po - pe) / (1 - pe) if pe < 1 else np.nan)


def wilson(k, n, z=1.96):
    if n <= 0:
        return (np.nan, np.nan)
    p = float(k) / float(n)
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(max(p * (1 - p), 0) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def cls_metrics(df, test_col, truth_col, pos):
    """未加权 + 设计加权两套指标。"""
    a = df[test_col].astype(str).values
    b = df[truth_col].astype(str).values
    w = df["w"].astype(float).values
    ap, bp = (a == pos), (b == pos)
    TP = int((ap & bp).sum())
    FP = int((ap & ~bp).sum())
    FN = int((~ap & bp).sum())
    TN = int((~ap & ~bp).sum())
    wTP = float(w[ap & bp].sum())
    wFP = float(w[ap & ~bp].sum())
    wFN = float(w[~ap & bp].sum())
    wTN = float(w[~ap & ~bp].sum())
    ppv = TP / (TP + FP) if TP + FP > 0 else np.nan
    ppvw = wTP / (wTP + wFP) if wTP + wFP > 0 else np.nan
    lo, hi = wilson(wTP, wTP + wFP)
    return dict(n=len(df), n_algo_pos=int(ap.sum()), TP=TP, FP=FP, FN=FN, TN=TN,
                ppr=float(ap.mean()), ppr_w=float(w[ap].sum() / w.sum()),
                ppv=ppv, ppv_w=ppvw, ppv_lo=lo, ppv_hi=hi,
                sens=TP / (TP + FN) if TP + FN > 0 else np.nan,
                spec=TN / (TN + FP) if TN + FP > 0 else np.nan,
                npv=TN / (TN + FN) if TN + FN > 0 else np.nan)


# ------------------------------------------------------------------ 判定（单例）
def judge(hc, dd, moa, full_text=""):
    """与 168 主流程同口径：narr = 住院经过 + 出院诊断（<200 字符退回全文）。"""
    if len(hc) + len(dd) < 200:
        narr = M168.norm(full_text)
    else:
        narr = M168.norm(hc + " " + dd)
    ia = M168.eval_inactivity(narr)
    ac = M168.eval_activity(narr)
    df = M168.eval_defer(narr + " " + M168.norm(moa))
    if ia["inact_n"] > 0:
        lab = "INACTIVE"
    elif df["defer_n"] > 0:
        lab = "INF_DEFER"
    else:
        lab = "UNDERDOC"
    labB = "ACTIVITY-POSITIVE" if ac["act_pos"] > 0 else "PURELY-NEGATED"
    return lab, labB, ia["drop"], ac["drop"]


def main():
    # ---------------------------------------------------------------- 金标准
    g7 = pd.read_csv(os.path.join(OUT, "adjud_merged.csv"))
    g7["stay_key"] = g7.stay_key.astype(int)
    g9 = pd.read_csv(os.path.join(OUT, "adjud2_merged.csv"))
    g9["stay_key"] = g9.stay_key.astype(int)

    P("=" * 100)
    P("v10 ① 规则消融 —— R1 / R2 / R3 逐条拨动，看 PPV 怎么变")
    P("=" * 100)
    P("")
    P("--- 0. 两批人类金标准 ---")
    for nm, g in [("样本内（v7b）", g7), (S_OUT + "（v9）", g9)]:
        P("  %-16s n=%3d（任务 A %2d / 任务 B %2d；有参考标准 %3d）"
          % (nm, len(g), int((g.task == "A").sum()), int((g.task == "B").sum()),
             int(g.ref_std.notna().sum())))
        dup = int(g.duplicated(["task", "stay_key"]).sum())
        P("      (task, stay_key) 重复 = %d（应为 0）" % dup)
        assert dup == 0, "金标准内 (task, stay_key) 必须唯一"
    ov = set(g7.stay_key) & set(g9.stay_key)
    P("  两批 stay_key 重叠 = %d（应为 0）" % len(ov))
    assert not ov, "两批样本必须零重叠"

    # ---------------------------------------------------------------- 原文
    keys = sorted(set(g7.stay_key) | set(g9.stay_key))
    k7, k9 = set(g7.stay_key), set(g9.stay_key)
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
    notes = pd.concat(parts, ignore_index=True)
    notes = (notes.sort_values(["hadm_id", "note_seq"])
                  .drop_duplicates("hadm_id", keep="last"))
    notes["hadm_id"] = notes.hadm_id.astype(int)
    txt = dict(zip(notes.hadm_id, notes.text))
    P("  取到出院记录 %d / %d 例（缺记录 %d 例 → narr 为空 → UNDERDOC）"
      % (len(txt), len(keys), len(keys) - len(txt)))

    blobs = {}
    for k in keys:
        t = txt.get(k, "")
        blobs[k] = M165.build_blob(t) + (t,)

    # ---------------------------------------------------------------- 消融
    rows, blocks = [], []
    cache = {}
    for vname, r1, r2, r3 in VARIANTS:
        M168.set_rules(r1=r1, r2=r2, r3=r3)
        recs = []
        din = {"invert": 0, "domain": 0, "hypoth": 0, "sent": 0}
        dac = {"neg": 0, "hypoth": 0, "blacklist": 0, "past": 0, "domain": 0,
               "radius": 0, "chronic": 0}
        for k in keys:
            hc, dd, moa, t = blobs[k]
            lab, labB, dia, dac1 = judge(hc, dd, moa, t)
            recs.append(dict(stay_key=k, labA=lab, labB=labB))
            if vname == "FULL":
                for kk in din:
                    din[kk] += dia[kk]
                for kk in dac:
                    dac[kk] += dac1[kk]
        RD = pd.DataFrame(recs)
        cache[vname] = RD
        if vname == "FULL":
            blocks = [
                dict(side="不活动线索", rule="R2 反转语境", n=din["invert"]),
                dict(side="不活动线索", rule="R2 假说语境", n=din["hypoth"]),
                dict(side="不活动线索", rule="R3 非风湿领域", n=din["domain"]),
                dict(side="不活动线索", rule="R1 非同句实体", n=din["sent"]),
                dict(side="活动线索", rule="R2 否定检测", n=dac["neg"]),
                dict(side="活动线索", rule="R2 假说语境", n=dac["hypoth"]),
                dict(side="活动线索", rule="R3 黑名单", n=dac["blacklist"]),
                dict(side="活动线索", rule="R3 既往/sequela", n=dac["past"]),
                dict(side="活动线索", rule="R3 维持用药列表", n=dac["chronic"]),
                dict(side="活动线索", rule="R3 共现半径", n=dac["radius"]),
                dict(side="活动线索", rule="R3 领域判断", n=dac["domain"]),
            ]

    P("")
    P("=" * 100)
    P("--- 1. 消融逐例指标 ---")
    P("=" * 100)
    for vname, _, _, _ in VARIANTS:
        for sname, gold, kset in [(S_IN, g7, k7), (S_OUT, g9, k9)]:
            d = cache[vname][cache[vname].stay_key.isin(kset)].merge(
                gold[["stay_key", "task", "ref_std", "w"]], on="stay_key",
                how="inner")
            d = d[d.ref_std.notna()].copy()
            if not len(d):
                continue
            P("")
            P("  [%-7s | %s] 可评 %d 例" % (vname, sname, len(d)))
            for task, legal, col in [("A", LBL_A, "labA"), ("B", LBL_B, "labB")]:
                sub = d[d.task == task]
                if not len(sub):
                    continue
                po, pe, kap = cohen_kappa(sub[col], sub.ref_std, legal)
                P("    任务 %s：κ=%.3f  一致率=%.3f  期望=%.3f" % (task, kap, po, pe))
                for c in legal:
                    m = cls_metrics(sub, col, "ref_std", c)
                    rows.append(dict(variant=vname, sample=sname, task=task,
                                     cls=c, kappa=kap, po=po, pe=pe, **m))
                    P("      %-18s n+=%3d TP=%3d FP=%3d PPV=%6s "
                      "(%.3f-%.3f) 判阳率=%.3f 灵敏度=%5s"
                      % (c, m["n_algo_pos"], m["TP"], m["FP"],
                         "%.3f" % m["ppv"] if np.isfinite(m["ppv"]) else "  —  ",
                         m["ppv_lo"], m["ppv_hi"], m["ppr"],
                         "%.3f" % m["sens"] if np.isfinite(m["sens"]) else "  —  "))

    AB = pd.DataFrame(rows)
    AB.to_csv(os.path.join(OUT, "table_v10_ablation.csv"), index=False)
    pd.DataFrame(blocks).to_csv(
        os.path.join(OUT, "table_v10_ablation_blocks.csv"), index=False)
    # 逐例标签落盘（供 201 做配对检验、202 画图，避免重复查库）
    pc = pd.concat([cache[v].assign(variant=v) for v, _, _, _ in VARIANTS],
                   ignore_index=True)
    pc.to_csv(os.path.join(OUT, "table_v10_percase.csv"), index=False)
    P("")
    P("  逐例标签已落盘：out/table_v10_percase.csv %s" % str(pc.shape))

    def get(vname, sname, task, cls, field):
        r = AB[(AB.variant == vname) & (AB["sample"] == sname) &
               (AB.task == task) & (AB.cls == cls)]
        return (float(r.iloc[0][field]), int(r.iloc[0].n_algo_pos)) if len(r) \
            else (np.nan, 0)

    # ---------------------------------------------------------------- 自洽校验
    M168.set_rules(True, True, True)
    P("")
    P("=" * 100)
    P("--- 2. 自洽校验：FULL 必须复现 168 落盘标签 ---")
    P("=" * 100)
    den = pd.read_csv(os.path.join(DATA, "gc_activity_density.csv"))
    den["stay_key"] = den.stay_key.astype(int)
    dm = dict(zip(den.stay_key, den.v8_label))
    bad7 = bad9 = tot7 = tot9 = 0
    FULL = dict(zip(cache["FULL"].stay_key, cache["FULL"].labA))
    for k in keys:
        if dm.get(k) is None:
            continue
        if k in k7:
            tot7 += 1
            bad7 += int(FULL[k] != dm[k])
        if k in k9:
            tot9 += 1
            bad9 += int(FULL[k] != dm[k])
    P("  样本内 %d 例中不一致 %d；样本外 %d 例中不一致 %d" % (tot7, bad7, tot9, bad9))
    assert bad7 == 0 and bad9 == 0, "FULL 未能复现 168 标签，消融结论不成立"

    # ---------------------------------------------------------------- 主表
    P("")
    P("=" * 100)
    P("--- 3. 消融主表：逐类 PPV（样本外才是判据）---")
    P("=" * 100)
    P("")
    P("  %-8s | %-36s | %-36s" % ("变体", "样本外(v9)  PPV(n+)", "样本内(v7b) PPV(n+)"))
    P("  " + "-" * 88)
    P("  %-8s | %-36s | %-36s" % ("", "INACT/INF_DEF/UNDERDOC/A+",
                                  "INACT/INF_DEF/UNDERDOC/A+"))
    summ = []
    for vname, _, _, _ in VARIANTS:
        row = dict(variant=vname)
        cells = {}
        for sname in [S_OUT, S_IN]:
            cs, ks = [], {}
            for cls, task in SUM_COLS:
                v, n = get(vname, sname, task, cls, "ppv")
                row["%s|%s" % (cls, sname)] = v
                row["n_%s|%s" % (cls, sname)] = n
                cs.append("%.3f(%d)" % (v, n) if np.isfinite(v) else " — (%d)" % n)
            for task, ref_cls, tag in [("A", "INACTIVE", "kappaA"),
                                       ("B", "ACTIVITY-POSITIVE", "kappaB")]:
                # κ 按任务整段保存，故取该任务下的任一类别行即可
                v, _ = get(vname, sname, task, ref_cls, "kappa")
                row["%s|%s" % (tag, sname)] = v
                ks[tag] = v
            cells[sname] = "/".join(cs)
        summ.append(row)
        P("  %-8s | %-36s | %-36s" % (vname, cells[S_OUT], cells[S_IN]))
    SM = pd.DataFrame(summ)
    SM.to_csv(os.path.join(OUT, "table_v10_ablation_summary.csv"), index=False)

    P("")
    P("  κ（算法 vs 人类）")
    P("  %-8s %13s %13s %13s %13s" % ("变体", "样本外 A", "样本外 B",
                                      "样本内 A", "样本内 B"))
    for row in summ:
        P("  %-8s %13.3f %13.3f %13.3f %13.3f"
          % (row["variant"], row["kappaA|" + S_OUT], row["kappaB|" + S_OUT],
             row["kappaA|" + S_IN], row["kappaB|" + S_IN]))

    P("")
    P("--- 4. 失效模式：算法判阳率（样本外）---")
    P("  %-8s %11s %11s %11s %11s" % ("变体", "INACTIVE", "INF_DEFER",
                                      "UNDERDOC", "A+"))
    for vname, _, _, _ in VARIANTS:
        ppr = []
        for cls, task in SUM_COLS:
            v, _ = get(vname, S_OUT, task, cls, "ppr")
            ppr.append(v)
        P("  %-8s %11.3f %11.3f %11.3f %11.3f" % tuple([vname] + ppr))
    P("  （对照：人类真阳性率 —— INACTIVE/INF_DEFER/UNDERDOC 见 172 的失效模式表）")

    P("")
    P("--- 5. 各规则在 FULL 配置下拦下的命中数（两批样本合计）---")
    for b in blocks:
        P("  %-10s %-18s %6d" % (b["side"], b["rule"], b["n"]))

    P("")
    P("saved: out/200_rule_ablation.txt, table_v10_ablation.csv,")
    P("       table_v10_ablation_summary.csv, table_v10_ablation_blocks.csv")

    with open(os.path.join(OUT, "200_rule_ablation.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
