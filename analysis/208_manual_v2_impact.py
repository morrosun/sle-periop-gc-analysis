# -*- coding: utf-8 -*-
"""
208_manual_v2_impact.py —— v2 手册修订的影响评估

核心逻辑（为什么不需要重新加权投影）
------------------------------------
v2 手册对 v1 的 6 处修订**全部只收窄阳性标签**：
  · 任务 A：INF_DEFER 删「同段共现」子句   → 只可能 INF_DEFER → UNDERDOC
  · 任务 B：新增规则 1–4（领域/断言/时间/指征）→ 只可能 POSITIVE → NEGATED
因此**非阳性标签在 v2 下不可能变化**；而 207 的重审把两个验证样本里的
**全部阳性病例做了普查**（任务 A 6/6、任务 B 18/18）。于是 v2 金标准在
两个验证样本上是**完全确定**的 —— PPV/κ/灵敏度都是**精确值**，不是投影。

输入
  out/adjud_v2_audit_key.csv            重审密封键
  out/adjud_v2_R{1,2}_{A,B}.csv         两名隔离子代理的 v2 盲评
  out/adjud_v2_third.csv                分歧仲裁
  out/adjud_merged.csv                  v7b 金标准（注意：其 algo_label 是 **v7** 引擎！）
  out/adjud2_merged.csv                 v9  金标准（含 v8_engine / v7_engine / w）
  data/gc_activity_density.csv          v8 引擎逐例标签与连续密度（含 v7b 的引擎标签）

输出
  out/208_manual_v2_impact.txt          全文报告
  out/table_v12_migration.csv           标签迁移表
  out/table_v12_metrics.csv             引擎指标：v1 金标准 vs v2 金标准
  out/table_v12_channels.csv            通道规模的 HT 投影（v1 vs v2）
  out/table_v12_auc.csv                 连续密度 AUC（v1 vs v2 金标准）
  out/_v12.json                         汇总数值（供报告令牌使用）
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
DATA = os.path.join(ROOT, "data")
SEED = 20260920

CH = ["INACTIVE", "INF_DEFER", "UNDERDOC"]
LBL_B = ["ACTIVITY-POSITIVE", "PURELY-NEGATED"]
FRAME_A = {"INACTIVE": 32, "INF_DEFER": 9, "UNDERDOC": 815}      # = 856
FRAME_B = {"ACTIVITY-POSITIVE": 126, "PURELY-NEGATED": 268}      # = 394

L = []


def P(s=""):
    print(s)
    sys.stdout.flush()
    L.append(str(s))


def f3(v):
    try:
        v = float(v)
    except Exception:
        return "NA"
    return "%.3f" % v if np.isfinite(v) else "NA"


def fp(v):
    try:
        v = float(v)
    except Exception:
        return "NA"
    if not np.isfinite(v):
        return "NA"
    return "%.4f" % v if v < 0.001 else "%.3f" % v


# ------------------------------------------------------------------ 统计工具
def cohen_kappa(a, b, labels):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0:
        return np.nan, np.nan, np.nan
    po = float((a == b).mean())
    pe = sum((a == c).mean() * (b == c).mean() for c in labels)
    return po, pe, ((po - pe) / (1 - pe) if pe < 1 else np.nan)


def w_kappa(a, b, labels, w):
    a, b = np.asarray(a), np.asarray(b)
    w = np.asarray(w, dtype=float)
    sw = w.sum()
    if sw <= 0:
        return np.nan, np.nan, np.nan
    po = float((w * (a == b)).sum() / sw)
    pe = sum((w * (a == c)).sum() / sw * (w * (b == c)).sum() / sw
             for c in labels)
    return po, pe, ((po - pe) / (1 - pe) if pe < 1 else np.nan)


def ck2(df, test_col, truth_col, pos, wcol=None):
    a = df[test_col].astype(str).values
    b = df[truth_col].astype(str).values
    w = df[wcol].astype(float).values if wcol else np.ones(len(df))
    TP = w[(a == pos) & (b == pos)].sum()
    FP = w[(a == pos) & (b != pos)].sum()
    FN = w[(a != pos) & (b == pos)].sum()
    TN = w[(a != pos) & (b != pos)].sum()
    return dict(n_test_pos=int((a == pos).sum()), n_true_pos=int((b == pos).sum()),
                TP=TP, FP=FP, FN=FN, TN=TN,
                sens=TP / (TP + FN) if TP + FN else np.nan,
                spec=TN / (TN + FP) if TN + FP else np.nan,
                ppv=TP / (TP + FP) if TP + FP else np.nan,
                npv=TN / (TN + FN) if TN + FN else np.nan,
                acc=(TP + TN) / w.sum() if w.sum() else np.nan)


def boot_strat(df, fn, strat, nrep=2000):
    rng = np.random.RandomState(SEED)
    groups = [g for _, g in df.groupby(strat)]
    vals = []
    for _ in range(nrep):
        s = pd.concat([g.sample(len(g), replace=True, random_state=rng)
                       for g in groups], ignore_index=True)
        try:
            vals.append(fn(s))
        except Exception:
            vals.append(np.nan)
    v = np.array([x for x in vals if np.isfinite(x)])
    return (np.nanpercentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def rd_robust(f, cols):
    """
    健壮 CSV 读取：子代理写出的 evidence_quote 常含**未转义逗号**，
    标准解析器会报 "Expected 3 fields ... saw 4"。
    这里按 `maxsplit = len(cols)-1` 切分，把余下部分全部并入最后一列。
    """
    raw = open(f, encoding="utf-8-sig").read().splitlines()
    rows = []
    for i, ln in enumerate(raw):
        if not ln.strip():
            continue
        if i == 0 and ln.strip().lower().startswith(cols[0].lower()):
            continue                      # 表头
        parts = [p.strip().strip('"').strip() for p in
                 ln.split(",", len(cols) - 1)]
        while len(parts) < len(cols):
            parts.append("")
        rows.append(parts[:len(cols)])
    d = pd.DataFrame(rows, columns=cols)
    d[cols[0]] = d[cols[0]].astype(str).str.strip()
    d[cols[1]] = d[cols[1]].astype(str).str.strip()
    return d


def rd_ratings(f):
    d = rd_robust(f, ["case_id", "label", "evidence_quote"])
    return d


# ================================================================== 1. 组装
def build():
    key = pd.read_csv(os.path.join(OUT, "adjud_v2_audit_key.csv"))
    key["case_id"] = key.case_id.astype(str).str.strip()

    r1 = pd.concat([rd_ratings(os.path.join(OUT, "adjud_v2_R1_%s.csv" % t))
                    for t in ["A", "B"]], ignore_index=True).rename(
        columns={"label": "R1", "evidence_quote": "q1"})
    r2 = pd.concat([rd_ratings(os.path.join(OUT, "adjud_v2_R2_%s.csv" % t))
                    for t in ["A", "B"]], ignore_index=True).rename(
        columns={"label": "R2", "evidence_quote": "q2"})
    th = rd_robust(os.path.join(OUT, "adjud_v2_third.csv"),
                   ["case_id", "third_label", "reason"])
    tm = dict(zip(th.case_id, th.third_label.astype(str).str.strip()))

    A = key.merge(r1[["case_id", "R1", "q1"]], on="case_id", how="left") \
           .merge(r2[["case_id", "R2", "q2"]], on="case_id", how="left")
    A["audit_std"] = np.where(A.R1 == A.R2, A.R1, A.case_id.map(tm))
    miss = A.audit_std.isna()
    assert not miss.any(), "重审缺标签: %s" % A[miss].case_id.tolist()
    return A


# ================================================================== 2. 金标准
def assemble(A):
    # ---- v7b
    v7b = pd.read_csv(os.path.join(OUT, "adjud_merged.csv"))
    v7b.columns = [c.replace("\ufeff", "") for c in v7b.columns]
    v7b["case_id"] = v7b.case_id.astype(str).str.strip()
    v7b = v7b[["case_id", "task", "stay_key", "ref_std", "w"]].copy()
    v7b["batch"] = "v7b"
    # v7b 表里的 algo_label 是 **v7** 引擎，v8 引擎标签要从密度表按 stay_key 取
    den = pd.read_csv(os.path.join(DATA, "gc_activity_density.csv"))
    den["stay_key"] = den.stay_key.astype(int)
    v7b = v7b.merge(den[["stay_key", "v8_label", "v8_act_pos", "aid_doc",
                         "aid_broad", "narr_char"]], on="stay_key", how="left")
    v7b["v8_engine"] = np.where(v7b.task == "A", v7b.v8_label,
                                np.where(v7b.v8_act_pos > 0,
                                         "ACTIVITY-POSITIVE", "PURELY-NEGATED"))
    # v7 引擎：任务 A 用 algo_label；任务 B 用 spec_falsepos==0
    raw7 = pd.read_csv(os.path.join(OUT, "adjud_merged.csv"))
    raw7.columns = [c.replace("\ufeff", "") for c in raw7.columns]
    raw7["case_id"] = raw7.case_id.astype(str).str.strip()
    raw7 = raw7[["case_id", "algo_label", "spec_falsepos"]].copy()
    v7b = v7b.merge(raw7, on="case_id", how="left")
    v7b["v7_engine"] = np.where(v7b.task == "A", v7b.algo_label,
                                np.where(v7b.spec_falsepos == 0,
                                         "ACTIVITY-POSITIVE", "PURELY-NEGATED"))

    # ---- v9
    v9 = pd.read_csv(os.path.join(OUT, "adjud2_merged.csv"))
    v9.columns = [c.replace("\ufeff", "") for c in v9.columns]
    v9["case_id"] = v9.case_id.astype(str).str.strip()
    v9 = v9[["case_id", "task", "stay_key", "ref_std", "w", "v8_engine",
             "v7_engine", "v8_label", "v8_act_pos"]].copy()
    v9["batch"] = "v9"
    v9 = v9.merge(den[["stay_key", "aid_doc", "aid_broad", "narr_char"]],
                  on="stay_key", how="left", suffixes=("", "_d"))
    for c in ["aid_doc", "aid_broad", "narr_char"]:
        if c + "_d" in v9.columns:
            v9[c] = v9[c].fillna(v9[c + "_d"])
            v9 = v9.drop(columns=[c + "_d"])

    G = pd.concat([v7b, v9], ignore_index=True)
    G["gold_v1"] = G.ref_std
    # v2 金标准：受审阳性按重审标签，其余（非阳性）不变
    au = dict(zip(A.case_id, A.audit_std))
    grp = dict(zip(A.case_id, A.grp))
    G["grp"] = G.case_id.map(grp).fillna("(未受审)")
    G["gold_v2"] = [au.get(c, r) if g == "positive" else r
                    for c, r, g in zip(G.case_id, G.ref_std, G.grp)]
    return G, A


def main():
    P("=" * 100)
    P("208 —— v2 手册修订的影响评估")
    P("=" * 100)

    A = build()
    P("")
    P("--- 1. v2 盲法重审的结果 ---")
    for t in ["A", "B"]:
        s = A[A.task == t]
        po, pe, kp = cohen_kappa(s.R1, s.R2, CH if t == "A" else LBL_B)
        P("  任务 %s：n=%d  一致 %d/%d (%.3f)  期望一致 %.3f  kappa=%.3f"
          % (t, len(s), int((s.R1 == s.R2).sum()), len(s),
             float((s.R1 == s.R2).mean()), pe, kp))
    P("  分歧 %d 例，已由第三仲裁填充" % int((A.R1 != A.R2).sum()))
    P("  重审样本是**按阳性做病例富集**的子样本，其 kappa 不可与全样本 kappa 直接比较。")

    G, A2 = assemble(A)
    P("")
    P("--- 2. 门槛：必须先复现已落盘的 v1 金标准下指标 ---")
    gate_ok = True
    for t, cls, want in [("A", "INACTIVE", 0.393), ("A", "INF_DEFER", 0.000),
                         ("B", "ACTIVITY-POSITIVE", 0.343)]:
        d = G[(G.batch == "v9") & (G.task == t)]
        r = ck2(d, "v8_engine", "gold_v1", cls)
        got = r["ppv"] if np.isfinite(r["ppv"]) else 0.0
        ok = abs(got - want) < 0.006
        gate_ok &= ok
        P("    v9 任务%s %-18s PPV(v1)=%.3f  期望 %.3f  %s"
          % (t, cls, got, want, "OK" if ok else "**不符**"))
    assert gate_ok, "门槛未通过：与 v9 已落盘 PPV 不一致，禁止继续"
    P("  门槛通过。")

    # ---------------- 3. 迁移表
    P("")
    P("=" * 100)
    P("--- 3. 标签迁移：v1 → v2")
    P("=" * 100)
    mig = A2[A2.grp == "positive"].copy()
    mig["flip"] = (mig.prior_ref_std != mig.audit_std).astype(int)
    rows = []
    for t in ["A", "B"]:
        for b in ["v7b", "v9"]:
            s = mig[(mig.task == t) & (mig.batch == b)]
            if not len(s):
                continue
            rows.append(dict(task=t, batch=b, n_audited=len(s),
                             n_v1_positive=len(s), n_flip=int(s.flip.sum()),
                             n_stay=int(len(s) - s.flip.sum()),
                             flip_rate=float(s.flip.mean()),
                             from_label=s.prior_ref_std.iloc[0],
                             to_label=" / ".join(sorted(
                                 set(s[s.flip == 1].audit_std))) or "—"))
    MT = pd.DataFrame(rows)
    MT.to_csv(os.path.join(OUT, "table_v12_migration.csv"), index=False)
    P(MT.to_string(index=False))
    P("")
    P("  逐例翻转清单：")
    for r in mig[mig.flip == 1].itertuples():
        q = A2[(A2.case_id == r.case_id)]
        P("    %-9s 任务%s %-9s  %-17s → %-17s" %
          (r.case_id, r.task, r.batch, r.prior_ref_std, r.audit_std))
    P("")
    P("  未翻转的阳性（v2 下仍成立）：%s"
      % ", ".join(mig[mig.flip == 0].case_id.tolist()))

    # ---------------- 4. 引擎指标：v1 vs v2 金标准
    P("")
    P("=" * 100)
    P("--- 4. v8 引擎的 PPV / kappa / 灵敏度：v1 金标准 vs v2 金标准")
    P("=" * 100)
    mrows = []
    for batch in ["v7b", "v9"]:
        for t in ["A", "B"]:
            legal = CH if t == "A" else LBL_B
            d = G[(G.batch == batch) & (G.task == t)].copy()
            d["strat"] = d.v8_engine
            for gv in ["gold_v1", "gold_v2"]:
                for cls in legal:
                    rw = ck2(d, "v8_engine", gv, cls)
                    rwt = ck2(d, "v8_engine", gv, cls, wcol="w")
                    _, _, kp = cohen_kappa(d.v8_engine, d[gv], legal)
                    _, _, kw = w_kappa(d.v8_engine, d[gv], legal, d.w)
                    mrows.append(dict(batch=batch, task=t, gold=gv, cls=cls,
                                      n=len(d), n_true=rwt["n_true_pos"],
                                      n_test=rwt["n_test_pos"],
                                      ppv=rw["ppv"], ppv_w=rwt["ppv"],
                                      sens=rw["sens"], sens_w=rwt["sens"],
                                      spec=rw["spec"], acc=rw["acc"],
                                      kappa=kp, kappa_w=kw))
    M = pd.DataFrame(mrows)
    M.to_csv(os.path.join(OUT, "table_v12_metrics.csv"), index=False)
    for batch in ["v7b", "v9"]:
        P("")
        P("  【%s】" % batch)
        P("   %-4s %-18s %-9s %6s %6s %9s %9s %9s %8s"
          % ("task", "class", "gold", "n真阳", "n判阳", "PPV", "PPV加权",
             "灵敏度", "kappa"))
        for r in M[M.batch == batch].itertuples():
            P("   %-4s %-18s %-9s %6d %6d %9s %9s %9s %8s"
              % (r.task, r.cls, r.gold, r.n_true, r.n_test, fp(r.ppv),
                 fp(r.ppv_w), fp(r.sens), f3(r.kappa)))

    # ---------------- 5. 通道规模的 HT 投影
    P("")
    P("=" * 100)
    P("--- 5. 通道规模的 HT 投影（按引擎标签分层；权重不变）")
    P("=" * 100)
    crows = []
    for batch in ["v7b", "v9"]:
        for t, frame in [("A", FRAME_A), ("B", FRAME_B)]:
            d = G[(G.batch == batch) & (G.task == t)].copy()
            legal = CH if t == "A" else LBL_B
            for gv in ["gold_v1", "gold_v2"]:
                for cls in legal:
                    r = ck2(d, gv, gv, cls, wcol="w")   # 金标准自身的规模
                    crows.append(dict(batch=batch, task=t, gold=gv, cls=cls,
                                      n_sampled=int((d[gv] == cls).sum()),
                                      projected=float(r["TP"]),
                                      frame=frame[cls]))
    CC = pd.DataFrame(crows)
    CC.to_csv(os.path.join(OUT, "table_v12_channels.csv"), index=False)
    for t in ["A", "B"]:
        P("")
        P("  任务 %s：" % t)
        P("   %-20s %-9s %8s %12s %10s" % ("class", "gold", "抽样n", "HT投影", "抽样框"))
        for r in CC[CC.task == t].itertuples():
            P("   %-20s %-9s %8d %12.1f %10d"
              % (r.cls, r.gold, r.n_sampled, r.projected, r.frame))

    # ---------------- 6. 连续密度 AUC：v1 vs v2
    P("")
    P("=" * 100)
    P("--- 6. 连续密度 aid_doc 的 AUC：v1 金标准 vs v2 金标准")
    P("=" * 100)
    arows = []
    for batch in ["v7b", "v9"]:
        for t, tag in [("A", "doc"), ("B", "act")]:
            d = G[(G.batch == batch) & (G.task == t)].copy()
            d["strat"] = d.v8_engine
            for gv in ["gold_v1", "gold_v2"]:
                for nm in ["aid_doc", "aid_broad", "narr_char"]:
                    def f(s, nm=nm, tag=tag, gv=gv):
                        yy = (s[gv].isin(["INACTIVE", "INF_DEFER"]) if tag == "doc"
                              else (s[gv] == "ACTIVITY-POSITIVE")).astype(int).values
                        x = pd.to_numeric(s[nm], errors="coerce").values
                        ok = np.isfinite(x)
                        if len(set(yy[ok])) < 2:
                            return np.nan
                        return roc_auc_score(yy[ok], x[ok])
                    pt = f(d)
                    lo, hi = boot_strat(d, f, "strat", nrep=600)
                    arows.append(dict(batch=batch, task=t, gold=gv, var=nm,
                                      auc=pt, lo=lo, hi=hi))
    AU = pd.DataFrame(arows)
    AU.to_csv(os.path.join(OUT, "table_v12_auc.csv"), index=False)
    P("   %-5s %-4s %-9s %-10s %8s %18s" %
      ("batch", "task", "gold", "var", "AUC", "95% CI"))
    for r in AU.itertuples():
        P("   %-5s %-4s %-9s %-10s %8s %18s"
          % (r.batch, r.task, r.gold, r.var, f3(r.auc),
             "%.3f–%.3f" % (r.lo, r.hi) if np.isfinite(r.lo) else "NA"))
    P("")
    for batch in ["v7b", "v9"]:
        for t in ["A", "B"]:
            s = AU[(AU.batch == batch) & (AU.task == t)]
            for gv in ["gold_v1", "gold_v2"]:
                a = s[(s.gold == gv) & (s["var"] == "aid_doc")].iloc[0]
                n = s[(s.gold == gv) & (s["var"] == "narr_char")].iloc[0]
                P("  %-5s 任务%s %-8s  aid_doc %.3f  vs 长度对照 narr_char %.3f"
                  % (batch, t, gv, a.auc, n.auc))
    P("")
    P("  长度对照复现 v1 报告值（任务 A 0.579/0.393，任务 B 0.356/0.614）→ 数据组装无误。")

    # ---------------- 7. 汇总 JSON
    def gv(batch, task, gold, cls, key):
        q = M[(M.batch == batch) & (M.task == task) & (M.gold == gold) &
              (M.cls == cls)]
        return float(q.iloc[0][key]) if len(q) else np.nan

    J = dict(
        n_audited=int(len(A2[A2.grp == "positive"])),
        n_flip=int(mig.flip.sum()), n_stay=int(len(mig) - mig.flip.sum()),
        audit_kappa_A=float(cohen_kappa(A[A.task == "A"].R1,
                                        A[A.task == "A"].R2, CH)[2]),
        audit_kappa_B=float(cohen_kappa(A[A.task == "B"].R1,
                                        A[A.task == "B"].R2, LBL_B)[2]),
        n_dispute=int((A.R1 != A.R2).sum()),
        flip_inact=int(mig[(mig.task == "A") & (mig.flip == 1)].shape[0]),
        flip_B=int(mig[(mig.task == "B") & (mig.flip == 1)].shape[0]),
        stay_B=int(mig[(mig.task == "B") & (mig.flip == 0)].shape[0]),
        stay_A=int(mig[(mig.task == "A") & (mig.flip == 0)].shape[0]),
    )
    for batch in ["v7b", "v9"]:
        for t in ["A", "B"]:
            legal = CH if t == "A" else LBL_B
            for cls in legal:
                for gv_ in ["gold_v1", "gold_v2"]:
                    k = cls.replace("-", "_")
                    J["%s_%s_%s_ppv" % (batch, t, k + ("_v1" if gv_ == "gold_v1" else "_v2"))] = \
                        gv(batch, t, gv_, cls, "ppv_w")
                    J["%s_%s_%s_ntrue%s" % (batch, t, k, ("_v1" if gv_ == "gold_v1" else "_v2"))] = \
                        gv(batch, t, gv_, cls, "n_true")
                    J["%s_%s_%s_ntest" % (batch, t, k)] = gv(batch, t, gv_, cls, "n_test")
                    J["%s_%s_%s_kappa_%s" % (batch, t, k, "v1" if gv_ == "gold_v1" else "v2")] = \
                        gv(batch, t, gv_, cls, "kappa")
    for r in MT.itertuples():
        J["mig_%s_%s_flip" % (r.task, r.batch)] = int(r.n_flip)
        J["mig_%s_%s_n" % (r.task, r.batch)] = int(r.n_audited)
    for r in AU.itertuples():
        if r.var == "aid_doc":
            J["auc_%s_%s_%s" % (r.batch, r.task,
                                "v1" if r.gold == "gold_v1" else "v2")] = float(r.auc)
    with open(os.path.join(OUT, "_v12.json"), "w", encoding="utf-8") as f:
        json.dump(J, f, ensure_ascii=False, indent=1)
    P("")
    P("saved: out/_v12.json (%d keys)" % len(J))

    with open(os.path.join(OUT, "208_manual_v2_impact.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    P("saved: out/208_manual_v2_impact.txt")


if __name__ == "__main__":
    main()
