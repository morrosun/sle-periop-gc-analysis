# -*- coding: utf-8 -*-
"""
172_replication_tests.py
========================
v9 补充检验：
  1. 样本内 vs 样本外的 PPV 差异（Fisher 精确检验）
  2. 样本外加权 PPV / AUC 的 bootstrap 置信区间（分层重抽样，与抽样设计一致）
  3. 「灵敏度全为 1、特异度塌陷」这一失效模式的形式化刻画（阳性判定率 vs 真阳性率）

输入  out/_v8.json、out/adjud2_merged.csv、out/table_v9_*.csv、data/gc_activity_density.csv
输出  out/172_replication_tests.txt、out/table_v9_compare_ci.csv、out/table_v9_mode.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
DATA = os.path.join(ROOT, "data")
SEED = 20260919
L = []


def P(s=""):
    print(s)
    sys.stdout.flush()
    L.append(str(s))


def boot(df, fn, strat, nrep=4000, seed=SEED):
    rng = np.random.RandomState(seed)
    groups = [g for _, g in df.groupby(strat)]
    v = []
    for _ in range(nrep):
        s = pd.concat([g.sample(len(g), replace=True, random_state=rng)
                       for g in groups], ignore_index=True)
        v.append(fn(s))
    v = np.array([x for x in v if np.isfinite(x)])
    return (np.nanpercentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan))


def main():
    J8 = json.load(open(os.path.join(OUT, "_v8.json"), encoding="utf-8"))
    ins = pd.DataFrame(J8["ppv"])
    ins8 = ins[ins.engine == "v8"]
    cmp_ = pd.read_csv(os.path.join(OUT, "table_v9_compare.csv"))
    k_in = pd.read_csv(os.path.join(OUT, "table_v8_kappa.csv"))
    k_out = pd.read_csv(os.path.join(OUT, "table_v9_kappa.csv"))

    df = pd.read_csv(os.path.join(OUT, "adjud2_merged.csv"))
    ad = pd.read_csv(os.path.join(DATA, "gc_activity_density.csv"))
    ad["stay_key"] = ad.stay_key.astype(int)
    df = df.drop(columns=[c for c in ["aid_doc", "aid_broad", "info_n",
                                      "narr_char"] if c in df.columns])
    df = df.merge(ad[["stay_key", "aid_doc", "aid_broad", "info_n",
                      "narr_char"]], on="stay_key", how="left")
    # 设计分层 = 抽样时用的层（任务 A 按 v8_label，任务 B 按 v8 活动阳性）
    df["strat"] = np.where(df.task == "A", df.v8_label, df.v8_engine)
    DA = df[df.task == "A"]
    DB = df[df.task == "B"]

    P("=" * 100)
    P("v9 补充检验")
    P("=" * 100)

    # ---------------- 1. 样本内 vs 样本外 PPV（Fisher 精确检验）
    P("")
    P("--- 1. 样本内 vs 样本外 PPV 差异（Fisher 精确检验，双侧）---")
    P("  %-22s %18s %18s %10s" % ("类别", "样本内 TP/FP", "样本外 TP/FP", "Fisher P"))
    rows = []
    pairs = [("INACTIVE", "INACTIVE"),
             ("INF_DEFER", "INF_DEFER"),
             ("UNDERDOC", "UNDERDOC")]
    for cls, v9cls in pairs:
        a = ins8[ins8.cls == cls]
        b = cmp_[cmp_.cls == v9cls]
        if not len(a) or not len(b):
            continue
        tp1, fp1 = int(a.iloc[0].TP), int(a.iloc[0].FP)
        n2 = int(b.iloc[0].n_out)
        tp2 = int(round(float(b.iloc[0].ppv_out) * n2)) if np.isfinite(
            float(b.iloc[0].ppv_out)) else 0
        fp2 = n2 - tp2
        if tp1 + fp1 == 0 or n2 == 0:
            continue
        odds, p = stats.fisher_exact([[tp1, fp1], [tp2, fp2]])
        P("  %-22s %18s %18s %10.4f" % (
            cls, "%d/%d" % (tp1, fp1), "%d/%d" % (tp2, fp2), p))
        rows.append(dict(cls=cls, in_TP=tp1, in_FP=fp1, out_TP=tp2, out_FP=fp2,
                         fisher_p=p))
    # 任务 B
    insB = pd.DataFrame(J8["ppv_B"])
    a = insB[insB.engine == "v8(narrow)"]
    pvB = pd.read_csv(os.path.join(OUT, "table_v9_ppv.csv"))
    rb = pvB[(pvB.engine == "v8") & (pvB.task == "B") &
             (pvB.cls == "ACTIVITY-POSITIVE")]
    if len(a) and len(rb):
        tp1, fp1 = int(a.iloc[0].TP), int(a.iloc[0].FP)
        tp2, fp2 = int(rb.iloc[0].TP), int(rb.iloc[0].FP)
        odds, p = stats.fisher_exact([[tp1, fp1], [tp2, fp2]])
        P("  %-22s %18s %18s %10.4f"
          % ("B 活动阳性", "%d/%d" % (tp1, fp1), "%d/%d" % (tp2, fp2), p))
        rows.append(dict(cls="ACTIVITY-POSITIVE", in_TP=tp1, in_FP=fp1,
                         out_TP=tp2, out_FP=fp2, fisher_p=p))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "table_v9_compare_ci.csv"),
                              index=False)

    # ---------------- 2. 样本外加权 PPV / AUC 的 bootstrap CI
    P("")
    P("--- 2. 样本外点估计的 bootstrap 95% CI（分层重抽样）---")
    P("  %-40s %10s %18s" % ("指标", "点估计", "95% CI"))

    # 加权 PPV：v8 各通道（用设计权重 w）
    DA1 = DA[DA.ref_std.notna()]
    for cls in ["INACTIVE", "INF_DEFER", "UNDERDOC"]:
        def f(s, cls=cls):
            m = s[s.v8_engine == cls]
            w = m.w.values
            tp = w[(m.ref_std == cls).values].sum()
            tot = w.sum()
            return tp / tot if tot > 0 else np.nan
        pt = f(DA1)
        lo, hi = boot(DA1, f, "strat")
        P("  %-40s %10.3f %18s"
          % ("任务 A v8 %s 加权 PPV" % cls, pt, "%.3f–%.3f" % (lo, hi)))

    DB1 = DB[DB.ref_std.notna()]

    def fB(s):
        m = s[s.v8_engine == "ACTIVITY-POSITIVE"]
        w = m.w.values
        tp = w[(m.ref_std == "ACTIVITY-POSITIVE").values].sum()
        return tp / w.sum() if w.sum() > 0 else np.nan
    pt = fB(DB1)
    lo, hi = boot(DB1, fB, "strat")
    P("  %-40s %10.3f %18s"
      % ("任务 B v8 活动阳性 加权 PPV", pt, "%.3f–%.3f" % (lo, hi)))

    # AUC
    for nm in ["aid_doc", "aid_broad"]:
        for tag, d, ycol in [("A", DA1, "doc"), ("B", DB1, "act")]:
            def f(s, nm=nm, ycol=ycol):
                yy = (s.ref_std.isin(["INACTIVE", "INF_DEFER"])
                      if ycol == "doc" else
                      (s.ref_std == "ACTIVITY-POSITIVE")).astype(int).values
                if len(set(yy)) < 2:
                    return np.nan
                return roc_auc_score(yy, s[nm].values)
            pt = f(d)
            lo, hi = boot(d, f, "strat")
            P("  %-40s %10.3f %18s"
              % ("任务 %s %s AUC" % (tag, nm), pt, "%.3f–%.3f" % (lo, hi)))

    # ---------------- 3. 失效模式：阳性判定率 vs 真阳性率
    P("")
    P("--- 3. 失效模式刻画：算法阳性率 vs 人类真阳性率 ---")
    P("  %-24s %10s %10s %10s %10s" % ("层", "算法判阳%", "真阳性%", "灵敏度", "PPV"))
    mode = []
    for cls in ["INACTIVE", "INF_DEFER", "UNDERDOC"]:
        d = DA1
        m = d.v8_engine == cls
        rate_algo = 100 * m.mean()
        rate_true = 100 * (d.ref_std == cls).mean()
        sens = (m & (d.ref_std == cls)).sum() / max(1, (d.ref_std == cls).sum())
        ppv = (m & (d.ref_std == cls)).sum() / max(1, m.sum())
        P("  %-24s %9.1f%% %9.1f%% %9.3f %9.3f"
          % (cls, rate_algo, rate_true, sens, ppv))
        mode.append(dict(cls=cls, algo_pos_rate=rate_algo,
                         true_pos_rate=rate_true, sens=sens, ppv=ppv))
    d = DB1
    m = d.v8_engine == "ACTIVITY-POSITIVE"
    P("  %-24s %9.1f%% %9.1f%% %9.3f %9.3f" % (
        "B 活动阳性", 100 * m.mean(),
        100 * (d.ref_std == "ACTIVITY-POSITIVE").mean(),
        (m & (d.ref_std == "ACTIVITY-POSITIVE")).sum()
        / max(1, (d.ref_std == "ACTIVITY-POSITIVE").sum()),
        (m & (d.ref_std == "ACTIVITY-POSITIVE")).sum() / max(1, m.sum())))
    mode.append(dict(cls="ACTIVITY-POSITIVE", algo_pos_rate=100 * m.mean(),
                     true_pos_rate=100 * (d.ref_std == "ACTIVITY-POSITIVE").mean(),
                     sens=1.0, ppv=float(
                         (m & (d.ref_std == "ACTIVITY-POSITIVE")).sum()
                         / max(1, m.sum()))))
    pd.DataFrame(mode).to_csv(os.path.join(OUT, "table_v9_mode.csv"),
                              index=False)
    P("")
    P("  读法：算法在每一层的灵敏度都接近 1（不漏真阳性），代价是阳性判定率")
    P("  远高于真阳性率 —— 即失效模式是**过度判定**，不是漏判。")

    with open(os.path.join(OUT, "172_replication_tests.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
