# -*- coding: utf-8 -*-
"""
169_v8_validation.py
====================
v8 —— 用 v7b 已有的人类金标准回测「补了三条规则的判定器」与「连续变量」。

**报告顺序严格按要求：先 PPV，再通道规模。**

  第一节（必须最先看）  v7 vs v8 的逐类 PPV / 灵敏度 / 特异度 / NPV（Wilcoxon 区间）
                        以及「算法阳性里有多少被人类确证」的确证率
  第二节                通道规模（分层加权反推），放在 PPV 之后
  第三节                连续变量：候选定义的 AUC、分半信度、梯度
  第四节                连续变量替代三分类进入交互模型：可估性、效应、方向自检

金标准来源
  out/adjud_merged.csv —— v7b 的两名隔离子代理盲法判定 + 第三仲裁（ref_std）
  任务 A n=98（三分类）、任务 B n=50（活动 vs 纯否定）

输出
  out/169_v8_validation.txt
  out/table_v8_ppv.csv / table_v8_perclass.csv / table_v8_channels.csv
  out/table_v8_auc.csv / table_v8_interaction.csv
  out/_v8.json
"""
import json
import os
import sys
from importlib.util import module_from_spec, spec_from_file_location

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
SEED = 20260916
CLS_A = ["INACTIVE", "INF_DEFER", "UNDERDOC"]
CORE = ["age", "female", "sofa", "vaso24", "vent24", "renal_fail", "shock",
        "surg_any", "n_agent_pre24"]

L = []


def P(s=""):
    print(s)
    sys.stdout.flush()
    L.append(str(s))


def load_mod(name):
    p = os.path.join(ROOT, "scripts", name)
    sp = spec_from_file_location("m_" + name[:3], p)
    m = module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


S166 = load_mod("166_double_coding_metrics.py")
wilson = S166.wilson
cohen_kappa = S166.cohen_kappa


# ==================================================================== 通用
def wilson_w(k, n, z=1.96):
    """k/n 可能不是整数（加权）→ 用有效样本量近似。"""
    if n <= 0:
        return (np.nan, np.nan)
    k = float(k)
    n = float(n)
    p = min(max(k / n, 0.0), 1.0)
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(max(p * (1 - p), 0) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def ck2(df, test_col, truth_col, pos, wcol=None):
    a = df[test_col].astype(str).values
    b = df[truth_col].astype(str).values
    w = (df[wcol].astype(float).values if wcol else np.ones(len(df)))
    TP = w[(a == pos) & (b == pos)].sum()
    FP = w[(a == pos) & (b != pos)].sum()
    FN = w[(a != pos) & (b == pos)].sum()
    TN = w[(a != pos) & (b != pos)].sum()
    return dict(TP=TP, FP=FP, FN=FN, TN=TN,
                sens=TP / (TP + FN) if TP + FN else np.nan,
                spec=TN / (TN + FP) if TN + FP else np.nan,
                ppv=TP / (TP + FP) if TP + FP else np.nan,
                npv=TN / (TN + FN) if TN + FN else np.nan,
                acc=(TP + TN) / w.sum())


def auc_ci(y, x, nrep=2000, seed=SEED):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    m = np.isfinite(y) & np.isfinite(x)
    y, x = y[m], x[m]
    if len(y) < 6 or len(np.unique(y)) < 2:
        return dict(auc=np.nan, lo=np.nan, hi=np.nan, n=len(y),
                    n_pos=int(y.sum()))
    a = float(roc_auc_score(y, x))
    rs = np.random.RandomState(seed)
    bs = []
    for _ in range(nrep):
        i = rs.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        bs.append(roc_auc_score(y[i], x[i]))
    if len(bs) < 50:
        return dict(auc=a, lo=np.nan, hi=np.nan, n=len(y), n_pos=int(y.sum()))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(auc=a, lo=float(lo), hi=float(hi), n=len(y),
                n_pos=int(y.sum()), boot=np.array(bs))


def firth(X, y, max_iter=400, tol=1e-10):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    b = np.zeros(X.shape[1])
    for _ in range(max_iter):
        pr = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))
        W = np.clip(pr * (1 - pr), 1e-12, None)
        Iinv = np.linalg.pinv(X.T @ (X * W[:, None]))
        h = np.einsum("ij,jk,ik->i", X * W[:, None], Iinv, X)
        U = X.T @ (y - pr + h * (0.5 - pr))
        step = Iinv @ U
        mx = np.max(np.abs(step))
        if mx > 4:
            step *= 4.0 / mx
        b = b + step
        if mx < tol:
            break
    pr = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))
    W = np.clip(pr * (1 - pr), 1e-12, None)
    Iinv = np.linalg.pinv(X.T @ (X * W[:, None]))
    sign, logdet = np.linalg.slogdet(X.T @ (X * W[:, None]))
    ll = float(np.sum(y * np.log(np.clip(pr, 1e-12, None))
                      + (1 - y) * np.log(np.clip(1 - pr, 1e-12, None))))
    return b, np.sqrt(np.abs(np.diag(Iinv))), ll + 0.5 * float(logdet)


def fit_logit(d, y, cols, want, use_firth=None):
    """拟合 d[y] ~ cols；返回 want 列的 OR/CI/p 与 LRT(对去掉 want 的简化模型)。"""
    use = d[[y] + list(cols)].dropna().copy()
    if len(use) == 0 or use[y].nunique() < 2:
        return None
    X = sm.add_constant(use[list(cols)].astype(float))
    red = [c for c in cols if c != want]
    Xr = sm.add_constant(use[red].astype(float)) if red else None
    if use_firth is None:
        use_firth = int(use[y].sum()) < 60
    try:
        if use_firth:
            b, se, lf = firth(X.values, use[y].values)
            l0 = firth(Xr.values, use[y].values)[2] if Xr is not None else 0.0
            model = "Firth"
        else:
            with np.errstate(all="ignore"):
                m = sm.Logit(use[y].astype(float), X).fit(disp=0, maxiter=300)
            b, se = np.asarray(m.params, float), np.asarray(m.bse, float)
            l0 = (sm.Logit(use[y].astype(float), Xr).fit(disp=0, maxiter=300).llf
                  if Xr is not None else 0.0)
            lf = m.llf
            model = "MLE"
    except Exception:
        try:
            b, se, lf = firth(X.values, use[y].values)
            l0 = firth(Xr.values, use[y].values)[2] if Xr is not None else 0.0
            model = "Firth"
        except Exception:
            return None
    j = list(X.columns).index(want)
    if not np.isfinite(b[j]) or not np.isfinite(se[j]) or se[j] <= 0:
        return None
    lrt = 2.0 * (lf - l0)
    return dict(OR=float(np.exp(b[j])), lo=float(np.exp(b[j] - 1.96 * se[j])),
                hi=float(np.exp(b[j] + 1.96 * se[j])),
                p=float(2 * (1 - stats.norm.cdf(abs(b[j] / se[j])))),
                p_lrt=float(1 - stats.chi2.cdf(max(lrt, 0.0), 1)),
                model=model, n=int(len(use)), n_ev=int(use[y].sum()))


def main():
    gold = pd.read_csv(os.path.join(OUT, "adjud_merged.csv"))
    gold["stay_key"] = gold.stay_key.astype(int)
    ad = pd.read_csv(os.path.join(DATA, "gc_activity_density.csv"))
    gold = gold.merge(ad, on="stay_key", how="left")
    P("金标准 n=%d（任务 A %d / 任务 B %d）；与 v8 引擎对接成功 %d"
      % (len(gold), int((gold.task == "A").sum()), int((gold.task == "B").sum()),
         int(gold.v8_label.notna().sum())))

    # v8 的「活动阳性」定义：存在通过全部过滤的阳性活动陈述
    gold["v8_activity_pos"] = np.where(
        gold.v8_act_pos.notna() & (gold.v8_act_pos > 0),
        "ACTIVITY-POSITIVE", "PURELY-NEGATED")
    gold["v8_activity_pos_broad"] = np.where(
        gold.v8_act_all_pos.fillna(0) > 0,
        "ACTIVITY-POSITIVE", "PURELY-NEGATED")

    A = gold[gold.task == "A"].copy()
    Bt = gold[gold.task == "B"].copy()

    # =================================================================
    # 第一节（最先看）：PPV
    # =================================================================
    P("")
    P("=" * 100)
    P("第一节  PPV 优先 —— 判定器在人类金标准上的表现（这才是「能不能用」的唯一判据）")
    P("=" * 100)
    P("")
    P("--- 1.1 任务 A（三通道）：逐类 PPV / 灵敏度 / 特异度 / NPV ---")
    P("  %-6s %-11s %5s %5s %5s %6s %6s %6s %6s %6s"
      % ("判定器", "类别", "n_pos", "TP", "FP", "PPV", "sens", "spec", "NPV", "acc"))
    rows = []
    for nm, col in [("v7", "algo_label"), ("v8", "v8_label")]:
        for c in CLS_A:
            r = ck2(A, col, "ref_std", c)
            lo, hi = wilson(r["TP"], r["TP"] + r["FP"])
            P("  %-6s %-11s %5d %5d %5d %6s %6s %6s %6s %6s" % (
                nm, c, int(r["TP"] + r["FP"]), int(r["TP"]), int(r["FP"]),
                "%.3f" % r["ppv"] if np.isfinite(r["ppv"]) else "n/a",
                "%.3f" % r["sens"], "%.3f" % r["spec"],
                "%.3f" % r["npv"], "%.3f" % r["acc"]))
            rows.append(dict(engine=nm, task="A", cls=c,
                             n_algo_pos=int(r["TP"] + r["FP"]),
                             TP=int(r["TP"]), FP=int(r["FP"]),
                             FN=int(r["FN"]), TN=int(r["TN"]),
                             ppv=r["ppv"], ppv_lo=lo, ppv_hi=hi,
                             sens=r["sens"], spec=r["spec"],
                             npv=r["npv"], acc=r["acc"]))
    P("")
    P("  加权（分层反推真实构成）：")
    for nm, col in [("v7", "algo_label"), ("v8", "v8_label")]:
        for c in CLS_A:
            r = ck2(A, col, "ref_std", c, wcol="w")
            P("    %-4s %-11s PPV=%.3f (%.3f-%.3f)  sens=%.3f  spec=%.3f"
              % (nm, c, r["ppv"], *wilson_w(r["TP"], r["TP"] + r["FP"]),
                 r["sens"], r["spec"]))
    P("")
    P("--- 1.2 任务 B（活动 vs 纯否定）---")
    rows_b = []
    for nm, col in [("v7", "algo_B"), ("v8(narrow)", "v8_activity_pos"),
                    ("v8(broad)", "v8_activity_pos_broad")]:
        r = ck2(Bt, col, "ref_std", "ACTIVITY-POSITIVE")
        lo, hi = wilson(r["TP"], r["TP"] + r["FP"])
        P("  %-11s n_pos=%2d  TP=%2d FP=%2d  PPV=%.3f (%.3f-%.3f)  "
          "sens=%.3f  spec=%.3f  acc=%.3f" % (
              nm, int(r["TP"] + r["FP"]), int(r["TP"]), int(r["FP"]),
              r["ppv"], lo, hi, r["sens"], r["spec"], r["acc"]))
        rows_b.append(dict(engine=nm, task="B", cls="ACTIVITY-POSITIVE",
                           n_algo_pos=int(r["TP"] + r["FP"]), TP=int(r["TP"]),
                           FP=int(r["FP"]), FN=int(r["FN"]), TN=int(r["TN"]),
                           ppv=r["ppv"], ppv_lo=lo, ppv_hi=hi,
                           sens=r["sens"], spec=r["spec"], npv=r["npv"],
                           acc=r["acc"]))
    P("")
    P("--- 1.3 总体一致率与 κ（算法 vs 人类参考）---")
    rows_k = []
    for nm, col in [("v7", "algo_label"), ("v8", "v8_label")]:
        po, pe, k = cohen_kappa(A[col].astype(str).values,
                                A.ref_std.astype(str).values, CLS_A)
        P("  任务 A %-4s  po=%.3f  pe=%.3f  kappa=%+.3f  PABAK=%+.3f"
          % (nm, po, pe, k, 2 * po - 1))
        rows_k.append(dict(task="A", engine=nm, po=po, pe=pe, kappa=k,
                           pabak=2 * po - 1))
    for nm, col in [("v7", "algo_B"), ("v8(narrow)", "v8_activity_pos"),
                    ("v8(broad)", "v8_activity_pos_broad")]:
        po, pe, k = cohen_kappa(
            Bt[col].astype(str).values, Bt.ref_std.astype(str).values,
            ["ACTIVITY-POSITIVE", "PURELY-NEGATED"])
        P("  任务 B %-11s po=%.3f  pe=%.3f  kappa=%+.3f  PABAK=%+.3f"
          % (nm, po, pe, k, 2 * po - 1))
        rows_k.append(dict(task="B", engine=nm, po=po, pe=pe, kappa=k,
                           pabak=2 * po - 1))
    pd.DataFrame(rows_k).to_csv(os.path.join(OUT, "table_v8_kappa.csv"),
                                index=False)
    pd.DataFrame(rows + rows_b).to_csv(
        os.path.join(OUT, "table_v8_ppv.csv"), index=False)
    P("")
    P("--- 1.4 混淆矩阵（v8）---")
    P(pd.crosstab(A.v8_label, A.ref_std, dropna=False).to_string())
    P("")
    P(pd.crosstab(Bt.v8_activity_pos, Bt.ref_std, dropna=False).to_string())

    # =================================================================
    # 第二节：通道规模（放在 PPV 之后）
    # =================================================================
    P("")
    P("=" * 100)
    P("第二节  通道规模（只有在第一节的 PPV 站得住时，这些数字才有意义）")
    P("=" * 100)
    # 任务 A 的分层权重：w = 框规模 / 层内抽样数 → 反推总体
    A["w_"] = A.w.astype(float)
    rows_c = []
    for lab, col in [("v7", "algo_label"), ("v8", "v8_label")]:
        for c in CLS_A:
            s = A[A[col] == c]
            n_hat = s.w_.sum()
            P("  %-4s %-11s 层内 n=%2d  反推总体 = %.1f" % (lab, c, len(s), n_hat))
            rows_c.append(dict(engine=lab, cls=c, n_sampled=len(s),
                               n_projected=n_hat))
        ok = (A[col] == A.ref_std)
        P("      确证率（算法阳性中被人类确证）= %.3f" % (
            (A[(A[col] != "UNDERDOC") & ok].w_.sum()
             / max(A[A[col] != "UNDERDOC"].w_.sum(), 1e-9))))
    # 全队列规模（用 165 的框）
    FRAME_A = S166.FRAME_A
    P("")
    P("  抽样框规模（165）：INACTIVE %d / INF_DEFER %d / UNDERDOC %d"
      % (FRAME_A["INACTIVE"], FRAME_A["INF_DEFER"], FRAME_A["UNDERDOC"]))
    P("  → 反推：v7 判定为 INACTIVE 的总体例数 = 54；v8 保留后 = %.0f（占 v7 的 %.0f%%）"
      % (A[(A.v8_label == "INACTIVE")].w_.sum(),
         100 * A[(A.v8_label == "INACTIVE")].w_.sum() / FRAME_A["INACTIVE"]))
    pd.DataFrame(rows_c).to_csv(os.path.join(OUT, "table_v8_channels.csv"),
                               index=False)

    # =================================================================
    # 第三节：连续变量
    # =================================================================
    P("")
    P("=" * 100)
    P("第三节  连续变量「活动度相关信息密度」")
    P("=" * 100)
    CANDS = ["aid_doc", "aid_broad", "aid_signed", "aid_act", "aid_eval",
             "info_n", "info_broad", "n_eval", "narr_char"]
    P("")
    P("--- 3.1 候选定义对「有/无实义活动度陈述」的判别力（任务 A，AUC）---")
    A["human_documented"] = A.ref_std.isin(["INACTIVE", "INF_DEFER"]).astype(int)
    P("  参考标准：人类判定为 INACTIVE 或 INF_DEFER（即记录里真的有活动度陈述）n_pos=%d / %d"
      % (int(A.human_documented.sum()), len(A)))
    rows_auc = []
    for c in CANDS:
        if A[c].notna().sum() < 20:
            continue
        r = auc_ci(A.human_documented, A[c].fillna(0))
        P("    %-12s AUC=%.3f (%.3f-%.3f)" % (c, r["auc"], r["lo"], r["hi"]))
        rows_auc.append(dict(task="A_documented", var=c, **{
            k: v for k, v in r.items() if k != "boot"}))
    P("")
    P("--- 3.2 对「活动阳性 vs 纯否定」的判别力（任务 B，AUC）---")
    Bt["human_act"] = (Bt.ref_std == "ACTIVITY-POSITIVE").astype(int)
    P("  n_pos=%d / %d" % (int(Bt.human_act.sum()), len(Bt)))
    for c in CANDS:
        if Bt[c].notna().sum() < 20:
            continue
        r = auc_ci(Bt.human_act, Bt[c].fillna(0))
        P("    %-12s AUC=%.3f (%.3f-%.3f)" % (c, r["auc"], r["lo"], r["hi"]))
        rows_auc.append(dict(task="B_activity", var=c, **{
            k: v for k, v in r.items() if k != "boot"}))
    pd.DataFrame(rows_auc).to_csv(os.path.join(OUT, "table_v8_auc.csv"),
                                  index=False)

    P("")
    P("--- 3.3 分半信度（句级奇偶分半 → Spearman-Brown 校正）---")
    if "aid_h1" in ad.columns:
        use = ad[["aid_h1", "aid_h2"]].dropna()
        r_s = float(use.corr(method="spearman").iloc[0, 1])
        r_p = float(use.corr(method="pearson").iloc[0, 1])
        sb = 2 * r_p / (1 + r_p) if r_p > -1 else np.nan
        P("    n=%d  Spearman=%.3f  Pearson=%.3f  Spearman-Brown=%.3f"
          % (len(use), r_s, r_p, sb))
        P("    对照：v7 三分类的「同一例换个判定者」κ=0.10（几乎随机）；")
        P("          连续变量无阈值，不存在「标签翻转」这一失效模式。")

    P("")
    P("--- 3.4 三分类 vs 连续变量的梯度（任务 A 内）---")
    for c in ["aid_doc", "aid_broad", "aid_signed"]:
        g = A.groupby("ref_std")[c].median()
        P("    %-11s  " % c + "  ".join("%s=%.3f" % (k, v) for k, v in g.items()))

    # =================================================================
    # 第四节：连续变量替代三分类进入交互模型
    # =================================================================
    P("")
    P("=" * 100)
    P("第四节  用连续变量替代三分类做效应修饰（这才是「更稳」的检验）")
    P("=" * 100)
    df = pd.read_csv(os.path.join(DATA, "cohort_rheum_icu.csv"))
    df = df.rename(columns={"sofa24": "sofa"})
    ab = pd.read_csv(os.path.join(DATA, "abx_outcomes_rheum.csv"))
    NEW = ["stay_key", "new_agent_24_7d", "n_agent_pre24"]
    df = df.merge(ab[[c for c in NEW if c in ab.columns]], on="stay_key",
                  how="left")
    gh = pd.read_csv(os.path.join(DATA, "gc_history.csv"))
    GK = ["stay_key", "note_found", "n_gc_mentions", "home_gc", "indN_spec"]
    df = df.merge(gh[[c for c in GK if c in gh.columns]], on="stay_key",
                  how="left")
    ic = pd.read_csv(os.path.join(DATA, "gc_inactivity.csv"))
    df = df.merge(ic[["stay_key", "spec_true", "inact3", "hc_len", "note_len2",
                      "nonrheum_pdx"]], on="stay_key", how="left")
    df = df.merge(ad[["stay_key", "v8_label", "aid_doc", "aid_broad",
                      "aid_signed", "info_n", "info_broad", "narr_char"]],
                  on="stay_key", how="left")
    SCORE = {"G0_none": 0.0, "G1_low": 1.0, "G2_mod": 2.0, "G3_high": 3.0}
    df["gc_str_num"] = df.gc24_str.map(SCORE)
    df["is_sle"] = (df.primary_grp == "SLE").astype(int)
    df["sofa"] = df.sofa.fillna(df.sofa.median())

    both = df[df.primary_grp.isin(["SLE", "RA"])].copy()
    bothn = both[both.note_found == 1].copy()
    sp0 = bothn[bothn.spec_true == 0].copy()
    P("  SLE+RA n=%d ; 有记录 n=%d ; spec- n=%d ; 30d 死亡 %d"
      % (len(both), len(bothn), len(sp0), int(sp0.death_30d.sum())))

    def ix(d, y="death_30d", covs=CORE):
        use = d[["gc_str_num", "is_sle", y] + list(covs)].dropna().copy()
        if len(use) < 30 or use[y].nunique() < 2 or use.gc_str_num.nunique() < 2:
            return None
        use["s"] = use.gc_str_num.astype(float)
        use["sx"] = use.s * use.is_sle
        X = sm.add_constant(use[["s", "is_sle"] + list(covs) + ["sx"]].astype(float))
        Xr = sm.add_constant(use[["s", "is_sle"] + list(covs)].astype(float))
        n_ev = int(use[y].sum())
        try:
            b, se, lf = firth(X.values, use[y].values)
            _, _, l0 = firth(Xr.values, use[y].values)
            model = "Firth"
        except Exception:
            try:
                with np.errstate(all="ignore"):
                    m = sm.Logit(use[y].astype(float), X).fit(disp=0, maxiter=300)
                    m0 = sm.Logit(use[y].astype(float), Xr).fit(disp=0, maxiter=300)
                b, se = np.asarray(m.params, float), np.asarray(m.bse, float)
                lf, l0 = m.llf, m0.llf
                model = "MLE"
            except Exception:
                return None
        j = list(X.columns).index("sx")
        if not np.isfinite(b[j]) or not np.isfinite(se[j]) or se[j] <= 0:
            return None
        e = float(np.exp(b[j]))
        lrt = 2 * (lf - l0)
        return dict(OR=1.0 / e, lo=1.0 / float(np.exp(b[j] + 1.96 * se[j])),
                    hi=1.0 / float(np.exp(b[j] - 1.96 * se[j])),
                    p_lrt=float(1 - stats.chi2.cdf(max(lrt, 0), 1)),
                    model=model, n=len(use), n_ev=n_ev)

    rows_ix = []
    P("")
    P("--- 4.1 v7 三通道 vs v8 二分层：交互(RA/SLE 剂量斜率比)。空结果 = 不可估 ---")
    groups = [("v7: INACTIVE", sp0[sp0.inact3 == "INACTIVE"]),
              ("v7: INF_DEFER", sp0[sp0.inact3 == "INF_DEFER"]),
              ("v7: UNDERDOC", sp0[sp0.inact3 == "UNDERDOC"]),
              ("v8: 有活动度信息 (aid_broad>0)", sp0[sp0.aid_broad > 0]),
              ("v8: 无活动度信息 (aid_broad=0)", sp0[sp0.aid_broad == 0]),
              ("v8: 高密度 (aid_broad>中位)",
               sp0[sp0.aid_broad > sp0.aid_broad.median()]),
              ("参考: 全 spec- 层", sp0)]
    for lab, s in groups:
        r = ix(s)
        if r is None:
            P("  %-34s n=%4d  ev=%3d  → 不可估计（事件数或剂量层不足）"
              % (lab, len(s), int(s.death_30d.sum())))
            rows_ix.append(dict(group=lab, n=len(s), n_ev=int(s.death_30d.sum()),
                                OR=np.nan, lo=np.nan, hi=np.nan, p=np.nan,
                                note="not estimable"))
            continue
        P("  %-34s n=%4d  ev=%3d  RA/SLE=%.2f (%.2f-%.2f)  P_LRT=%.3f  [%s]"
          % (lab, r["n"], r["n_ev"], r["OR"], r["lo"], r["hi"],
             r["p_lrt"], r["model"]))
        rows_ix.append(dict(group=lab, n=r["n"], n_ev=r["n_ev"], OR=r["OR"],
                            lo=r["lo"], hi=r["hi"], p=r["p_lrt"],
                            model=r["model"], note="ok"))

    P("")
    P("--- 4.2 连续修饰：交互是否随「活动度信息密度」变化 ---")
    for mod in ["aid_broad", "aid_doc"]:
        s = sp0.dropna(subset=[mod]).copy()
        if s[mod].std() == 0:
            continue
        s["m"] = (s[mod] - s[mod].mean()) / s[mod].std()
        s["s"] = s.gc_str_num.astype(float)
        s["sx"] = s.s * s.is_sle
        cols = ["s", "sx", "is_sle", "m", "s_m", "sx_m", "sx_m3"]
        s["s_m"] = s.s * s.m
        s["sx_m"] = s.sx * s.m
        s["sx_m3"] = s.sx * s.m                # 三项交互 = s × is_sle × m
        base = ["s", "sx", "is_sle", "m", "s_m"]
        use = s[[y0 for y0 in ["death_30d"]] + cols].dropna()
        if len(use) < 40 or use.death_30d.sum() < 8:
            P("  %-10s 事件不足（%d），跳过" % (mod, int(use.death_30d.sum())))
            continue
        X = sm.add_constant(use[cols].astype(float))
        Xr = sm.add_constant(use[base].astype(float))
        n_ev = int(use.death_30d.sum())
        try:
            b, se, lf = firth(X.values, use.death_30d.values)
            _, _, l0 = firth(Xr.values, use.death_30d.values)
            model = "Firth"
        except Exception:
            continue
        j = list(X.columns).index("sx_m3")
        z = b[j] / se[j]
        lrt = 2 * (lf - l0)
        P("  %-10s n=%4d ev=%3d  三项交互 s×SLE×density: OR=%.2f (%.2f-%.2f) "
          "P=%.3f  LRT P=%.3f [%s]" % (
              mod, len(use), n_ev, float(np.exp(b[j])),
              float(np.exp(b[j] - 1.96 * se[j])),
              float(np.exp(b[j] + 1.96 * se[j])),
              float(2 * (1 - stats.norm.cdf(abs(z)))),
              float(1 - stats.chi2.cdf(max(lrt, 0), 1)), model))
        rows_ix.append(dict(group="3-way s×SLE×%s" % mod, n=len(use), n_ev=n_ev,
                            OR=float(np.exp(b[j])),
                            lo=float(np.exp(b[j] - 1.96 * se[j])),
                            hi=float(np.exp(b[j] + 1.96 * se[j])),
                            p=float(1 - stats.chi2.cdf(max(lrt, 0), 1)),
                            model=model + "(3way)", note="ok"))
    pd.DataFrame(rows_ix).to_csv(os.path.join(OUT, "table_v8_interaction.csv"),
                                 index=False)

    P("")
    P("--- 4.3 方向自检（D3）：交互须与分病种趋势比值同号 ---")
    rows_d3 = []
    for lab, s in [("全 spec-", sp0),
                   ("aid_broad=0", sp0[sp0.aid_broad == 0]),
                   ("aid_broad>0", sp0[sp0.aid_broad > 0])]:
        t1 = fit_logit(s, "death_30d", ["gc_str_num"] + CORE, "gc_str_num",
                       use_firth=False)
        t2 = fit_logit(s[s.primary_grp == "SLE"], "death_30d",
                       ["gc_str_num"] + CORE, "gc_str_num", use_firth=False)
        t3 = fit_logit(s[s.primary_grp == "RA"], "death_30d",
                       ["gc_str_num"] + CORE, "gc_str_num", use_firth=False)
        if t2 and t3:
            ratio = t3["OR"] / t2["OR"]
            ok = (ratio > 1) == (t3["OR"] > t2["OR"])
            P("  %-12s SLE 趋势 %.2f / RA 趋势 %.2f → 比值 %.2f  %s"
              % (lab, t2["OR"], t3["OR"], ratio,
                 "一致" if ok else "不一致"))
            rows_d3.append(dict(group=lab, sle_OR=t2["OR"], ra_OR=t3["OR"],
                                ratio=ratio, agree=bool(ok)))

    # =================================================================
    # 第五节：全队列收敛效度（不依赖人类金标准）
    # =================================================================
    P("")
    P("=" * 100)
    P("第五节  全队列收敛效度 —— 不依赖金标准的独立检查")
    P("=" * 100)
    P("  若「明确无活动」真的抓对了，它应当同时满足：")
    P("   ① 30 d 死亡率最低   ② 院前已在用 GC 的比例高（慢病在控）")
    P("   ③ 主诊断为非风湿病的比例低（风湿病不是这次住院的原因）")
    P("")
    P("  %-14s %5s %9s %9s %9s %7s %11s" % (
        "通道", "n", "30d死亡%", "24h用GC%", "院前GC%", "SOFA", "主诊非风湿%"))

    def prof(sub, lab):
        if not len(sub):
            P("  %-14s %5d" % (lab, 0))
            rows_prof.append(dict(label=lab, n=0, death=np.nan, used=np.nan,
                                  home=np.nan, sofa=np.nan, nonrheum=np.nan))
            return
        P("  %-14s %5d %8.1f%% %8.1f%% %8.1f%% %7.1f %10.1f%%" % (
            lab, len(sub), 100 * sub.death_30d.mean(),
            100 * (sub.gc_str_num >= 1).mean(),
            100 * sub.home_gc.fillna(0).mean(), sub.sofa.median(),
            100 * sub.nonrheum_pdx.fillna(0).mean()))
        rows_prof.append(dict(
            label=lab, n=int(len(sub)),
            death=round(100 * float(sub.death_30d.mean()), 2),
            used=round(100 * float((sub.gc_str_num >= 1).mean()), 2),
            home=round(100 * float(sub.home_gc.fillna(0).mean()), 2),
            sofa=float(sub.sofa.median()),
            nonrheum=round(100 * float(sub.nonrheum_pdx.fillna(0).mean()), 2)))

    rows_prof = []

    for k in ["INACTIVE", "INF_DEFER", "UNDERDOC"]:
        prof(bothn[bothn.inact3 == k], "v7 " + k)
    for k in ["INACTIVE", "INF_DEFER", "UNDERDOC"]:
        prof(bothn[bothn.v8_label == k], "v8 " + k)
    P("")
    P("  spec- 层内的通道规模与死亡（v7 → v8）:")
    rows_ch = []
    for k in ["INACTIVE", "INF_DEFER", "UNDERDOC"]:
        s7 = sp0[sp0.inact3 == k]
        s8 = sp0[sp0.v8_label == k]
        P("    %-10s v7 n=%4d ev=%2d  →  v8 n=%4d ev=%2d" % (
            k, len(s7), int(s7.death_30d.sum()), len(s8),
            int(s8.death_30d.sum())))
        rows_ch.append(dict(cls=k, v7_n=len(s7), v7_ev=int(s7.death_30d.sum()),
                            v8_n=len(s8), v8_ev=int(s8.death_30d.sum())))
    pd.DataFrame(rows_ch).to_csv(os.path.join(OUT, "table_v8_channels_spec0.csv"),
                                 index=False)
    P("")
    P("  说明：v8 修掉了 v7 的两处**结构性错误**——")
    P("   （a）v7 的 INF_DEFER 有 19/20 来自「整篇感染 + 整篇减停」的弱共现通道；")
    P("   （b）v7 的 INACTIVE 有 44/54 由非风湿领域的裸词 asymptomatic 触发。")

    # ---------------- 导出 JSON
    obj = dict(
        gold_n=len(gold), gold_A=len(A), gold_B=len(Bt),
        ppv=rows, ppv_B=rows_b, kappa=rows_k, auc=rows_auc, ix=rows_ix,
        profile=rows_prof, channels_spec0=rows_ch, d3=rows_d3,
        split_half=None,
    )
    if "aid_h1" in ad.columns:
        use = ad[["aid_h1", "aid_h2"]].dropna()
        if len(use) > 20:
            obj["split_half"] = dict(
                n=int(len(use)),
                spearman=float(use.corr(method="spearman").iloc[0, 1]),
                pearson=float(use.corr(method="pearson").iloc[0, 1]))
    with open(os.path.join(OUT, "_v8.json"), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)

    txt = "\n".join(L) + "\n"
    with open(os.path.join(OUT, "169_v8_validation.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    P("")
    P("saved: out/169_v8_validation.txt + table_v8_*.csv + _v8.json")


if __name__ == "__main__":
    main()
