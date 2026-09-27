# -*- coding: utf-8 -*-
"""
155_head_to_head_v5.py

v5：主结局换成「新起始广谱抗菌药」，并把模型与多重比较做成稿件可直接用的形式。

为什么要换主结局（以及换的是什么）
----------------------------------
v4 的主结局是血培养阳性 >24 h —— 在 SLE+RA 头对头里只有 87 个事件，脆弱性
= 3 个事件。但它**没有被换掉**的替代品：v2 记录的"新广谱抗生素 243 事件"
其实是

    MAX(CASE WHEN starttime >= intime + 48 h THEN 1 ELSE 0 END)

即"住院期间任意一剂出现在 48 h 之后"。本队列 87.3% 的人在 24 h 内就已经用上
抗菌药（中位 2.3 h），所以该口径主要在测**继续用药（continuation）**，
不是新发事件 —— 实测 SLE 组命中率 56.1%，正好复现 v2 的 243/433。

v5 因此定义真正 incident 的版本（154 号脚本已提取）：

  主结局   new_agent_24_7d   某广谱抗菌药**首次启用**落在 ICU 入室 24 h–7 d，
                             且该 agent 在 24 h landmark 前从未用过（agent 级
                             new-user）。剔除外科预防类（cefazolin 等）。
 二级(特异) culture_pos_24_7d 同窗口血培养阳性（菌血症）
 二级(硬结局) death_30d / death_hosp
 敏感性    incident_strict   再加上"landmark 时完全未用抗菌药"
          legacy_any48      v2 口径（对齐历史，已知含继续用药）
          infect_icd        宽感染 ICD（无计时）
 对照      hyper_48h        GC -> 高血糖（阳性对照）
          gi_bleed          GC -> 消化道出血（阴性结局对照）

三条硬规则（用户指定）
----------------------
 1  每个估计值都**并列报告两个模型**：未加权（adjusted logistic / Firth）
    与 IPTW；不再只挑显著的那个。
 2  **预先指定主结局**：确认性检验 = 主结局上的疾病×剂量交互，单次、不校正。
 3  其余结局做 **BH（Benjamini-Hochberg）** 多重校正，按结局族分别报 q 值。

协变量集在 v4 基础上加一项 baseline 抗菌药负荷 n_agent_pre24
（既影响"能否再起一个新药"，也影响感染/死亡风险）。
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")

ORDER = ["G0_none", "G1_low", "G2_mod", "G3_high"]
LABEL = {"G0_none": "none", "G1_low": ">0-<10", "G2_mod": "10-<50", "G3_high": ">=50"}
SCORE = {"G0_none": 0.0, "G1_low": 1.0, "G2_mod": 2.0, "G3_high": 3.0}

CORE_BASE = ["age", "female", "sofa", "vaso24", "vent24", "renal_fail", "shock", "surg_any"]
EXTRA_COV = ["n_agent_pre24"]                    # v5 新增：baseline 抗菌药负荷
CORE = CORE_BASE + EXTRA_COV                     # 预先指定的 v5 调整集
PSCOV = CORE + ["resp_fail", "coagulop", "liver_fail", "immuno_any", "hcq",
                "cytopenia", "serositis", "n_proc", "n_hosp"]

# (变量, 标签, 族, 角色说明)
OUTCOMES = [
    ("new_agent_24_7d", "New broad-spectrum antimicrobial agent, 24 h-7 d", "primary", "PRIMARY"),
    ("culture_pos_24_7d", "Blood culture positive, 24 h-7 d", "secondary", "specificity"),
    ("death_30d", "30-day mortality", "secondary", "hard outcome"),
    ("death_hosp", "In-hospital mortality", "secondary", "hard outcome"),
    ("incident_strict", "Strict incident (antimicrobial-naive + new agent)", "sensitivity", "strict"),
    ("legacy_any48", "Legacy: any antimicrobial dose >=48 h (v2 definition)", "sensitivity", "legacy"),
    ("new_agent_anytime", "New agent, no upper window bound", "sensitivity", "window: unbounded"),
    ("culture_pos_after24", "Blood culture positive, >24 h (v4 window)", "sensitivity", "window: v4"),
    ("infect_icd", "Infection ICD code (wide, untimed)", "sensitivity", "wide"),
    ("hyper_48h", "Glucose >=180 mg/dL", "control", "POSITIVE CONTROL"),
    ("gi_bleed", "GI bleeding", "control", "NEGATIVE CONTROL"),
]
ROLE = {o[0]: o[3] for o in OUTCOMES}
FAMILY = {o[0]: o[2] for o in OUTCOMES}
NICE = {o[0]: o[1] for o in OUTCOMES}

PRIMARY = "new_agent_24_7d"
KEY_SECONDARY = "death_30d"

lines = []


def P(s=""):
    print(s)
    lines.append(str(s))


# ------------------------------------------------------------------ helpers
def fit_logit(df, y, xvars):
    use = df[[y] + xvars].dropna()
    if len(use) == 0 or use[y].nunique() < 2:
        return None
    X = sm.add_constant(use[xvars].astype(float))
    try:
        with np.errstate(all="ignore"):
            m = sm.Logit(use[y].astype(float), X).fit(disp=0, maxiter=300)
        b, se = float(m.params.iloc[1]), float(m.bse.iloc[1])
    except Exception:
        return None
    if not np.isfinite(b) or not np.isfinite(se) or se <= 0 or abs(b) > 12:
        return None
    z = b / se
    return dict(OR=float(np.exp(b)), lo=float(np.exp(b - 1.96 * se)),
                hi=float(np.exp(b + 1.96 * se)),
                p=float(2 * (1 - stats.norm.cdf(abs(z)))), beta=b, se=se,
                n=int(len(use)), n_ev=int(use[y].sum()))


def firth(X, y, max_iter=400, tol=1e-10):
    """Firth penalised logistic regression (sparse / separated data)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    b = np.zeros(p)
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


def iptw(d, covs, trim=0.01, reg_C=0.3):
    """Stabilised IPTW for the 4-level exposure; returns d with 'sw' + ESS."""
    X = d[covs].astype(float).values
    keep = np.isfinite(X).all(1)
    d = d[keep].copy()
    X = d[covs].astype(float).values
    X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    T = pd.Categorical(d["gc24_str"], categories=ORDER).codes
    ps = LogisticRegression(solver="lbfgs", max_iter=8000, C=reg_C).fit(X, T)
    Pm = ps.predict_proba(X)
    marg = np.array([(T == g).mean() for g in range(len(ORDER))])
    w = marg[T] / np.clip(Pm[np.arange(len(T)), T], 1e-6, 1)
    if trim:
        lo, hi = np.quantile(w, [trim, 1 - trim])
        w = np.clip(w, lo, hi)
    d = d.copy()
    d["sw"] = w
    return d, float(w.sum() ** 2 / (w ** 2).sum())


def maxsmd(dd, covs, wcol, ref="G0_none"):
    ww = np.ones(len(dd)) if wcol is None else dd[wcol].values
    worst = 0.0
    for v in covs:
        m = dd[v].astype(float).values
        sd = np.sqrt(np.average((m - np.average(m, weights=ww)) ** 2, weights=ww))
        if sd == 0:
            continue
        m0 = np.average(m[(dd.gc24_str == ref).values], weights=ww[(dd.gc24_str == ref).values])
        for g in ORDER[1:]:
            i = (dd.gc24_str == g).values
            worst = max(worst, abs((np.average(m[i], weights=ww[i]) - m0) / sd))
    return worst


def trend_firth(d, y, extra=None, col="gc_str_num"):
    cols = [col, y] + (extra or [])
    use = d[cols].dropna()
    if len(use) == 0 or use[y].nunique() < 2 or use[col].nunique() < 2:
        return None
    X = sm.add_constant(use[[col] + (extra or [])].astype(float))
    b, se, ll = firth(X.values, use[y].values)
    j = list(X.columns).index(col)
    z = b[j] / se[j]
    return dict(OR=float(np.exp(b[j])), lo=float(np.exp(b[j] - 1.96 * se[j])),
                hi=float(np.exp(b[j] + 1.96 * se[j])),
                p=float(2 * (1 - stats.norm.cdf(abs(z)))), n=int(len(use)),
                n_ev=int(use[y].sum()), beta=float(b[j]), se=float(se[j]), ll=ll)


def interaction_test(d, y, covs, use_firth=None):
    """疾病×剂量交互 -> RA/SLE 剂量趋势 OR 之比（每层增量）。
    返回未加权（MLE 或 Firth，视事件数）结果。"""
    cols = ["gc_str_num", "is_sle", y] + covs
    use = d[cols].dropna().copy()
    if len(use) == 0 or use[y].nunique() < 2:
        return None
    use["s"] = use.gc_str_num.astype(float)
    use["sx"] = use.s * use.is_sle.astype(float)
    X = sm.add_constant(use[["s", "is_sle", "sx"] + covs].astype(float))
    Xr = sm.add_constant(use[["s", "is_sle"] + covs].astype(float))
    if use_firth is None:
        use_firth = int(use[y].sum()) < 60
    model = "Firth" if use_firth else "MLE"
    if use_firth:
        b, se, ll_full = firth(X.values, use[y].values)
        _, _, ll_red = firth(Xr.values, use[y].values)
        lrt = 2.0 * (ll_full - ll_red)
    else:
        try:
            with np.errstate(all="ignore"):
                m = sm.Logit(use[y].astype(float), X).fit(disp=0, maxiter=300)
                m0 = sm.Logit(use[y].astype(float), Xr).fit(disp=0, maxiter=300)
            b, se = np.asarray(m.params, float), np.asarray(m.bse, float)
            lrt = 2.0 * (m.llf - m0.llf)
        except Exception:
            model = "Firth"
            b, se, ll_full = firth(X.values, use[y].values)
            _, _, ll_red = firth(Xr.values, use[y].values)
            lrt = 2.0 * (ll_full - ll_red)
    p_lrt = float(1 - stats.chi2.cdf(max(lrt, 0.0), 1))
    j = list(X.columns).index("sx")
    z = b[j] / se[j]
    e = float(np.exp(b[j]))            # SLE/RA slope ratio
    return dict(OR=1.0 / e, lo=1.0 / float(np.exp(b[j] + 1.96 * se[j])),
                hi=1.0 / float(np.exp(b[j] - 1.96 * se[j])),
                or_sle_vs_ra=e, log=float(-b[j]),
                p_wald=float(2 * (1 - stats.norm.cdf(abs(z)))),
                p_lrt=p_lrt, lrt=float(lrt), model=model,
                n=int(len(use)), n_ev=int(use[y].sum()))


def interaction_iptw(pool, y, covs=None):
    """IPTW 版交互（权重按病种内重新拟合后合并）。"""
    use = pool[["s", "is_ra", y, "sw"]].dropna()
    if len(use) == 0 or use[y].nunique() < 2:
        return None
    try:
        with np.errstate(all="ignore"):
            m = smf.glm("%s ~ s + is_ra + s:is_ra" % y, data=use,
                        family=sm.families.Binomial(), freq_weights=use["sw"]).fit()
        t = [k for k in m.params.index if "s:is_ra" in k][0]
        b, se = float(m.params[t]), float(m.bse[t])
        return dict(OR=float(np.exp(b)), lo=float(np.exp(b - 1.96 * se)),
                    hi=float(np.exp(b + 1.96 * se)), log=b,
                    p=float(m.pvalues[t]), n=int(len(use)), n_ev=int(use[y].sum()))
    except Exception:
        return None


def bh(pvals, q=0.05):
    """Benjamini-Hochberg: 返回 (q_value 数组, 是否拒绝)。"""
    p = np.asarray([np.nan if v is None else float(v) for v in pvals], float)
    ok = np.isfinite(p)
    qv = np.full(len(p), np.nan)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return qv, np.zeros(len(p), bool)
    o = idx[np.argsort(p[idx])]
    m = len(o)
    ranked = p[o] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qv[o] = np.clip(ranked, 0, 1)
    return qv, qv <= q


def line(r, key="p"):
    if r is None:
        return "not estimable"
    return "%.2f (%.2f-%.2f) P=%.3f" % (r["OR"], r["lo"], r["hi"], r.get(key, r.get("p")))


def evalue(rr):
    if rr is None or not np.isfinite(rr) or rr <= 0:
        return np.nan
    rr = rr if rr >= 1 else 1.0 / rr
    return rr + np.sqrt(rr * (rr - 1))


# ------------------------------------------------------------------ load
df = pd.read_csv(os.path.join(DATA, "cohort_rheum_icu.csv"))
df = df.rename(columns={"sofa24": "sofa"})
ab = pd.read_csv(os.path.join(DATA, "abx_outcomes_rheum.csv"))
NEW = ["stay_key", "legacy_any48", "n_agent_pre24", "abx_naive_at24", "n_agent_new_24_7d",
       "t_new_agent_h", "new_agent_24_7d", "incident_strict", "new_agent_anytime",
       "n_culture_24_7d", "culture_pos_24_7d", "h_disch", "h_death", "h_censor",
       "death_within7d", "window_complete", "died_before_window_end",
       "discharged_before_window_end", "ondan24", "docusate24", "apap24", "laxative24"]
df = df.merge(ab[NEW], on="stay_key", how="left")
df["sofa"] = df.sofa.fillna(df.sofa.median())
df["gc_str_num"] = df.gc24_str.map(SCORE)
df["hyper_48h"] = df.hyper_48h.astype(float)
for c in ["new_agent_24_7d", "incident_strict", "legacy_any48", "culture_pos_24_7d",
          "culture_pos_after24", "n_agent_pre24", "abx_naive_at24", "death_30d",
          "death_hosp", "gi_bleed", "infect_icd", "window_complete",
          "new_agent_anytime"]:
    df[c] = df[c].astype(float)
assert df[PRIMARY].notna().all(), "primary outcome has missing values"

P("=" * 104)
P("SLE vs RA HEAD-TO-HEAD  v5   (MIMIC-IV rheumatic ICU cohort, 24-h landmark)")
P("=" * 104)
P("  cohort n = %d   SLE n = %d   RA n = %d   diseases = %d" % (
    len(df), (df.primary_grp == "SLE").sum(), (df.primary_grp == "RA").sum(),
    df.primary_grp.nunique()))
P("  PRE-SPECIFIED PRIMARY OUTCOME : %s" % NICE[PRIMARY])
P("  adjustment set (v5) = v4 set + baseline antimicrobial burden (n_agent_pre24)")
P("  every estimate below is reported for BOTH models (unweighted / IPTW)")

# =====================================================================
P("")
P("=" * 104)
P("0. WHY THE DEFINITION CHANGED: event counts of each candidate definition")
P("=" * 104)
P("  %-46s %10s %10s %10s" % ("definition", "ALL", "SLE", "RA"))
sle = df[df.primary_grp == "SLE"]
ra = df[df.primary_grp == "RA"]
both = df[df.primary_grp.isin(["SLE", "RA"])].copy()
DEFS = [("legacy_any48", "A any antimicrobial dose >=48 h  (v2; = continuation)"),
        ("new_agent_24_7d", "B new agent started 24 h-7 d  (PRIMARY)"),
        ("incident_strict", "C B + antimicrobial-naive at landmark  (strict)"),
        ("culture_pos_24_7d", "D blood culture + 24 h-7 d  (specificity)"),
        ("death_30d", "E 30-day mortality"),
        ("infect_icd", "F infection ICD code (wide, untimed)")]
def_rows = []
for c, lab in DEFS:
    P("  %-46s %4d(%4.1f%%) %4d(%4.1f%%) %4d(%4.1f%%)" % (
        lab, df[c].sum(), 100 * df[c].mean(), sle[c].sum(), 100 * sle[c].mean(),
        ra[c].sum(), 100 * ra[c].mean()))
    def_rows.append(dict(var=c, label=lab, all_n=int(df[c].sum()),
                         all_pct=round(100 * df[c].mean(), 2),
                         sle_n=int(sle[c].sum()), sle_pct=round(100 * sle[c].mean(), 2),
                         ra_n=int(ra[c].sum()), ra_pct=round(100 * ra[c].mean(), 2)))
P("")
P("  head-to-head subset (SLE+RA, n=%d):" % len(both))
for c, lab in DEFS:
    P("    %-46s %5d events" % (c, int(both[c].sum())))
P("")
P("  window completeness (landmark -> 7 d), whole cohort:")
P("    complete (in hospital / discharged >= 7 d) %d (%.1f%%) ; died before end %d (%.1f%%)"
  " ; discharged alive before end %d (%.1f%%)" % (
      df.window_complete.sum(), 100 * df.window_complete.mean(),
      df.died_before_window_end.sum(), 100 * df.died_before_window_end.mean(),
      df.discharged_before_window_end.sum(), 100 * df.discharged_before_window_end.mean()))

# =====================================================================
P("")
P("=" * 104)
P("1. WHO REACHES THE ICU?  (the two populations differ before exposure starts)")
P("=" * 104)
VARS = [("age", "age, median", "med"), ("female", "female, %", "pct"),
        ("sofa", "SOFA, median", "med"), ("vaso24", "vasopressor <=24 h, %", "pct"),
        ("vent24", "ventilation <=24 h, %", "pct"), ("renal_fail", "renal failure, %", "pct"),
        ("shock", "shock, %", "pct"), ("resp_fail", "respiratory failure, %", "pct"),
        ("surg_any", "surgical service, %", "pct"), ("elective", "elective admission, %", "pct"),
        ("immuno_any", "non-GC immunosuppressant, %", "pct"), ("hcq", "hydroxychloroquine, %", "pct"),
        ("n_agent_pre24", "antimicrobial agents before 24 h, mean", "med"),
        ("gc_any24", "ANY GC first 24 h, %", "pct"),
        ("gc24_daily_pe_mg", "GC dose, median mg/day", "med")]
P("  %-40s %14s %14s %10s" % ("", "SLE (n=%d)" % len(sle), "RA (n=%d)" % len(ra), "|SMD|"))
base_rows = []
for v, lab, kind in VARS:
    a, b = sle[v].astype(float), ra[v].astype(float)
    if kind == "pct":
        va, vb, s = 100 * a.mean(), 100 * b.mean(), ("%.1f%%" % (100 * a.mean()), "%.1f%%" % (100 * b.mean()))
    else:
        va, vb, s = a.median(), b.median(), ("%.1f" % a.median(), "%.1f" % b.median())
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    smd = abs(a.mean() - b.mean()) / sd if sd > 0 else np.nan
    P("  %-40s %14s %14s %10.3f" % (lab, s[0], s[1], smd))
    base_rows.append(dict(variable=lab, sle=s[0], ra=s[1], smd=round(float(smd), 4)))

# =====================================================================
P("")
P("=" * 104)
P("2. DOSE-RESPONSE WITHIN EACH DISEASE -- crude / adjusted / IPTW side by side")
P("=" * 104)
P("  dose-trend = OR per stratum increment across 0 / >0-<10 / 10-<50 / >=50 mg/day")
fits = {}
for g in ["SLE", "RA"]:
    d, ess = iptw(df[df.primary_grp == g].copy(), PSCOV)
    fits[g] = (d, ess)
    P("  PS model %s: n=%d  ESS=%.0f  max|SMD| after weighting = %.3f" % (
        g, len(d), ess, maxsmd(d, CORE, "sw")))
P("")
P("  %-26s %-5s %7s %7s %25s %25s %25s" % (
    "outcome", "dis", "n", "events", "crude", "adjusted", "IPTW"))
dose_rows = []
for y, lab, fam, role in OUTCOMES:
    for g in ["SLE", "RA"]:
        d, ess = fits[g]
        sub = d[[y, "gc_str_num"]].dropna()
        c0 = fit_logit(d, y, ["gc_str_num"])
        c1 = fit_logit(d, y, ["gc_str_num"] + CORE)
        c2 = None
        try:
            with np.errstate(all="ignore"):
                m = smf.glm("%s ~ gc_str_num" % y, data=d, family=sm.families.Binomial(),
                            freq_weights=d["sw"]).fit()
            b, se = float(m.params["gc_str_num"]), float(m.bse["gc_str_num"])
            c2 = dict(OR=float(np.exp(b)), lo=float(np.exp(b - 1.96 * se)),
                      hi=float(np.exp(b + 1.96 * se)), p=float(m.pvalues["gc_str_num"]))
        except Exception:
            pass
        P("  %-26s %-5s %7d %7d %25s %25s %25s" % (
            lab[:26], g, len(sub), int(sub[y].sum()), line(c0), line(c1), line(c2)))
        dose_rows.append(dict(outcome=lab, var=y, family=fam, role=role, disease=g,
                              n=len(sub), n_ev=int(sub[y].sum()),
                              crude_or=(c0 or {}).get("OR"), crude_lo=(c0 or {}).get("lo"),
                              crude_hi=(c0 or {}).get("hi"), crude_p=(c0 or {}).get("p"),
                              adj_or=(c1 or {}).get("OR"), adj_lo=(c1 or {}).get("lo"),
                              adj_hi=(c1 or {}).get("hi"), adj_p=(c1 or {}).get("p"),
                              iptw_or=(c2 or {}).get("OR"), iptw_lo=(c2 or {}).get("lo"),
                              iptw_hi=(c2 or {}).get("hi"), iptw_p=(c2 or {}).get("p")))
    P("")

# =====================================================================
P("")
P("=" * 104)
P("3. THE HEAD-TO-HEAD TEST: disease x dose interaction  (RA/SLE ratio of trend ORs)")
P("=" * 104)
P("  BOTH models reported.  D3 self-check compares the interaction coefficient with")
P("  the ratio of the separately fitted per-disease trends (catches an inverted term).")
P("")
pool = pd.concat([fits["SLE"][0], fits["RA"][0]], ignore_index=True)
pool["is_ra"] = (pool.primary_grp == "RA").astype(int)
pool["s"] = pool.gc_str_num.astype(float)
P("  %-26s %-11s %8s %26s %9s %9s %26s %9s" % (
    "outcome", "role", "events", "unweighted", "P Wald", "P LRT", "IPTW", "P"))
int_rows = []
for y, lab, fam, role in OUTCOMES:
    a = fit_logit(df[df.primary_grp == "SLE"], y, ["gc_str_num"] + CORE)
    b2 = fit_logit(df[df.primary_grp == "RA"], y, ["gc_str_num"] + CORE)
    r = interaction_test(both, y, CORE)
    ri = interaction_iptw(pool, y)
    if r is None:
        P("  %-26s %-11s   not estimable" % (lab[:26], role))
        continue
    chk = ""
    if a and b2 and a["OR"] > 0 and b2["OR"] > 0:
        ratio = b2["OR"] / a["OR"]
        ok = abs(np.log(ratio) - np.log(r["OR"])) < 0.25
        chk = "OK" if ok else "<<< CHECK"
    P("  %-26s %-11s %8d %26s %9.3f %9.3f %26s %9.3f  %s" % (
        lab[:26], role, r["n_ev"], "%.2f (%.2f-%.2f)" % (r["OR"], r["lo"], r["hi"]),
        r["p_wald"], r["p_lrt"],
        (("%.2f (%.2f-%.2f)" % (ri["OR"], ri["lo"], ri["hi"])) if ri else "not estimable"),
        (ri["p"] if ri else np.nan), chk))
    ratio_emp = (b2["OR"] / a["OR"]) if (a and b2 and a["OR"] > 0) else np.nan
    int_rows.append(dict(outcome=lab, var=y, family=fam, role=role,
                         or_uw=r["OR"], lo_uw=r["lo"], hi_uw=r["hi"],
                         p_wald=r["p_wald"], p_lrt=r["p_lrt"], model_uw=r["model"],
                         or_iptw=(ri or {}).get("OR"), lo_iptw=(ri or {}).get("lo"),
                         hi_iptw=(ri or {}).get("hi"), p_iptw=(ri or {}).get("p"),
                         sle_trend=(a or {}).get("OR"), ra_trend=(b2 or {}).get("OR"),
                         ratio_emp=float(ratio_emp) if np.isfinite(ratio_emp) else np.nan,
                         n=r["n"], n_ev=r["n_ev"]))

# ------------------------------------------------ 3b. multiplicity
P("")
P("=" * 104)
P("3b. MULTIPLICITY: pre-specified primary vs BH-corrected family")
P("=" * 104)
P("  (1) PRE-SPECIFIED confirmatory test = interaction on the PRIMARY outcome,")
P("      tested once, uncorrected.  (2) All remaining interaction tests are")
P("      secondary/sensitivity and are BH-corrected within their family.")
P("      BH over the FULL family is also shown as a sensitivity.")
P("")
P("  %-26s %-11s %-11s %9s %9s %9s %9s" % (
    "outcome", "family", "role", "P (adj)", "P (IPTW)", "q in fam", "q full"))
p_fam = {}
for r in int_rows:
    p_fam.setdefault(r["family"], []).append(r["p_lrt"])
q_fam = {}
for fam, ps in p_fam.items():
    qv, _ = bh(ps)
    for i, r_ in enumerate([x for x in int_rows if x["family"] == fam]):
        r_["q_family"] = float(qv[i])
pv_all = [r["p_lrt"] for r in int_rows]
qv_all, rej_all = bh(pv_all)
for r, qa in zip(int_rows, qv_all):
    r["q_full"] = float(qa)
    r["reject_full"] = bool(qa <= 0.05)
    P("  %-26s %-11s %-11s %9.3f %9.3f %9.3f %9.3f%s" % (
        r["outcome"][:26], r["family"], r["role"], r["p_lrt"],
        r["p_iptw"] if r["p_iptw"] is not None else np.nan,
        r["q_family"], r["q_full"], "  <-- survives full BH" if r["reject_full"] else ""))

# ------------------------------------------------ 3c. 4-stratum ORs
P("")
P("  same interaction, expressed as 4-level factor ORs (ref = no GC), BOTH models:")
P("  %-26s %-5s %-9s %24s %24s" % ("outcome", "dis", "stratum", "adjusted", "IPTW"))
strata_rows = []
for y, lab, fam, role in OUTCOMES:
    for g in ["SLE", "RA"]:
        d, ess = fits[g]
        for gs in ORDER[1:]:
            sub = d[d.gc24_str.isin(["G0_none", gs])].copy()
            sub["e"] = (sub.gc24_str == gs).astype(int)
            r1 = fit_logit(sub, y, ["e"] + CORE)
            r2 = None
            try:
                with np.errstate(all="ignore"):
                    m = smf.glm("%s ~ e" % y, data=sub, family=sm.families.Binomial(),
                                freq_weights=sub["sw"]).fit()
                b, se = float(m.params["e"]), float(m.bse["e"])
                r2 = dict(OR=float(np.exp(b)), lo=float(np.exp(b - 1.96 * se)),
                          hi=float(np.exp(b + 1.96 * se)), p=float(m.pvalues["e"]))
            except Exception:
                pass
            P("  %-26s %-5s %-9s %24s %24s" % (lab[:26], g, LABEL[gs], line(r1), line(r2)))
            strata_rows.append(dict(outcome=lab, var=y, disease=g, stratum=LABEL[gs],
                                    adj_or=(r1 or {}).get("OR"), adj_lo=(r1 or {}).get("lo"),
                                    adj_hi=(r1 or {}).get("hi"), adj_p=(r1 or {}).get("p"),
                                    iptw_or=(r2 or {}).get("OR"), iptw_lo=(r2 or {}).get("lo"),
                                    iptw_hi=(r2 or {}).get("hi"), iptw_p=(r2 or {}).get("p"),
                                    n_ev=(r1 or {}).get("n_ev"), n=(r1 or {}).get("n")))
    P("")

# =====================================================================
P("")
P("=" * 104)
P("4. CONTROLS  (both models; a control that fails invalidates the rows above)")
P("=" * 104)
P("  %-26s %-8s %-5s %26s %26s" % ("outcome", "role", "dis", "adjusted trend", "IPTW trend"))
P("")
pc_rows = []
for y, lab, fam, role in [o for o in OUTCOMES if o[2] == "control"]:
    for g in ["SLE", "RA"]:
        d, ess = fits[g]
        r1 = fit_logit(d, y, ["gc_str_num"] + CORE)
        r2 = None
        try:
            with np.errstate(all="ignore"):
                m = smf.glm("%s ~ gc_str_num" % y, data=d, family=sm.families.Binomial(),
                            freq_weights=d["sw"]).fit()
            b, se = float(m.params["gc_str_num"]), float(m.bse["gc_str_num"])
            r2 = dict(OR=float(np.exp(b)), lo=float(np.exp(b - 1.96 * se)),
                      hi=float(np.exp(b + 1.96 * se)), p=float(m.pvalues["gc_str_num"]))
        except Exception:
            pass
        P("  %-26s %-8s %-5s %26s %26s" % (lab[:26], role.split()[0], g, line(r1), line(r2)))
        pc_rows.append(dict(outcome=lab, disease=g, role=role,
                            adj_or=(r1 or {}).get("OR"), adj_lo=(r1 or {}).get("lo"),
                            adj_hi=(r1 or {}).get("hi"), adj_p=(r1 or {}).get("p"),
                            iptw_or=(r2 or {}).get("OR"), iptw_lo=(r2 or {}).get("lo"),
                            iptw_hi=(r2 or {}).get("hi"), iptw_p=(r2 or {}).get("p")))
P("")
P("  independent negative-EXPOSURE controls (routine ICU drugs with no plausible")
P("  infection effect, given in the first 24 h).  If these reproduce the GC signal,")
P("  what is being measured is care-process confounding, not drug effect.")
NEGEXP = [("ondan24", "ondansetron <=24 h"), ("docusate24", "docusate <=24 h"),
          ("apap24", "acetaminophen <=24 h"), ("laxative24", "senna/bisacodyl <=24 h")]
P("  %-26s %-8s %-26s %-26s" % ("outcome", "neg-exp", "adjusted", "IPTW (unweighted scale)"))
neg_rows = []
for y in [PRIMARY, "culture_pos_24_7d", "incident_strict", "legacy_any48",
          "death_30d", "hyper_48h", "gi_bleed"]:
    for c, lab in NEGEXP:
        r1 = fit_logit(both, y, [c] + CORE)
        P("    %-24s %-8s %-26s" % (NICE[y][:24], lab.split()[0], line(r1)))
        neg_rows.append(dict(outcome=NICE[y], var=y, negexp=lab,
                             adj_or=(r1 or {}).get("OR"), adj_lo=(r1 or {}).get("lo"),
                             adj_hi=(r1 or {}).get("hi"), adj_p=(r1 or {}).get("p"),
                             n=(r1 or {}).get("n"), n_ev=(r1 or {}).get("n_ev")))
    P("")

# =====================================================================
P("")
P("=" * 104)
P("5. E-VALUE for the pre-specified primary outcome and 30-day death")
P("=" * 104)
P("  OR -> RR by RR = OR / (1 - p0 + p0*OR), p0 = risk in the no-GC stratum.")
ev_rows = []
for y in [PRIMARY, KEY_SECONDARY]:
    for g in ["SLE", "RA"]:
        d = df[df.primary_grp == g]
        r = fit_logit(d, y, ["gc_str_num"] + CORE)
        if r is None:
            continue
        p0 = float(d.loc[d.gc24_str == "G0_none", y].mean())
        rr = r["OR"] / (1 - p0 + p0 * r["OR"])
        lo_rr = r["lo"] / (1 - p0 + p0 * r["lo"])
        P("  %-6s %-30s OR %.2f  p0 %.3f  RR %.2f  E-value %.2f (CI limit %.2f)" % (
            g, NICE[y][:30], r["OR"], p0, rr, evalue(rr), evalue(lo_rr)))
        ev_rows.append(dict(disease=g, outcome=NICE[y], var=y, OR=r["OR"], p0=p0, RR=rr,
                            evalue=evalue(rr), evalue_lo=evalue(lo_rr), p=r["p"]))

# ------------------------------------------------------------------ save
pd.DataFrame(def_rows).to_csv(os.path.join(OUT, "table_v5_definitions.csv"), index=False)
pd.DataFrame(base_rows).to_csv(os.path.join(OUT, "table_v5_baseline.csv"), index=False)
pd.DataFrame(dose_rows).to_csv(os.path.join(OUT, "table_v5_dose.csv"), index=False)
pd.DataFrame(int_rows).to_csv(os.path.join(OUT, "table_v5_interaction.csv"), index=False)
pd.DataFrame(strata_rows).to_csv(os.path.join(OUT, "table_v5_strata.csv"), index=False)
pd.DataFrame(pc_rows).to_csv(os.path.join(OUT, "table_v5_controls.csv"), index=False)
pd.DataFrame(neg_rows).to_csv(os.path.join(OUT, "table_v5_negexp.csv"), index=False)
pd.DataFrame(ev_rows).to_csv(os.path.join(OUT, "table_v5_evalue.csv"), index=False)

with open(os.path.join(OUT, "_v5_results.json"), "w", encoding="utf-8") as f:
    json.dump(dict(definitions=def_rows, baseline=base_rows, dose=dose_rows,
                   interaction=int_rows, strata=strata_rows, controls=pc_rows,
                   negexp=neg_rows, evalue=ev_rows,
                   ess={"SLE": fits["SLE"][1], "RA": fits["RA"][1]},
                   maxsmd={"SLE": maxsmd(fits["SLE"][0], CORE, "sw"),
                           "RA": maxsmd(fits["RA"][0], CORE, "sw"),
                           "SLE_unw": maxsmd(fits["SLE"][0], CORE, None),
                           "RA_unw": maxsmd(fits["RA"][0], CORE, None)},
                   prespecified_primary=PRIMARY, key_secondary=KEY_SECONDARY,
                   core=CORE), f, ensure_ascii=False, indent=1, default=str)

with open(os.path.join(OUT, "155_head2head_v5.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nsaved -> out/155_head2head_v5.txt")
