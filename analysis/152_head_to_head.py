# -*- coding: utf-8 -*-
"""
152_head_to_head.py

Plan B, step 2: SLE vs RA head-to-head on the multi-disease cohort.

The question
------------
The SLE-only analysis (v2) found that the SAME exposure -- glucocorticoid
dose intensity in the first ICU day -- points in OPPOSITE directions in
the two diseases: in RA, a static high cumulative dose was crudely
associated with death (OR 3.31, dismantled to 1.09 by MSM), whereas in
SLE even the crude association was reversed (OR 0.61).  That was inferred
from two separate studies.  This script tests it directly in one cohort.

Three things have to be separated before any disease contrast is credible
------------------------------------------------------------------------
1. AGE.  RA patients in this cohort have a median age of 72.0 y versus
   57.1 y for SLE -- a 15-year gap.  Age drives both infection and death,
   so a naive head-to-head is largely an age comparison.  The age gap is
   therefore reported explicitly and adjusted for; a second model adds
   age-interaction to see whether the disease contrast survives.
2. INDICATION STRUCTURE.  Whether high-dose patients are sicker or
   healthier than untreated patients can differ by disease.  This is
   quantified as a signed severity gap (high-dose minus no-GC) in SOFA,
   age, vasopressor use, renal failure and immunosuppressant use.
3. WHICH OUTCOME.  Infection ICD codes are well powered (948 events);
   blood-culture-proven bacteraemia is the specific signal (87 events).
   Both are reported, because the wide and narrow outcomes disagreed in v2.

Analysis ladder
---------------
  A  cohort description, SLE vs RA, overall and by dose stratum (+SMD)
  A2 how much of the crude SLE-vs-RA outcome gap is age?
  B  indication-structure gradient per disease
  C  within-disease dose-response (crude, adjusted, IPTW)
  D  disease x dose interaction -- the head-to-head test
  E  positive control (GC -> hyperglycaemia) per disease
  F  the wider SARD spectrum, ordered by GC dependence
"""
import json
import os

import numpy as np
import pandas as pd
import psycopg2
import statsmodels.api as sm
import statsmodels.formula.api as smf
from psycopg2.extras import execute_values
from scipy import stats
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)
CONFIG = dict(
    host=os.environ.get("MIMIC_DB_HOST", "localhost"),
    port=int(os.environ.get("MIMIC_DB_PORT", "5432")),
    user=os.environ.get("MIMIC_DB_USER", "postgres"),
    password=os.environ.get("MIMIC_DB_PASSWORD", ""),
)

ORDER = ["G0_none", "G1_low", "G2_mod", "G3_high"]
LABEL = {"G0_none": "none", "G1_low": ">0-<10", "G2_mod": "10-<50", "G3_high": ">=50"}
SCORE = {"G0_none": 0.0, "G1_low": 1.0, "G2_mod": 2.0, "G3_high": 3.0}

# covariates used to adjust the outcome model (parsimonious: the sparse
# bacteraemia outcome has only 87 events, so keep the parameter count low)
CORE = ["age", "female", "sofa", "vaso24", "vent24", "renal_fail", "shock", "surg_any"]
# richer set for the propensity model (it models the exposure, not the outcome)
PSCOV = CORE + ["resp_fail", "coagulop", "liver_fail", "immuno_any", "hcq",
                "cytopenia", "serositis", "n_proc", "n_hosp"]

OUTCOMES = [("culture_pos_after24", "blood culture + >24 h (specific)"),
            ("infect_icd", "infection ICD code (wide)"),
            ("death_hosp", "hospital death"),
            ("hyper_48h", "glucose >=180 (POSITIVE CONTROL)"),
            ("gi_bleed", "GI bleeding (NEGATIVE CONTROL)")]

lines = []


def P(s=""):
    print(s)
    lines.append(str(s))


# ------------------------------------------------------------------ helpers
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


def iptw(d, covs, trim=0.01, reg_C=0.3):
    """Stabilised IPTW for a 4-level exposure; returns d with 'sw' and the ESS."""
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


def fit_logit(df, y, xvars):
    use = df[[y] + xvars].dropna()
    if len(use) == 0 or use[y].nunique() < 2:
        return None
    X = sm.add_constant(use[xvars].astype(float))
    try:
        m = sm.Logit(use[y].astype(float), X).fit(disp=0)
        b, se = float(m.params.iloc[1]), float(m.bse.iloc[1])
    except Exception:
        return None
    if not np.isfinite(b) or abs(b) > 12 or np.isnan(se) or se == 0:
        return None
    z = b / se
    return dict(OR=float(np.exp(b)), lo=float(np.exp(b - 1.96 * se)),
                hi=float(np.exp(b + 1.96 * se)),
                p=float(2 * (1 - stats.norm.cdf(abs(z)))),
                n=int(len(use)), n_ev=int(use[y].sum()), beta=b, se=se)


def firth(X, y, max_iter=400, tol=1e-10):
    """Firth penalised logistic regression (sparse / separated data).
    X must already contain an intercept column."""
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


def firth_trend(d, y, extra=None, disease_col=None):
    """Firth dose-trend (per stratum increment) within a subgroup.
    Returns OR per stratum increment (0->3)."""
    cols = ["gc_str_num", y] + (extra or [])
    use = d[cols].dropna()
    if len(use) == 0 or use[y].nunique() < 2 or use.gc_str_num.nunique() < 2:
        return None
    X = sm.add_constant(use[["gc_str_num"] + (extra or [])].astype(float))
    b, se, ll = firth(X.values, use[y].values)
    j = list(X.columns).index("gc_str_num")
    z = b[j] / se[j]
    return dict(OR=float(np.exp(b[j])), lo=float(np.exp(b[j] - 1.96 * se[j])),
                hi=float(np.exp(b[j] + 1.96 * se[j])),
                p=float(2 * (1 - stats.norm.cdf(abs(z)))),
                n=int(len(use)), n_ev=int(use[y].sum()), beta=float(b[j]),
                se=float(se[j]), ll=ll)


def interaction_test(d, y, covs, use_firth=False):
    """disease x dose-score interaction -> ratio of dose-trend ORs (RA vs SLE).
    Returns the interaction OR (per stratum increment, RA relative to SLE)."""
    cols = ["gc_str_num", "is_sle", y] + covs
    use = d[cols].dropna()
    if len(use) == 0 or use[y].nunique() < 2:
        return None
    use = use.copy()
    use["s"] = use.gc_str_num.astype(float)
    use["sx"] = use.s * use.is_sle.astype(float)     # interaction term
    X = sm.add_constant(use[["s", "is_sle", "sx"] + covs].astype(float))
    if use_firth or use[y].sum() < 60:
        b, se, ll_full = firth(X.values, use[y].values)
        # reduced model (no interaction) for a penalised LRT
        Xr = sm.add_constant(use[["s", "is_sle"] + covs].astype(float))
        _, _, ll_red = firth(Xr.values, use[y].values)
        lrt = 2.0 * (ll_full - ll_red)
        p_lrt = float(1 - stats.chi2.cdf(max(lrt, 0.0), 1))
    else:
        try:
            m = sm.Logit(use[y].astype(float), X).fit(disp=0)
            Xr = sm.add_constant(use[["s", "is_sle"] + covs].astype(float))
            m0 = sm.Logit(use[y].astype(float), Xr).fit(disp=0)
            b = np.asarray(m.params)
            se = np.asarray(m.bse)
            lrt = 2.0 * (m.llf - m0.llf)
            p_lrt = float(1 - stats.chi2.cdf(max(lrt, 0.0), 1))
        except Exception:
            b, se, ll_full = firth(X.values, use[y].values)
            Xr = sm.add_constant(use[["s", "is_sle"] + covs].astype(float))
            _, _, ll_red = firth(Xr.values, use[y].values)
            lrt = 2.0 * (ll_full - ll_red)
            p_lrt = float(1 - stats.chi2.cdf(max(lrt, 0.0), 1))
    j = list(X.columns).index("sx")
    z = b[j] / se[j]
    # sx = s * is_sle, so exp(b_sx) is the ratio of slopes for SLE relative
    # to RA.  Report the more intuitive orientation -- RA relative to SLE --
    # by reciprocating (the two-sided P is unchanged by inversion).
    e = float(np.exp(b[j]))
    return dict(OR=1.0 / e, lo=1.0 / float(np.exp(b[j] + 1.96 * se[j])),
                hi=1.0 / float(np.exp(b[j] - 1.96 * se[j])),
                or_sle_vs_ra=e,
                p_wald=float(2 * (1 - stats.norm.cdf(abs(z)))),
                p_lrt=p_lrt, lrt=float(lrt), n=int(len(use)), n_ev=int(use[y].sum()))


def or_line(r):
    if r is None:
        return "not estimable"
    return "%.2f (%.2f-%.2f) P=%.3f" % (r["OR"], r["lo"], r["hi"], r["p"])


# ------------------------------------------------------------------ load
df = pd.read_csv(os.path.join(DATA, "cohort_rheum_icu.csv"))
df = df.rename(columns={"sofa24": "sofa"})          # SQL alias is sofa24
df["sofa"] = df.sofa.fillna(df.sofa.median())
df["gc_str_num"] = df.gc24_str.map(SCORE)
df["hyper_48h"] = df.hyper_48h.astype(float)

P("=" * 100)
P("SLE vs RA HEAD-TO-HEAD   (MIMIC-IV, multi-disease rheumatic ICU cohort)")
P("=" * 100)
P("  landmark cohort n = %d   diseases = %d" % (len(df), df.primary_grp.nunique()))
P("  any GC in first 24 h: %d (%.1f%%)" % (df.gc_any24.sum(), 100 * df.gc_any24.mean()))

# =====================================================================
# A. cohort description, SLE vs RA
# =====================================================================
P("")
P("=" * 100)
P("A. WHO REACHES THE ICU?  SLE vs RA  (this is the first obstacle to any")
P("   disease contrast: the two populations differ before exposure starts)")
P("=" * 100)
sle = df[df.primary_grp == "SLE"].copy()
ra = df[df.primary_grp == "RA"].copy()
both = df[df.primary_grp.isin(["SLE", "RA"])].copy()

VARS = [("age", "age, median (IQR)", "med"), ("female", "female, %", "pct"),
        ("sofa", "SOFA, median", "med"), ("vaso24", "vasopressor <=24 h, %", "pct"),
        ("vent24", "ventilation <=24 h, %", "pct"), ("renal_fail", "renal failure, %", "pct"),
        ("shock", "shock, %", "pct"), ("resp_fail", "respiratory failure, %", "pct"),
        ("liver_fail", "hepatic failure, %", "pct"), ("coagulop", "coagulopathy, %", "pct"),
        ("cytopenia", "cytopenia, %", "pct"), ("serositis", "serositis/effusion, %", "pct"),
        ("surg_any", "surgical service, %", "pct"), ("elective", "elective admission, %", "pct"),
        ("immuno_any", "non-GC immunosuppressant, %", "pct"), ("hcq", "hydroxychloroquine, %", "pct"),
        ("n_proc", "procedures, median", "med"), ("n_hosp", "admissions, median", "med"),
        ("gc_any24", "ANY GC first 24 h, %", "pct"),
        ("gc24_daily_pe_mg", "GC dose, median mg/day*", "med")]

P("  %-34s %14s %14s %10s" % ("", "SLE (n=%d)" % len(sle), "RA (n=%d)" % len(ra), "|SMD|"))
base_rows = []
for v, lab, kind in VARS:
    a = sle[v].astype(float)
    b = ra[v].astype(float)
    if kind == "pct":
        va, vb = 100 * a.mean(), 100 * b.mean()
        s = "%.1f%%" % va, "%.1f%%" % vb
    else:
        va, vb = a.median(), b.median()
        s = "%.1f" % va, "%.1f" % vb
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    smd = abs(a.mean() - b.mean()) / sd if sd > 0 else np.nan
    P("  %-34s %14s %14s %10.3f" % (lab, s[0], s[1], smd))
    base_rows.append(dict(variable=lab, sle=s[0], ra=s[1], smd=round(float(smd), 4)))

P("  * among exposed only: SLE median %.1f (IQR %.1f-%.1f, n=%d); "
  "RA median %.1f (IQR %.1f-%.1f, n=%d)" % (
      sle.loc[sle.gc24_daily_pe_mg > 0, "gc24_daily_pe_mg"].median(),
      sle.loc[sle.gc24_daily_pe_mg > 0, "gc24_daily_pe_mg"].quantile(.25),
      sle.loc[sle.gc24_daily_pe_mg > 0, "gc24_daily_pe_mg"].quantile(.75),
      (sle.gc24_daily_pe_mg > 0).sum(),
      ra.loc[ra.gc24_daily_pe_mg > 0, "gc24_daily_pe_mg"].median(),
      ra.loc[ra.gc24_daily_pe_mg > 0, "gc24_daily_pe_mg"].quantile(.25),
      ra.loc[ra.gc24_daily_pe_mg > 0, "gc24_daily_pe_mg"].quantile(.75),
      (ra.gc24_daily_pe_mg > 0).sum()))

P("")
P("  dose strata by disease:")
P("  %-8s %8s %10s %10s %10s %10s" % ("", "n", "none", ">0-<10", "10-<50", ">=50"))
for g, s in both.groupby("primary_grp"):
    vc = s.gc24_str.value_counts().reindex(ORDER).fillna(0).astype(int)
    P("  %-8s %8d %10d %10d %10d %10d" % (g, len(s), vc["G0_none"], vc["G1_low"],
                                          vc["G2_mod"], vc["G3_high"]))

# ---------------------------------------------------------------- A2: age
P("")
P("=" * 100)
P("A2. HOW MUCH OF THE CRUDE SLE-vs-RA GAP IS JUST AGE?")
P("=" * 100)
P("  Age is the dominant difference (SLE 57.1 y vs RA 72.0 y).  Below: the")
P("  crude SLE-vs-RA odds ratio for each outcome, and what remains after")
P("  age+sex adjustment.  A large swing means the 'disease effect' is age.")
P("  %-34s %22s %22s %8s" % ("outcome", "crude OR SLE vs RA", "+age+sex", "P"))
age_rows = []
for y, lab in OUTCOMES:
    if y == "hyper_48h":
        pass
    use = both[[y, "is_sle", "age", "female"]].dropna().copy()
    if use[y].nunique() < 2:
        continue
    c0 = fit_logit(use, y, ["is_sle"])
    c1 = fit_logit(use, y, ["is_sle", "age", "female"])
    if c0 is None or c1 is None:
        continue
    P("  %-34s %22s %22s %8.3f" % (lab, or_line(c0), or_line(c1), c1["p"]))
    age_rows.append(dict(outcome=lab, crude=c0["OR"], crude_lo=c0["lo"], crude_hi=c0["hi"],
                         adj=c1["OR"], adj_lo=c1["lo"], adj_hi=c1["hi"], p=c1["p"],
                         n=c1["n"], n_ev=c1["n_ev"]))

# =====================================================================
# B. indication structure
# =====================================================================
P("")
P("=" * 100)
P("B. INDICATION STRUCTURE: are high-dose patients sicker or healthier than")
P("   untreated patients?  (high-dose minus no-GC, within each disease)")
P("=" * 100)
P("  Positive = high-dose stratum is MORE severe than the untreated stratum")
P("  (confounding inflates any apparent harm).")
P("  Negative = high-dose stratum is LESS severe (confounding masks harm,")
P("  or the untreated group is the untreated-severe / contraindicated group).")
ind_rows = []
IND_V = [("age", "age"), ("sofa", "SOFA"), ("vaso24", "vasopressor"),
         ("renal_fail", "renal failure"), ("immuno_any", "immunosuppressant"),
         ("shock", "shock"), ("surg_any", "surgical")]

MIN_N = 25
for g in ["SLE", "RA", "Vasculitis", "PMR_GCA", "Sarcoidosis", "axSpA", "APS", "SSc"]:
    s = df[df.primary_grp == g]
    if len(s) < 100:
        continue
    g0 = s[s.gc24_str == "G0_none"]
    g3 = s[s.gc24_str == "G3_high"]
    if len(g0) < MIN_N or len(g3) < MIN_N:
        P("  %-12s skipped (none n=%d, high n=%d)" % (g, len(g0), len(g3)))
        continue
    P("")
    P("  %s  (no GC n=%d, >=50 mg/day n=%d)" % (g, len(g0), len(g3)))
    P("    %-20s %10s %10s %10s %10s" % ("", "no GC", ">=50", "gap", "SMD"))
    smds = []
    for v, lab in IND_V:
        a, b = g0[v].astype(float), g3[v].astype(float)
        if v in ("vaso24", "renal_fail", "immuno_any", "shock", "surg_any"):
            va, vb = 100 * a.mean(), 100 * b.mean()
            sa, sb = "%.1f%%" % va, "%.1f%%" % vb
        else:
            va, vb = a.median(), b.median()
            sa, sb = "%.1f" % va, "%.1f" % vb
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        smd = (b.mean() - a.mean()) / sd if sd > 0 else np.nan
        smds.append(smd)
        P("    %-20s %10s %10s %10.1f %10.3f" % (lab, sa, sb, vb - va, smd))
        ind_rows.append(dict(disease=g, variable=lab, none=sa, high=sb,
                             gap=round(float(vb - va), 3), smd=round(float(smd), 3)))
    P("    %-20s %10s %10s %10s %10.3f" % ("MEAN |SMD| severity gap", "", "",
                                          "", np.nanmean(np.abs(smds))))

# =====================================================================
# C. within-disease dose response
# =====================================================================
P("")
P("=" * 100)
P("C. DOSE-RESPONSE WITHIN EACH DISEASE   (dose-trend = OR per stratum")
P("   increment across 0 / >0-<10 / 10-<50 / >=50 mg/day)")
P("=" * 100)

fits = {}
for g in ["SLE", "RA"]:
    d = df[df.primary_grp == g].copy()
    d, ess = iptw(d, PSCOV)
    fits[g] = (d, ess)

dose_rows = []
for y, lab in OUTCOMES:
    P("")
    P("  outcome = %s" % lab)
    P("  %-8s %6s %7s %24s %24s %24s" % ("disease", "n", "events", "crude trend",
                                          "adjusted trend", "IPTW trend"))
    for g in ["SLE", "RA"]:
        d, ess = fits[g]
        sub = d[[y, "gc_str_num"]].dropna()
        if sub[y].nunique() < 2 or sub.gc_str_num.nunique() < 2:
            P("  %-8s %6d %7d  not estimable" % (g, len(sub), int(sub[y].sum())))
            continue
        c0 = fit_logit(d, y, ["gc_str_num"])
        c1 = fit_logit(d, y, ["gc_str_num"] + CORE)
        # IPTW: weighted GLM on the ordinal score
        try:
            m = smf.glm("%s ~ gc_str_num" % y, data=d, family=sm.families.Binomial(),
                        freq_weights=d["sw"]).fit()
            b, se = float(m.params["gc_str_num"]), float(m.bse["gc_str_num"])
            c2 = dict(OR=float(np.exp(b)), lo=float(np.exp(b - 1.96 * se)),
                      hi=float(np.exp(b + 1.96 * se)),
                      p=float(2 * (1 - stats.norm.cdf(abs(b / se)))))
        except Exception:
            c2 = None
        P("  %-8s %6d %7d %24s %24s %24s" % (g, len(sub), int(sub[y].sum()),
                                              or_line(c0), or_line(c1), or_line(c2)))
        dose_rows.append(dict(outcome=lab, disease=g, n=len(sub), n_ev=int(sub[y].sum()),
                              crude_or=(c0 or {}).get("OR"), crude_p=(c0 or {}).get("p"),
                              adj_or=(c1 or {}).get("OR"), adj_p=(c1 or {}).get("p"),
                              adj_lo=(c1 or {}).get("lo"), adj_hi=(c1 or {}).get("hi"),
                              iptw_or=(c2 or {}).get("OR"), iptw_p=(c2 or {}).get("p")))

# =====================================================================
# D. interaction -- the head-to-head test
# =====================================================================
P("")
P("=" * 100)
P("D. THE HEAD-TO-HEAD TEST: disease x dose interaction")
P("=" * 100)
P("  Interaction OR = ratio of dose-trend ORs (RA relative to SLE) per")
P("  stratum increment.  OR 1 = the dose-response is the same in both")
P("  diseases.  Values >1 mean the dose-response is steeper in RA.")
P("")
P("  %-34s %6s %7s %26s %10s %10s" % ("outcome", "n", "events",
                                        "RA/SLE trend OR (95% CI)", "P Wald", "P LRT"))
int_rows = []
for y, lab in OUTCOMES:
    r = interaction_test(both, y, CORE)
    if r is None:
        P("  %-34s   not estimable" % lab)
        continue
    P("  %-34s %6d %7d %26s %10.3f %10.3f" % (lab, r["n"], r["n_ev"],
                                              "%.2f (%.2f-%.2f)" % (r["OR"], r["lo"], r["hi"]),
                                              r["p_wald"], r["p_lrt"]))
    int_rows.append(dict(outcome=lab, **{k: r[k] for k in
                                         ("OR", "lo", "hi", "p_wald", "p_lrt", "n", "n_ev")}))

P("")
P("  same, with dose as a 4-level factor (SAFE per-disease ORs, ref = none):")
P("  %-10s %-10s %22s %22s %22s" % ("outcome", "disease", ">0-<10", "10-<50", ">=50"))
per_rows = []
for y, lab in OUTCOMES:
    for g in ["SLE", "RA"]:
        d = df[df.primary_grp == g]
        parts = []
        for gs in ORDER[1:]:
            sub = d[d.gc24_str.isin(["G0_none", gs])].copy()
            sub["e"] = (sub.gc24_str == gs).astype(int)
            r = fit_logit(sub, y, ["e"])
            parts.append(or_line(r))
            if r:
                per_rows.append(dict(outcome=lab, disease=g, stratum=LABEL[gs],
                                     OR=r["OR"], lo=r["lo"], hi=r["hi"], p=r["p"],
                                     n_ev=r["n_ev"], n=r["n"]))
        P("  %-10s %-10s %22s %22s %22s" % (y[:10], g, parts[0], parts[1], parts[2]))

# ------------------------------------------------- D2: IPTW interaction
P("")
P("  D2. interaction under IPTW weights (weights refit within each disease,")
P("      pooled model with s:is_ra -- the coefficient is the RA/SLE ratio")
P("  %-34s %6s %7s %26s %10s" % ("outcome", "n", "events", "RA/SLE trend OR (95% CI)", "P"))
pool = pd.concat([fits["SLE"][0], fits["RA"][0]], ignore_index=True)
pool["is_ra"] = (pool.primary_grp == "RA").astype(int)
pool["s"] = pool.gc_str_num.astype(float)
iptw_int = []
for y, lab in OUTCOMES:
    use = pool[["s", "is_ra", y, "sw"]].dropna()
    if use[y].nunique() < 2:
        continue
    try:
        m = smf.glm("%s ~ s + is_ra + s:is_ra" % y, data=use,
                    family=sm.families.Binomial(), freq_weights=use["sw"]).fit()
        t = [k for k in m.params.index if "s:is_ra" in k][0]
        b, se = float(m.params[t]), float(m.bse[t])
        P("  %-34s %6d %7d %26s %10.3f" % (
            lab, len(use), int(use[y].sum()),
            "%.2f (%.2f-%.2f)" % (np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)),
            float(m.pvalues[t])))
        iptw_int.append(dict(outcome=lab, OR=float(np.exp(b)),
                             lo=float(np.exp(b - 1.96 * se)), hi=float(np.exp(b + 1.96 * se)),
                             p=float(m.pvalues[t]), n=int(len(use)), n_ev=int(use[y].sum())))
    except Exception as e:
        P("  %-34s failed: %s" % (lab, str(e)[:60]))

# -------------------------------------- D3: consistency self-check
P("")
P("  D3. SELF-CHECK: the interaction coefficient must agree with the ratio of")
P("      the separately fitted per-disease trends.  This is what catches an")
P("      inverted interaction orientation -- always run it.")
P("  %-10s %12s %12s %12s %12s" % ("outcome", "SLE trend", "RA trend",
                                    "RA/SLE ratio", "interaction"))
P("  %-10s %12s %12s %12s %12s" % ("", "(adj)", "(adj)", "from strata", "OR (adj)"))
for (y, lab) in OUTCOMES:
    d_sle = df[df.primary_grp == "SLE"]
    d_ra = df[df.primary_grp == "RA"]
    a = fit_logit(d_sle, y, ["gc_str_num"] + CORE)
    b2 = fit_logit(d_ra, y, ["gc_str_num"] + CORE)
    r = interaction_test(both, y, CORE)
    if a is None or b2 is None or r is None:
        continue
    P("  %-10s %12.2f %12.2f %12.2f %12.2f   %s" % (
        y[:10], a["OR"], b2["OR"], b2["OR"] / a["OR"], r["OR"],
        "OK" if abs(np.log(b2["OR"] / a["OR"]) - np.log(r["OR"])) < 0.25 else "<<< CHECK"))
P("")
P("  (exact equality is not expected: the pooled interaction model imposes a")
P("   common covariate effect, the stratified fits do not.  Only the")
P("   DIRECTION and rough magnitude should agree -- a sign flip means the")
P("   interaction coefficient has been inverted.)")

# =====================================================================
# E. positive control
# =====================================================================
P("")
P("=" * 100)
P("E. POSITIVE CONTROL: does the exposure variable still 'work' in both")
P("   diseases?  (GC -> hyperglycaemia; a failed control invalidates the rows")
P("   above for that disease)")
P("=" * 100)
P("  %-10s %6s %8s %26s %26s" % ("disease", "n", "events", "any-GC OR",
                                  "dose-trend OR per stratum"))
pc = []
for g in ["SLE", "RA"]:
    d = df[df.primary_grp == g]
    r_any = fit_logit(d, "hyper_48h", ["gc_any24"])
    r_tr = fit_logit(d, "hyper_48h", ["gc_str_num"])
    P("  %-10s %6d %8d %26s %26s" % (g, len(d), int(d.hyper_48h.sum()),
                                     or_line(r_any), or_line(r_tr)))
    pc.append(dict(disease=g, n=len(d), n_ev=int(d.hyper_48h.sum()),
                   any_or=(r_any or {}).get("OR"), any_p=(r_any or {}).get("p"),
                   trend_or=(r_tr or {}).get("OR"), trend_p=(r_tr or {}).get("p")))
P("")
P("  glucose measurement coverage by disease (the control needs a glucose):")
for g in ["SLE", "RA"]:
    d = df[df.primary_grp == g]
    P("    %-6s %d / %d measured (%.1f%%)" % (g, d.glu_max_48h.notna().sum(), len(d),
                                              100 * d.glu_max_48h.notna().mean()))

# =====================================================================
# F. wider spectrum: does the association track GC dependence?
# =====================================================================
P("")
P("=" * 100)
P("F. WIDER SPECTRUM: does the dose-infection association track how much the")
P("   disease DEPENDS on glucocorticoids as baseline therapy?")
P("=" * 100)
P("  GC dependence is measured outside the ICU cohort: the share of ALL")
P("  hospitalisations for that disease in which a systemic GC was ordered.")
sp = []
try:
    cls = pd.read_csv(os.path.join(DATA, "rheum_hadm_classified.csv"))
    cls = cls[cls.primary != "none"][["hadm_id", "primary"]]
    cc = psycopg2.connect(dbname="mimiciv", **CONFIG)
    cur = cc.cursor()
    cur.execute("CREATE TEMP TABLE rheum(hadm_id BIGINT PRIMARY KEY, primary_grp TEXT)")
    execute_values(cur, "INSERT INTO rheum (hadm_id, primary_grp) VALUES %s",
                   list(cls.itertuples(index=False, name=None)), page_size=5000)
    cc.commit()
    q = r"""
    SELECT r.primary_grp,
           COUNT(DISTINCT r.hadm_id) AS n_hadm,
           COUNT(DISTINCT g.hadm_id) AS n_gc_hadm
    FROM rheum r
    LEFT JOIN (SELECT DISTINCT pr.hadm_id FROM mimiciv_hosp.prescriptions pr
               WHERE pr.dose_unit_rx = 'mg' AND pr.route IN ('IV','PO','PO/NG','IM')
                 AND (pr.drug ILIKE '%%predni%%' OR pr.drug ILIKE '%%methylpred%%'
                   OR pr.drug ILIKE '%%dexameth%%' OR pr.drug ILIKE '%%hydrocort%%'
                   OR pr.drug ILIKE '%%betameth%%' OR pr.drug ILIKE '%%triamcin%%')) g
         ON r.hadm_id = g.hadm_id
    GROUP BY r.primary_grp
    """
    dep = pd.read_sql_query(q, cc)
    cc.close()
    dep["gc_dep"] = dep.n_gc_hadm / dep.n_hadm
    P("")
    P("  %-14s %8s %10s %12s %10s %10s %26s" % (
        "disease", "n hadm", "n w/ GC", "GC dependence", "n24", "events",
        "dose-trend OR (infection ICD)"))
    for _, r in dep.sort_values("gc_dep", ascending=False).iterrows():
        g = r.primary_grp
        s = df[df.primary_grp == g]
        if len(s) < 80:
            P("  %-14s %8d %10d %11.1f%% %10d %10d   (too few for a trend)" % (
                g, r.n_hadm, r.n_gc_hadm, 100 * r.gc_dep, len(s), int(s.infect_icd.sum())))
            continue
        tr = fit_logit(s, "infect_icd", ["gc_str_num"])
        P("  %-14s %8d %10d %11.1f%% %10d %10d %26s" % (
            g, r.n_hadm, r.n_gc_hadm, 100 * r.gc_dep, len(s), int(s.infect_icd.sum()),
            or_line(tr)))
        sp.append(dict(disease=g, n_hadm=int(r.n_hadm), gc_dep=float(r.gc_dep),
                       n24=len(s), n_ev=int(s.infect_icd.sum()),
                       trend_or=(tr or {}).get("OR"), trend_lo=(tr or {}).get("lo"),
                       trend_hi=(tr or {}).get("hi"), trend_p=(tr or {}).get("p")))
    dep.to_csv(os.path.join(OUT, "table_gc_dependence.csv"), index=False)

    # ------------------------------------------------ formal gradient test
    if len(sp) >= 5:
        sd_ = pd.DataFrame(sp).dropna(subset=["trend_or", "trend_lo", "trend_hi"])
        sd_ = sd_[sd_.trend_lo > 0].copy()
        sd_["logOR"] = np.log(sd_.trend_or)
        sd_["se"] = (np.log(sd_.trend_hi) - np.log(sd_.trend_lo)) / (2 * 1.96)
        sd_ = sd_[sd_.se > 0]
        rho, prho = stats.spearmanr(sd_.gc_dep, sd_.logOR)
        w = 1.0 / sd_.se.values ** 2
        Xw = np.column_stack([np.ones(len(sd_)), sd_.gc_dep.values])
        cov = np.linalg.pinv(Xw.T @ (w[:, None] * Xw))
        beta = cov @ (Xw.T @ (w * sd_.logOR.values))
        se_b = np.sqrt(np.abs(cov[1, 1]))
        z = beta[1] / se_b
        P("")
        P("  FORMAL GRADIENT TEST  (k=%d diseases with an estimable trend)" % len(sd_))
        P("    Spearman rho (GC dependence vs log dose-trend OR) = %.3f   P=%.3f" % (rho, prho))
        P("    inverse-variance weighted slope per +10 pp GC dependence:")
        P("      OR %.3f (%.3f-%.3f)  P=%.4f" % (
            np.exp(0.10 * beta[1]), np.exp(0.10 * (beta[1] - 1.96 * se_b)),
            np.exp(0.10 * (beta[1] + 1.96 * se_b)),
            float(2 * (1 - stats.norm.cdf(abs(z))))))
        P("    -> where GC is baseline therapy a dose increment carries little")
        P("       infection signal; where GC is not baseline it does.")
        json_grad = dict(k=int(len(sd_)), spearman=float(rho), spearman_p=float(prho),
                         slope_per_10pp=float(np.exp(0.10 * beta[1])),
                         lo=float(np.exp(0.10 * (beta[1] - 1.96 * se_b))),
                         hi=float(np.exp(0.10 * (beta[1] + 1.96 * se_b))),
                         p=float(2 * (1 - stats.norm.cdf(abs(z)))))
    else:
        json_grad = None
except Exception as e:
    P("  GC-dependence query failed: %s" % str(e)[:200])
    json_grad = None

# ---------------------------------------------------------------- save
pd.DataFrame(base_rows).to_csv(os.path.join(OUT, "table_head2head_baseline.csv"), index=False)
pd.DataFrame(ind_rows).to_csv(os.path.join(OUT, "table_head2head_indication.csv"), index=False)
pd.DataFrame(age_rows).to_csv(os.path.join(OUT, "table_head2head_age.csv"), index=False)
pd.DataFrame(dose_rows).to_csv(os.path.join(OUT, "table_head2head_dose.csv"), index=False)
pd.DataFrame(int_rows).to_csv(os.path.join(OUT, "table_head2head_interaction.csv"), index=False)
pd.DataFrame(per_rows).to_csv(os.path.join(OUT, "table_head2head_strata.csv"), index=False)
pd.DataFrame(sp).to_csv(os.path.join(OUT, "table_head2head_spectrum.csv"), index=False)

with open(os.path.join(OUT, "_head2head_results.json"), "w", encoding="utf-8") as f:
    json.dump(dict(baseline=base_rows, indication=ind_rows, age=age_rows,
                   dose=dose_rows, interaction=int_rows, iptw_interaction=iptw_int,
                   strata=per_rows, positive_control=pc, spectrum=sp,
                   gc_dependence_gradient=json_grad), f,
              ensure_ascii=False, indent=1, default=str)

with open(os.path.join(OUT, "152_head2head.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nsaved -> out/152_head2head.txt")
