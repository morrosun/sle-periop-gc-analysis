# -*- coding: utf-8 -*-
"""
Plan A - main analysis
======================
Exposure : prednisone-equivalent dose intensity (mg/day) in the ICU exposure window
           strata 0 / >0-<10 / 10-<50 / >=50  (sensitivity: tertiles among exposed)
Outcomes : in-hospital infection (primary), 30-day death (secondary)
Models   : crude / age-sex / +severity / +organ support & SLE proxies
           non-monotonic dose-response via restricted cubic splines + quadratic term
           bootstrap marginal standardisation for adjusted risk differences
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "out")
rng = np.random.default_rng(20260916)

df = pd.read_csv(os.path.join(DATA, "cohort_sle_icu_v3.csv"))

# ------------------------------------------------------------------ exposure strata
CUTS = [0, 10, 50]
def stratum(v):
    if v <= 0:
        return "G0_none"
    if v < 10:
        return "G1_low"
    if v < 50:
        return "G2_mod"
    return "G3_high"

df["gc_str"] = df["gc_daily_pe_mg"].apply(stratum)
df["gc_any"] = (df["gc_daily_pe_mg"] > 0).astype(int)
df["imm_any"] = (df[["imm_cyc", "imm_mmf", "imm_aza", "imm_cni", "imm_mtx", "imm_bio"]].max(axis=1) == 1).astype(int)
df["emerg"] = df["admission_type"].astype(str).str.contains("EMER", case=False, na=False).astype(int)
df["female"] = (df["gender"] == "F").astype(int)
df["sofa"] = df["sofa"].fillna(df["sofa"].median())
df["charlson"] = df["charlson"].fillna(df["charlson"].median())
df["apsiii"] = df["apsiii"].fillna(df["apsiii"].median())
df["pni"] = 10 * df["albumin_min"] + 0.005 * df["abs_lymphocytes_min"] * 1000.0

ORDER = ["G0_none", "G1_low", "G2_mod", "G3_high"]
LABEL = {"G0_none": "0 (none)", "G1_low": ">0-<10", "G2_mod": "10-<50", "G3_high": ">=50"}
df["gc_str"] = pd.Categorical(df["gc_str"], categories=ORDER, ordered=True)

lines = []
def P(s=""):
    print(s)
    lines.append(str(s))

P("=" * 96)
P("SLE ICU cohort - Plan A main analysis")
P("=" * 96)
P("N stays = %d | N unique admissions = %d | N subjects = %d" %
  (len(df), df.hadm_id.nunique(), df.subject_id.nunique()))

# ------------------------------------------------------------------ Table 1
P("\n" + "=" * 96)
P("TABLE 1  Baseline characteristics by glucocorticoid dose intensity (mg/day prednisone-equiv)")
P("=" * 96)

def desc(series, dec=1):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return "-"
    if s.nunique() <= 2 and set(s.unique()).issubset({0, 1}):
        return "%d (%.1f%%)" % (int(s.sum()), 100 * s.mean())
    return "%.1f (%.1f)" % (s.mean(), s.std())

rows = [
    ("n", lambda d: pd.Series([len(d)] * len(d))),
    ("Age, years", "age"),
    ("Female", "female"),
    ("Charlson index", "charlson"),
    ("SOFA (max 48 h)", "sofa"),
    ("APS III", "apsiii"),
    ("OASIS", "oasis"),
    ("Mechanical ventilation (48 h)", "on_vent"),
    ("Vasopressor (48 h)", "on_vaso"),
    ("Renal replacement therapy", "on_rrt"),
    ("Lupus nephritis", "lupus_nephritis"),
    ("Any immunosuppressant", "imm_any"),
    ("Hydroxychloroquine", "imm_hcq"),
    ("Emergency admission", "emerg"),
    ("GC before ICU admission", "gc_preicu"),
    ("WBC min, x10^9/L", "wbc_min"),
    ("Platelets min, x10^9/L", "platelets_min"),
    ("Creatinine max, mg/dL", "creatinine_max"),
    ("Albumin min, g/dL", "albumin_min"),
    ("PNI", "pni"),
]

hdr = "%-34s" % "Variable"
for g in ORDER:
    hdr += "%18s" % (LABEL[g] + " (n=%d)" % int((df.gc_str == g).sum()))
hdr += "%12s" % "P value"
P(hdr)
P("-" * 96)

t1_rows = []
for item in rows:
    name = item[0] if isinstance(item, tuple) else item
    col = item[1] if isinstance(item, tuple) else item
    line = "%-34s" % name
    groups = []
    for g in ORDER:
        d = df[df.gc_str == g]
        if name == "n":
            line += "%18s" % len(d)
            continue
        line += "%18s" % desc(d[col])
        groups.append(pd.to_numeric(d[col], errors="coerce").dropna())
    if name != "n":
        try:
            if len(groups) == 4 and all(len(x) > 1 for x in groups):
                if set(pd.concat(groups).unique()).issubset({0, 1}):
                    tab = np.array([[int(x.sum()), len(x) - int(x.sum())] for x in groups])
                    chi2, p, _, _ = stats.chi2_contingency(tab)
                else:
                    f, p = stats.f_oneway(*groups)
            else:
                p = np.nan
        except Exception:
            p = np.nan
        line += "%12s" % ("%.3f" % p if p == p else "-")
        t1_rows.append((name, p))
    P(line)

# ------------------------------------------------------------------ outcome by stratum
P("\n" + "=" * 96)
P("Crude outcome rates by stratum")
P("=" * 96)
P("%-12s %6s %14s %14s %14s %10s" % ("stratum", "n", "infection n (%)", "d30 n (%)", "d90 n (%)", "SOFA"))
for g in ORDER:
    d = df[df.gc_str == g]
    P("%-12s %6d %14s %14s %14s %10.1f" % (
        LABEL[g], len(d),
        "%d (%.1f)" % (d.infection.sum(), 100 * d.infection.mean()),
        "%d (%.1f)" % (d.death_30d.sum(), 100 * d.death_30d.mean()),
        "%d (%.1f)" % (d.death_90d.sum(), 100 * d.death_90d.mean()),
        d.sofa.mean()))

# ------------------------------------------------------------------ regression models
BASE_COV = "age + female + charlson + sofa + on_vent + on_vaso + on_rrt + lupus_nephritis + imm_any + emerg"

def fit_logit(formula, data, label):
    m = smf.logit(formula, data=data).fit(disp=0)
    return m

def or_table(m, var_prefix, ref_label, labels):
    out = []
    for name in m.params.index:
        if not name.startswith(var_prefix):
            continue
        beta = m.params[name]
        se = m.bse[name]
        p = m.pvalues[name]
        orr = np.exp(beta)
        lo, hi = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
        key = name.replace(var_prefix, "").replace("]", "").replace("[", "")
        out.append((LABEL.get(key, key), orr, lo, hi, p))
    return out

P("\n" + "=" * 96)
P("PRIMARY OUTCOME - in-hospital infection (logistic regression, reference = no GC)")
P("=" * 96)

models = {
    "Crude":                 "infection ~ C(gc_str, Treatment(reference='G0_none'))",
    "Model 1 (age, sex)":    "infection ~ C(gc_str, Treatment(reference='G0_none')) + age + female",
    "Model 2 (+ severity)":  "infection ~ C(gc_str, Treatment(reference='G0_none')) + age + female + charlson + sofa + apsiii",
    "Model 3 (full)":        "infection ~ C(gc_str, Treatment(reference='G0_none')) + " + BASE_COV,
}

main_res = {}
for label, f in models.items():
    m = fit_logit(f, df, label)
    main_res[label] = m
    P("\n%s   (n=%d, events=%d)" % (label, int(m.nobs), int(df.infection.sum())))
    P("%-16s %10s %20s %10s" % ("stratum", "OR", "95% CI", "P"))
    for lab, orr, lo, hi, p in or_table(m, "C(gc_str, Treatment(reference='G0_none'))[T.", "G0_none", ORDER):
        P("%-16s %10.2f %20s %10.3f" % (lab, orr, "%.2f-%.2f" % (lo, hi), p))

mfull = main_res["Model 3 (full)"]

# ------------------------------------------------------------------ non-monotonic dose-response
P("\n" + "=" * 96)
P("NON-MONOTONIC DOSE-RESPONSE TEST")
P("=" * 96)

sub = df[df.gc_any == 1].copy()
sub["ldose"] = np.log(sub["gc_daily_pe_mg"] + 1)

# (a) quadratic term on log-dose among exposed
try:
    mq = smf.logit("infection ~ ldose + I(ldose**2) + " + BASE_COV, data=sub).fit(disp=0)
    b2, se2, p2 = mq.params["I(ldose ** 2)"], mq.bse["I(ldose ** 2)"], mq.pvalues["I(ldose ** 2)"]
    P("\n(a) Quadratic term for log-dose among exposed (n=%d, events=%d)" % (len(sub), int(sub.infection.sum())))
    P("    beta_quadratic = %.4f (SE %.4f), P = %.4f" % (b2, se2, p2))
    # vertex
    b1 = mq.params["ldose"]
    if b2 < 0:
        vtx = np.exp(-b1 / (2 * b2) - 1)
        P("    concave (inverted-U); vertex at dose = %.1f mg/day" % vtx)
    else:
        P("    convex (U-shaped) on log scale")
except Exception as e:
    P("    quadratic model failed: %s" % e)

# (b) restricted cubic spline on all patients with a zero-dose reference
try:
    from patsy import dmatrix
    def rcs_basis(x, knots):
        x = np.asarray(x, dtype=float)
        k = np.asarray(knots, dtype=float)
        n = len(k)
        out = [x]
        for j in range(n - 2):
            d = (np.maximum(x - k[j], 0) ** 3
                 - np.maximum(x - k[-2], 0) ** 3 * (k[n - 1] - k[j]) / (k[n - 1] - k[n - 2])
                 + np.maximum(x - k[n - 1], 0) ** 3 * (k[n - 2] - k[j]) / (k[n - 1] - k[n - 2]))
            out.append(d / (k[n - 1] - k[0]) ** 2)
        return np.column_stack(out)

    allp = df.copy()
    z = allp["gc_daily_pe_mg"].values
    exposed = z > 0
    knots = np.percentile(z[exposed], [5, 35, 65, 95])
    B = rcs_basis(np.log(z[exposed] + 1), np.log(knots + 1))
    Xrcs = np.zeros((len(allp), B.shape[1]))
    Xrcs[exposed] = B
    for i in range(B.shape[1]):
        allp["rcs%d" % i] = Xrcs[:, i]
    rcs_terms = " + ".join(["rcs%d" % i for i in range(B.shape[1])])
    f_rcs = "infection ~ " + rcs_terms + " + " + BASE_COV
    m_rcs = smf.logit(f_rcs, data=allp).fit(disp=0)

    # overall / nonlinear Wald tests on the spline block
    names = ["rcs%d" % i for i in range(B.shape[1])]
    R = np.zeros((len(names), len(m_rcs.params)))
    for i, nm in enumerate(names):
        R[i, list(m_rcs.params.index).index(nm)] = 1
    w_overall = m_rcs.wald_test(R, scalar=True)
    R2 = np.zeros((len(names) - 1, len(m_rcs.params)))
    for i in range(1, len(names)):
        R2[i - 1, list(m_rcs.params.index).index(names[i])] = 1
    w_nonlin = m_rcs.wald_test(R2, scalar=True)
    P("\n(b) Restricted cubic spline (4 knots at 5/35/65/95th pct of exposed dose), reference = zero dose")
    P("    overall dose-response  : chi2 = %.2f, df = %d, P = %.4f" %
      (w_overall.statistic, len(names), w_overall.pvalue))
    P("    NON-LINEAR component   : chi2 = %.2f, df = %d, P = %.4f" %
      (w_nonlin.statistic, len(names) - 1, w_nonlin.pvalue))

    # predicted OR vs zero dose across the dose range
    grid = np.percentile(z[exposed], np.linspace(1, 99, 60))
    Bg = rcs_basis(np.log(grid + 1), np.log(knots + 1))
    beta = np.array([m_rcs.params["rcs%d" % i] for i in range(B.shape[1])])
    lp = Bg @ beta
    se_lp = np.sqrt(np.einsum("ij,jk,ik->i", Bg, m_rcs.cov_params().loc[names, names].values, Bg))
    spline_curve = pd.DataFrame({
        "dose": grid, "logOR": lp,
        "lo": lp - 1.96 * se_lp, "hi": lp + 1.96 * se_lp,
        "OR": np.exp(lp), "ORlo": np.exp(lp - 1.96 * se_lp), "ORhi": np.exp(lp + 1.96 * se_lp)})
    spline_curve.to_csv(os.path.join(OUT, "spline_infection.csv"), index=False)
    P("    spline curve saved -> out/spline_infection.csv")
    P("    OR at selected doses vs zero dose:")
    for d in [5, 10, 25, 50, 100, 250, 500]:
        i = (np.abs(grid - d)).argmin()
        r = spline_curve.iloc[i]
        P("      %6.0f mg/day : OR %.2f (%.2f-%.2f)" % (d, r.OR, r.ORlo, r.ORhi))
except Exception as e:
    import traceback
    P("    spline failed: %s" % e)
    traceback.print_exc()

# (c) stratum pattern: is any non-reference stratum below 1 while a higher one is above?
P("\n(c) Stratum pattern from the full model (monotonicity check)")
ests = []
for lab, orr, lo, hi, p in or_table(mfull, "C(gc_str, Treatment(reference='G0_none'))[T.", "G0_none", ORDER):
    ests.append((lab, orr, p))
for lab, orr, p in ests:
    P("    %-10s OR %.2f  P=%.3f" % (lab, orr, p))

# ------------------------------------------------------------------ adjusted risk difference
P("\n" + "=" * 96)
P("ADJUSTED ABSOLUTE RISK DIFFERENCE vs no GC (bootstrap marginal standardisation, 1000 reps)")
P("=" * 96)

def ard_bootstrap(data, formula, group_col, target_levels, ref, n_boot=1000):
    res = {}
    d0 = data.copy()
    for lvl in target_levels:
        d0[group_col] = pd.Categorical(d0[group_col], categories=ORDER, ordered=False)
    m = smf.logit(formula, data=d0).fit(disp=0)
    preds = {}
    for lvl in ORDER:
        dd = d0.copy()
        dd[group_col] = lvl
        preds[lvl] = m.predict(dd).mean()
    for lvl in target_levels:
        res[lvl] = (preds[lvl], preds[lvl] - preds[ref])
    boots = {lvl: [] for lvl in target_levels}
    n = len(d0)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        db = d0.iloc[idx].copy()
        try:
            mb = smf.logit(formula, data=db).fit(disp=0)
            pb = {}
            for lvl in ORDER:
                dd = db.copy()
                dd[group_col] = lvl
                pb[lvl] = mb.predict(dd).mean()
            for lvl in target_levels:
                boots[lvl].append(pb[lvl] - pb[ref])
        except Exception:
            continue
    out = []
    for lvl in target_levels:
        arr = np.array(boots[lvl])
        lo, hi = np.percentile(arr, [2.5, 97.5]) if len(arr) else (np.nan, np.nan)
        out.append((lvl, res[lvl][0], res[lvl][1], lo, hi, len(arr)))
    return out

ard = ard_bootstrap(df, "infection ~ C(gc_str) + " + BASE_COV, "gc_str",
                    ["G1_low", "G2_mod", "G3_high"], "G0_none", n_boot=1000)
P("%-12s %22s %26s" % ("stratum", "standardised risk", "ARD vs none (95% CI)"))
for lvl, risk, d, lo, hi, nb in ard:
    P("%-12s %22.1f%% %26s" % (LABEL[lvl], 100 * risk, "%+.1f pp (%+.1f to %+.1f)" % (100 * d, 100 * lo, 100 * hi)))
P("(bootstrap reps = %d)" % (ard[0][5] if ard else 0))

# ------------------------------------------------------------------ secondary: 30-day death
P("\n" + "=" * 96)
P("SECONDARY OUTCOME - 30-day mortality")
P("=" * 96)
P("NOTE: %d events; models are exploratory (events-per-variable limit reached)" % int(df.death_30d.sum()))

P("\nBinary exposure (any GC vs none):")
for label, f in [
    ("Crude", "death_30d ~ gc_any"),
    ("Age-sex", "death_30d ~ gc_any + age + female"),
    ("+ severity", "death_30d ~ gc_any + age + female + charlson + sofa"),
    ("+ full", "death_30d ~ gc_any + " + BASE_COV)]:
    m = fit_logit(f, df, label)
    b, se, p = m.params["gc_any"], m.bse["gc_any"], m.pvalues["gc_any"]
    P("  %-12s OR %.2f (%.2f-%.2f)  P=%.3f" % (label, np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), p))

P("\nDose strata (exploratory, reference = no GC):")
for label, f in [
    ("Crude", "death_30d ~ C(gc_str, Treatment(reference='G0_none'))"),
    ("Adjusted", "death_30d ~ C(gc_str, Treatment(reference='G0_none')) + age + female + charlson + sofa + on_vaso + on_rrt")]:
    m = fit_logit(f, df, label)
    P("  %s" % label)
    for lab, orr, lo, hi, p in or_table(m, "C(gc_str, Treatment(reference='G0_none'))[T.", "G0_none", ORDER):
        P("     %-10s OR %.2f (%.2f-%.2f)  P=%.3f" % (lab, orr, lo, hi, p))

# ------------------------------------------------------------------ save
with open(os.path.join(OUT, "20_main_results.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nsaved -> out/20_main_results.txt")
