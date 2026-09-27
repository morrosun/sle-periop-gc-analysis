# -*- coding: utf-8 -*-
"""
The bacteraemia signal survived the control panel, but it can still be an artefact
of surveillance intensity: patients on higher steroid doses are monitored harder,
so more blood cultures are drawn, so more positives are found.

This script measures that directly and adjusts for it.

Also: Fix 1 as first implemented made covariate balance WORSE (max|SMD| 0.160 -> 0.264),
because 12 extra covariates inflate the weighting problem at n=433. Here the activity
set is made parsimonious and the weights are trimmed.
"""
import os
import numpy as np
import pandas as pd
import psycopg2
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, OUT = os.path.join(BASE, "data"), os.path.join(BASE, "out")
CONFIG = dict(
    host=os.environ.get("MIMIC_DB_HOST", "localhost"),
    port=int(os.environ.get("MIMIC_DB_PORT", "5432")),
    user=os.environ.get("MIMIC_DB_USER", "postgres"),
    password=os.environ.get("MIMIC_DB_PASSWORD", ""),
)
rng = np.random.default_rng(20260916)

ORDER = ["G0_none", "G1_low", "G2_mod", "G3_high"]
LABEL = {"G0_none": "0 (none)", "G1_low": ">0-<10", "G2_mod": "10-<50", "G3_high": ">=50"}

# ------------------------------------------------------------------ culture intensity
SQL = r"""
WITH sle AS (
    SELECT DISTINCT d.hadm_id FROM mimiciv_hosp.diagnoses_icd d
    WHERE (d.icd_version=10 AND d.icd_code LIKE 'M32%') OR (d.icd_version=9 AND d.icd_code LIKE '7100%')
),
icu AS (
    SELECT i.hadm_id, i.stay_id, i.intime,
           ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.intime) AS rn
    FROM mimiciv_icu.icustays i JOIN sle ON i.hadm_id = sle.hadm_id
),
lm AS (SELECT * FROM icu WHERE rn=1)
SELECT m.hadm_id,
       COUNT(*) FILTER (WHERE m.charttime >= lm.intime + INTERVAL '24 hours') AS ncx_after24,
       COUNT(DISTINCT m.test_seq) FILTER (WHERE m.charttime >= lm.intime + INTERVAL '24 hours') AS nspec_after24,
       COUNT(*) FILTER (WHERE m.charttime >= lm.intime + INTERVAL '24 hours' AND m.org_name IS NOT NULL) AS npos_after24,
       COUNT(*) FILTER (WHERE m.charttime >= lm.intime) AS ncx_after0
FROM mimiciv_hosp.microbiologyevents m JOIN lm ON m.hadm_id = lm.hadm_id
WHERE m.spec_type_desc ILIKE '%BLOOD%' AND m.charttime IS NOT NULL
GROUP BY m.hadm_id
"""

lines = []
def P(s=""):
    print(s); lines.append(str(s))

c = psycopg2.connect(dbname="mimiciv", **CONFIG)
cx = pd.read_sql_query(SQL, c)
c.close()
cx.to_csv(os.path.join(DATA, "culture_intensity.csv"), index=False)

v4 = pd.read_csv(os.path.join(DATA, "cohort_v4_fixes.csv"))
v3 = pd.read_csv(os.path.join(DATA, "cohort_sle_icu_v3.csv"))
keep = ["hadm_id", "stay_id", "gender", "charlson", "sofa", "apsiii", "on_vent", "on_vaso",
        "on_rrt", "admission_type", "wbc_min", "platelets_min", "creatinine_max",
        "imm_cyc", "imm_mmf", "imm_aza", "imm_cni", "imm_mtx", "imm_hcq", "imm_bio"]
df = v4.merge(v3[keep], on=["hadm_id", "stay_id"], how="left").merge(cx, on="hadm_id", how="left")
for v in ["ncx_after24", "nspec_after24", "npos_after24", "ncx_after0"]:
    df[v] = df[v].fillna(0)
df["female"] = (df["gender"] == "F").astype(int)
df["emerg"] = df["admission_type"].astype(str).str.contains("EMER", case=False, na=False).astype(int)
df["imm_any"] = (df[["imm_cyc", "imm_mmf", "imm_aza", "imm_cni", "imm_mtx", "imm_bio"]].max(axis=1) == 1).astype(int)
df["imm_hcq"] = df["imm_hcq"].fillna(0)
for v in ["sofa", "charlson", "apsiii", "wbc_min", "platelets_min", "creatinine_max"]:
    df[v] = df[v].fillna(df[v].median())

lm = df[(df.los24 == 1) & (df.abx_before_icu == 0) & (df.culture_pre_icu == 0)].copy()
lm["gc_str"] = pd.Categorical(lm["gc24_str"], categories=ORDER)
# how many patients had ANY blood culture drawn after the landmark
lm["any_cx_after24"] = (lm.ncx_after24 > 0).astype(int)

P("=" * 100)
P("DETECTION / SURVEILLANCE BIAS IN THE BACTERAEMIA OUTCOME")
P("=" * 100)
P("\nlandmark cohort without prevalent infection: n = %d" % len(lm))
P("\n%-14s %6s %10s %12s %12s %12s" % ("GC dose", "n", "any cx", "mean cx", "n pos", "% pos (of drawn)"))
for g in ORDER:
    s = lm[lm.gc_str == g]
    drawn = s[s.any_cx_after24 == 1]
    P("%-14s %6d %10s %12.2f %12d %12s" % (
        LABEL[g], len(s), "%.1f%%" % (100 * s.any_cx_after24.mean()), s.ncx_after24.mean(),
        int(s.culture_after24.sum()),
        "%.1f%%" % (100 * drawn.culture_after24.mean()) if len(drawn) else "n/a"))

P("\nInterpretation: if the higher bacteraemia rate in the high-dose group simply reflects"
  "\nmore cultures being drawn, the percentage positive AMONG patients actually cultured"
  "\nshould be flat across dose strata.")

m = smf.glm("any_cx_after24 ~ C(gc_str, Treatment(reference='G0_none')) + age + sofa + on_vaso",
            data=lm, family=sm.families.Binomial()).fit()
P("\nAdjusted OR for HAVING A BLOOD CULTURE DRAWN after the landmark (surveillance intensity):")
for g in ["G1_low", "G2_mod", "G3_high"]:
    t = "C(gc_str, Treatment(reference='G0_none'))[T.%s]" % g
    if t in m.params.index:
        b, se, p = m.params[t], m.bse[t], m.pvalues[t]
        P("   %-10s OR %.2f (%.2f-%.2f)  P=%.3f" % (LABEL[g], np.exp(b), np.exp(b - 1.96 * se),
                                                    np.exp(b + 1.96 * se), p))

m2 = smf.glm("ncx_after24 ~ C(gc_str, Treatment(reference='G0_none')) + age + sofa + on_vaso",
             data=lm, family=sm.families.Poisson()).fit()
P("\nAdjusted rate ratio for the NUMBER of blood cultures drawn:")
for g in ["G1_low", "G2_mod", "G3_high"]:
    t = "C(gc_str, Treatment(reference='G0_none'))[T.%s]" % g
    if t in m2.params.index:
        b, se, p = m2.params[t], m2.bse[t], m2.pvalues[t]
        P("   %-10s RR %.2f (%.2f-%.2f)  P=%.3f" % (LABEL[g], np.exp(b), np.exp(b - 1.96 * se),
                                                    np.exp(b + 1.96 * se), p))

# ------------------------------------------------------------------ adjusted for intensity
ADJ = "age + female + charlson + sofa + on_vent + on_vaso + on_rrt"
P("\n" + "-" * 100)
P("Bacteraemia outcome, progressively accounting for surveillance intensity")
P("-" * 100)
P("%-48s %10s %24s %8s" % ("model", "n", "10-<50 mg/day OR (95% CI)", "P"))

specs = [
    ("unadjusted", "culture_after24 ~ C(gc_str, Treatment(reference='G0_none'))", lm),
    ("+ clinical covariates", "culture_after24 ~ C(gc_str, Treatment(reference='G0_none')) + " + ADJ, lm),
    ("+ was a culture drawn at all", "culture_after24 ~ C(gc_str, Treatment(reference='G0_none')) + " + ADJ + " + any_cx_after24", lm),
    ("+ number of cultures drawn", "culture_after24 ~ C(gc_str, Treatment(reference='G0_none')) + " + ADJ + " + ncx_after24", lm),
    ("RESTRICTED to patients who were cultured", "culture_after24 ~ C(gc_str, Treatment(reference='G0_none')) + " + ADJ, lm[lm.any_cx_after24 == 1]),
]
for lab, f, d in specs:
    try:
        mm = smf.glm(f, data=d, family=sm.families.Binomial()).fit()
        t = "C(gc_str, Treatment(reference='G0_none'))[T.G2_mod]"
        if t in mm.params.index:
            b, se, p = mm.params[t], mm.bse[t], mm.pvalues[t]
            P("%-48s %10d %14.2f (%.2f-%.2f) %8.3f" % (lab, len(d), np.exp(b),
                                                       np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), p))
        else:
            P("%-48s %10d %14s" % (lab, len(d), "estimand unavailable"))
    except Exception as e:
        P("%-48s failed: %s" % (lab, str(e)[:70]))

P("\nAll three dose strata, restricted to patients who actually had a blood culture drawn:")
d = lm[lm.any_cx_after24 == 1]
P("   n = %d, events = %d" % (len(d), int(d.culture_after24.sum())))
try:
    mm = smf.glm("culture_after24 ~ C(gc_str, Treatment(reference='G0_none')) + " + ADJ,
                 data=d, family=sm.families.Binomial()).fit()
    for g in ["G1_low", "G2_mod", "G3_high"]:
        t = "C(gc_str, Treatment(reference='G0_none'))[T.%s]" % g
        if t in mm.params.index:
            b, se, p = mm.params[t], mm.bse[t], mm.pvalues[t]
            P("   %-10s n=%3d  OR %.2f (%.2f-%.2f)  P=%.3f" % (
                LABEL[g], int((d.gc_str == g).sum()), np.exp(b), np.exp(b - 1.96 * se),
                np.exp(b + 1.96 * se), p))
except Exception as e:
    P("   failed: %s" % e)

# ------------------------------------------------------------------ refined IPTW
P("\n" + "=" * 100)
P("REFINED IPTW: parsimonious activity set + weight trimming")
P("=" * 100)

COV_CORE = ["age", "female", "charlson", "sofa", "apsiii", "on_vent", "on_vaso", "on_rrt",
            "emerg", "wbc_min", "platelets_min", "creatinine_max"]
# drop the near-constant activity flags (serositis 1.4%, other-organ 4.8%) - they buy
# nothing and destabilise the weights
COV_ACT_PAR = ["act_score", "act_nephritis", "act_cytopenia", "act_renal_fail",
               "imm_any", "imm_hcq", "n_prior_adm"]


def maxsmd(dd, covs, wcol):
    ww = np.ones(len(dd)) if wcol is None else dd[wcol].values
    worst = 0.0
    for v in covs:
        m = dd[v].astype(float).values
        sd = np.sqrt(np.average((m - np.average(m, weights=ww)) ** 2, weights=ww))
        if sd == 0:
            continue
        m0 = np.average(m[(dd.gc_str == "G0_none").values], weights=ww[(dd.gc_str == "G0_none").values])
        for g in ["G1_low", "G2_mod", "G3_high"]:
            i = (dd.gc_str == g).values
            worst = max(worst, abs((np.average(m[i], weights=ww[i]) - m0) / sd))
    return worst


def iptw(d, covs, trim=0.01, reg_C=0.3):
    X = d[covs].astype(float).values
    X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    T = pd.Categorical(d["gc_str"], categories=ORDER).codes
    ps = LogisticRegression(solver="lbfgs", max_iter=8000, C=reg_C).fit(X, T)
    Pm = ps.predict_proba(X)
    marg = np.array([(T == g).mean() for g in range(len(ORDER))])
    w = marg[T] / np.clip(Pm[np.arange(len(T)), T], 1e-6, 1)
    if trim:
        lo, hi = np.quantile(w, [trim, 1 - trim])
        w = np.clip(w, lo, hi)
    d = d.copy(); d["sw"] = w
    return d, w.sum() ** 2 / (w ** 2).sum()


P("\n%-40s %8s %14s %14s %14s" % ("propensity specification", "ESS", "max|SMD| core raw", "max|SMD| core wtd", "max|SMD| all wtd"))
configs = [
    ("core covariates only", COV_CORE, 0.01, 0.3),
    ("core + all 12 activity proxies", COV_CORE + ["act_score", "act_nephritis", "act_serositis",
     "act_other_organ", "act_aps", "act_cytopenia", "act_renal_fail", "act_effusion",
     "n_dx_codes", "n_prior_adm", "imm_any", "imm_hcq"], 0.01, 0.3),
    ("core + parsimonious activity (7)", COV_CORE + COV_ACT_PAR, 0.01, 0.3),
    ("core + parsimonious, trimmed 5%", COV_CORE + COV_ACT_PAR, 0.05, 0.3),
    ("core + parsimonious, stronger reg", COV_CORE + COV_ACT_PAR, 0.05, 0.1),
]
fitted = {}
for lab, covs, tr, reg_C in configs:
    d, ess = iptw(lm, covs, tr, reg_C)
    raw, wtd = maxsmd(d, COV_CORE, None), maxsmd(d, COV_CORE, "sw")
    wtd_all = maxsmd(d, covs, "sw")
    P("%-40s %8.0f %14.3f %14.3f %14.3f" % (lab, ess, raw, wtd, wtd_all))
    fitted[lab] = d

P("\nDose-response for bacteraemia under each specification (10-<50 mg/day stratum):")
P("%-40s %10s %24s %8s" % ("specification", "n", "OR (95% CI)", "P"))
for out in ["culture_after24", "abx_new_after48", "death_30d"]:
    P("\n  outcome = %s" % out)
    P("  %-40s %10s %-24s %8s" % ("specification", "n", "10-<50 mg/day OR (95% CI)", "P"))
    for cfg in configs:
        lab = cfg[0]
        d = fitted[lab]
        try:
            m = smf.glm("%s ~ C(gc_str, Treatment(reference='G0_none'))" % out, data=d,
                        family=sm.families.Binomial(), freq_weights=d["sw"]).fit()
            t = "C(gc_str, Treatment(reference='G0_none'))[T.G2_mod]"
            if t in m.params.index:
                b, se, p = m.params[t], m.bse[t], m.pvalues[t]
                P("  %-40s %10d %14.2f (%.2f-%.2f) %8.3f" % (lab, len(d), np.exp(b),
                                                              np.exp(b - 1.96 * se),
                                                              np.exp(b + 1.96 * se), p))
        except Exception as e:
            P("  %-40s failed: %s" % (lab, str(e)[:60]))

P("\nAll strata, best-balanced specification (core covariates only):")
d = fitted["core covariates only"]
for out in ["culture_after24", "abx_new_after48", "death_30d"]:
    P("\n  outcome = %s" % out)
    m = smf.glm("%s ~ C(gc_str, Treatment(reference='G0_none'))" % out, data=d,
                family=sm.families.Binomial(), freq_weights=d["sw"]).fit()
    for g in ["G1_low", "G2_mod", "G3_high"]:
        t = "C(gc_str, Treatment(reference='G0_none'))[T.%s]" % g
        if t in m.params.index:
            b, se, p = m.params[t], m.bse[t], m.pvalues[t]
            P("     %-10s OR %.2f (%.2f-%.2f)  P=%.3f" % (LABEL[g], np.exp(b),
                np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), p))

with open(os.path.join(OUT, "81_detection_bias.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nsaved -> out/81_detection_bias.txt")
