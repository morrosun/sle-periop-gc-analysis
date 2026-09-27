# -*- coding: utf-8 -*-
"""
Repairing the failed positive control
=====================================
Original result: GC->infection OR 1.26 vs ondansetron->infection OR 1.27.
Two candidate explanations, tested separately:

  H1  confounding by SLE disease activity (untreated)   -> Fix 1
  H2  exposure and outcome measured in the same window  -> Fix 2

and one design flaw that was there from the start:

  H3  the "positive control" was not a positive control at all. GC->infection
      is the study hypothesis. A positive control must be an association that
      is already known. We use GC -> hyperglycaemia (a direct pharmacological
      effect). It separates "the exposure variable is broken" from "the
      association is absent".

Control panel (2 x 2)
---------------------
                     | infection (hypothesis) | hyperglycaemia (known effect)
   GC exposure       |  ?                     |  expect +
   ondansetron (neg) |  expect ~1             |  expect ~1
   GI bleed (neg outcome) should also be ~1 for GC.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, OUT = os.path.join(BASE, "data"), os.path.join(BASE, "out")
rng = np.random.default_rng(20260916)

ORDER = ["G0_none", "G1_low", "G2_mod", "G3_high"]
LABEL = {"G0_none": "0 (none)", "G1_low": ">0-<10", "G2_mod": "10-<50", "G3_high": ">=50"}

v4 = pd.read_csv(os.path.join(DATA, "cohort_v4_fixes.csv"))
v3 = pd.read_csv(os.path.join(DATA, "cohort_sle_icu_v3.csv"))

# carry over the severity / comorbidity covariates already built in v3
keep = ["hadm_id", "stay_id", "gender", "charlson", "sofa", "apsiii", "on_vent", "on_vaso",
        "on_rrt", "lupus_nephritis", "admission_type", "wbc_min", "platelets_min",
        "creatinine_max", "albumin_min", "abs_lymphocytes_min", "imm_cyc", "imm_mmf",
        "imm_aza", "imm_cni", "imm_mtx", "imm_hcq", "imm_bio"]
df = v4.merge(v3[keep], on=["hadm_id", "stay_id"], how="left")

df["female"] = (df["gender"] == "F").astype(int)
df["emerg"] = df["admission_type"].astype(str).str.contains("EMER", case=False, na=False).astype(int)
df["imm_any"] = (df[["imm_cyc", "imm_mmf", "imm_aza", "imm_cni", "imm_mtx", "imm_bio"]].max(axis=1) == 1).astype(int)
df["imm_hcq"] = df["imm_hcq"].fillna(0)
for v in ["sofa", "charlson", "apsiii", "wbc_min", "platelets_min", "creatinine_max"]:
    df[v] = df[v].fillna(df[v].median())

# landmark cohort: outcome can only be observed if the patient is still there
lm = df[df.los24 == 1].copy()
lm["gc_str"] = pd.Categorical(lm["gc24_str"], categories=ORDER)

lines = []
def P(s=""):
    print(s); lines.append(str(s))

def orstr(m, term):
    b, se, p = m.params[term], m.bse[term], m.pvalues[term]
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), p

P("=" * 100)
P("POSITIVE-CONTROL REPAIR - SLE ICU cohort")
P("=" * 100)
P("\nfull cohort %d stays | landmark cohort (ICU LOS >= 24 h) %d stays | %d patients" % (
    len(df), len(lm), lm.subject_id.nunique()))
P("exposure strata (GC in first 24 h of ICU): " +
  "  ".join("%s n=%d" % (LABEL[g], int((lm.gc_str == g).sum())) for g in ORDER))
P("\noutcome availability in the landmark cohort:")
for v, lab in [("abx_new_after48", "new broad-spectrum antibiotic > 48 h after ICU"),
               ("culture_after24", "positive blood culture > 24 h after ICU"),
               ("gi_bleed", "GI bleeding (negative outcome control)"),
               ("death_30d", "30-day death"),
               ("infection_any", "any infection ICD code (untimed, original)")]:
    P("   %-42s %3d (%.1f%%)" % (lab, int(lm[v].sum()), 100 * lm[v].mean()))
P("   %-42s %3d (%.1f%%)" % ("ondansetron in first 24 h (negative exposure)",
                             int(lm.ondan24.sum()), 100 * lm.ondan24.mean()))

# =====================================================================
# 0.  CONTROL PANEL
# =====================================================================
P("\n" + "=" * 100)
P("0. CONTROL PANEL  (why the original positive control was not one)")
P("=" * 100)

ADJ = "age + female + charlson + sofa + on_vent + on_vaso + on_rrt"

P("\n-- panel A: any GC in first 24 h --")
P("%-46s %10s %10s" % ("", "OR (95% CI)", "P"))
for out, lab in [("hyper_post", "POSITIVE CONTROL: hyperglycaemia >=180"),
                 ("insulin_after_icu", "POSITIVE CONTROL: insulin in first 48 h"),
                 ("abx_new_after48", "hypothesis: treated infection >48 h"),
                 ("culture_after24", "hypothesis: bacteraemia >24 h"),
                 ("gi_bleed", "negative outcome: GI bleeding")]:
    d = lm[lm[out].notna()]
    m = smf.glm("%s ~ gc24_any + %s" % (out, ADJ), data=d, family=sm.families.Binomial()).fit()
    o, lo, hi, p = orstr(m, "gc24_any")
    P("%-46s %10.2f (%.2f-%.2f) %10.3f" % (lab, o, lo, hi, p))

P("\n-- panel B: negative exposure (ondansetron) against the SAME outcomes --")
for out, lab in [("hyper_post", "hyperglycaemia >=180"),
                 ("insulin_after_icu", "insulin in first 48 h"),
                 ("abx_new_after48", "treated infection >48 h"),
                 ("culture_after24", "bacteraemia >24 h")]:
    d = lm[lm[out].notna()]
    m = smf.glm("%s ~ ondan24 + %s" % (out, ADJ), data=d, family=sm.families.Binomial()).fit()
    o, lo, hi, p = orstr(m, "ondan24")
    P("%-46s %10.2f (%.2f-%.2f) %10.3f" % (lab, o, lo, hi, p))

P("\n-- panel C: dose gradient of the positive control (does the exposure variable work?) --")
P("%-14s %8s %14s %14s" % ("GC dose", "n", "% hyper>=180", "mean glu max"))
for g in ORDER:
    s = lm[lm.gc_str == g]
    P("%-14s %8d %14s %14.1f" % (LABEL[g], len(s),
        "%.1f%%" % (100 * s.hyper_post.mean(skipna=True)) if s.hyper_post.notna().any() else "n/a",
        s.glu_max_post48.mean()))
d = lm[lm.hyper_post.notna()]
m = smf.glm("hyper_post ~ C(gc_str, Treatment(reference='G0_none')) + %s" % ADJ,
            data=d, family=sm.families.Binomial()).fit()
P("\nadjusted hyperglycaemia OR by dose stratum:")
for g in ["G1_low", "G2_mod", "G3_high"]:
    t = "C(gc_str, Treatment(reference='G0_none'))[T.%s]" % g
    if t in m.params.index:
        o, lo, hi, p = orstr(m, t)
        P("   %-10s OR %.2f (%.2f-%.2f)  P=%.3f" % (LABEL[g], o, lo, hi, p))

# =====================================================================
# 1.  FIX 1 - ACTIVITY-AUGMENTED PROPENSITY SCORE
# =====================================================================
P("\n" + "=" * 100)
P("1. FIX 1: SLE DISEASE-ACTIVITY PROXIES IN THE PROPENSITY MODEL")
P("=" * 100)

COV_BASE = ["age", "female", "charlson", "sofa", "apsiii", "on_vent", "on_vaso", "on_rrt",
            "emerg", "wbc_min", "platelets_min", "creatinine_max"]
COV_ACT = ["act_score", "act_nephritis", "act_serositis", "act_other_organ", "act_aps",
           "act_cytopenia", "act_renal_fail", "act_effusion", "n_dx_codes", "n_prior_adm",
           "imm_any", "imm_hcq"]

P("\nactivity-proxy prevalence in the landmark cohort (n=%d):" % len(lm))
for v in COV_ACT:
    P("   %-18s mean %.3f   range %.0f-%.0f" % (v, lm[v].mean(), lm[v].min(), lm[v].max()))
P("\n   lab proxies are too sparse for the main model (C3 %.1f%%, C4 %.1f%%, CRP %.1f%%)"
  " -- used only in the sensitivity subset" % (
      100 * lm.c3_min.notna().mean(), 100 * lm.c4_min.notna().mean(), 100 * lm.crp_max.notna().mean()))


def fit_iptw(d, covs, label):
    X = d[covs].astype(float).values
    X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    T = pd.Categorical(d["gc_str"], categories=ORDER).codes
    ps = LogisticRegression(solver="lbfgs", max_iter=8000, C=1.0).fit(X, T)
    Pmat = ps.predict_proba(X)
    marg = np.array([(T == g).mean() for g in range(len(ORDER))])
    w = marg[T] / np.clip(Pmat[np.arange(len(T)), T], 1e-6, 1)
    d = d.copy(); d["sw"] = w
    ess = w.sum() ** 2 / (w ** 2).sum()

    def maxsmd(dd, wcol):
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
    P("\n[%s]  stabilised weights mean %.3f  max %.2f  ESS %.0f / %d   max|SMD| %.3f -> %.3f" % (
        label, w.mean(), w.max(), ess, len(d), maxsmd(d, None), maxsmd(d, "sw")))
    return d, maxsmd(d, "sw")


def dose_response(d, outcome, wcol, label):
    f = "%s ~ C(gc_str, Treatment(reference='G0_none'))" % outcome
    m = smf.glm(f, data=d, family=sm.families.Binomial(), freq_weights=d[wcol]).fit()
    # bootstrap over the whole pipeline would be ideal; here we bootstrap the outcome model
    boots = {g: [] for g in ["G1_low", "G2_mod", "G3_high"]}
    n = len(d)
    for _ in range(1500):
        i = rng.integers(0, n, n)
        db = d.iloc[i]
        try:
            mb = smf.glm(f, data=db, family=sm.families.Binomial(), freq_weights=db[wcol]).fit()
            for g in boots:
                t = "C(gc_str, Treatment(reference='G0_none'))[T.%s]" % g
                if t in mb.params.index:
                    boots[g].append(mb.params[t])
        except Exception:
            pass
    P("\n  %s (outcome = %s)" % (label, outcome))
    P("  %-12s %8s %20s %16s %8s" % ("stratum", "n", "IPTW OR (95% CI)", "bootstrap CI", "P"))
    rows = []
    for g in ["G1_low", "G2_mod", "G3_high"]:
        t = "C(gc_str, Treatment(reference='G0_none'))[T.%s]" % g
        if t not in m.params.index:
            continue
        o, lo, hi, p = orstr(m, t)
        arr = np.array(boots[g])
        blo, bhi = (np.exp(np.percentile(arr, [2.5, 97.5])) if len(arr) else (np.nan, np.nan))
        P("  %-12s %8d %10.2f (%.2f-%.2f) %12s %8.3f" % (
            LABEL[g], int((d.gc_str == g).sum()), o, lo, hi, "%.2f-%.2f" % (blo, bhi), p))
        rows.append((g, o, lo, hi, blo, bhi, p))
    return rows


P("\n--- 1a. IPTW with the ORIGINAL covariate set (reproducing the earlier model) ---")
d_base, _ = fit_iptw(lm, COV_BASE, "original covariates")
r_base = dose_response(d_base, "abx_new_after48", "sw", "original covariates")

P("\n--- 1b. IPTW with ACTIVITY-AUGMENTED covariates ---")
d_act, smd_act = fit_iptw(lm, COV_BASE + COV_ACT, "+ activity proxies")
r_act = dose_response(d_act, "abx_new_after48", "sw", "+ activity proxies")

# =====================================================================
# 2.  FIX 2 - TEMPORAL ORDERING (LANDMARK)
# =====================================================================
P("\n" + "=" * 100)
P("2. FIX 2: CLEAN TEMPORAL ORDERING (24 h landmark)")
P("=" * 100)
P("""
exposure : glucocorticoid dose intensity in the first 24 h of ICU
outcome  : events occurring AFTER the landmark
            - abx_new_after48   : new broad-spectrum antibiotic > 48 h after ICU
            - culture_after24   : positive blood culture > 24 h after ICU
            - death_30d         : death within 30 days of ICU admission
prevalent infection at ICU admission is adjusted for (and excluded in a sensitivity run)
""")

P("--- 2a. timed outcomes, excluding prevalent infection at ICU admission ---")
lm2 = lm[(lm.abx_before_icu == 0) & (lm.culture_pre_icu == 0)].copy()
P("n = %d" % len(lm2))
d_t, _ = fit_iptw(lm2, COV_BASE + COV_ACT, "landmark, activity-adjusted")
for out in ["abx_new_after48", "culture_after24", "death_30d"]:
    dose_response(d_t, out, "sw", "landmark -> %s" % out)

P("\n--- 2b. same, but keeping prevalent infection and adjusting for it ---")
lm3 = lm.copy()
for out in ["abx_new_after48", "culture_after24"]:
    d = lm3.copy()
    m = smf.glm("%s ~ C(gc_str, Treatment(reference='G0_none')) + abx_before_icu" % out,
                data=d, family=sm.families.Binomial()).fit()
    P("  %-24s abx_before_icu OR %.2f (%.2f-%.2f)" % (
        out, np.exp(m.params["abx_before_icu"]),
        np.exp(m.params["abx_before_icu"] - 1.96 * m.bse["abx_before_icu"]),
        np.exp(m.params["abx_before_icu"] + 1.96 * m.bse["abx_before_icu"])))

# =====================================================================
# 3.  COMBINED + SENSITIVITY
# =====================================================================
P("\n" + "=" * 100)
P("3. COMBINED MODEL AND SENSITIVITY")
P("=" * 100)

P("\n--- 3a. combined (activity-adjusted PS + landmark + timed outcome) ---")
for out in ["abx_new_after48", "culture_after24", "death_30d"]:
    dose_response(d_t, out, "sw", "combined -> %s" % out)

P("\n--- 3b. negative controls in the combined design ---")
d = lm2.copy()
for out, lab in [("gi_bleed", "negative outcome: GI bleed"),
                 ("death_30d", "30-day death")]:
    m = smf.glm("%s ~ C(gc_str, Treatment(reference='G0_none')) + %s" % (out, ADJ),
                data=d, family=sm.families.Binomial()).fit()
    P("  %s" % lab)
    for g in ["G1_low", "G2_mod", "G3_high"]:
        t = "C(gc_str, Treatment(reference='G0_none'))[T.%s]" % g
        if t in m.params.index:
            o, lo, hi, p = orstr(m, t)
            P("     %-10s OR %.2f (%.2f-%.2f)  P=%.3f" % (LABEL[g], o, lo, hi, p))

m = smf.glm("abx_new_after48 ~ ondan24 + %s" % ADJ, data=d, family=sm.families.Binomial()).fit()
o, lo, hi, p = orstr(m, "ondan24")
P("  negative exposure: ondansetron -> treated infection  OR %.2f (%.2f-%.2f)  P=%.3f" % (o, lo, hi, p))
m = smf.glm("hyper_post ~ ondan24 + %s" % ADJ, data=d[d.hyper_post.notna()], family=sm.families.Binomial()).fit()
o, lo, hi, p = orstr(m, "ondan24")
P("  negative exposure: ondansetron -> hyperglycaemia     OR %.2f (%.2f-%.2f)  P=%.3f" % (o, lo, hi, p))

P("\n--- 3c. lab-based activity subset (C3 / C4 / CRP measured) ---")
sub = lm2[lm2.c3_min.notna() | lm2.c4_min.notna() | lm2.crp_max.notna()].copy()
P("n = %d  (C3 %d, C4 %d, CRP %d)" % (len(sub), sub.c3_min.notna().sum(),
                                      sub.c4_min.notna().sum(), sub.crp_max.notna().sum()))
if len(sub) > 60:
    for v in ["c3_min", "c4_min", "crp_max"]:
        sub[v] = sub[v].fillna(sub[v].median())
    dsub, _ = fit_iptw(sub, COV_BASE + COV_ACT + ["c3_min", "c4_min", "crp_max"], "lab subset")
    dose_response(dsub, "abx_new_after48", "sw", "lab subset -> treated infection")

P("\n--- 3d. E-values for the combined estimates ---")
def evalue(o, lo):
    rr = o if o >= 1 else 1.0 / o
    rr = max(rr, 1.0001)
    lr = lo if lo >= 1 else 1.0 / lo
    lr = max(lr, 1.0001)
    return rr + np.sqrt(rr * (rr - 1)), lr + np.sqrt(lr * (lr - 1))

for out in ["abx_new_after48", "culture_after24", "death_30d"]:
    m = smf.glm("%s ~ C(gc_str, Treatment(reference='G0_none'))" % out,
                data=d_t, family=sm.families.Binomial(), freq_weights=d_t["sw"]).fit()
    P("  %s" % out)
    for g in ["G1_low", "G2_mod", "G3_high"]:
        t = "C(gc_str, Treatment(reference='G0_none'))[T.%s]" % g
        if t in m.params.index:
            o, lo, hi, p = orstr(m, t)
            E, Elo = evalue(o, lo)
            P("     %-10s OR %.2f  E-value point %.2f  CI limit %.2f" % (LABEL[g], o, E, Elo))

with open(os.path.join(OUT, "80_positive_control_fix.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nsaved -> out/80_positive_control_fix.txt")
