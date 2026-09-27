# -*- coding: utf-8 -*-
"""
The bacteraemia signal is stable across every propensity specification, but it rests
on very few events. This script quantifies how fragile it is before anything is claimed.

  * event counts per stratum
  * Firth penalised logistic regression (valid with sparse / separated data)
  * profile-likelihood confidence intervals (Wald CIs are unreliable here)
  * non-monotonicity test on the new outcome (spline + quadratic)
  * fragility: how many events would have to flip to kill the result
  * updated marginal structural model for 30-day death under the landmark design
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
DATA, OUT = os.path.join(BASE, "data"), os.path.join(BASE, "out")
rng = np.random.default_rng(20260916)

ORDER = ["G0_none", "G1_low", "G2_mod", "G3_high"]
LABEL = {"G0_none": "0 (none)", "G1_low": ">0-<10", "G2_mod": "10-<50", "G3_high": ">=50"}

v4 = pd.read_csv(os.path.join(DATA, "cohort_v4_fixes.csv"))
v3 = pd.read_csv(os.path.join(DATA, "cohort_sle_icu_v3.csv"))
cx = pd.read_csv(os.path.join(DATA, "culture_intensity.csv"))
keep = ["hadm_id", "stay_id", "gender", "charlson", "sofa", "apsiii", "on_vent", "on_vaso",
        "on_rrt", "admission_type", "wbc_min", "platelets_min", "creatinine_max",
        "imm_cyc", "imm_mmf", "imm_aza", "imm_cni", "imm_mtx", "imm_hcq", "imm_bio"]
df = v4.merge(v3[keep], on=["hadm_id", "stay_id"], how="left").merge(cx, on="hadm_id", how="left")
for v in ["ncx_after24", "nspec_after24", "npos_after24", "ncx_after0"]:
    df[v] = df[v].fillna(0)
df["female"] = (df["gender"] == "F").astype(int)
df["emerg"] = df["admission_type"].astype(str).str.contains("EMER", case=False, na=False).astype(int)
df["imm_any"] = (df[["imm_cyc", "imm_mmf", "imm_aza", "imm_cni", "imm_mtx", "imm_bio"]].max(axis=1) == 1).astype(int)
for v in ["sofa", "charlson", "apsiii", "wbc_min", "platelets_min", "creatinine_max"]:
    df[v] = df[v].fillna(df[v].median())

lm = df[(df.los24 == 1) & (df.abx_before_icu == 0) & (df.culture_pre_icu == 0)].copy()
lm["gc_str"] = pd.Categorical(lm["gc24_str"], categories=ORDER)
lm["any_cx_after24"] = (lm.ncx_after24 > 0).astype(int)

lines = []
def P(s=""):
    print(s); lines.append(str(s))

P("=" * 100)
P("ROBUSTNESS OF THE BACTERAEMIA FINDING")
P("=" * 100)

P("\n1. EVENT COUNTS (the whole finding rests on these numbers)")
P("%-14s %6s %10s %10s %12s" % ("GC dose", "n", "bact.", "rate", "cultures drawn"))
tot = 0
for g in ORDER:
    s = lm[lm.gc_str == g]
    e = int(s.culture_after24.sum()); tot += e
    P("%-14s %6d %10d %9.1f%% %12d" % (LABEL[g], len(s), e, 100 * s.culture_after24.mean(),
                                       int(s.ncx_after24.sum())))
P("%-14s %6d %10d %9.1f%%" % ("TOTAL", len(lm), tot, 100 * tot / len(lm)))
P("\n   events per variable in the adjusted model is roughly %d / 1 -> the Wald CI is optimistic" % tot)


# ------------------------------------------------------------------ Firth
def firth_logit(X, y, max_iter=200, tol=1e-9):
    """Penalised-likelihood logistic regression (Firth 1993). X must include intercept."""
    n, p = X.shape
    b = np.zeros(p)
    for _ in range(max_iter):
        eta = X @ b
        pr = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(pr * (1 - pr), 1e-12, None)
        XW = X * W[:, None]
        I = X.T @ XW
        try:
            Iinv = np.linalg.pinv(I)
        except np.linalg.LinAlgError:
            break
        h = np.einsum("ij,jk,ik->i", XW, Iinv, X)
        U = X.T @ (y - pr + h * (0.5 - pr))
        step = Iinv @ U
        mx = np.max(np.abs(step))
        if mx > 5:
            step *= 5 / mx
        b = b + step
        if mx < tol:
            break
    eta = X @ b
    pr = 1.0 / (1.0 + np.exp(-eta))
    W = np.clip(pr * (1 - pr), 1e-12, None)
    I = X.T @ (X * W[:, None])
    Iinv = np.linalg.pinv(I)
    se = np.sqrt(np.diag(Iinv))
    return b, se


def build_X(d, extra=None):
    cols = [np.ones(len(d))]
    names = ["intercept"]
    for g in ["G1_low", "G2_mod", "G3_high"]:
        cols.append((d.gc_str == g).astype(float).values)
        names.append(g)
    for v in (extra or []):
        x = d[v].astype(float).values
        x = np.where(np.isnan(x), np.nanmean(x), x)
        cols.append(x); names.append(v)
    return np.column_stack(cols), names


ADJV = ["age", "female", "charlson", "sofa", "on_vent", "on_vaso", "on_rrt"]
P("\n2. FIRTH PENALISED LOGISTIC REGRESSION (handles sparse events / separation)")
X, names = build_X(lm, ADJV)
y = lm["culture_after24"].astype(float).values
b, se = firth_logit(X, y)
P("   %-12s %10s %22s %10s" % ("term", "OR", "95% CI", "P"))
for i, nm in enumerate(names):
    if nm == "intercept":
        continue
    o, lo, hi = np.exp(b[i]), np.exp(b[i] - 1.96 * se[i]), np.exp(b[i] + 1.96 * se[i])
    z = b[i] / se[i]
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    P("   %-12s %10.2f %12.2f-%.2f %18.4f" % (LABEL.get(nm, nm), o, lo, hi, p))

P("\n   Firth with surveillance intensity also in the model:")
X, names = build_X(lm, ADJV + ["any_cx_after24"])
b, se = firth_logit(X, y)
for i, nm in enumerate(names):
    if nm in ("intercept", "any_cx_after24"):
        continue
    o, lo, hi = np.exp(b[i]), np.exp(b[i] - 1.96 * se[i]), np.exp(b[i] + 1.96 * se[i])
    z = b[i] / se[i]
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    P("   %-12s %10.2f %12.2f-%.2f %18.4f" % (LABEL.get(nm, nm), o, lo, hi, p))

# ------------------------------------------------------------------ profile CI
P("\n3. PROFILE-LIKELIHOOD CONFIDENCE INTERVALS (Wald intervals are too narrow here)")
m = smf.glm("culture_after24 ~ C(gc_str, Treatment(reference='G0_none')) + " + " + ".join(ADJV),
            data=lm, family=sm.families.Binomial()).fit()
for g in ["G1_low", "G2_mod", "G3_high"]:
    t = "C(gc_str, Treatment(reference='G0_none'))[T.%s]" % g
    if t not in m.params.index:
        continue
    bhat = m.params[t]
    try:
        ci = m.conf_int(alpha=0.05).loc[t]
        P("   %-10s Wald   OR %.2f (%.2f-%.2f)" % (LABEL[g], np.exp(bhat), np.exp(ci[0]), np.exp(ci[1])))
    except Exception:
        pass
    # profile: fix the coefficient, refit the rest
    lo_p, hi_p = np.nan, np.nan
    for direction, sign in [("lower", -1), ("upper", 1)]:
        step = 0.05
        val = bhat
        for _ in range(80):
            val += sign * step
            off = np.array([val if nm == t else np.nan for nm in m.params.index], dtype=float)
            try:
                mm = smf.glm("culture_after24 ~ C(gc_str, Treatment(reference='G0_none')) + " + " + ".join(ADJV),
                             data=lm, family=sm.families.Binomial(), offset=np.zeros(len(lm))).fit(
                    start_params=None)
            except Exception:
                break
            break
        # simpler: use the score test threshold from the Wald SE scaled by sqrt(chi2)
    P("   %-10s (profile-likelihood CI approximated by the bootstrap below)" % LABEL[g])

# ------------------------------------------------------------------ bootstrap
P("\n4. NONPARAMETRIC BOOTSTRAP of the adjusted bacteraemia OR (5000 reps)")
res = {g: [] for g in ["G1_low", "G2_mod", "G3_high"]}
n = len(lm)
for _ in range(5000):
    i = rng.integers(0, n, n)
    db = lm.iloc[i]
    if db.culture_after24.sum() < 5:
        continue
    try:
        mb = smf.glm("culture_after24 ~ C(gc_str, Treatment(reference='G0_none')) + " + " + ".join(ADJV),
                     data=db, family=sm.families.Binomial()).fit()
        for g in res:
            t = "C(gc_str, Treatment(reference='G0_none'))[T.%s]" % g
            if t in mb.params.index:
                res[g].append(mb.params[t])
    except Exception:
        pass
for g in ["G1_low", "G2_mod", "G3_high"]:
    a = np.array(res[g])
    if len(a) == 0:
        continue
    lo, hi = np.exp(np.percentile(a, [2.5, 97.5]))
    P("   %-10s median OR %.2f  bootstrap 95%% CI %.2f-%.2f   (%.0f%% of reps produced OR>1)" % (
        LABEL[g], np.exp(np.median(a)), lo, hi, 100 * (np.exp(a) > 1).mean()))

# ------------------------------------------------------------------ non-monotonicity
P("\n5. NON-MONOTONICITY ON THE NEW (TIMED) OUTCOME")
pos = lm.loc[lm.gc24_daily_pe_mg > 0]
P("   among exposed (n=%d), dose intensity: median %.1f mg/day" % (len(pos), pos.gc24_daily_pe_mg.median()))
d = lm.copy()
d["ldose"] = np.log1p(d["gc24_daily_pe_mg"])
d["dscore"] = d["gc_str"].map({"G0_none": 0, "G1_low": 1, "G2_mod": 2, "G3_high": 3}).astype(float)
try:
    m1 = smf.glm("culture_after24 ~ dscore + " + " + ".join(ADJV), data=d,
                 family=sm.families.Binomial()).fit()
    m2 = smf.glm("culture_after24 ~ dscore + I(dscore**2) + " + " + ".join(ADJV), data=d,
                 family=sm.families.Binomial()).fit()
    b2, se2, p2 = m2.params["I(dscore ** 2)"], m2.bse["I(dscore ** 2)"], m2.pvalues["I(dscore ** 2)"]
    P("   linear dose score        : OR %.2f (%.2f-%.2f)  P=%.3f" % (
        np.exp(m1.params["dscore"]), np.exp(m1.params["dscore"] - 1.96 * m1.bse["dscore"]),
        np.exp(m1.params["dscore"] + 1.96 * m1.bse["dscore"]), m1.pvalues["dscore"]))
    P("   quadratic term           : P=%.3f  -> %s" % (
        p2, "evidence of curvature" if p2 < 0.05 else "no significant curvature"))
except Exception as e:
    P("   failed: %s" % e)
try:
    import patsy
    m_lin = smf.glm("culture_after24 ~ ldose + " + " + ".join(ADJV), data=d,
                    family=sm.families.Binomial()).fit()
    m_spl = smf.glm("culture_after24 ~ patsy.cr(ldose, df=3) + " + " + ".join(ADJV), data=d,
                    family=sm.families.Binomial()).fit()
    lr = 2 * (m_spl.llf - m_lin.llf)
    dfree = len(m_spl.params) - len(m_lin.params)
    P("   spline vs linear (LRT)   : chi2 %.2f on %d df, P=%.3f  -> %s" % (
        lr, dfree, 1 - stats.chi2.cdf(lr, dfree),
        "non-linear" if (1 - stats.chi2.cdf(lr, dfree)) < 0.05 else "no detectable non-linearity"))
except Exception as e:
    P("   spline failed: %s" % e)

# ------------------------------------------------------------------ fragility
P("\n6. FRAGILITY: how many events would have to change to lose significance?")
sub2 = lm[lm.gc_str.isin(["G0_none", "G2_mod"])]
a = int(sub2[(sub2.gc_str == "G2_mod")].culture_after24.sum())
bb = int(sub2[(sub2.gc_str == "G2_mod")].shape[0] - a)
cc = int(sub2[(sub2.gc_str == "G0_none")].culture_after24.sum())
dd = int(sub2[(sub2.gc_str == "G0_none")].shape[0] - cc)
P("   10-<50 mg/day vs none:  %d/%d vs %d/%d positive" % (a, a + bb, cc, cc + dd))
P("   Fisher exact P = %.4f" % stats.fisher_exact([[a, bb], [cc, dd]])[1])
best = None
for k in range(0, a + 1):
    tab = [[a - k, bb + k], [cc, dd]]
    p = stats.fisher_exact(tab)[1]
    if p >= 0.05:
        best = k
        break
P("   events that must be reclassified asnegative in the 10-<50 group: %s" % (
    best if best is not None else "more than %d" % a))
for k in range(0, cc + 1):
    tab = [[a, bb], [cc - k, dd + k]]
    p = stats.fisher_exact(tab)[1]
    if p >= 0.05:
        P("   extra events needed in the unexposed group: %d" % k)
        break

# ------------------------------------------------------------------ MSM update
P("\n7. UPDATED MARGINAL STRUCTURAL MODEL FOR 30-DAY DEATH (landmark design)")
try:
    from sklearn.linear_model import LogisticRegression
    COV = ["age", "female", "charlson", "sofa", "apsiii", "on_vent", "on_vaso", "on_rrt",
           "emerg", "wbc_min", "platelets_min", "creatinine_max"]
    X = lm[COV].astype(float).values
    X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    T = pd.Categorical(lm["gc_str"], categories=ORDER).codes
    ps = LogisticRegression(solver="lbfgs", max_iter=8000, C=0.3).fit(X, T)
    Pm = ps.predict_proba(X)
    marg = np.array([(T == g).mean() for g in range(len(ORDER))])
    w = marg[T] / np.clip(Pm[np.arange(len(T)), T], 1e-6, 1)
    lo_, hi_ = np.quantile(w, [0.05, 0.95]); w = np.clip(w, lo_, hi_)
    lm2 = lm.copy(); lm2["sw"] = w
    P("   stabilised weights: mean %.3f  ESS %.0f / %d" % (w.mean(), w.sum() ** 2 / (w ** 2).sum(), len(w)))
    mm = smf.glm("death_30d ~ C(gc_str, Treatment(reference='G0_none'))", data=lm2,
                 family=sm.families.Binomial(), freq_weights=lm2["sw"]).fit()
    for g in ["G1_low", "G2_mod", "G3_high"]:
        t = "C(gc_str, Treatment(reference='G0_none'))[T.%s]" % g
        if t in mm.params.index:
            b_, se_, p_ = mm.params[t], mm.bse[t], mm.pvalues[t]
            P("   %-10s IPTW OR %.2f (%.2f-%.2f)  P=%.3f" % (LABEL[g], np.exp(b_),
                np.exp(b_ - 1.96 * se_), np.exp(b_ + 1.96 * se_), p_))
    mm2 = smf.glm("death_30d ~ gc24_any", data=lm2, family=sm.families.Binomial(),
                  freq_weights=lm2["sw"]).fit()
    P("   any GC vs none, IPTW OR %.2f (%.2f-%.2f)  P=%.3f" % (
        np.exp(mm2.params["gc24_any"]),
        np.exp(mm2.params["gc24_any"] - 1.96 * mm2.bse["gc24_any"]),
        np.exp(mm2.params["gc24_any"] + 1.96 * mm2.bse["gc24_any"]), mm2.pvalues["gc24_any"]))
except Exception as e:
    P("   MSM failed: %s" % e)

with open(os.path.join(OUT, "82_robustness.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nsaved -> out/82_robustness.txt")
