# -*- coding: utf-8 -*-
"""
IPTW for a 4-level exposure + negative controls + E-values + sensitivity analyses
=================================================================================
Weighting : stabilised IPTW from a multinomial logistic propensity model
Inference : nonparametric bootstrap (2000 reps) on the weighted logistic fit
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

df = pd.read_csv(os.path.join(DATA, "cohort_sle_icu_v3.csv"))
nc = pd.read_csv(os.path.join(DATA, "negctrl_vars.csv"))
df = df.merge(nc, on=["hadm_id", "stay_id"], how="left")

COV = ["age", "female", "charlson", "sofa", "apsiii", "on_vent", "on_vaso", "on_rrt",
       "lupus_nephritis", "imm_any", "emerg", "wbc_min", "platelets_min", "creatinine_max"]


def build(df):
    d = df.copy()
    d["gc_str"] = pd.Categorical(d["gc_str"], categories=ORDER, ordered=False) if "gc_str" in d else None
    return d


def stratum(v):
    if v <= 0:
        return "G0_none"
    if v < 10:
        return "G1_low"
    if v < 50:
        return "G2_mod"
    return "G3_high"


ORDER = ["G0_none", "G1_low", "G2_mod", "G3_high"]
LABEL = {"G0_none": "0 (none)", "G1_low": ">0-<10", "G2_mod": "10-<50", "G3_high": ">=50"}

df["gc_str"] = df["gc_daily_pe_mg"].apply(stratum)
df["imm_any"] = (df[["imm_cyc", "imm_mmf", "imm_aza", "imm_cni", "imm_mtx", "imm_bio"]].max(axis=1) == 1).astype(int)
df["emerg"] = df["admission_type"].astype(str).str.contains("EMER", case=False, na=False).astype(int)
df["female"] = (df["gender"] == "F").astype(int)
for v in ["sofa", "charlson", "apsiii", "wbc_min", "platelets_min", "creatinine_max", "oasis"]:
    df[v] = df[v].fillna(df[v].median())

lines = []
def P(s=""):
    print(s); lines.append(str(s))

P("=" * 96)
P("IPTW / negative controls / sensitivity analyses - SLE ICU (n=%d)" % len(df))
P("=" * 96)

# ------------------------------------------------------------------ IPTW
P("\n" + "=" * 96)
P("1. STABILISED IPTW (4-level exposure)")
P("=" * 96)

X = df[COV].values
X = (X - X.mean(0)) / X.std(0)
T = pd.Categorical(df["gc_str"], categories=ORDER).codes

ps_model = LogisticRegression(solver="lbfgs", max_iter=5000, C=1.0)
ps_model.fit(X, T)
PS = ps_model.predict_proba(X)
marg = np.array([(T == g).mean() for g in range(len(ORDER))])
ps_i = PS[np.arange(len(T)), T]
sw = marg[T] / np.clip(ps_i, 1e-6, 1)
df["sw"] = sw

P("stabilised weights: mean %.3f  min %.3f  max %.3f  ESS %.1f / %d" % (
    sw.mean(), sw.min(), sw.max(), sw.sum() ** 2 / (sw ** 2).sum(), len(sw)))
for g, lab in enumerate(ORDER):
    s = sw[T == g]
    P("   %-10s n=%3d  weight mean %.3f  range %.2f-%.2f" % (lab, len(s), s.mean(), s.min(), s.max()))


def max_smd(d, wcol=None):
    w = np.ones(len(d)) if wcol is None else d[wcol].values
    out = {}
    ref = d[d.gc_str == "G0_none"]
    for v in COV:
        m = d[v].values.astype(float)
        sd = np.sqrt(np.average((m - np.average(m, weights=w)) ** 2, weights=w))
        if sd == 0:
            out[v] = 0.0
            continue
        worst = 0.0
        for g in ["G1_low", "G2_mod", "G3_high"]:
            idx = (d.gc_str == g).values
            idxr = (d.gc_str == "G0_none").values
            m1 = np.average(m[idx], weights=w[idx])
            m0 = np.average(m[idxr], weights=w[idxr])
            worst = max(worst, abs((m1 - m0) / sd))
        out[v] = worst
    return out


smd_un = max_smd(df, None)
smd_w = max_smd(df, "sw")
P("\n%-26s %10s %10s" % ("covariate", "SMD raw", "SMD wtd"))
for v in COV:
    P("%-26s %10.3f %10.3f" % (v, smd_un[v], smd_w[v]))
P("max |SMD|  raw %.3f -> weighted %.3f   (0.1 = conventional balance threshold)" % (
    max(smd_un.values()), max(smd_w.values())))

# ------------------------------------------------------------------ weighted models
def weighted_logit(d, formula, wcol="sw", n_boot=2000):
    m = smf.glm(formula, data=d, family=sm.families.Binomial(), freq_weights=d[wcol]).fit()
    res = {}
    for nm in m.params.index:
        if nm.startswith("C(gc_str"):
            key = nm.split("T.")[-1].rstrip("]")
            b, se, p = m.params[nm], m.bse[nm], m.pvalues[nm]
            res[key] = (np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), p)
    boots = {k: [] for k in res}
    n = len(d)
    ok = 0
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        db = d.iloc[idx]
        try:
            mb = smf.glm(formula, data=db, family=sm.families.Binomial(), freq_weights=db[wcol]).fit()
            for nm in mb.params.index:
                if nm.startswith("C(gc_str"):
                    key = nm.split("T.")[-1].rstrip("]")
                    boots[key].append(mb.params[nm])
            ok += 1
        except Exception:
            continue
    ci = {}
    for k, v in boots.items():
        arr = np.array(v)
        if len(arr):
            lo, hi = np.exp(np.percentile(arr, [2.5, 97.5]))
        else:
            lo, hi = np.nan, np.nan
        ci[k] = (lo, hi)
    return res, ci, ok


F_INF = "infection ~ C(gc_str, Treatment(reference='G0_none'))"
P("\n" + "=" * 96)
P("2. WEIGHTED DOSE-RESPONSE for in-hospital infection")
P("=" * 96)
P("%-12s %22s %22s" % ("stratum", "IPTW OR (model CI)", "bootstrap 95% CI"))
res, ci, ok = weighted_logit(df, F_INF)
for g in ["G1_low", "G2_mod", "G3_high"]:
    if g in res:
        orr, lo, hi, p = res[g]
        blo, bhi = ci[g]
        P("%-12s %10.2f (%.2f-%.2f) %14s  P=%.3f" % (LABEL[g], orr, lo, hi, "%.2f-%.2f" % (blo, bhi), p))
P("(bootstrap reps successful = %d / 2000)" % ok)

# weighted ARD
P("\nWeighted standardised risk and absolute risk difference vs no GC:")
m = smf.glm(F_INF, data=df, family=sm.families.Binomial(), freq_weights=df["sw"]).fit()
risks = {}
for g in ORDER:
    dd = df.copy()
    dd["gc_str"] = g
    risks[g] = np.average(m.predict(dd), weights=dd["sw"])
for g in ["G1_low", "G2_mod", "G3_high"]:
    P("  %-10s standardised risk %5.1f%%   ARD %+5.1f pp" % (LABEL[g], 100 * risks[g], 100 * (risks[g] - risks["G0_none"])))
P("  %-10s standardised risk %5.1f%%" % (LABEL["G0_none"], 100 * risks["G0_none"]))

# ------------------------------------------------------------------ negative controls
P("\n" + "=" * 96)
P("3. NEGATIVE CONTROLS")
P("=" * 96)

# positive control: any GC -> infection (should be positive or at least directionally so)
mm = smf.glm("infection ~ gc_any + age + female + charlson + sofa + on_vent + on_vaso + on_rrt",
             data=df, family=sm.families.Binomial()).fit()
b, se = mm.params["gc_any"], mm.bse["gc_any"]
P("positive control  GC(any) -> infection   : OR %.2f (%.2f-%.2f)  P=%.3f" %
  (np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), mm.pvalues["gc_any"]))

# negative outcome: GI bleeding
P("\nnegative outcome  GI bleeding (n events = %d):" % int(df.gi_bleed.sum()))
for lab, f in [("any GC", "gi_bleed ~ gc_any + age + female + charlson + sofa"),
               ("G1", "gi_bleed ~ C(gc_str, Treatment(reference='G0_none')) + age + female + charlson + sofa")]:
    try:
        m2 = smf.glm(f, data=df, family=sm.families.Binomial()).fit()
        for nm in m2.params.index:
            if nm == "gc_any" or nm.startswith("C(gc_str"):
                b, se, p = m2.params[nm], m2.bse[nm], m2.pvalues[nm]
                key = LABEL.get(nm.split("T.")[-1].rstrip("]"), "any GC")
                P("   %-10s OR %.2f (%.2f-%.2f)  P=%.3f" % (key, np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), p))
    except Exception as e:
        P("   failed: %s" % e)

# negative exposure: ondansetron -> infection
P("\nnegative exposure  ondansetron -> infection:")
m3 = smf.glm("infection ~ ondansetron + age + female + charlson + sofa + on_vent + on_vaso",
             data=df, family=sm.families.Binomial()).fit()
b, se, p = m3.params["ondansetron"], m3.bse["ondansetron"], m3.pvalues["ondansetron"]
P("   ondansetron  OR %.2f (%.2f-%.2f)  P=%.3f" % (np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), p))


# ------------------------------------------------------------------ E-values
def evalue(orr, lo):
    rr = orr if orr >= 1 else 1.0 / orr
    rr = max(rr, 1.0001)
    E = rr + np.sqrt(rr * (rr - 1))
    lo_r = lo if lo >= 1 else 1.0 / lo
    lo_r = max(lo_r, 1.0001)
    Elo = lo_r + np.sqrt(lo_r * (lo_r - 1))
    return E, Elo


P("\n" + "=" * 96)
P("4. E-VALUES (for the IPTW estimates)")
P("=" * 96)
for g in ["G1_low", "G2_mod", "G3_high"]:
    if g in res:
        orr, lo, hi, p = res[g]
        E, Elo = evalue(orr, lo)
        P("  %-10s OR %.2f  E-value(point) %.2f  E-value(CI limit) %.2f" % (LABEL[g], orr, E, Elo))

# ------------------------------------------------------------------ sensitivity
P("\n" + "=" * 96)
P("5. SENSITIVITY ANALYSES")
P("=" * 96)

def simple_logit(d, formula, label):
    try:
        m = smf.logit(formula, data=d).fit(disp=0)
        out = []
        for nm in m.params.index:
            if nm.startswith("C(gc_str"):
                key = nm.split("T.")[-1].rstrip("]")
                b, se, p = m.params[nm], m.bse[nm], m.pvalues[nm]
                out.append("%s %.2f (%.2f-%.2f)" % (LABEL.get(key, key), np.exp(b),
                                                    np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)))
        P("  %-38s n=%3d  %s" % (label, len(d), " | ".join(out) if out else "n/a"))
    except Exception as e:
        P("  %-38s failed: %s" % (label, str(e)[:80]))


F_ADJ = "C(gc_str, Treatment(reference='G0_none')) + age + female + charlson + sofa + on_vent + on_vaso + on_rrt + lupus_nephritis + imm_any + emerg"

P("\n(a) cohort restrictions")
simple_logit(df[df.los48 == 1], "infection ~ " + F_ADJ, "ICU LOS >= 48 h")
simple_logit(df[df.gc_any_hosp == 0].assign(gc_str="G0_none"), "infection ~ " + F_ADJ, "never-GC vs exposed (sens)")
simple_logit(df[df.abx_before_icu == 0], "infection ~ " + F_ADJ, "exclude pre-ICU antibiotics")
simple_logit(df[df.on_vaso == 0], "infection ~ " + F_ADJ, "exclude vasopressor users")
simple_logit(df[df.lupus_nephritis == 1], "infection ~ " + F_ADJ, "lupus nephritis subgroup")

P("\n(b) alternative outcome definitions")
df["inf_culture"] = df["culture_after_icu"]
df["inf_abx_new"] = df["abx_new_after48"]
simple_logit(df, "inf_culture ~ " + F_ADJ, "positive blood culture (narrow)")
simple_logit(df, "inf_abx_new ~ " + F_ADJ, "new antibiotic > 48 h (treated)")

P("\n(c) alternative exposure cut-points (tertiles among exposed)")
pos = df.loc[df.gc_daily_pe_mg > 0, "gc_daily_pe_mg"]
try:
    t = pd.qcut(pos, 3, labels=["T1", "T2", "T3"])
    d2 = df.copy()
    d2.loc[pos.index, "gc_t"] = t.astype(str)
    d2["gc_t"] = d2["gc_t"].fillna("G0")
    d2["gc_t"] = pd.Categorical(d2["gc_t"], categories=["G0", "T1", "T2", "T3"])
    m = smf.logit("infection ~ C(gc_t, Treatment(reference='G0')) + age + female + charlson + sofa + on_vent + on_vaso + on_rrt + lupus_nephritis + imm_any + emerg", data=d2).fit(disp=0)
    out = []
    for nm in m.params.index:
        if nm.startswith("C(gc_t"):
            key = nm.split("T.")[-1].rstrip("]")
            b, se, p = m.params[nm], m.bse[nm], m.pvalues[nm]
            out.append("%s %.2f (%.2f-%.2f) P=%.3f" % (key, np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), p))
    P("  " + " | ".join(out))
except Exception as e:
    P("  tertile failed: %s" % str(e)[:120])

P("\n(d) PNI subset (albumin + lymphocyte available)")
sub = df[df.albumin_min.notna() & df.abs_lymphocytes_min.notna()].copy()
sub["pni"] = 10 * sub["albumin_min"] + 5 * sub["abs_lymphocytes_min"]
P("  n = %d" % len(sub))
simple_logit(sub, "infection ~ " + F_ADJ, "without PNI")
simple_logit(sub, "infection ~ " + F_ADJ + " + pni", "with PNI")

with open(os.path.join(OUT, "30_iptw_sensitivity.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nsaved -> out/30_iptw_sensitivity.txt")
