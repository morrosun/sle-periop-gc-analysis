# -*- coding: utf-8 -*-
"""
Time-dependent exposure + marginal structural model for 30-day mortality
========================================================================
Exposure is reconstructed as a time-dependent binary indicator that flips from 0 to 1 at the
first glucocorticoid administration inside the exposure window, which removes the immortal-time
artefact inherent in a cumulative-dose definition. Stabilised IPTW then addresses time-varying
indication confounding measured by baseline severity.

Caveat carried forward: only 54 deaths are available, so every estimate here is exploratory.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from lifelines import CoxTimeVaryingFitter
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, OUT = os.path.join(BASE, "data"), os.path.join(BASE, "out")
rng = np.random.default_rng(7)

df = pd.read_csv(os.path.join(DATA, "cohort_sle_icu_v3.csv"))
nc = pd.read_csv(os.path.join(DATA, "negctrl_vars.csv"))
df = df.merge(nc, on=["hadm_id", "stay_id"], how="left")

df["imm_any"] = (df[["imm_cyc", "imm_mmf", "imm_aza", "imm_cni", "imm_mtx", "imm_bio"]].max(axis=1) == 1).astype(int)
df["emerg"] = df["admission_type"].astype(str).str.contains("EMER", case=False, na=False).astype(int)
df["female"] = (df["gender"] == "F").astype(int)
for v in ["sofa", "charlson", "apsiii", "oasis", "wbc_min", "platelets_min", "creatinine_max"]:
    df[v] = df[v].fillna(df[v].median())

lines = []
def P(s=""):
    print(s); lines.append(str(s))

P("=" * 94)
P("TIME-DEPENDENT EXPOSURE + MARGINAL STRUCTURAL MODEL - 30-day mortality")
P("=" * 94)

# ------------------------------------------------------------------ build long format
rows = []
for _, r in df.iterrows():
    # follow-up time in days, censored at 30 d / discharge / death
    if pd.notna(r["deathtime"]):
        t_death = (pd.to_datetime(r["deathtime"]) - pd.to_datetime(r["intime"])).total_seconds() / 86400.0
    else:
        t_death = np.nan
    t_disch = (pd.to_datetime(r["dischtime"]) - pd.to_datetime(r["intime"])).total_seconds() / 86400.0
    t_max = 30.0

    if pd.notna(t_death) and t_death <= t_max:
        t_end, ev = max(t_death, 1e-6), 1
    else:
        t_end, ev = min(t_disch if pd.notna(t_disch) else t_max, t_max), 0
    if t_end <= 0:
        continue

    start = max(float(r["gc_first_hour"]) / 24.0, 0.0) if pd.notna(r["gc_first_hour"]) else np.nan
    exposed = (r["gc_any"] == 1) and pd.notna(start)

    if exposed and start > 0 and start < t_end:
        rows.append(dict(stay_id=r.stay_id, start=0.0, stop=start, gc=0, event=0,
                         age=r.age, female=r.female, charlson=r.charlson, sofa=r.sofa,
                         apsiii=r.apsiii, on_vent=r.on_vent, on_vaso=r.on_vaso, on_rrt=r.on_rrt,
                         lupus_nephritis=r.lupus_nephritis, imm_any=r.imm_any, emerg=r.emerg))
        rows.append(dict(stay_id=r.stay_id, start=start, stop=t_end, gc=1, event=ev,
                         age=r.age, female=r.female, charlson=r.charlson, sofa=r.sofa,
                         apsiii=r.apsiii, on_vent=r.on_vent, on_vaso=r.on_vaso, on_rrt=r.on_rrt,
                         lupus_nephritis=r.lupus_nephritis, imm_any=r.imm_any, emerg=r.emerg))
    else:
        gc0 = 1 if (exposed and start <= 0) else 0
        rows.append(dict(stay_id=r.stay_id, start=0.0, stop=t_end, gc=gc0, event=ev,
                         age=r.age, female=r.female, charlson=r.charlson, sofa=r.sofa,
                         apsiii=r.apsiii, on_vent=r.on_vent, on_vaso=r.on_vaso, on_rrt=r.on_rrt,
                         lupus_nephritis=r.lupus_nephritis, imm_any=r.imm_any, emerg=r.emerg))

long = pd.DataFrame(rows)
P("\nlong format: %d rows, %d stays, %d events" % (len(long), long.stay_id.nunique(), int(long.event.sum())))
P("person-time exposed / unexposed (days): %.0f / %.0f" % (
    long.loc[long.gc == 1, "stop"].sum() - long.loc[long.gc == 1, "start"].sum(),
    long.loc[long.gc == 0, "stop"].sum() - long.loc[long.gc == 0, "start"].sum()))

# ------------------------------------------------------------------ unweighted time-dependent Cox
ctv = CoxTimeVaryingFitter()
cov = ["gc", "age", "female", "charlson", "sofa", "on_vaso", "on_rrt", "lupus_nephritis", "imm_any"]
ctv.fit(long, id_col="stay_id", event_col="event", start_col="start", stop_col="stop",
        formula=" + ".join(cov))
P("\nUnweighted time-dependent Cox (30-day mortality):")
P("   HR %.2f  (95%% CI %.2f-%.2f)  P=%.3f" % (
    ctv.hazard_ratios_["gc"],
    np.exp(ctv.confidence_intervals_.loc["gc"].iloc[0]),
    np.exp(ctv.confidence_intervals_.loc["gc"].iloc[1]),
    ctv.summary.loc["gc", "p"]))

# ------------------------------------------------------------------ stabilised IPTW
P("\n" + "=" * 94)
P("STABILISED IPTW (marginal structural Cox)")
P("=" * 94)

COV = ["age", "female", "charlson", "sofa", "apsiii", "on_vent", "on_vaso", "on_rrt",
       "lupus_nephritis", "imm_any", "emerg"]

# denominator model: P(gc | baseline covariates, time)
X = long[COV + ["start"]].values
X = (X - X.mean(0)) / np.std(X, axis=0)
y = long["gc"].values
mdl = LogisticRegression(solver="lbfgs", max_iter=5000, C=1.0).fit(X, y)
p_den = mdl.predict_proba(X)[:, 1]

# numerator model: P(gc | time only)
Xn = long[["start"]].values
Xn = (Xn - Xn.mean(0)) / np.std(Xn, axis=0)
mdl_n = LogisticRegression(solver="lbfgs", max_iter=5000, C=1.0).fit(Xn, y)
p_num = mdl_n.predict_proba(Xn)[:, 1]

long["sw"] = np.where(y == 1, p_num / np.clip(p_den, 1e-6, 1),
                      (1 - p_num) / np.clip(1 - p_den, 1e-6, 1))
long["sw"] = long["sw"].clip(0.05, 20)

P("weights: mean %.3f  min %.3f  max %.3f  ESS %.1f / %d" % (
    long.sw.mean(), long.sw.min(), long.sw.max(),
    long.sw.sum() ** 2 / (long.sw ** 2).sum(), len(long)))
P("exposure model AUC-class separation: mean p_den exposed %.3f vs unexposed %.3f" % (
    p_den[y == 1].mean(), p_den[y == 0].mean()))

ctv_w = CoxTimeVaryingFitter()
ctv_w.fit(long, id_col="stay_id", event_col="event", start_col="start", stop_col="stop",
          formula="gc", weights_col="sw")
P("\nMSM (IPTW-weighted, marginal) - 30-day mortality:")
P("   HR %.2f  (95%% CI %.2f-%.2f)  P=%.3f" % (
    ctv_w.hazard_ratios_["gc"],
    np.exp(ctv_w.confidence_intervals_.loc["gc"].iloc[0]),
    np.exp(ctv_w.confidence_intervals_.loc["gc"].iloc[1]),
    ctv_w.summary.loc["gc", "p"]))

# ------------------------------------------------------------------ compare with the static model
P("\n" + "=" * 94)
P("CONTRAST: static vs time-dependent exposure definition")
P("=" * 94)
m_static = smf.logit("death_30d ~ gc_any + age + female + charlson + sofa + on_vaso + on_rrt + lupus_nephritis + imm_any",
                     data=df).fit(disp=0)
b, se = m_static.params["gc_any"], m_static.bse["gc_any"]
P("static  any GC (logistic)          : OR %.2f (%.2f-%.2f)  P=%.3f" % (
    np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), m_static.pvalues["gc_any"]))

# static high cumulative dose (the definition that inflates associations)
df["gc_high"] = (df["gc_cum_pe_mg"] > 250).astype(int)
m_hi = smf.logit("death_30d ~ gc_high + age + female + charlson + sofa + on_vaso + on_rrt", data=df).fit(disp=0)
b, se = m_hi.params["gc_high"], m_hi.bse["gc_high"]
P("static  high cumulative dose >250mg: OR %.2f (%.2f-%.2f)  P=%.3f" % (
    np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), m_hi.pvalues["gc_high"]))

P("time-dependent Cox (unweighted)    : HR %.2f" % ctv.hazard_ratios_["gc"])
P("MSM (IPTW-weighted)                : HR %.2f" % ctv_w.hazard_ratios_["gc"])

P("\nEvents available for the mortality models: %d. "
  "Interpretation is restricted to 'no evidence of an independent association'." % int(df.death_30d.sum()))

with open(os.path.join(OUT, "40_msm_results.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nsaved -> out/40_msm_results.txt")
