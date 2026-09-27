# -*- coding: utf-8 -*-
"""
153_head_to_head_robustness.py

Plan B, step 3: is the disease x dose interaction real, or is it the
15-year age gap wearing a disease costume?

152 reported, for the interaction RA-vs-SLE in the dose-trend:
    blood culture + >24 h   adjusted 0.73 (0.48-1.10)  P=0.131
                            IPTW     0.62 (0.40-0.97)  P=0.037
    hospital death          adjusted 1.54 (1.05-2.26)  P=0.027
                            IPTW     1.25 (0.88-1.78)  P=0.208
Note the awkward pattern: which outcome reaches significance depends on
the adjustment strategy.  That alone means the headline cannot be "the
dose-response differs by disease" without support.

Five attacks on the interaction
-------------------------------
 1  AGE.  RA is 15 y older.  (a) restrict both diseases to a common age
    window; (b) let the dose effect itself vary with age (s:age) so that
    any age-driven dose-response is absorbed before the disease contrast.
 2  INDICATION.  surgical admissions differ (RA 41% vs SLE 34%); drop them.
 3  EXPOSURE DEFINITION.  the 4-stratum ordinal score is arbitrary; retest
    with any-GC and with >=50 vs none.
 4  RESAMPLING.  bootstrap the interaction within disease to get its
    sampling distribution and the share of replicates agreeing in sign.
 5  UNMEASURED CONFOUNDING.  E-value for the per-disease estimates.

Controls (hyperglycaemia positive, GI bleeding negative) are carried
through every attack.  If a model produces a control interaction, the
model is wrong, not the disease.
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
ORDER = ["G0_none", "G1_low", "G2_mod", "G3_high"]

CORE = ["age", "female", "sofa", "vaso24", "vent24", "renal_fail", "shock", "surg_any"]
OUTCOMES = [("culture_pos_after24", "blood culture + >24 h"),
            ("infect_icd", "infection ICD code"),
            ("death_hosp", "hospital death"),
            ("hyper_48h", "glucose >=180 (POS CTRL)"),
            ("gi_bleed", "GI bleeding (NEG CTRL)")]

lines = []


def P(s=""):
    print(s)
    lines.append(str(s))


def quiet_logit(y, X):
    """Logistic MLE without the convergence noise; returns (b, se) or None."""
    try:
        with np.errstate(all="ignore"):
            m = sm.Logit(y, X).fit(disp=0, maxiter=200)
        b, se = np.asarray(m.params, float), np.asarray(m.bse, float)
        if not np.all(np.isfinite(b)) or not np.all(np.isfinite(se)):
            return None
        return b, se
    except Exception:
        return None


def interaction(d, y, covs, extra_int=None):
    """RA-vs-SLE ratio of dose-trend ORs per stratum increment.

    The model is  y ~ s + is_sle + s:is_sle + covs  (+ optional extra
    interactions of s with covariates, given as column names in
    `extra_int`).  exp(b[s:is_sle]) is the SLE/RA slope ratio, which is
    reciprocated so that OR > 1 means the dose-response is steeper in RA.
    """
    need = ["gc_str_num", "is_sle", y] + covs + (extra_int or [])
    need = list(dict.fromkeys(need))        # extra_int may repeat a covariate
    use = d[need].dropna().copy()
    if len(use) == 0 or use[y].nunique() < 2:
        return None
    use["s"] = use.gc_str_num.astype(float)
    use["sx"] = use.s * use.is_sle.astype(float)
    terms = ["s", "is_sle", "sx"] + covs
    for v in (extra_int or []):
        use["s_" + v] = use.s * use[v].astype(float)
        terms.append("s_" + v)
    X = sm.add_constant(use[terms].astype(float))
    r = quiet_logit(use[y].astype(float).values, X.values)
    if r is None:
        return None
    b, se = r
    j = list(X.columns).index("sx")
    if se[j] <= 0:
        return None
    e = float(np.exp(b[j]))
    return dict(OR=1.0 / e, lo=1.0 / float(np.exp(b[j] + 1.96 * se[j])),
                hi=1.0 / float(np.exp(b[j] - 1.96 * se[j])),
                p=float(2 * (1 - stats.norm.cdf(abs(b[j] / se[j])))),
                n=int(len(use)), n_ev=int(use[y].sum()))


def trend(d, y, covs):
    need = ["gc_str_num", y] + covs
    use = d[need].dropna()
    if len(use) == 0 or use[y].nunique() < 2:
        return None
    X = sm.add_constant(use[["gc_str_num"] + covs].astype(float))
    r = quiet_logit(use[y].astype(float).values, X.values)
    if r is None:
        return None
    b, se = r
    j = list(X.columns).index("gc_str_num")
    return dict(OR=float(np.exp(b[j])), lo=float(np.exp(b[j] - 1.96 * se[j])),
                hi=float(np.exp(b[j] + 1.96 * se[j])),
                p=float(2 * (1 - stats.norm.cdf(abs(b[j] / se[j])))),
                n=int(len(use)), n_ev=int(use[y].sum()))


def line(r):
    if r is None:
        return "n.e."
    return "%.2f (%.2f-%.2f) P=%.3f" % (r["OR"], r["lo"], r["hi"], r["p"])


df = pd.read_csv(os.path.join(DATA, "cohort_rheum_icu.csv"))
df = df.rename(columns={"sofa24": "sofa"})
df["sofa"] = df.sofa.fillna(df.sofa.median())
df["gc_str_num"] = df.gc24_str.map({"G0_none": 0.0, "G1_low": 1.0, "G2_mod": 2.0, "G3_high": 3.0})
both = df[df.primary_grp.isin(["SLE", "RA"])].copy()

P("=" * 100)
P("HEAD-TO-HEAD ROBUSTNESS")
P("=" * 100)
P("  SLE n=%d (age median %.1f)   RA n=%d (age median %.1f)   gap %.1f y" % (
    (both.is_sle == 1).sum(), both.loc[both.is_sle == 1, "age"].median(),
    (both.is_sle == 0).sum(), both.loc[both.is_sle == 0, "age"].median(),
    both.loc[both.is_sle == 0, "age"].median() - both.loc[both.is_sle == 1, "age"].median()))

# =====================================================================
# 1a. common age window
# =====================================================================
P("")
P("=" * 100)
P("1a. AGE OVERLAP: restrict BOTH diseases to the same age window")
P("=" * 100)
P("  %-14s %8s %8s %8s %8s %8s" % ("window", "SLE n", "RA n", "SLE ev*", "RA ev*",
                                    "PA" ))
for lo, hi in [(18, 120), (40, 85), (50, 80), (55, 75), (60, 80)]:
    sub = both[(both.age >= lo) & (both.age <= hi)]
    a = sub[sub.is_sle == 1]
    b2 = sub[sub.is_sle == 0]
    if len(a) < 80 or len(b2) < 200:
        P("  %-14s %8d %8d   (too small)" % ("%d-%d y" % (lo, hi), len(a), len(b2)))
        continue
    P("  %-14s %8d %8d %8d %8d" % ("%d-%d y" % (lo, hi), len(a), len(b2),
                                   int(a.culture_pos_after24.sum()),
                                   int(b2.culture_pos_after24.sum())))
P("  * events are blood-culture positive >24 h")

P("")
P("  interaction under each age window (RA vs SLE trend ratio):")
P("  %-10s %-16s %26s %8s   %26s %8s" % ("window", "outcome", "adjusted", "P",
                                          "with s:age", "P"))
age_rows = []
for lo, hi in [(18, 120), (50, 80), (55, 75)]:
    sub = both[(both.age >= lo) & (both.age <= hi)]
    if len(sub[sub.is_sle == 1]) < 80:
        continue
    for y, lab in OUTCOMES:
        r1 = interaction(sub, y, CORE)
        r2 = interaction(sub, y, CORE, extra_int=["age"])
        if r1 is None:
            continue
        P("  %-10s %-16s %26s %8.3f   %26s %8.3f" % (
            "%d-%d" % (lo, hi), lab[:16], line(r1), r1["p"], line(r2),
            r2["p"] if r2 else np.nan))
        age_rows.append(dict(window="%d-%d" % (lo, hi), outcome=lab,
                             adj_or=r1["OR"], adj_p=r1["p"],
                             adj_lo=r1["lo"], adj_hi=r1["hi"],
                             with_s_age_or=(r2 or {}).get("OR"),
                             with_s_age_p=(r2 or {}).get("p"),
                             with_s_age_lo=(r2 or {}).get("lo"),
                             with_s_age_hi=(r2 or {}).get("hi")))

# =====================================================================
# 1b. let the dose effect vary with age, full cohort
# =====================================================================
P("")
P("=" * 100)
P("1b. ABSORB AGE: add s:age and s:sofa to the interaction model.")
P("    If the disease contrast is really an age contrast, it should vanish.")
P("=" * 100)
P("  %-34s %26s %26s %26s" % ("outcome", "base adj", "+ s:age", "+ s:age + s:sofa"))
absorb = []
for y, lab in OUTCOMES:
    r0 = interaction(both, y, CORE)
    r1 = interaction(both, y, CORE, extra_int=["age"])
    r2 = interaction(both, y, CORE, extra_int=["age", "sofa"])
    if r0 is None:
        continue
    P("  %-34s %26s %26s %26s" % (lab, line(r0), line(r1), line(r2)))
    absorb.append(dict(outcome=lab, base_or=r0["OR"], base_p=r0["p"],
                       base_lo=r0["lo"], base_hi=r0["hi"],
                       s_age_or=(r1 or {}).get("OR"), s_age_p=(r1 or {}).get("p"),
                       s_age_lo=(r1 or {}).get("lo"), s_age_hi=(r1 or {}).get("hi"),
                       s_age_sofa_or=(r2 or {}).get("OR"),
                       s_age_sofa_p=(r2 or {}).get("p"),
                       s_age_sofa_lo=(r2 or {}).get("lo"),
                       s_age_sofa_hi=(r2 or {}).get("hi")))

# =====================================================================
# 2. surgical admissions
# =====================================================================
P("")
P("=" * 100)
P("2. INDICATION: drop surgical admissions (RA 41.3% vs SLE 34.2%)")
P("=" * 100)
P("  %-34s %26s %26s" % ("outcome", "all", "non-surgical only"))
surg = []
for y, lab in OUTCOMES:
    r0 = interaction(both, y, CORE)
    r1 = interaction(both[both.surg_any == 0], y, [c for c in CORE if c != "surg_any"])
    if r0 is None:
        continue
    P("  %-34s %26s %26s" % (lab, line(r0), line(r1)))
    surg.append(dict(outcome=lab, all_or=r0["OR"], all_p=r0["p"],
                     all_lo=r0["lo"], all_hi=r0["hi"],
                     nonsurg_or=(r1 or {}).get("OR"), nonsurg_p=(r1 or {}).get("p"),
                     nonsurg_lo=(r1 or {}).get("lo"), nonsurg_hi=(r1 or {}).get("hi"),
                     n=(r1 or {}).get("n")))

# =====================================================================
# 3. exposure definition
# =====================================================================
P("")
P("=" * 100)
P("3. EXPOSURE DEFINITION: is the ordinal score doing the work?")
P("=" * 100)
P("  %-34s %26s %26s %26s" % ("outcome", "ordinal score", "any GC", ">=50 vs none"))
expo = []
for y, lab in OUTCOMES:
    r0 = interaction(both, y, CORE)
    d = both.copy()
    d["gc_str_num"] = d.gc_any24.astype(float)          # any vs none
    r1 = interaction(d, y, CORE)
    d2 = both[both.gc24_str.isin(["G0_none", "G3_high"])].copy()
    d2["gc_str_num"] = (d2.gc24_str == "G3_high").astype(float)
    r2 = interaction(d2, y, CORE)
    if r0 is None:
        continue
    P("  %-34s %26s %26s %26s" % (lab, line(r0), line(r1), line(r2)))
    expo.append(dict(outcome=lab, ordinal_or=r0["OR"], ordinal_p=r0["p"],
                     ordinal_lo=r0["lo"], ordinal_hi=r0["hi"],
                     any_or=(r1 or {}).get("OR"), any_p=(r1 or {}).get("p"),
                     any_lo=(r1 or {}).get("lo"), any_hi=(r1 or {}).get("hi"),
                     high_or=(r2 or {}).get("OR"), high_p=(r2 or {}).get("p"),
                     high_lo=(r2 or {}).get("lo"), high_hi=(r2 or {}).get("hi")))

# =====================================================================
# 4. bootstrap
# =====================================================================
P("")
P("=" * 100)
P("4. BOOTSTRAP the interaction (2000 resamples, stratified by disease)")
P("=" * 100)
rng = np.random.default_rng(20260916)
B = 2000
boot = []
for y, lab in [("culture_pos_after24", "blood culture + >24 h"),
               ("death_hosp", "hospital death"),
               ("hyper_48h", "glucose >=180 (POS CTRL)")]:
    a = both[both.is_sle == 1]
    b2 = both[both.is_sle == 0]
    vals = []
    for _ in range(B):
        sa = a.iloc[rng.integers(0, len(a), len(a))]
        sb = b2.iloc[rng.integers(0, len(b2), len(b2))]
        r = interaction(pd.concat([sa, sb], ignore_index=True), y, CORE)
        if r is not None and np.isfinite(r["OR"]):
            vals.append(r["OR"])
    if len(vals) < 100:
        P("  %-34s bootstrap failed" % lab)
        continue
    v = np.array(vals)
    lv = np.log(v)
    P("  %-34s median %.2f  (%.2f-%.2f percentile)  share OR>1 = %.1f%%  k=%d" % (
        lab, np.exp(np.median(lv)), np.exp(np.percentile(lv, 2.5)),
        np.exp(np.percentile(lv, 97.5)), 100 * (v > 1).mean(), len(v)))
    boot.append(dict(outcome=lab, median=float(np.exp(np.median(lv))),
                     lo=float(np.exp(np.percentile(lv, 2.5))),
                     hi=float(np.exp(np.percentile(lv, 97.5))),
                     frac_gt1=float((v > 1).mean()), k=int(len(v))))

# =====================================================================
# 5. E-value
# =====================================================================
P("")
P("=" * 100)
P("5. UNMEASURED CONFOUNDING: E-value for the per-disease dose-trend")
P("=" * 100)
P("  E-value = minimum strength of association (on the risk-ratio scale)")
P("  that an unmeasured confounder would need with BOTH the exposure and")
P("  the outcome to explain the estimate away.  OR -> RR by")
P("  RR = OR / (1 - p0 + p0*OR) with p0 = risk in the reference stratum.")
P("")
P("  %-8s %-24s %8s %8s %8s %10s" % ("disease", "outcome", "OR", "p0", "RR", "E-value"))


def evalue(rr):
    if rr < 1:
        rr = 1.0 / rr
    return rr + np.sqrt(rr * (rr - 1))


ev_rows = []
for g in ["SLE", "RA"]:
    d = df[df.primary_grp == g]
    for y, lab in [("culture_pos_after24", "blood culture + >24 h"),
                   ("infect_icd", "infection ICD code"),
                   ("death_hosp", "hospital death")]:
        r = trend(d, y, CORE)
        if r is None:
            continue
        p0 = d.loc[d.gc24_str == "G0_none", y].mean()
        rr = r["OR"] / (1 - p0 + p0 * r["OR"])
        P("  %-8s %-24s %8.3f %8.3f %8.3f %10.2f" % (g, lab, r["OR"], p0, rr, evalue(rr)))
        ev_rows.append(dict(disease=g, outcome=lab, OR=r["OR"], p0=float(p0),
                            RR=float(rr), e_value=float(evalue(rr)), n_ev=r["n_ev"]))

# per-stratum increment is not a natural risk contrast; also give the
# >=50 vs none binary E-value, which is the interpretable one
P("")
P("  same for the interpretable binary contrast (>=50 vs none):")
P("  %-8s %-24s %8s %8s %8s %10s" % ("disease", "outcome", "OR", "p0", "RR", "E-value"))
for g in ["SLE", "RA"]:
    d = df[df.primary_grp == g]
    for y, lab in [("culture_pos_after24", "blood culture + >24 h"),
                   ("death_hosp", "hospital death")]:
        sub = d[d.gc24_str.isin(["G0_none", "G3_high"])].copy()
        sub["e"] = (sub.gc24_str == "G3_high").astype(int)
        X = sm.add_constant(sub[["e"] + CORE].astype(float))
        rr_ = quiet_logit(sub[y].astype(float).values, X.values)
        if rr_ is None:
            continue
        b, se = rr_
        j = list(X.columns).index("e")
        orv = float(np.exp(b[j]))
        p0 = sub.loc[sub.e == 0, y].mean()
        rrv = orv / (1 - p0 + p0 * orv)
        P("  %-8s %-24s %8.3f %8.3f %8.3f %10.2f" % (g, lab, orv, p0, rrv, evalue(rrv)))
        ev_rows.append(dict(disease=g, outcome=lab + " (>=50 vs none)", OR=orv,
                            p0=float(p0), RR=float(rrv), e_value=float(evalue(rrv)),
                            n_ev=int(sub[y].sum()),
                            lo=float(np.exp(b[j] - 1.96 * se[j])),
                            hi=float(np.exp(b[j] + 1.96 * se[j]))))

pd.DataFrame(age_rows).to_csv(os.path.join(OUT, "table_h2h_robust_age.csv"), index=False)
pd.DataFrame(absorb).to_csv(os.path.join(OUT, "table_h2h_robust_absorb.csv"), index=False)
pd.DataFrame(surg).to_csv(os.path.join(OUT, "table_h2h_robust_surg.csv"), index=False)
pd.DataFrame(expo).to_csv(os.path.join(OUT, "table_h2h_robust_expo.csv"), index=False)
pd.DataFrame(boot).to_csv(os.path.join(OUT, "table_h2h_robust_boot.csv"), index=False)
pd.DataFrame(ev_rows).to_csv(os.path.join(OUT, "table_h2h_evalue.csv"), index=False)

with open(os.path.join(OUT, "153_robustness.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nsaved -> out/153_robustness.txt")
