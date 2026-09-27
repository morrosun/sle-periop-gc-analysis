# -*- coding: utf-8 -*-
"""
162_external_head_to_head.py

Plan B, step 4: test the SLE-vs-RA contrast outside MIMIC-IV.

What can and cannot be tested externally
----------------------------------------
   outcome                MIMIC-IV   eICU-CRD   NWICU
   blood culture + >24 h  36 / 51 *  0 / 0      no microbiology table
   infection ICD code     906        139        35
   hospital death         179        38         0
   glucose >=180 (ctrl)   493        122        41
   * SLE / RA events

So the specific bacteraemia signal -- the one outcome that drives the
MIMIC head-to-head -- CANNOT be tested externally at all.  What can be
tested is whether the WIDE infection outcome and mortality show the same
disease-dependent dose-response, and whether the exposure variable
(positive control) behaves the same way in both diseases in both
databases.

A separate, already-known problem applies to eICU: only 18% of these
stays have any glucocorticoid order in the first ICU day, against 41% in
MIMIC-IV and 47% in NWICU.  Exposure misclassification of that size is
directional -- it drags every estimate toward the null -- so an eICU
null is close to uninformative.  It is reported rather than hidden.

Minimum detectable effect (MDE) is reported for every cell so that
"no signal" can be told apart from "no power".
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")

DBCOL = {"mimiciv": "MIMIC-IV", "eicu": "eICU-CRD", "nwicu": "NWICU"}
COV = ["age", "female", "renal_fail", "vaso24", "vent24"]
OUTCOMES = [("infect_icd", "infection ICD code"),
            ("death_hosp", "hospital death"),
            ("hyper_48h", "glucose >=180 (POS CTRL)"),
            ("gi_bleed", "GI bleeding (NEG CTRL)")]

lines = []


def P(s=""):
    print(s)
    lines.append(str(s))


def fit(df, y, xvars):
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
    return dict(OR=float(np.exp(b)), lo=float(np.exp(b - 1.96 * se)),
                hi=float(np.exp(b + 1.96 * se)),
                p=float(2 * (1 - stats.norm.cdf(abs(b / se)))),
                n=int(len(use)), n_ev=int(use[y].sum()))


def line(r):
    if r is None:
        return "not estimable"
    return "%.2f (%.2f-%.2f) P=%.3f" % (r["OR"], r["lo"], r["hi"], r["p"])


def interaction(d, y, covs):
    """RA-vs-SLE ratio of dose-trend ORs.  exp(b[s:is_ra]) *is* the RA/SLE
    ratio because here the interaction is built as s * is_ra (not is_sle):
    getting this orientation wrong silently inverts the conclusion, so it is
    written this way on purpose."""
    need = list(dict.fromkeys(["gc_str_num", "is_ra", y] + covs))
    use = d[need].dropna().copy()
    if len(use) < 40 or use[y].nunique() < 2 or use.gc_str_num.nunique() < 2:
        return None
    use["s"] = use.gc_str_num.astype(float)
    use["sx"] = use.s * use.is_ra.astype(float)
    X = sm.add_constant(use[["s", "is_ra", "sx"] + covs].astype(float))
    try:
        m = sm.Logit(use[y].astype(float), X).fit(disp=0)
        b, se = np.asarray(m.params), np.asarray(m.bse)
    except Exception:
        return None
    j = list(X.columns).index("sx")
    if not np.isfinite(b[j]) or se[j] <= 0 or abs(b[j]) > 12:
        return None
    return dict(OR=float(np.exp(b[j])), lo=float(np.exp(b[j] - 1.96 * se[j])),
                hi=float(np.exp(b[j] + 1.96 * se[j])),
                p=float(2 * (1 - stats.norm.cdf(abs(b[j] / se[j])))),
                n=int(len(use)), n_ev=int(use[y].sum()))


def mde(n1, n0, p0, alpha=0.05, power=0.80):
    if n1 <= 0 or n0 <= 0 or p0 <= 0:
        return None
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)

    def pw(orv):
        p1 = orv * p0 / (1 - p0 + orv * p0)
        pb = (n1 * p1 + n0 * p0) / (n1 + n0)
        se = np.sqrt(pb * (1 - pb) * (1 / n1 + 1 / n0))
        if se <= 0:
            return 0.0
        return float(stats.norm.cdf(abs(p1 - p0) / se - z_a))
    lo, hi = 1.0, 500.0
    if pw(hi) < power:
        return None
    for _ in range(80):
        mid = (lo + hi) / 2
        if pw(mid) < power:
            lo = mid
        else:
            hi = mid
    return hi


# ------------------------------------------------------------------- load
def load():
    d = {}
    x = pd.read_csv(os.path.join(DATA, "cohort_rheum_icu.csv"))
    x = x.rename(columns={"sofa24": "sofa"})
    x = x[x.primary_grp.isin(["SLE", "RA"])].copy()
    x["db"] = "mimiciv"
    d["mimiciv"] = x

    for k in ["eicu", "nwicu"]:
        y = pd.read_csv(os.path.join(DATA, "cohort_%s_rheum.csv" % k))
        y["db"] = k
        d[k] = y
    for k, y in d.items():                       # needed for the interaction model
        y["is_ra"] = (y.primary_grp == "RA").astype(int)
        y["is_sle"] = (y.primary_grp == "SLE").astype(int)
    return d


D = load()

P("=" * 100)
P("EXTERNAL HEAD-TO-HEAD:  SLE vs RA in MIMIC-IV, eICU-CRD, NWICU")
P("=" * 100)
P("  %-10s %8s %8s %10s %12s %10s" % ("database", "SLE n", "RA n", "SLE GC%", "RA GC%",
                                       "events*"))
for k in ["mimiciv", "eicu", "nwicu"]:
    x = D[k]
    a, b = x[x.primary_grp == "SLE"], x[x.primary_grp == "RA"]
    P("  %-10s %8d %8d %9.1f%% %11.1f%% %10d" % (
        DBCOL[k], len(a), len(b), 100 * a.gc_any24.mean(), 100 * b.gc_any24.mean(),
        int(x.infect_icd.sum())))

# =====================================================================
# 1. positive control, by disease and database
# =====================================================================
P("")
P("=" * 100)
P("1. POSITIVE CONTROL by disease and database  (GC -> glucose >=180)")
P("   This must be positive in BOTH diseases wherever the exposure is")
P("   recorded well.  Where it is not, that database cannot adjudicate the")
P("   head-to-head -- the exposure, not the disease, is the problem.")
P("=" * 100)
P("  %-10s %-6s %6s %7s %10s %26s" % ("database", "disease", "n", "events",
                                       "any GC %", "dose-trend OR per stratum"))
pc_rows = []
for k in ["mimiciv", "eicu", "nwicu"]:
    x = D[k]
    for g in ["SLE", "RA"]:
        s = x[x.primary_grp == g]
        r = fit(s, "hyper_48h", ["gc_str_num"])
        P("  %-10s %-6s %6d %7d %9.1f%% %26s" % (
            DBCOL[k], g, len(s), int(s.hyper_48h.sum()), 100 * s.gc_any24.mean(), line(r)))
        pc_rows.append(dict(db=k, disease=g, n=len(s), n_ev=int(s.hyper_48h.sum()),
                            gc_pct=float(100 * s.gc_any24.mean()),
                            trend_or=(r or {}).get("OR"), trend_lo=(r or {}).get("lo"),
                            trend_hi=(r or {}).get("hi"), trend_p=(r or {}).get("p")))

# =====================================================================
# 2. outcomes by disease and database
# =====================================================================
P("")
P("=" * 100)
P("2. DOSE-TREND BY DISEASE AND DATABASE  (OR per dose-stratum increment)")
P("=" * 100)
for y, lab in OUTCOMES:
    P("")
    P("  outcome = %s" % lab)
    P("  %-10s %-6s %6s %7s %26s %26s" % ("database", "disease", "n", "events",
                                           "crude", "+age,sex,severity"))
    for k in ["mimiciv", "eicu", "nwicu"]:
        x = D[k]
        for g in ["SLE", "RA"]:
            s = x[x.primary_grp == g]
            if y == "hyper_48h":
                s = s[s.glu_max_48h.notna()]
            r0 = fit(s, y, ["gc_str_num"])
            r1 = fit(s, y, ["gc_str_num"] + COV)
            if r0 is None and r1 is None:
                P("  %-10s %-6s %6d %7d   %-24s %-24s" % (
                    DBCOL[k], g, len(s), int(s[y].sum()), "not estimable", "not estimable"))
                continue
            P("  %-10s %-6s %6d %7d %26s %26s" % (
                DBCOL[k], g, len(s), int(s[y].sum()), line(r0), line(r1)))

# =====================================================================
# 3. the head-to-head interaction, per database
# =====================================================================
P("")
P("=" * 100)
P("3. HEAD-TO-HEAD INTERACTION PER DATABASE  (RA/SLE ratio of dose-trends)")
P("   >1 = the dose-response is steeper in RA.  Controls must sit at 1.")
P("=" * 100)
P("  %-24s %-10s %6s %7s %26s %10s" % ("outcome", "database", "n", "events",
                                        "RA/SLE OR (95% CI)", "P"))
int_rows = []
for y, lab in OUTCOMES:
    for k in ["mimiciv", "eicu", "nwicu"]:
        x = D[k]
        if y == "hyper_48h":
            x = x[x.glu_max_48h.notna()]
        r0 = interaction(x, y, [])
        r1 = interaction(x, y, COV)
        if r0 is None and r1 is None:
            P("  %-24s %-10s   not estimable" % (lab, DBCOL[k]))
            continue
        P("  %-24s %-10s %6d %7d %26s %10.3f   crude %s" % (
            lab, DBCOL[k], (r1 or r0)["n"], (r1 or r0)["n_ev"], line(r1),
            (r1 or r0)["p"], ("%.2f" % r0["OR"]) if r0 else "-"))
        int_rows.append(dict(outcome=lab, db=k, n=(r1 or r0)["n"], n_ev=(r1 or r0)["n_ev"],
                             adj_or=(r1 or {}).get("OR"), adj_lo=(r1 or {}).get("lo"),
                             adj_hi=(r1 or {}).get("hi"), adj_p=(r1 or {}).get("p"),
                             crude_or=(r0 or {}).get("OR"), crude_p=(r0 or {}).get("p")))

# =====================================================================
# 4. MDE
# =====================================================================
P("")
P("=" * 100)
P("4. WHAT COULD EACH DATABASE x DISEASE DETECT?  (80% power, alpha=0.05,")
P("   binary any-GC vs none, on the unit's own reference risk)")
P("=" * 100)
P("  %-24s %-10s %-6s %8s %8s %10s %12s" % (
    "outcome", "database", "disease", "n gc", "n none", "risk none", "MDE OR"))
mde_rows = []
for y, lab in [("infect_icd", "infection ICD code"), ("death_hosp", "hospital death"),
               ("hyper_48h", "glucose >=180")]:
    for k in ["mimiciv", "eicu", "nwicu"]:
        x = D[k]
        for g in ["SLE", "RA"]:
            s = x[x.primary_grp == g].dropna(subset=[y])
            n1 = int((s.gc_any24 == 1).sum())
            n0 = int((s.gc_any24 == 0).sum())
            p0 = s.loc[s.gc_any24 == 0, y].mean() if n0 else np.nan
            m = mde(n1, n0, p0) if (n0 and not np.isnan(p0)) else None
            P("  %-24s %-10s %-6s %8d %8d %9.3f %12s" % (
                lab, DBCOL[k], g, n1, n0, p0 if n0 else np.nan,
                ("%.2f" % m) if m else ("no events in reference group" if n0 else "-")))
            mde_rows.append(dict(outcome=lab, db=k, disease=g, n_exposed=n1,
                                 n_unexposed=n0,
                                 risk_unexposed=(float(p0) if n0 else None),
                                 mde_or=(float(m) if m else None)))

pd.DataFrame(pc_rows).to_csv(os.path.join(OUT, "table_ext_h2h_positive_control.csv"), index=False)
pd.DataFrame(int_rows).to_csv(os.path.join(OUT, "table_ext_h2h_interaction.csv"), index=False)
pd.DataFrame(mde_rows).to_csv(os.path.join(OUT, "table_ext_h2h_mde.csv"), index=False)

with open(os.path.join(OUT, "162_external_head2head.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nsaved -> out/162_external_head2head.txt")
