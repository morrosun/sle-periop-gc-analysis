# -*- coding: utf-8 -*-
"""
External validation of the MIMIC-IV glucocorticoid findings in eICU-CRD and NWICU.

What this script answers
------------------------
  Q1  Can the PRIMARY finding (GC dose -> bloodstream infection after the 24 h
      landmark) be tested outside MIMIC-IV?        [spoiler: no -- 0 events]
  Q2  Does the EXPOSURE VARIABLE transport? Tested with the positive control
      (GC -> hyperglycaemia), which is feasible in all three databases.
  Q3  On the outcomes that do have events everywhere, do the estimates agree?
      Reported per database and then combined with random-effects meta-analysis.
  Q4  How large an effect could each external database actually have detected?

Design choices fixed in advance
-------------------------------
  * crude ORs are primary (covariate sets are not fully transportable and the
    external event counts are far too small for multivariable adjustment);
    age+sex adjustment is reported as a sensitivity analysis
  * per-database estimates are reported separately; pooling is random-effects
    (DerSimonian-Laird) on the log scale
  * an outcome is declared NOT ESTIMABLE in a database when it has 0 events
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import os
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

P = print
DBS = [("mimiciv", "MIMIC-IV"), ("eicu", "eICU-CRD"), ("nwicu", "NWICU")]


# ------------------------------------------------------------------ helpers
def load():
    d = {}
    for k, _ in DBS:
        p = os.path.join(DATA, "cohort_%s.csv" % ("mimic_harmonized" if k == "mimiciv" else k))
        x = pd.read_csv(p)
        x["db"] = k
        d[k] = x
    return d


def fit_logit(df, y, xvars):
    """Logistic regression; returns (or, lo, hi, p, n, n_ev, method)."""
    use = df[[y] + xvars].dropna()
    if len(use) == 0 or use[y].nunique() < 2:
        return None
    X = sm.add_constant(use[xvars].astype(float))
    try:
        m = sm.Logit(use[y].astype(float), X).fit(disp=0)
        b, se = float(m.params.iloc[1]), float(m.bse.iloc[1])
        method = "ML"
    except Exception:
        try:
            m = sm.Logit(use[y].astype(float), X).fit_regularized(alpha=1e-2, L1_wt=0.0, disp=0)
            b = float(np.asarray(m.params).ravel()[1])
            se = np.nan
            method = "ridge"
        except Exception:
            return None
    if np.isnan(se) or se == 0:
        return None
    # |beta| > 12 means the fit ran away (complete separation) -> not interpretable
    if not np.isfinite(b) or abs(b) > 12:
        return None
    orr = float(np.exp(b))
    lo, hi = float(np.exp(b - 1.96 * se)), float(np.exp(b + 1.96 * se))
    z = b / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return dict(OR=orr, lo=lo, hi=hi, p=float(p), n=len(use),
                n_ev=int(use[y].sum()), method=method, beta=b, se=se)


def two_by_two(df, y):
    use = df[["gc_any24", y]].dropna()
    a = int(((use.gc_any24 == 1) & (use[y] == 1)).sum())
    b = int(((use.gc_any24 == 1) & (use[y] == 0)).sum())
    c = int(((use.gc_any24 == 0) & (use[y] == 1)).sum())
    dd = int(((use.gc_any24 == 0) & (use[y] == 0)).sum())
    return a, b, c, dd


def fisher_or(df, y):
    """Unconditional OR with a Haldane-Anscombe continuity correction whenever
    ANY cell of the 2x2 table is empty, plus Fisher's exact two-sided P."""
    a0, b0, c0, d0 = two_by_two(df, y)
    if (a0 + b0) == 0 or (c0 + d0) == 0 or (a0 + c0) == 0:
        return None
    p = float(stats.fisher_exact([[a0, c0], [b0, d0]])[1])
    a, b, c, dd = [float(v) for v in (a0, b0, c0, d0)]
    if min(a, b, c, dd) == 0:              # ANY empty cell, not just b*c
        a, b, c, dd = a + .5, b + .5, c + .5, dd + .5
    logor = float(np.log((a * dd) / (b * c)))
    se = float(np.sqrt(1 / a + 1 / b + 1 / c + 1 / dd))
    return dict(OR=float(np.exp(logor)),
                lo=float(np.exp(logor - 1.96 * se)),
                hi=float(np.exp(logor + 1.96 * se)),
                p=p, n=int(a0 + b0 + c0 + d0), n_ev=int(a0 + c0),
                method="Fisher+HA", beta=logor, se=se,
                table=(int(a0), int(b0), int(c0), int(d0)))


def dose_trend(df, y):
    """Wald test for linear trend across the four dose strata."""
    use = df[["gc24_str", y]].dropna().copy()
    if use[y].nunique() < 2 or len(use) == 0:
        return None
    score = {"G0_none": 0.0, "G1_low": 1.0, "G2_mod": 2.0, "G3_high": 3.0}
    use["s"] = use.gc24_str.map(score)
    X = sm.add_constant(use[["s"]].astype(float))
    try:
        m = sm.Logit(use[y].astype(float), X).fit(disp=0)
        b, se = m.params.iloc[1], m.bse.iloc[1]
        b, se = float(b), float(se)
        z = b / se
        return dict(OR=float(np.exp(b)), lo=float(np.exp(b - 1.96 * se)),
                    hi=float(np.exp(b + 1.96 * se)),
                    p=float(2 * (1 - stats.norm.cdf(abs(z)))), n=len(use),
                    n_ev=int(use[y].sum()), beta=b, se=se)
    except Exception:
        return None


def dl_meta(ests):
    """DerSimonian-Laird random-effects pooling on the log-OR scale."""
    ests = [e for e in ests if e is not None and np.isfinite(e["beta"]) and e["se"] > 0]
    if not ests:
        return None
    b = np.array([e["beta"] for e in ests])
    s = np.array([e["se"] for e in ests])
    w = 1.0 / s ** 2
    mu_f = float(np.sum(w * b) / np.sum(w))
    Q = float(np.sum(w * (b - mu_f) ** 2))
    k = len(ests)
    dfq = max(k - 1, 1)
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - dfq) / C) if C > 0 else 0.0
    w2 = 1.0 / (s ** 2 + tau2)
    mu = float(np.sum(w2 * b) / np.sum(w2))
    se = float(np.sqrt(1.0 / np.sum(w2)))
    z = mu / se
    return dict(OR=float(np.exp(mu)), lo=float(np.exp(mu - 1.96 * se)),
                hi=float(np.exp(mu + 1.96 * se)),
                p=float(2 * (1 - stats.norm.cdf(abs(z)))),
                tau2=float(tau2), Q=Q, df=dfq,
                pQ=float(1 - stats.chi2.cdf(Q, dfq)), k=k,
                I2=float(max(0.0, (Q - dfq) / Q) * 100) if Q > 0 else 0.0)


def mde(n1, n0, p0, alpha=0.05, power=0.80):
    """Smallest OR detectable at 80% power, given n exposed / n unexposed and
    the event risk p0 in the unexposed."""
    if min(n1, n0) < 2 or p0 <= 0 or p0 >= 1:
        return None
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)

    def pw(orv):
        p1 = orv * p0 / (1 - p0 + orv * p0)
        pb = (n1 * p1 + n0 * p0) / (n1 + n0)
        se = np.sqrt(pb * (1 - pb) * (1 / n1 + 1 / n0))
        d = abs(p1 - p0)
        return stats.norm.cdf(d / se - z_a)
    lo, hi = 1.0, 200.0
    if pw(hi) < power:
        return None
    for _ in range(80):
        mid = (lo + hi) / 2
        if pw(mid) < power:
            lo = mid
        else:
            hi = mid
    return float(hi)


# ------------------------------------------------------------------ main
def main():
    d = load()
    res = {}

    # ---------------------------------------------------------------- 0. feasibility
    P("=" * 78)
    P("0. FEASIBILITY OF EXTERNAL VALIDATION PER OUTCOME")
    P("=" * 78)
    feas = []
    outcomes = [("culture_pos_after24", "Blood culture positive >24 h (timed)"),
                ("abx_new_after48", "New antibiotic >48 h (timed)"),
                ("infect_icd", "Infection diagnosis code (UNTIMED)"),
                ("hyper_48h", "Glucose >=180 mg/dL <48 h (POSITIVE CONTROL)"),
                ("gi_bleed", "GI bleeding code (NEGATIVE OUTCOME)"),
                ("death_hosp", "Hospital death")]
    for var, lab in outcomes:
        row = dict(outcome=lab, var=var)
        for k, _ in DBS:
            x = d[k]
            if var not in x.columns:
                row[k] = "N/A"
                continue
            if var in ("culture_pos_after24",) and k == "nwicu":
                row[k] = "no micro table"
                continue
            n_ev = int(x[var].sum(skipna=True))
            n_av = int(x[var].notna().sum())
            row[k] = "%d / %d" % (n_ev, n_av)
        feas.append(row)
    ft = pd.DataFrame(feas)
    P(ft.to_string(index=False))
    ft.to_csv(os.path.join(OUT, "table_external_feasibility.csv"), index=False)

    # ---------------------------------------------------------------- 1. baseline
    P("\n" + "=" * 78)
    P("1. HARMONISED COHORTS")
    P("=" * 78)
    rows = []
    for k, lab in DBS:
        x = d[k]
        rows.append(dict(db=lab, n=len(x),
                         gc_any="%d (%.1f%%)" % (x.gc_any24.sum(), 100 * x.gc_any24.mean()),
                         age_med=round(float(x.age.median()), 1),
                         female="%.1f%%" % (100 * x.female.mean()),
                         vaso="%.1f%%" % (100 * x.vaso24.mean()),
                         vent="%.1f%%" % (100 * x.vent24.mean()),
                         renal="%.1f%%" % (100 * x.renal_fail.mean()),
                         los_med=round(float(x.icu_los_days.median()), 2)))
    bt = pd.DataFrame(rows)
    P(bt.to_string(index=False))
    bt.to_csv(os.path.join(OUT, "table_external_baseline.csv"), index=False)

    # ---------------------------------------------------------------- 2. positive control
    P("\n" + "=" * 78)
    P("2. POSITIVE CONTROL  GC -> hyperglycaemia (>=180 mg/dL within 48 h)")
    P("=" * 78)
    pc_rows, pc_ests, tr_ests = [], [], []
    for k, lab in DBS:
        x = d[k]
        r = fit_logit(x, "hyper_48h", ["gc_any24"])
        tr = dose_trend(x, "hyper_48h")
        if tr:
            t2 = dict(tr); t2["db"] = lab
            tr_ests.append(t2)
        n1 = int(x.gc_any24.sum())
        pc_rows.append(dict(db=lab, n=len(x), n_exposed=n1,
                            pct_hyper_none="%.1f%%" % (100 * x.loc[x.gc_any24 == 0, "hyper_48h"].mean(skipna=True)),
                            pct_hyper_gc="%.1f%%" % (100 * x.loc[x.gc_any24 == 1, "hyper_48h"].mean(skipna=True)),
                            OR_crude=("%.2f (%.2f-%.2f)" % (r["OR"], r["lo"], r["hi"])) if r else "n.e.",
                            p=("%.3f" % r["p"]) if r else "n.e.",
                            OR_trend_per_stratum=("%.2f (%.2f-%.2f)" % (tr["OR"], tr["lo"], tr["hi"])) if tr else "n.e.",
                            p_trend=("%.4f" % tr["p"]) if tr else "n.e."))
        if r:
            r2 = dict(r); r2["db"] = lab
            pc_ests.append(r2)
    pct = pd.DataFrame(pc_rows)
    P(pct.to_string(index=False))
    m = dl_meta(pc_ests)
    if m:
        P("\n  random-effects pooled OR (any GC -> hyperglycaemia): %.2f (%.2f-%.2f), P=%.4g" % (
            m["OR"], m["lo"], m["hi"], m["p"]))
        P("  heterogeneity: tau2=%.3f  Q=%.2f (df=%d)  P=%.3f  I2=%.1f%%" % (
            m["tau2"], m["Q"], m["df"], m["pQ"], m["I2"]))
    pct.to_csv(os.path.join(OUT, "table_positive_control_external.csv"), index=False)
    res["positive_control"] = dict(rows=pc_rows, meta=m)

    # per-stratum estimates of the positive control
    mt = dl_meta(tr_ests)
    if mt:
        P("\n  pooled DOSE-TREND per stratum increment (random effects, k=%d): "
          "OR %.2f (%.2f-%.2f), P=%.4g" % (mt["k"], mt["OR"], mt["lo"], mt["hi"], mt["p"]))
        P("  heterogeneity: tau2=%.3f  Q=%.2f (df=%d)  P=%.3f  I2=%.1f%%" % (
            mt["tau2"], mt["Q"], mt["df"], mt["pQ"], mt["I2"]))
    res["pc_trend_meta"] = mt

    P("\n  per-dose-stratum OR (ref = no GC):")
    for k, lab in DBS:
        x = d[k]
        parts = []
        for g in ["G1_low", "G2_mod", "G3_high"]:
            sub = x[x.gc24_str.isin(["G0_none", g])].copy()
            sub["e"] = (sub.gc24_str == g).astype(int)
            r = fit_logit(sub, "hyper_48h", ["e"])
            ne = int(sub.loc[sub.e == 1, "hyper_48h"].sum())
            nn = int(sub.e.sum())
            parts.append("%s n=%d/%d ev %s" % (g, nn, ne,
                        ("%.2f (%.2f-%.2f)" % (r["OR"], r["lo"], r["hi"])) if r else "n.e."))
        P("    %-10s %s" % (lab, " | ".join(parts)))

    # ---------------------------------------------------------------- 3. infection outcomes
    P("\n" + "=" * 78)
    P("3. INFECTION OUTCOMES  (any GC vs none, crude)")
    P("=" * 78)
    oc_rows = {}
    for var, lab in [("culture_pos_after24", "Blood culture positive >24 h"),
                     ("abx_new_after48", "New antibiotic >48 h"),
                     ("infect_icd", "Infection diagnosis code"),
                     ("gi_bleed", "GI bleeding (negative outcome)"),
                     ("death_hosp", "Hospital death")]:
        ests = []
        P("\n  --- %s ---" % lab)
        for k, dlab in DBS:
            x = d[k]
            if var not in x.columns or x[var].notna().sum() == 0:
                P("    %-10s not estimable (outcome unavailable)" % dlab)
                continue
            r = fisher_or(x, var)
            if r is None:
                P("    %-10s not estimable (0 events in a margin)" % dlab)
                continue
            r2 = dict(r); r2["db"] = dlab
            ests.append(r2)
            a, b, c, dd = r["table"]
            P("    %-10s 2x2 [exp %d/%d vs unexp %d/%d]  OR %.2f (%.2f-%.2f)  P=%.3f" % (
                dlab, a, a + b, c, c + dd, r["OR"], r["lo"], r["hi"], r["p"]))
        mm = dl_meta(ests)
        if mm:
            P("    POOLED (random effects, k=%d): OR %.2f (%.2f-%.2f)  P=%.4g  I2=%.1f%%  P_het=%.3f" % (
                mm["k"], mm["OR"], mm["lo"], mm["hi"], mm["p"], mm["I2"], mm["pQ"]))
        oc_rows[var] = dict(lab=lab, ests=[{kk: vv for kk, vv in e.items()} for e in ests],
                            meta=mm)

    # ---------------------------------------------------------------- 4. sensitivity: age/sex adjusted
    P("\n" + "=" * 78)
    P("4. SENSITIVITY: age + sex adjusted (events permitting)")
    P("=" * 78)
    for var in ["hyper_48h", "infect_icd", "abx_new_after48", "culture_pos_after24"]:
        P("\n  --- %s ---" % var)
        for k, dlab in DBS:
            x = d[k]
            x = x.copy()
            x["age10"] = (x.age - x.age.median()) / 10.0
            if var not in x.columns or x[var].notna().sum() == 0:
                P("    %-10s n.e." % dlab)
                continue
            n_ev = int(x[var].sum())
            if n_ev < 10:
                P("    %-10s not fitted (%d events, EPV too low)" % (dlab, n_ev))
                continue
            r = fit_logit(x, var, ["gc_any24", "age10", "female"])
            if r is None:
                P("    %-10s not fitted (separation)" % dlab)
                continue
            P("    %-10s OR %.2f (%.2f-%.2f)  P=%.3f  [n=%d, %d events]" % (
                dlab, r["OR"], r["lo"], r["hi"], r["p"], r["n"], r["n_ev"]))

    # ---------------------------------------------------------------- 5. dose strata in MIMIC (reference signal)
    P("\n" + "=" * 78)
    P("5. REFERENCE: MIMIC-IV dose strata for the primary outcome")
    P("=" * 78)
    x = d["mimiciv"]
    for g in ["G1_low", "G2_mod", "G3_high"]:
        sub = x[x.gc24_str.isin(["G0_none", g])].copy()
        sub["e"] = (sub.gc24_str == g).astype(int)
        r = fit_logit(sub, "culture_pos_after24", ["e"])
        P("  %-8s vs none: %s   (events %d/%d vs %d/%d)" % (
            g, ("%.2f (%.2f-%.2f)" % (r["OR"], r["lo"], r["hi"])) if r else "n.e.",
            int(sub.loc[sub.e == 1, "culture_pos_after24"].sum()), int(sub.e.sum()),
            int(sub.loc[sub.e == 0, "culture_pos_after24"].sum()), int((sub.e == 0).sum())))
    tr = dose_trend(x, "culture_pos_after24")
    P("  trend per stratum increment: OR %.2f (%.2f-%.2f)  P=%.4f" % (
        tr["OR"], tr["lo"], tr["hi"], tr["p"]))

    # ---------------------------------------------------------------- 6. minimum detectable effect
    P("\n" + "=" * 78)
    P("6. WHAT COULD EACH DATABASE HAVE DETECTED? (80% power, alpha=0.05)")
    P("=" * 78)
    md_rows = []
    for var in ["culture_pos_after24", "abx_new_after48", "infect_icd"]:
        for k, dlab in DBS:
            x = d[k]
            if var not in x.columns or x[var].notna().sum() == 0:
                continue
            n1 = int(x.gc_any24.sum())
            n0 = int((x.gc_any24 == 0).sum())
            p0 = float(x.loc[x.gc_any24 == 0, var].mean(skipna=True))
            if not np.isfinite(p0) or p0 <= 0:
                md_rows.append(dict(outcome=var, db=dlab, n_exposed=n1, n_unexposed=n0,
                                    risk_unexposed="%.1f%%" % (100 * p0) if np.isfinite(p0) else "-",
                                    MDE_OR="no event in reference group"))
                continue
            v = mde(n1, n0, p0)
            md_rows.append(dict(outcome=var, db=dlab, n_exposed=n1, n_unexposed=n0,
                                risk_unexposed="%.1f%%" % (100 * p0),
                                MDE_OR=("%.2f" % v) if v else ">200"))
    mdt = pd.DataFrame(md_rows)
    P(mdt.to_string(index=False))
    mdt.to_csv(os.path.join(OUT, "table_mde_external.csv"), index=False)

    # ---------------------------------------------------------------- 7. exposure-recording density
    P("\n" + "=" * 78)
    P("7. HOW DENSELY IS THE EXPOSURE RECORDED? (steroid order rows per SLE stay)")
    P("=" * 78)
    dens = [("MIMIC-IV", 4166, 550), ("eICU-CRD", 195, 209), ("NWICU", 259, 49)]
    for lab, nrow, nstay in dens:
        P("  %-10s %5d rows / %4d stays = %.2f rows per stay" % (lab, nrow, nstay, nrow / nstay))

    # ---------------------------------------------------------------- 8. save
    mim_strata = []
    for g in ["G1_low", "G2_mod", "G3_high"]:
        sub = x[x.gc24_str.isin(["G0_none", g])].copy()
        sub["e"] = (sub.gc24_str == g).astype(int)
        r = fit_logit(sub, "culture_pos_after24", ["e"])
        mim_strata.append(dict(stratum=g,
                               OR=r["OR"] if r else None, lo=r["lo"] if r else None,
                               hi=r["hi"] if r else None, p=r["p"] if r else None,
                               n=int(sub.e.sum()),
                               ev=int(sub.loc[sub.e == 1, "culture_pos_after24"].sum())))
    ref_ev = int(x.loc[x.gc24_str == "G0_none", "culture_pos_after24"].sum())
    ref_n = int((x.gc24_str == "G0_none").sum())

    payload = dict(
        feasibility=feas, baseline=rows, positive_control=res["positive_control"],
        pc_trend_meta=res.get("pc_trend_meta"), outcomes=oc_rows,
        mde=md_rows, mimic_culture_strata=dict(ref_n=ref_n, ref_ev=ref_ev, strata=mim_strata),
        density=[dict(db=l, rows=r, stays=s, per_stay=round(r / s, 2)) for l, r, s in dens],
    )
    with open(os.path.join(OUT, "_external_validation_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    P("\n  saved -> out/_external_validation_results.json")


if __name__ == "__main__":
    main()
