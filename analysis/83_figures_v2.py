# -*- coding: utf-8 -*-
"""
Figures for the positive-control repair (report v2). Light theme.

Cohort convention (one cohort throughout, so the control panel and the hypothesis
tests are directly comparable):
  PRIMARY  = full landmark cohort (ICU LOS >= 24 h), n = 433
             prevalent infection at ICU admission is adjusted for, not excluded
  (the restricted cohort, n = 352, is a sensitivity analysis)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, OUT = os.path.join(BASE, "data"), os.path.join(BASE, "out")
FIG = os.path.join(OUT, "fig")
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(20260916)

ORDER = ["G0_none", "G1_low", "G2_mod", "G3_high"]
LABEL = {"G0_none": "0 (none)", "G1_low": ">0-<10", "G2_mod": "10-<50", "G3_high": ">=50"}
SHORT = {"G1_low": ">0-<10", "G2_mod": "10-<50", "G3_high": ">=50"}
COL = {"G1_low": "#1D9E75", "G2_mod": "#D85A30", "G3_high": "#7F77DD"}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.labelcolor": "#1a1a1a",
    "text.color": "#1a1a1a", "xtick.color": "#333333", "ytick.color": "#333333",
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})

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

lm = df[df.los24 == 1].copy()
lm["gc_str"] = pd.Categorical(lm["gc24_str"], categories=ORDER)
lm["any_cx_after24"] = (lm.ncx_after24 > 0).astype(int)
ADJ = "age + female + charlson + sofa + on_vent + on_vaso + on_rrt"
ADJ_INF = ADJ + " + abx_before_icu"


def orrow(m, term):
    b, se, p = m.params[term], m.bse[term], m.pvalues[term]
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), p


def iptw(d, covs, C=0.3, trim=0.05):
    X = d[covs].astype(float).values
    X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    T = pd.Categorical(d["gc_str"], categories=ORDER).codes
    ps = LogisticRegression(solver="lbfgs", max_iter=8000, C=C).fit(X, T)
    Pm = ps.predict_proba(X)
    marg = np.array([(T == g).mean() for g in range(len(ORDER))])
    w = marg[T] / np.clip(Pm[np.arange(len(T)), T], 1e-6, 1)
    lo_, hi_ = np.quantile(w, [trim, 1 - trim])
    d = d.copy(); d["sw"] = np.clip(w, lo_, hi_)
    return d, d["sw"].sum() ** 2 / (d["sw"] ** 2).sum()


COV = ["age", "female", "charlson", "sofa", "apsiii", "on_vent", "on_vaso", "on_rrt",
       "emerg", "wbc_min", "platelets_min", "creatinine_max"]


# ===================================================================== Figure 5
def fig_controls():
    spec = [
        ("GC \u2192 hyperglycaemia", "[positive control]", "hyper_post", "gc24_any", "pos"),
        ("GC \u2192 insulin, first 48 h", "[positive control]", "insulin_after_icu", "gc24_any", "pos"),
        ("Ondansetron \u2192 hyperglycaemia", "[negative exp.]", "hyper_post", "ondan24", "neg"),
        ("Ondansetron \u2192 treated infection", "[negative exp.]", "abx_new_after48", "ondan24", "neg"),
        ("GC \u2192 treated infection >48 h", "[hypothesis]", "abx_new_after48", "gc24_any", "hyp"),
        ("GC \u2192 bacteraemia >24 h", "[hypothesis]", "culture_after24", "gc24_any", "hyp"),
        ("GC \u2192 GI bleeding", "[negative outcome]", "gi_bleed", "gc24_any", "negout"),
    ]
    rows = []
    for lab, tag, out, exp, kind in spec:
        d = lm[lm[out].notna()]
        m = smf.glm("%s ~ %s + %s" % (out, exp, ADJ), data=d, family=sm.families.Binomial()).fit()
        o, lo, hi, p = orrow(m, exp)
        rows.append((lab, tag, o, lo, hi, p, kind))

    fig, ax = plt.subplots(figsize=(10.0, 4.3))
    plt.subplots_adjust(left=0.29, right=0.72, top=0.87, bottom=0.17)
    y = 0
    for lab, tag, o, lo, hi, p, kind in rows:
        y -= 1
        col = {"pos": "#D85A30", "neg": "#8A8A8A", "hyp": "#2B6CB0", "negout": "#8A8A8A"}[kind]
        ax.plot([lo, hi], [y, y], color=col, lw=1.7, zorder=2, solid_capstyle="round")
        ax.plot(o, y, "o", color=col, ms=6.4, zorder=3,
                markerfacecolor="white" if p >= 0.05 else col, markeredgewidth=1.7)
        ax.text(1.04, y, "%.2f (%.2f\u2013%.2f)" % (o, lo, hi),
                transform=ax.get_yaxis_transform(), fontsize=8, va="center", ha="left")
        ax.text(1.42, y, "P=%.3f" % p, transform=ax.get_yaxis_transform(), fontsize=8,
                va="center", ha="left", color="#1a1a1a" if p < 0.05 else "#a8a8a8")
    ax.set_xscale("log")
    ax.set_xlim(0.3, 13)
    ax.set_xticks([0.5, 1, 2, 4, 8])
    ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
    ax.axvline(1.0, color="#333", lw=0.9, ls="--", zorder=1)
    ax.set_yticks([-i for i in range(1, len(rows) + 1)])
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.6)
    ax.set_ylim(-len(rows) - 0.55, -0.45)
    for i, r in enumerate(rows):
        ax.text(1.015, -i - 1 - 0.30, r[1], transform=ax.get_yaxis_transform(),
                fontsize=6.9, va="center", ha="left", color="#8a8a8a", style="italic")
    ax.set_xlabel("Adjusted odds ratio (95% CI)", fontsize=9)
    ax.set_title("Control panel: the positive control now behaves as it should", fontsize=10.5)
    fig.text(0.06, 0.035,
             "filled marker = P<0.05;  open marker = P\u22650.05;  landmark cohort, n=433;"
             "  models adjusted for age, sex, Charlson, SOFA, ventilation, vasopressors, RRT",
             fontsize=7.2, ha="left", color="#666")
    for ext in ["png", "pdf"]:
        plt.savefig(os.path.join(FIG, "Figure5_control_panel.%s" % ext), dpi=300, facecolor="white")
    plt.close()
    return rows


# ===================================================================== Figure 6
def fig_dose():
    lm2, ess = iptw(lm, COV)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))
    plt.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.30, wspace=0.45)

    ax = axes[0]
    rates, ns = [], []
    for g in ORDER:
        s = lm[lm.gc_str == g]
        rates.append(100 * s.culture_after24.mean()); ns.append(len(s))
    ax.bar(range(4), rates, color=["#B0B0B0"] + [COL[g] for g in ["G1_low", "G2_mod", "G3_high"]],
           width=0.62, edgecolor="#333", lw=0.6)
    for i, (r, g) in enumerate(zip(rates, ORDER)):
        e = int(lm[lm.gc_str == g].culture_after24.sum())
        ax.text(i, r + 0.6, "%.1f%%\n%d/%d" % (r, e, ns[i]), ha="center", fontsize=7.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels([LABEL[g] for g in ORDER], fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Bacteraemia after 24 h (%)", fontsize=8.6)
    ax.set_xlabel("Glucocorticoid dose (mg/day)", fontsize=8.6)
    ax.set_title("Crude rates", fontsize=9.6)
    ax.set_ylim(0, 21)

    ax = axes[1]
    outs = [("culture_after24", "Bacteraemia", ADJ_INF),
            ("abx_new_after48", "Treated infection", ADJ_INF),
            ("death_30d", "30-day death", ADJ)]
    yy = 0
    for out, lab, adj in outs:
        yy -= 1
        m = smf.glm("%s ~ C(gc_str, Treatment(reference='G0_none')) + %s" % (out, adj),
                    data=lm2, family=sm.families.Binomial(), freq_weights=lm2["sw"]).fit()
        ax.axhspan(yy - 0.5, yy + 0.5, color="#F4F4F2", zorder=0)
        for k, g in enumerate(["G1_low", "G2_mod", "G3_high"]):
            t = "C(gc_str, Treatment(reference='G0_none'))[T.%s]" % g
            if t not in m.params.index:
                continue
            o, lo, hi, p = orrow(m, t)
            yv = yy + (0.27 - 0.27 * k)
            ax.plot([lo, hi], [yv, yv], color=COL[g], lw=1.4, zorder=2)
            ax.plot(o, yv, "s", color=COL[g], ms=5.0, zorder=3,
                    markerfacecolor="white" if p >= 0.05 else COL[g], markeredgewidth=1.5)
    for g in ["G1_low", "G2_mod", "G3_high"]:
        ax.plot([], [], "-s", color=COL[g], label=SHORT[g] + " mg/day")
    ax.axvline(1.0, color="#333", lw=0.9, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.015, 30)
    ax.set_xticks([0.03, 0.1, 0.3, 1, 3, 10, 30])
    ax.set_xticklabels(["0.03", "0.1", "0.3", "1", "3", "10", "30"], fontsize=8)
    ax.set_yticks([-1, -2, -3])
    ax.set_yticklabels(["Bacteraemia", "Treated infection", "30-day death"], fontsize=8.6)
    ax.set_ylim(-3.5, -0.5)
    ax.set_xlabel("IPTW-adjusted odds ratio vs no glucocorticoid (95% CI)", fontsize=8.6)
    ax.set_title("IPTW-adjusted estimates  (ESS %.0f / %d)" % (ess, len(lm2)), fontsize=9.6)
    ax.legend(fontsize=7.6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3)
    for ext in ["png", "pdf"]:
        plt.savefig(os.path.join(FIG, "Figure6_dose_response_v2.%s" % ext), dpi=300, facecolor="white")
    plt.close()


# ===================================================================== Figure 7
def fig_surveillance():
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    plt.subplots_adjust(left=0.13, right=0.83, top=0.87, bottom=0.19)
    x = np.arange(4)
    drawn, posrate = [], []
    for g in ORDER:
        s = lm[lm.gc_str == g]
        drawn.append(s.ncx_after24.mean())
        sub = s[s.any_cx_after24 == 1]
        posrate.append(100 * sub.culture_after24.mean() if len(sub) else np.nan)
    ax.bar(x - 0.19, drawn, width=0.36, color="#9BB7D4", edgecolor="#333", lw=0.6,
           label="blood cultures drawn per patient")
    ax.set_ylabel("Mean blood cultures drawn", color="#2B6CB0", fontsize=8.6)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[g] for g in ORDER], fontsize=8.2)
    ax.set_xlabel("Glucocorticoid dose (mg/day prednisone-equivalent)", fontsize=8.8)
    ax2 = ax.twinx()
    ax2.bar(x + 0.19, posrate, width=0.36, color="#D85A30", edgecolor="#333", lw=0.6,
            label="% positive among those cultured")
    ax2.set_ylabel("% of cultured patients with a positive culture", color="#D85A30", fontsize=8.6)
    ax2.set_ylim(0, 34)
    for i, (d_, p_) in enumerate(zip(drawn, posrate)):
        ax.text(i - 0.19, d_ + 0.15, "%.1f" % d_, ha="center", fontsize=7.4, color="#2B6CB0")
        ax2.text(i + 0.19, p_ + 0.9, "%.1f%%" % p_, ha="center", fontsize=7.4, color="#D85A30")
    ax.set_title("Surveillance check: more cultures do not explain the signal")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=False, loc="upper left")
    ax2.spines["top"].set_visible(False)
    for ext in ["png", "pdf"]:
        plt.savefig(os.path.join(FIG, "Figure7_surveillance.%s" % ext), dpi=300, facecolor="white")
    plt.close()


# ===================================================================== Figure 8
def fig_fragility():
    res = []
    n = len(lm)
    for _ in range(4000):
        i = rng.integers(0, n, n)
        db = lm.iloc[i]
        if db.culture_after24.sum() < 5:
            continue
        try:
            mb = smf.glm("culture_after24 ~ C(gc_str, Treatment(reference='G0_none')) + " + ADJ_INF,
                         data=db, family=sm.families.Binomial()).fit()
            t = "C(gc_str, Treatment(reference='G0_none'))[T.G2_mod]"
            if t in mb.params.index:
                res.append(mb.params[t])
        except Exception:
            pass
    res = np.exp(np.array(res))
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6))
    plt.subplots_adjust(left=0.09, right=0.97, top=0.85, bottom=0.20, wspace=0.34)

    ax = axes[0]
    # a handful of bootstrap replicates reach very large ORs and stretch the axis;
    # clip the display range and say so
    XHI = 40.0
    lo_b = max(np.percentile(res, 0.2), 0.6)
    bins = np.logspace(np.log10(lo_b), np.log10(XHI), 48)
    ax.hist(np.clip(res, lo_b, XHI), bins=bins, color="#9BB7D4",
            edgecolor="#4a6f92", lw=0.35)
    ax.axvline(1.0, color="#333", ls="--", lw=1.0)
    ax.axvline(np.median(res), color="#D85A30", lw=1.8)
    ax.set_xscale("log")
    ax.set_xlim(lo_b, XHI)
    ax.set_xticks([1, 3, 10, 30])
    ax.set_xticklabels(["1", "3", "10", "30"], fontsize=8)
    ax.set_xticks([], minor=True)
    ax.text(0.98, 0.62, "%.1f%% of replicates\nabove 40 (axis clipped)" % (
        100 * (res > XHI).mean()), transform=ax.transAxes, fontsize=7.2,
        ha="right", va="top", color="#8a8a8a")
    ax.set_xlabel("Bootstrap odds ratio, 10\u2013<50 mg/day vs none", fontsize=8.5)
    ax.set_ylabel("Bootstrap replicates", fontsize=8.5)
    ax.set_title("Bootstrap distribution (%d replicates)" % len(res), fontsize=9.4)
    ax.text(0.03, 0.96, "%.0f%% gave OR>1" % (100 * (res > 1).mean()),
            transform=ax.transAxes, fontsize=8.2, va="top", color="#D85A30")

    ax = axes[1]
    a0 = int(lm[(lm.gc_str == "G2_mod")].culture_after24.sum())
    b0 = int((lm.gc_str == "G2_mod").sum()) - a0
    c0 = int(lm[(lm.gc_str == "G0_none")].culture_after24.sum())
    d0 = int((lm.gc_str == "G0_none").sum()) - c0
    ks, ps = [], []
    for k in range(0, a0 + 1):
        tab = [[a0 - k, b0 + k], [c0, d0]]
        ks.append(k); ps.append(stats.fisher_exact(tab)[1])
    ax.plot(ks, ps, "-o", color="#D85A30", ms=5, lw=1.5)
    ax.axhline(0.05, color="#333", ls="--", lw=1.0)
    ax.set_xlabel("Events in the 10\u2013<50 group reclassified as negative", fontsize=8.5)
    ax.set_ylabel("Fisher exact P", fontsize=8.5)
    ax.set_title("Fragility of the bacteraemia result", fontsize=9.4)
    ax.set_ylim(0, max(ps) * 1.18)
    ax.set_xlim(-0.4, a0 + 0.4)
    ax.text(0.05, 0.96, "%d/%d vs %d/%d positive" % (a0, a0 + b0, c0, c0 + d0),
            transform=ax.transAxes, fontsize=8.2, va="top")
    for ext in ["png", "pdf"]:
        plt.savefig(os.path.join(FIG, "Figure8_fragility.%s" % ext), dpi=300, facecolor="white")
    plt.close()


if __name__ == "__main__":
    rows = fig_controls()
    for r in rows:
        print("%-42s %-20s OR %.2f (%.2f-%.2f) P=%.3f" % (r[0], r[1], r[2], r[3], r[4], r[5]))
    print("\n--- cohort checks ---")
    lmr = df[(df.los24 == 1) & (df.abx_before_icu == 0) & (df.culture_pre_icu == 0)].copy()
    lmr["gc_str"] = pd.Categorical(lmr["gc24_str"], categories=ORDER)
    for lab, d in [("primary n=%d" % len(lm), lm), ("restricted n=%d" % len(lmr), lmr)]:
        m = smf.glm("culture_after24 ~ C(gc_str, Treatment(reference='G0_none')) + " + ADJ_INF,
                    data=d, family=sm.families.Binomial()).fit()
        t = "C(gc_str, Treatment(reference='G0_none'))[T.G2_mod]"
        o, lo, hi, p = orrow(m, t)
        print("  %-22s bacteraemia 10-<50 OR %.2f (%.2f-%.2f) P=%.3f  events=%d" % (
            lab, o, lo, hi, p, int(d.culture_after24.sum())))
    fig_dose(); fig_surveillance(); fig_fragility()
    print("\nfigures written to", FIG)
