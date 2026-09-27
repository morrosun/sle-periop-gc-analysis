# -*- coding: utf-8 -*-
"""Figures for the v5 report (primary outcome = new broad-spectrum antimicrobial).

  Figure 18  outcome-definition audit: why "any dose >=48 h" is not an incident event
  Figure 19  PRIMARY outcome dose-response within each disease, three models
  Figure 20  head-to-head interaction across the whole outcome hierarchy + BH
  Figure 21  robustness of the interaction (age / surgery / exposure / bootstrap)
  Figure 22  competing risk, and the control panel

Layout rules kept from 170_figures_v4 (they prevented every collision seen earlier):
  * explicit y = -row index, never a running counter
  * right-hand text goes OUTSIDE the axes with clip_on=False
  * log axes get explicit ticks + NullLocator/NullFormatter
  * if a label set can collide, use a legend
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator, NullFormatter
import matplotlib.ticker as mticker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
FIG = os.path.join(OUT, "fig")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": "#555555", "axes.linewidth": 0.8,
    "axes.labelcolor": "#222222", "text.color": "#222222",
    "xtick.color": "#333333", "ytick.color": "#333333",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})

C_SLE, C_RA = "#2B6CB0", "#D85A30"
C_PRIM, C_SEC, C_SENS, C_CTRL = "#B03A2E", "#2F7A4F", "#7F77DD", "#9AA0A6"
C_UW, C_IPTW = "#1F4E79", "#C99A2E"
GREY = "#9AA0A6"
FAMCOL = {"primary": C_PRIM, "secondary": C_SEC, "sensitivity": C_SENS, "control": C_CTRL}

JB = json.load(open(os.path.join(OUT, "_v5_results.json"), encoding="utf-8"))
DOSE = pd.DataFrame(JB["dose"]).rename(columns={"var": "ov"})
IX = pd.DataFrame(JB["interaction"]).rename(columns={"var": "ov"})
CTRL = pd.DataFrame(JB["controls"]).rename(columns={"var": "ov"})
NEG = pd.DataFrame(JB["negexp"]).rename(columns={"var": "ov"})
STRATA = pd.DataFrame(JB["strata"]).rename(columns={"var": "ov"})
PRIMARY = JB["prespecified_primary"]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, "%s.%s" % (name, ext)), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def plain_log(ax, ticks):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(["%g" % t for t in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())


def plain_logy(ax, ticks):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(mticker.FixedFormatter(["%g" % t for t in ticks]))
    ax.yaxis.set_minor_formatter(NullFormatter())


def forest(ax, rows, xlim, xticks, xlabel):
    """rows = list of dict(label, or, lo, hi, col, marker, y, bold)"""
    plain_log(ax, xticks)
    ax.axvline(1.0, color=GREY, lw=1.0, ls="--", zorder=1)
    for r in rows:
        y = r["y"]
        if r.get("bars"):
            ax.plot([r["lo"], r["hi"]], [y, y], color=r["col"], lw=1.6,
                    solid_capstyle="butt", zorder=2, alpha=0.9)
        ax.plot([r["or_"]], [y], marker=r.get("marker", "o"), ms=r.get("ms", 6),
                color=r["col"], mec="white", mew=0.8, zorder=3)
        if r.get("right"):
            ax.text(xlim[1] * 1.06, y, r["right"], va="center", ha="left",
                    fontsize=7.6, color="#333333", clip_on=False)
    ax.set_ylim(min(r["y"] for r in rows) - 0.9, max(r["y"] for r in rows) + 0.9)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)


def ors(r, key="or"):
    if r is None or not np.isfinite(r.get(key, np.nan)):
        return "not estimable"
    return "%.2f (%.2f-%.2f)" % (r[key], r[key.replace("or", "lo")], r[key.replace("or", "hi")])


# =====================================================================
# Figure 18 -- outcome-definition audit
# =====================================================================
coh = pd.read_csv(os.path.join(DATA, "cohort_rheum_icu.csv"), low_memory=False)
ab = pd.read_csv(os.path.join(DATA, "abx_outcomes_rheum.csv"))
coh["intime"] = pd.to_datetime(coh.intime)
coh["abx_first_time"] = pd.to_datetime(coh.abx_first_time)
coh["h_abx"] = (coh.abx_first_time - coh.intime).dt.total_seconds() / 3600.0
m = coh.merge(ab[["stay_key", "legacy_any48", "n_agent_pre24", "new_agent_24_7d",
                  "incident_strict", "culture_pos_24_7d"]], on="stay_key", how="left")
m["culture_pos_after24"] = m.culture_pos_after24.astype(float)
m["death_30d"] = m.death_30d.astype(float)
m["infect_icd"] = m.infect_icd.astype(float)
sle = m[m.primary_grp == "SLE"]
ra = m[m.primary_grp == "RA"]

DEFS = [("legacy_any48", "A  any dose >=48 h\n(v2 definition)"),
        ("new_agent_24_7d", "B  new agent, 24 h-7 d\n(PRIMARY)"),
        ("incident_strict", "C  B + antimicrobial-\nnaive at landmark"),
        ("culture_pos_24_7d", "D  blood culture +\n24 h-7 d (specificity)"),
        ("culture_pos_after24", "D2  blood culture +\n>24 h (v4 window)"),
        ("death_30d", "E  30-day mortality"),
        ("infect_icd", "F  infection ICD code\n(wide, untimed)")]

fig = plt.figure(figsize=(11.4, 3.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 1.0], wspace=0.32)

axA = fig.add_subplot(gs[0, 0])
ylab, ypos = [], []
for i, (c, lab) in enumerate(DEFS):
    y = -i
    ypos.append((y, c))
    ylab.append(lab)
    for j, (sub, col) in enumerate([(m, "#4A4A4A"), (sle, C_SLE), (ra, C_RA)]):
        v = 100 * sub[c].astype(float).mean()
        axA.barh(y + (1 - j) * 0.26, v, height=0.24, color=col,
                 alpha=1.0 if j == 0 else 0.85,
                 edgecolor="white", lw=0.5)
    axA.text(100 * m[c].astype(float).mean() + 2.0, y, "%d" % int(m[c].astype(float).sum()),
             va="center", ha="left", fontsize=7.2, color="#333333")
axA.set_yticks([y for y, _ in ypos])
axA.set_yticklabels(ylab, fontsize=7.4)
axA.set_xlabel("patients with the event (%)")
axA.set_xlim(0, 72)
axA.spines[["top", "right"]].set_visible(False)
axA.set_title("A  event counts of each candidate outcome definition", fontsize=8.4, loc="left")
axA.legend(handles=[Line2D([], [], color="#4A4A4A", lw=6, label="all rheumatic (n=%d)" % len(m)),
                    Line2D([], [], color=C_SLE, lw=6, label="SLE (n=%d)" % len(sle)),
                    Line2D([], [], color=C_RA, lw=6, label="RA (n=%d)" % len(ra))],
           fontsize=7.2, loc="lower right", frameon=False)

axB = fig.add_subplot(gs[0, 1])
h = m.h_abx.dropna().clip(-24, 168)
axB.hist(h, bins=np.arange(-24, 174, 6), color="#4A4A4A", alpha=0.85, edgecolor="white", lw=0.4)
axB.axvline(24, color=C_PRIM, lw=1.5, ls="--")
axB.text(24, axB.get_ylim()[1] * 0.97, " 24 h landmark", color=C_PRIM, fontsize=7.4, va="top")
axB.set_xlabel("first broad-spectrum antimicrobial\nrelative to ICU admission (h)")
axB.set_ylabel("patients")
axB.spines[["top", "right"]].set_visible(False)
axB.set_title("B  when the first antimicrobial starts\n    (among those ever treated)", fontsize=8.4, loc="left")
share24 = 100 * (m.h_abx < 24).mean()
axB.text(0.98, 0.62, "%.1f%% of treated patients\nstart before 24 h\n(median %.1f h)"
         % (share24, m.h_abx.median()), transform=axB.transAxes, ha="right", va="top",
         fontsize=7.4, color="#333333")

axC = fig.add_subplot(gs[0, 2])
ev = m[m.legacy_any48 == 1]
cont = 100 * (ev.n_agent_pre24 > 0).mean()
newp = 100 - cont
axC.barh([1], [cont], color="#C7C7C7", edgecolor="white", height=0.5)
axC.barh([1], [newp], left=[cont], color=C_PRIM, edgecolor="white", height=0.5)
axC.text(cont / 2, 1, "%.1f%%" % cont, ha="center", va="center", fontsize=8, color="#333333")
axC.text(cont + newp / 2, 1, "%.1f%%" % newp, ha="center", va="center", fontsize=8, color="white")
axC.set_ylim(0.4, 1.6)
axC.set_xlim(0, 100)
axC.set_yticks([])
axC.set_xlabel("share of the %d 'any dose >=48 h' events" % len(ev))
axC.spines[["top", "right", "left"]].set_visible(False)
axC.tick_params(axis="y", length=0)
axC.set_title("C  what definition A actually captures", fontsize=8.4, loc="left")
axC.legend(handles=[Line2D([], [], color="#C7C7C7", lw=8,
                           label="already on antimicrobials at the landmark"),
                    Line2D([], [], color=C_PRIM, lw=8,
                           label="antimicrobial-free at the landmark")],
           fontsize=7.0, loc="lower center", frameon=False, ncol=1)
save(fig, "Figure18_outcome_definition_audit")

# =====================================================================
# Figure 19 -- primary outcome dose-response, three models
# =====================================================================
fig = plt.figure(figsize=(11.4, 3.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.25], wspace=0.30)
for k, (g, col) in enumerate([("SLE", C_SLE), ("RA", C_RA)]):
    ax = fig.add_subplot(gs[0, k])
    d = DOSE[(DOSE.ov == PRIMARY) & (DOSE.disease == g)]
    st = STRATA[(STRATA.ov == PRIMARY) & (STRATA.disease == g)]
    xs = np.arange(len(st))
    ax.axhline(1.0, color=GREY, lw=1.0, ls="--")
    for i, (_, r) in enumerate(st.iterrows()):
        for off, kk, c_, mk in [(-0.17, "adj", C_UW, "o"), (0.17, "iptw", C_IPTW, "s")]:
            if np.isfinite(r.get(kk + "_or", np.nan)):
                ax.plot([i + off, i + off], [r[kk + "_lo"], r[kk + "_hi"]], color=c_, lw=1.6)
                ax.plot([i + off], [r[kk + "_or"]], marker=mk, ms=6, color=c_,
                        mec="white", mew=0.8)
                ax.text(i + off, r[kk + "_lo"] * 0.92, "%.2f" % r[kk + "_or"],
                        ha="center", va="top", fontsize=6.6, color=c_)
    plain_logy(ax, [0.1, 0.25, 0.5, 1, 2, 5, 10])
    ax.set_xticks(xs)
    ax.set_xticklabels(list(st.stratum), fontsize=7.2)
    ax.set_xlim(-0.6, len(st) - 0.4)
    ax.set_ylim(0.12, 9)
    ax.set_ylabel("OR vs no GC" if k == 0 else "")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("%s   (n=%d, events=%d)" % (g, int(d.n.iloc[0]), int(d.n_ev.iloc[0])),
                 fontsize=8.4, loc="left", color=col)
    ax.set_xlabel("dose stratum (mg/day prednisone equivalent)")
    tr = d.iloc[0]
    ax.text(0.02, 0.98, "dose trend (per stratum)\ncrude   %.2f (%.2f-%.2f)\nadjusted %.2f (%.2f-%.2f)\nIPTW     %.2f (%.2f-%.2f)"
            % (tr.crude_or, tr.crude_lo, tr.crude_hi, tr.adj_or, tr.adj_lo, tr.adj_hi,
               tr.iptw_or, tr.iptw_lo, tr.iptw_hi),
            transform=ax.transAxes, va="top", ha="left", fontsize=7.0, color="#333333")
    if k == 0:
        ax.legend(handles=[Line2D([], [], color=C_UW, marker="o", ls="-", label="adjusted"),
                           Line2D([], [], color=C_IPTW, marker="s", ls="-", label="IPTW")],
                  fontsize=7.2, loc="upper right", frameon=False)

axC = fig.add_subplot(gs[0, 2])
rows = []
ci_lab = []
for i, (_, r) in enumerate(DOSE[DOSE.ov == PRIMARY].iterrows()):
    y = -i
    rows.append(dict(y=y, or_=r.adj_or, lo=r.adj_lo, hi=r.adj_hi, col=C_UW, marker="o"))
    rows.append(dict(y=y - 0.30, or_=r.iptw_or, lo=r.iptw_lo, hi=r.iptw_hi, col=C_IPTW, marker="s"))
    ci_lab.append((y - 0.15, "%s   adj %.2f / IPTW %.2f" % (r.disease, r.adj_or, r.iptw_or)))
forest(axC, rows, (0.55, 1.9), [0.6, 0.8, 1.0, 1.3, 1.8], "dose-trend OR per stratum increment")
axC.spines["bottom"].set_visible(False)
axC.set_title("C  dose-trend OR per stratum increment\n    (both models, both diseases)",
              fontsize=8.4, loc="left")
for y, s in ci_lab:
    axC.text(0.60, y, s, fontsize=7.6, va="center", ha="left", color="#333333")
save(fig, "Figure19_primary_dose_response")

# =====================================================================
# Figure 20 -- interaction forest across the hierarchy, with BH
# =====================================================================
fig, ax = plt.subplots(figsize=(11.4, 0.42 * len(IX) + 1.9))
rows, labels = [], []
for i, (_, r) in enumerate(IX.iterrows()):
    y = -i
    col = FAMCOL[r.family]
    rows.append(dict(y=y + 0.16, or_=r.or_uw, lo=r.lo_uw, hi=r.hi_uw, col=col,
                     marker="o", ms=6.5 if r.family == "primary" else 5.5))
    if np.isfinite(r.or_iptw):
        rows.append(dict(y=y - 0.16, or_=r.or_iptw, lo=r.lo_iptw, hi=r.hi_iptw,
                         col=col, marker="s", ms=5.0, alpha=None))
    lab = "%s\n%s" % (r.outcome, r.role)
    labels.append((y, lab, col, r))
forest(ax, rows, (0.30, 3.2), [0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0],
       "RA / SLE ratio of dose-trend ORs   (per stratum increment)")
ax.spines["bottom"].set_visible(False)
ax.set_title("Figure 20  the head-to-head test across the whole outcome hierarchy\n"
             "circle = unweighted (MLE/Firth), square = IPTW;  "
             "red = pre-specified primary, green = secondary, purple = sensitivity, grey = control",
             fontsize=9.2, loc="left")
for y, lab, col, r in labels:
    ax.text(0.305, y, lab, fontsize=7.4, va="center", ha="left", color=col)
    q = r.q_full
    tag = ("q=%.3f" % q) if np.isfinite(q) else "q=NA"
    star = "  *" if (np.isfinite(r.p_lrt) and r.p_lrt < 0.05) else ""
    ax.text(3.35, y, "%s (BH, full family)%s" % (tag, star), fontsize=7.2, va="center",
            ha="left", clip_on=False,
            color="#B03A2E" if (np.isfinite(r.p_lrt) and r.p_lrt < 0.05) else "#666666")
ax.text(0.305, 0.6, "no outcome survives BH correction over the full family;"
                    " the pre-specified primary is null",
        fontsize=7.6, va="center", ha="left", color="#B03A2E", style="italic")
save(fig, "Figure20_interaction_forest_bh")

# =====================================================================
# Figure 21 -- robustness of the interaction
# =====================================================================
age = pd.read_csv(os.path.join(OUT, "table_v5_robust_age.csv"))
absb = pd.read_csv(os.path.join(OUT, "table_v5_robust_absorb.csv"))
surg = pd.read_csv(os.path.join(OUT, "table_v5_robust_surg.csv"))
expo = pd.read_csv(os.path.join(OUT, "table_v5_robust_expo.csv"))
boot = pd.read_csv(os.path.join(OUT, "table_v5_robust_boot.csv"))

fig = plt.figure(figsize=(11.4, 6.2))
gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.30)
TWO = ["new antimicrobial (PRIMARY)", "30-day death"]
PCOL = {"new antimicrobial (PRIMARY)": C_PRIM, "30-day death": C_RA}

ax1 = fig.add_subplot(gs[0, 0])
rows = []
i = 0
for w in ["all ages", "50-80 y", "55-75 y", "60-70 y"]:
    for o in TWO:
        r = age[(age.window == w) & (age.outcome == o)]
        if not len(r):
            continue
        r = r.iloc[0]
        rows.append(dict(y=-i, or_=r.or_, lo=r.lo, hi=r.hi, col=PCOL[o], marker="o"))
        ax1.text(0.32, -i, "%s  %s  (%d/%d)" % (w, o.split("(")[0].strip(), r.sle_n, r.ra_n),
                 fontsize=6.9, va="center", ha="left", color=PCOL[o])
        i += 1
    i += 0.5
forest(ax1, rows, (0.35, 4.5), [0.5, 0.75, 1, 1.5, 2, 3, 4.5], "RA/SLE ratio")
ax1.spines["bottom"].set_visible(False)
ax1.set_title("A  AGE overlap: restrict both diseases to a common window", fontsize=8.4, loc="left")

ax2 = fig.add_subplot(gs[0, 1])
rows = []
i = 0
for o in TWO:
    r = absb[absb.outcome == o].iloc[0]
    for lab, o_, l_, h_ in [("base adjusted", r.base, r.base_lo, r.base_hi),
                            ("+ s:age", r.s_age, r.s_age_lo, r.s_age_hi),
                            ("+ s:age + s:sofa", r.s_age_sofa, r.s_age_sofa_lo, r.s_age_sofa_hi)]:
        rows.append(dict(y=-i, or_=o_, lo=l_, hi=h_, col=PCOL[o], marker="o"))
        ax2.text(0.32, -i, "%s  %s" % (lab, o.split("(")[0].strip()), fontsize=6.9,
                 va="center", ha="left", color=PCOL[o])
        i += 1
    i += 0.5
forest(ax2, rows, (0.35, 4.5), [0.5, 0.75, 1, 1.5, 2, 3, 4.5], "RA/SLE ratio")
ax2.spines["bottom"].set_visible(False)
ax2.set_title("B  ABSORB AGE: let the dose effect itself vary with age", fontsize=8.4, loc="left")

ax3 = fig.add_subplot(gs[1, 0])
rows = []
i = 0
for o in TWO:
    r = surg[surg.outcome == o].iloc[0]
    for lab, o_, l_, h_ in [("all admissions", r.all_, r.all_lo, r.all_hi),
                            ("non-surgical only", r.nonsurg, r.nonsurg_lo, r.nonsurg_hi)]:
        rows.append(dict(y=-i, or_=o_, lo=l_, hi=h_, col=PCOL[o], marker="o"))
        ax3.text(0.32, -i, "%s  %s" % (lab, o.split("(")[0].strip()), fontsize=6.9,
                 va="center", ha="left", color=PCOL[o])
        i += 1
    i += 0.5
for o in TWO:
    r = expo[expo.outcome == o].iloc[0]
    for lab, o_, l_, h_ in [("ordinal score", r.ordinal, r.ordinal_lo, r.ordinal_hi),
                            ("any GC vs none", r.any_gc, r.any_gc_lo, r.any_gc_hi),
                            (">=50 vs none", r.high, r.high_lo, r.high_hi)]:
        rows.append(dict(y=-i, or_=o_, lo=l_, hi=h_, col=PCOL[o], marker="D", ms=5))
        ax3.text(0.32, -i, "%s  %s" % (lab, o.split("(")[0].strip()), fontsize=6.9,
                 va="center", ha="left", color=PCOL[o])
        i += 1
    i += 0.5
forest(ax3, rows, (0.05, 60), [0.1, 0.3, 1, 3, 10, 40], "RA/SLE ratio")
ax3.spines["bottom"].set_visible(False)
ax3.set_title("C  drop surgical admissions, and re-code the exposure", fontsize=8.4, loc="left")

ax4 = fig.add_subplot(gs[1, 1])
plain_log(ax4, [0.4, 0.6, 0.8, 1, 1.5, 2, 3])
ax4.axvline(1.0, color=GREY, lw=1.0, ls="--")
rows = []
for i, (_, r) in enumerate(boot.iterrows()):
    col = PCOL.get(r.outcome, GREY)
    y = -i
    ax4.plot([r.boot_lo, r.boot_hi], [y, y], color=col, lw=1.8)
    ax4.plot([r.point], [y], marker="o", ms=6, color=col, mec="white", mew=0.8)
    ax4.text(0.42, y, "%s\n  P(ratio>1)=%.3f" % (r.outcome, r.share_gt1), fontsize=6.9,
             va="center", ha="left", color=col)
    rows.append(y)
ax4.set_ylim(min(rows) - 0.7, max(rows) + 0.7)
ax4.set_yticks([])
ax4.set_xlim(0.4, 3)
ax4.set_xlabel("bootstrap RA/SLE ratio (2000 resamples, stratified by disease)")
ax4.spines[["top", "right", "left"]].set_visible(False)
ax4.tick_params(axis="y", length=0)
ax4.set_title("D  bootstrap sampling distribution", fontsize=8.4, loc="left")
fig.suptitle("Figure 21  robustness of the pre-specified-primary (red) and 30-day-death (orange) interactions",
             fontsize=9.4, x=0.02, ha="left", y=0.99)
save(fig, "Figure21_interaction_robustness")

# =====================================================================
# Figure 22 -- competing risk + control panel
# =====================================================================
comp = pd.read_csv(os.path.join(OUT, "table_v5_robust_competing.csv"))
cox = pd.read_csv(os.path.join(OUT, "table_v5_robust_cox.csv"))

fig = plt.figure(figsize=(11.4, 5.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.62, wspace=0.30)

ax1 = fig.add_subplot(gs[0, :])
# JSON 交互表用长标签，稳健性 CSV 用短标签 —— 两套都要
LONG = dict(prim="New broad-spectrum antimicrobial agent, 24 h-7 d",
            d30="30-day mortality", cult="Blood culture positive, 24 h-7 d")
SHORT = dict(prim="new antimicrobial (PRIMARY)", d30="30-day death",
             cult="blood culture + (spec.)")
specs = [("full cohort (adjusted logistic)", IX, LONG),
         ("complete 7-d window only", comp[comp.analysis == "complete window"], SHORT),
         ("antimicrobial-naive subgroup", comp[comp.analysis == "antimicrobial-naive"], SHORT)]
rows, i = [], 0
for nm, tab, LB in specs:
    for key in ["prim", "d30", "cult"]:
        r = tab[tab.outcome == LB[key]]
        if not len(r):
            continue
        r = r.iloc[0]
        if key == "prim":
            col = C_PRIM
        elif key == "d30":
            col = C_RA
        else:
            col = GREY
        o_ = r.or_ if "or_" in r.index else r.or_uw
        lo_ = r.lo if "lo" in r.index else r.lo_uw
        hi_ = r.hi if "hi" in r.index else r.hi_uw
        rows.append(dict(y=-i, or_=o_, lo=lo_, hi=hi_, col=col, marker="o"))
        ax1.text(0.14, -i, "%-32s %s" % (nm, LB[key].split(",")[0][:26]), fontsize=7.0,
                 va="center", ha="left", color=col)
        i += 1
    i += 0.6
for _, r in cox.iterrows():
    col = C_RA if "disease" in r.term else "#1D9E75"
    rows.append(dict(y=-i, or_=r.hr, lo=r.lo, hi=r.hi, col=col, marker="D"))
    ax1.text(0.14, -i, "%-32s %s" % ("Cox PH (discharge/death/7 d censor)", r.term),
             fontsize=7.0, va="center", ha="left", color=col)
    i += 1
forest(ax1, rows, (0.1, 12), [0.2, 0.5, 1, 2, 5, 10], "estimate (RA/SLE ratio for interactions, HR for Cox)")
ax1.spines["bottom"].set_visible(False)
ax1.set_title("A  competing risk / informative censoring: the interaction disappears under every treatment",
              fontsize=8.6, loc="left")

ax2 = fig.add_subplot(gs[1, 0])
plain_log(ax2, [0.8, 1.0, 1.5, 2.0, 3.0])
ax2.axvline(1.0, color=GREY, lw=1.0, ls="--")
ORD = {"G0_none": "none", "G1_low": ">0-<10", "G2_mod": "10-<50", "G3_high": ">=50"}
for g, col in [("SLE", C_SLE), ("RA", C_RA)]:
    st = STRATA[(STRATA.ov == "hyper_48h") & (STRATA.disease == g)]
    xs = np.arange(len(st))
    ax2.plot(xs, st.adj_or, "-o", color=col, ms=6, mec="white", mew=0.8, label="%s adjusted" % g)
    for x, o_, l_, h_ in zip(xs, st.adj_or, st.adj_lo, st.adj_hi):
        ax2.plot([x, x], [l_, h_], color=col, lw=1.5)
    ax2.plot(xs, st.iptw_or, "--s", color=col, ms=5, alpha=0.7, label="%s IPTW" % g)
ax2.set_ylim(0.6, 5.5)
ax2.set_xticks(np.arange(3))
ax2.set_xticklabels([ORD[s] for s in ["G1_low", "G2_mod", "G3_high"]])
ax2.set_xlabel("dose stratum (mg/day prednisone equivalent)")
ax2.set_ylabel("OR for glucose >=180 mg/dL")
ax2.spines[["top", "right"]].set_visible(False)
ax2.legend(fontsize=6.8, frameon=False, ncol=2, loc="upper left")
ax2.set_title("B  POSITIVE CONTROL: GC -> hyperglycaemia\n    passes in both diseases, dose gradient present",
              fontsize=8.4, loc="left")

ax3 = fig.add_subplot(gs[1, 1])
rows = []
i = 0
for key, col in [("prim", C_PRIM), ("cult", "#4A4A4A"), ("d30", C_RA),
                 ("glu", GREY), ("gi", GREY)]:
    o = {"prim": LONG["prim"], "cult": LONG["cult"], "d30": LONG["d30"],
         "glu": "Glucose >=180 mg/dL", "gi": "GI bleeding"}[key]
    sub = NEG[NEG.outcome == o]
    for _, r in sub.iterrows():
        rows.append(dict(y=-i, or_=r.adj_or, lo=r.adj_lo, hi=r.adj_hi, col=col,
                         marker="o", ms=5))
        ax3.text(0.16, -i, "%-30s / %s" % (o.split(",")[0][:28], r.negexp.split()[0]),
                 fontsize=6.6, va="center", ha="left", color=col)
        i += 1
    i += 0.4
forest(ax3, rows, (0.15, 4.0), [0.2, 0.4, 0.6, 1, 2, 3], "OR for the outcome")
ax3.spines["bottom"].set_visible(False)
ax3.set_title("C  NEGATIVE-EXPOSURE controls: ordinary ICU drugs\n    also move the outcome -> care-process confounding",
              fontsize=8.4, loc="left")
fig.suptitle("Figure 22  competing risk and the control panel", fontsize=9.4, x=0.02, ha="left", y=0.99)
save(fig, "Figure22_competing_risk_and_controls")
print("figures done")
