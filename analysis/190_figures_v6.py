# -*- coding: utf-8 -*-
"""Figures for the v6 report (death interaction + indication mechanism).

  Figure 23  indication extraction: three operationalisations, audit, and how
             disease-specific activity vs generic organ failure move with dose
  Figure 24  reference-group validity (G0 SLE vs RA) + population structure of
             the four dose strata
  Figure 25  indication x exposure concordance 2x2 (n, mortality, RA/SLE)
  Figure 26  death dose-response within disease, stratified by indication/history,
             with the positive control repeated in every stratum
  Figure 27  the death interaction: adjustment ladder, within-stratum estimates,
             and multiplicity

Layout rules kept from 170/180 (they prevented every collision seen earlier):
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
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

C_SLE, C_RA = "#2B6CB0", "#D85A30"
C_SPEC, C_GEN, C_SHOCK = "#B03A2E", "#2F7A4F", "#7F77DD"
C_POS, C_NEG = "#B03A2E", "#2F7A4F"
GREY = "#9AA0A6"
STRCOL = {"none": "#9AA0A6", ">0-<10": "#7FB3D5", "10-<50": "#2E86C1",
          ">=50": "#1B3A5C"}

J = json.load(open(os.path.join(OUT, "_v6_results.json"), encoding="utf-8"))
REF = pd.DataFrame(J["refgroup"])
STR = pd.DataFrame(J["structure"])
CON = pd.DataFrame(J["concordance"])
DOSE = pd.DataFrame(J["dose"])
IXA = pd.DataFrame(J["ix_adjust"])
IXS = pd.DataFrame(J["ix_strata"])
BH = pd.DataFrame(J["bh"])
SEL = pd.DataFrame(J["selection"])
DXC = pd.DataFrame(J["dxconc"])
QC = pd.read_csv(os.path.join(DATA, "gc_indication.csv"))

PH = {"G0_none": "none", "G1_low": ">0-<10", "G2_mod": "10-<50",
      "G3_high": ">=50"}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, "%s.%s" % (name, ext)), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def plain_log(ax, ticks):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(["%g" % t for t in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())


def forest(ax, rows, xlim, xticks, xlabel, ref=1.0):
    """rows: list of dict(y, OR, lo, hi, col, marker, ms)."""
    ax.set_ylim(-max(1, len(rows)) + 0.4, 0.6)
    for r in rows:
        ax.plot([r["lo"], r["hi"]], [r["y"], r["y"]], color=r.get("col", "#333"),
                lw=1.5, solid_capstyle="butt", zorder=2)
        ax.plot([r["OR"]], [r["y"]], marker=r.get("marker", "o"),
                ms=r.get("ms", 6), color=r.get("col", "#333"),
                mec="white", mew=0.8, zorder=3)
    ax.axvline(ref, color="#888", lw=0.9, ls="--", zorder=1)
    if xlim:
        ax.set_xlim(*xlim)
    if xticks:
        plain_log(ax, xticks)
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=8.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


# ===================================================================== Fig 23
def fig23():
    fig = plt.figure(figsize=(11.2, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.62,
                          wspace=0.34)

    # --- A: audit of extraction coverage
    ax = fig.add_subplot(gs[0, 0])
    n = len(QC)
    nf = int((QC.note_found == 1).sum())
    items = [("discharge note available", 100 * nf / n),
             ("MOA section parsed", 100 * (QC.moa_len > 0).sum() / n),
             ("GC word in narrative", 100 * (QC.n_gc_mentions > 0).sum() / n),
             ("disease-specific activity", 100 * QC.indN_spec.mean()),
             ("activity in discharge dx", 100 * QC.indD_spec.mean()),
             ("home GC on admit list", 100 * QC.home_gc.mean())]
    ys = [-i for i in range(len(items))]
    for y, (lab, v) in zip(ys, items):
        ax.barh(y, v, height=0.62, color=C_SLE if v > 60 else C_SPEC,
                alpha=0.85)
        ax.text(v + 1.5, y, "%.1f%%" % v, va="center", fontsize=7.4, color="#333")
    ax.set_yticks(ys)
    ax.set_yticklabels([i[0] for i in items], fontsize=7.4)
    ax.set_xlim(0, 118)
    ax.set_xlabel("% of the rheumatic ICU cohort", fontsize=8)
    ax.set_title("A  what the notes yield", fontsize=8.6, loc="left")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    # --- B: three operationalisations x disease
    ax = fig.add_subplot(gs[0, 1])
    nn = pd.read_csv(os.path.join(DATA, "gc_history.csv"))
    nn = nn[nn.note_found == 1]
    _coh = pd.read_csv(os.path.join(DATA, "cohort_rheum_icu.csv"))[
        ["stay_key", "primary_grp"]].drop_duplicates("stay_key")
    nn = nn.merge(_coh, on="stay_key", how="left")
    nn = nn[nn.primary_grp.isin(["SLE", "RA"])]
    groups = [("SLE", C_SLE), ("RA", C_RA)]
    metrics = [("activity\n(narrative)", "indN_spec"),
               ("activity\n(discharge dx)", "indD_spec"),
               ("generic organ\nfailure", "indN_gen"),
               ("shock /\nstress-dose", "indN_shock")]
    x = np.arange(len(metrics))
    w = 0.36
    for k, (g, col) in enumerate(groups):
        sub = nn[nn.primary_grp == g]
        vals = [100 * sub[c].mean() for _, c in metrics]
        ax.bar(x + (k - 0.5) * w, vals, w, color=col, alpha=0.88, label=g)
        for xx, v in zip(x + (k - 0.5) * w, vals):
            ax.text(xx, v + 1.6, "%.1f" % v, ha="center", fontsize=6.8,
                    color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=7.0)
    ax.set_ylim(0, 68)
    ax.set_ylabel("% of patients", fontsize=8)
    ax.set_title("B  indication mix differs by disease, not by severity",
                 fontsize=8.6, loc="left")
    ax.legend(frameon=False, fontsize=7.6, ncol=2, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # --- C: spec / gen by dose stratum
    ax = fig.add_subplot(gs[0, 2])
    lab_order = ["none", ">0-<10", "10-<50", ">=50"]
    x = np.arange(len(lab_order))
    for key, col, mk in [("spec", C_SPEC, "o"), ("gen", C_GEN, "s"),
                         ("shock", C_SHOCK, "^")]:
        vals = []
        for lab in lab_order:
            sub = STR[STR.stratum == lab]
            vals.append(sub[key].mean())
        ax.plot(x, vals, marker=mk, color=col, lw=1.7, ms=6, mec="white",
                mew=0.8, label={"spec": "disease-specific activity",
                                "gen": "generic organ failure",
                                "shock": "shock / stress-dose"}[key])
        for xx, v in zip(x, vals):
            ax.text(xx, v + 1.8, "%.0f" % v, ha="center", fontsize=6.8,
                    color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(lab_order, fontsize=7.4)
    ax.set_ylim(15, 72)
    ax.set_xlabel("GC dose in first 24 h (prednisone-equivalent mg/day)",
                  fontsize=8)
    ax.set_ylabel("% of patients", fontsize=8)
    ax.set_title("C  documented activity tracks dose;\n     generic organ failure tracks it too",
                 fontsize=8.6, loc="left")
    ax.legend(frameon=False, fontsize=7.0, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # --- D: home GC by stratum, both diseases
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(len(lab_order))
    for k, (g, col) in enumerate(groups):
        vals = [STR[(STR.disease == g) & (STR.stratum == lab)].home_gc.iloc[0]
                for lab in lab_order]
        ax.bar(x + (k - 0.5) * 0.36, vals, 0.36, color=col, alpha=0.88,
               label=g)
        for xx, v in zip(x + (k - 0.5) * 0.36, vals):
            ax.text(xx, v + 2, "%.0f" % v, ha="center", fontsize=6.8, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(lab_order, fontsize=7.4)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% with home GC\non admission list", fontsize=8)
    ax.set_title("D  the low-dose stratum is continuation of a home dose",
                 fontsize=8.6, loc="left")
    ax.legend(frameon=False, fontsize=7.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # --- E: structural GC user by stratum
    ax = fig.add_subplot(gs[1, 1])
    for k, (g, col) in enumerate(groups):
        vals = [STR[(STR.disease == g) & (STR.stratum == lab)].gcu.iloc[0]
                for lab in lab_order]
        ax.bar(x + (k - 0.5) * 0.36, vals, 0.36, color=col, alpha=0.88, label=g)
        for xx, v in zip(x + (k - 0.5) * 0.36, vals):
            ax.text(xx, v + 2, "%.0f" % v, ha="center", fontsize=6.8, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(lab_order, fontsize=7.4)
    ax.set_ylim(0, 112)
    ax.set_ylabel("% structural GC user\n(home / prior / pre-ICU)", fontsize=8)
    ax.set_title("E  so the 'dose-response' is largely a\n     contrast between patient types",
                 fontsize=8.6, loc="left")
    ax.legend(frameon=False, fontsize=7.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # --- F: GC unmentioned in narrative
    ax = fig.add_subplot(gs[1, 2])
    for k, (g, col) in enumerate(groups):
        vals = [STR[(STR.disease == g) & (STR.stratum == lab)].gc_unmentioned.iloc[0]
                for lab in lab_order]
        ax.bar(x + (k - 0.5) * 0.36, vals, 0.36, color=col, alpha=0.88, label=g)
        for xx, v in zip(x + (k - 0.5) * 0.36, vals):
            ax.text(xx, v + 2, "%.0f" % v, ha="center", fontsize=6.8, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(lab_order, fontsize=7.4)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% where GC is never\nmentioned in the narrative", fontsize=8)
    ax.set_title("F  the reference group is a group\n     the note barely discusses",
                 fontsize=8.6, loc="left")
    ax.legend(frameon=False, fontsize=7.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle("Figure 23  Indication extraction from discharge notes: "
                 "coverage, operationalisations, and population structure of the dose strata",
                 fontsize=10.2, x=0.008, ha="left", y=1.005)
    save(fig, "Figure23_indication_extraction")


# ===================================================================== Fig 24
def fig24():
    fig = plt.figure(figsize=(11.0, 6.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 1.0], wspace=0.44)

    # --- A: |SMD| SLE vs RA within G0
    ax = fig.add_subplot(gs[0, 0])
    r = REF.dropna(subset=["smd"]).copy()
    r = r[r.smd > 0]
    r = r.sort_values("smd")
    r = pd.concat([r.head(6), r.tail(9)])
    ys = [-i for i in range(len(r))]
    for y, (_, row) in zip(ys, r.iterrows()):
        col = C_SPEC if row.smd >= 0.2 else (C_SLE if row.smd >= 0.1 else GREY)
        ax.plot([0, row.smd], [y, y], color=col, lw=1.6)
        ax.plot([row.smd], [y], marker="o", ms=5.5, color=col, mec="white",
                mew=0.7)
        ax.text(row.smd + 0.02, y, "%.2f" % row.smd, va="center", fontsize=6.8,
                color=col)
    ax.axvline(0.2, color="#333", lw=1.0, ls=":")
    ax.text(0.205, ys[0] + 0.75, "|SMD| = 0.2", fontsize=6.9, color="#333")
    ax.set_yticks(ys)
    ax.set_yticklabels([v.replace(", %", "").replace(", median", "")
                        for v in r.variable], fontsize=7.0)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("|standardised mean difference|, SLE vs RA", fontsize=8)
    ax.set_title("A  inside the no-GC reference group the two\n"
                 "     diseases are still different populations",
                 fontsize=8.8, loc="left")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    # --- B: concordance 2x2 n
    ax = fig.add_subplot(gs[0, 1])
    labs = list(CON.group)
    x = np.arange(len(labs))
    ax.bar(x - 0.19, CON.sle_n, 0.38, color=C_SLE, alpha=0.88, label="SLE")
    ax.bar(x + 0.19, CON.ra_n, 0.38, color=C_RA, alpha=0.88, label="RA")
    for xx, v in zip(x - 0.19, CON.sle_n):
        ax.text(xx, v + 8, "%d" % v, ha="center", fontsize=6.8, color=C_SLE)
    for xx, v in zip(x + 0.19, CON.ra_n):
        ax.text(xx, v + 8, "%d" % v, ha="center", fontsize=6.8, color=C_RA)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" ", "\n", 1) for l in labs], fontsize=7.0)
    ax.set_ylim(0, 520)
    ax.set_ylabel("patients", fontsize=8)
    ax.set_title("B  indication x exposure concordance\n     (n)", fontsize=8.8,
                 loc="left")
    ax.legend(frameon=False, fontsize=7.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # --- C: mortality by concordance cell
    ax = fig.add_subplot(gs[0, 2])
    ax.bar(x - 0.19, CON.sle_death, 0.38, color=C_SLE, alpha=0.88, label="SLE")
    ax.bar(x + 0.19, CON.ra_death, 0.38, color=C_RA, alpha=0.88, label="RA")
    for xx, v in zip(x - 0.19, CON.sle_death):
        ax.text(xx, v + 0.25, "%.1f" % v, ha="center", fontsize=6.8, color=C_SLE)
    for xx, v in zip(x + 0.19, CON.ra_death):
        ax.text(xx, v + 0.25, "%.1f" % v, ha="center", fontsize=6.8, color=C_RA)
    for i, (_, row) in enumerate(CON.iterrows()):
        if row.sle_death and row.ra_death and row.sle_death > 0:
            ax.text(i, max(row.sle_death, row.ra_death) + 1.5,
                    "RA/SLE %.2f" % (row.ra_death / row.sle_death),
                    ha="center", fontsize=6.7, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" ", "\n", 1) for l in labs], fontsize=7.0)
    ax.set_ylim(0, 17.5)
    ax.set_ylabel("30-day mortality (%)", fontsize=8)
    ax.set_title("C  the RA excess sits in 'GC given,\n     no activity documented'",
                 fontsize=8.8, loc="left")
    ax.legend(frameon=False, fontsize=7.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle("Figure 24  The reference group, and where the disease contrast "
                 "actually lives", fontsize=10.2, x=0.008, ha="left", y=1.02)
    save(fig, "Figure24_refgroup_and_concordance")


# ===================================================================== Fig 25
def fig25():
    fig = plt.figure(figsize=(11.0, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.10)

    strat_order = ["ALL (SLE+RA, with note)",
                   "spec+ (documented activity)",
                   "spec- (no documented activity)",
                   "GC-naive (gc_user_struct=0)",
                   "prior/home GC user"]

    # --- A: death trend within disease, by stratum
    ax = fig.add_subplot(gs[0, 0])
    rows = []
    y = 0
    for s in strat_order:
        for g, col, mk in [("SLE", C_SLE, "o"), ("RA", C_RA, "s")]:
            rr = DOSE[(DOSE.stratum == s) & (DOSE.disease == g) &
                      (DOSE.outcome == "death_30d")]
            if not len(rr):
                continue
            rr = rr.iloc[0]
            rows.append(dict(y=y, OR=rr.OR, lo=rr.lo, hi=rr.hi, col=col,
                             marker=mk, ms=6))
            ax.text(2.92, y, "%.2f (%.2f-%.2f)" % (rr.OR, rr.lo, rr.hi),
                    fontsize=6.8, va="center", ha="left", color=col,
                    clip_on=False)
            ax.text(3.72, y, "%d/%d" % (rr.n_ev, rr.n), fontsize=6.6,
                    va="center", ha="left", color="#666", clip_on=False)
            y -= 1
        ax.text(2.92, y + 0.5, s, fontsize=7.6, va="center", ha="left",
                color="#111", clip_on=False, fontweight="bold")
        y -= 0.7
    forest(ax, rows, (0.28, 4.2), [0.3, 0.5, 1, 2, 4],
           "dose-trend OR for 30-day death (per stratum increment)")
    ax.text(2.92, 0.75, "OR (95% CI)", fontsize=7.0, va="center", ha="left",
            clip_on=False, fontweight="bold")
    ax.text(3.72, 0.75, "events/n", fontsize=7.0, va="center", ha="left",
            clip_on=False, fontweight="bold")
    ax.set_title("A  30-day death dose-response within each disease",
                 fontsize=8.8, loc="left")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C_SLE, label="SLE",
                              mec="white"),
                       Line2D([], [], marker="s", ls="", color=C_RA, label="RA",
                              mec="white")],
              frameon=False, fontsize=7.6, loc="lower left")

    # --- B: same strata, positive control (glucose)
    ax = fig.add_subplot(gs[0, 1])
    rows2 = []
    y = 0
    for s in strat_order:
        for g, col, mk in [("SLE", C_SLE, "o"), ("RA", C_RA, "s")]:
            rr = DOSE[(DOSE.stratum == s) & (DOSE.disease == g) &
                      (DOSE.outcome == "hyper_48h")]
            if not len(rr):
                continue
            rr = rr.iloc[0]
            rows2.append(dict(y=y, OR=rr.OR, lo=rr.lo, hi=rr.hi, col=col,
                              marker=mk, ms=6))
            ax.text(2.92, y, "%.2f (%.2f-%.2f)" % (rr.OR, rr.lo, rr.hi),
                    fontsize=6.8, va="center", ha="left", color=col,
                    clip_on=False)
            y -= 1
        ax.text(2.92, y + 0.5, s, fontsize=7.6, va="center", ha="left",
                color="#111", clip_on=False, fontweight="bold")
        y -= 0.7
    forest(ax, rows2, (0.28, 4.2), [0.3, 0.5, 1, 2, 4],
           "dose-trend OR for glucose >=180 mg/dL (POSITIVE CONTROL)")
    ax.text(2.92, 0.75, "OR (95% CI)", fontsize=7.0, va="center", ha="left",
            clip_on=False, fontweight="bold")
    ax.set_title("B  the same strata, positive control", fontsize=8.8,
                 loc="left")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C_SLE, label="SLE",
                              mec="white"),
                       Line2D([], [], marker="s", ls="", color=C_RA, label="RA",
                              mec="white")],
              frameon=False, fontsize=7.6, loc="lower left")

    fig.text(0.008, -0.015,
             "Strata are nested subsets of the 1 250 SLE+RA admissions that have a "
             "discharge note.  B exists to show that restricting to a stratum did not "
             "break the exposure measurement: the pharmacological positive control "
             "survives everywhere except the smallest (GC-naive SLE).",
             fontsize=7.4, color="#444", ha="left")
    fig.suptitle("Figure 25  Stratifying the death dose-response by indication "
                 "and by GC history", fontsize=10.2, x=0.008, ha="left", y=1.03)
    save(fig, "Figure25_death_dose_by_stratum")


# ===================================================================== Fig 26
def fig26():
    fig = plt.figure(figsize=(11.2, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.72], wspace=0.42)

    # --- A: adjustment ladder
    ax = fig.add_subplot(gs[0, 0])
    rows = []
    for i, (_, r) in enumerate(IXA.iterrows()):
        y = -i
        col = C_SPEC if i == 0 else C_SLE
        rows.append(dict(y=y, OR=r.OR, lo=r.lo, hi=r.hi, col=col, ms=6))
        ax.text(3.05, y, "%.2f (%.2f-%.2f)" % (r.OR, r.lo, r.hi), fontsize=6.7,
                va="center", ha="left", color=col, clip_on=False)
        ax.text(4.30, y, "%.3f" % r.p_lrt, fontsize=6.7, va="center",
                ha="left", color="#555", clip_on=False)
    forest(ax, rows, (0.75, 3.0), [0.8, 1, 1.5, 2, 3],
           "RA/SLE ratio of dose-trend ORs for death")
    ax.set_yticklabels([])
    ax.text(3.05, 0.75, "RA/SLE (95% CI)", fontsize=6.9, va="center",
            ha="left", clip_on=False, fontweight="bold")
    ax.text(4.30, 0.75, "P (LRT)", fontsize=6.9, va="center", ha="left",
            clip_on=False, fontweight="bold")
    for i, lab in enumerate(IXA.label):
        ax.text(0.745, -i, lab.split("  ")[0], fontsize=7.2, va="center",
                ha="right", clip_on=False)
    ax.set_title("A  adding indication structure barely moves it",
                 fontsize=8.8, loc="left")

    # --- B: within-stratum interaction
    ax = fig.add_subplot(gs[0, 1])
    rows2 = []
    for i, (_, r) in enumerate(IXS.iterrows()):
        y = -i
        hot = r.stratum.startswith("spec-")
        col = C_SPEC if hot else C_SLE
        rows2.append(dict(y=y, OR=r.OR, lo=max(r.lo, 0.16), hi=min(r.hi, 14.0),
                          col=col, ms=6))
        ax.text(15.0, y, "%.2f (%.2f-%.2f)" % (r.OR, r.lo, r.hi), fontsize=6.6,
                va="center", ha="left", color=col, clip_on=False)
        ax.text(30.0, y, "%.3f" % r.p_lrt, fontsize=6.6, va="center", ha="left",
                color="#555", clip_on=False)
    forest(ax, rows2, (0.30, 14.0), [0.5, 1, 2, 4, 8],
           "RA/SLE ratio of dose-trend ORs for death")
    ax.text(15.0, 0.75, "RA/SLE (95% CI)", fontsize=6.9, va="center", ha="left",
            clip_on=False, fontweight="bold")
    ax.text(30.0, 0.75, "P (LRT)", fontsize=6.9, va="center", ha="left",
            clip_on=False, fontweight="bold")
    for i, lab in enumerate(IXS.stratum):
        ax.text(0.295, -i, lab, fontsize=7.0, va="center", ha="right",
                clip_on=False)
    ax.set_title("B  and it lives in the patients with no documented activity",
                 fontsize=8.8, loc="left")

    # --- C: multiplicity
    ax = fig.add_subplot(gs[0, 2])
    b = BH.copy()
    b = b.sort_values("p_lrt")
    x = np.arange(len(b))
    ax.bar(x - 0.19, b.p_lrt, 0.38, color=C_SLE, alpha=0.9, label="P (LRT)")
    ax.bar(x + 0.19, b.q, 0.38, color=C_SPEC, alpha=0.9, label="q (BH)")
    ax.axhline(0.05, color="#333", lw=1.0, ls="--")
    ax.text(len(b) - 0.5, 0.058, "0.05", fontsize=6.8, ha="right", color="#333")
    for xx, v in zip(x - 0.19, b.p_lrt):
        ax.text(xx, v + 0.02, "%.3f" % v, ha="center", fontsize=6.4,
                color=C_SLE, rotation=90)
    for xx, v in zip(x + 0.19, b.q):
        ax.text(xx, v + 0.02, "%.3f" % v, ha="center", fontsize=6.4,
                color=C_SPEC, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_", "\n") for v in b.outcome], fontsize=6.6)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("P / q", fontsize=8)
    ax.set_title("C  nothing else survives\n     family-wise correction",
                 fontsize=8.8, loc="left")
    ax.legend(frameon=False, fontsize=7.0, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.text(0.008, -0.015,
             "A: the pre-specified confirmatory test is the 30-day-death interaction "
             "under the base adjustment set (dark red). Attenuation is reported as the "
             "% change in log(OR) relative to that base.  B: subsets of the same 1 250 "
             "admissions.  C: BH across the remaining interaction tests.",
             fontsize=7.4, color="#444", ha="left")
    fig.suptitle("Figure 26  The death interaction survives indication adjustment, "
                 "and is concentrated where no activity was documented",
                 fontsize=10.2, x=0.008, ha="left", y=1.03)
    save(fig, "Figure26_death_interaction_mechanism")


# ===================================================================== Fig 27
def fig27():
    fig = plt.figure(figsize=(11.0, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.40)

    # --- A: selection check
    ax = fig.add_subplot(gs[0, 0])
    s = SEL.dropna(subset=["smd"])
    s = s.sort_values("smd")
    ys = [-i for i in range(len(s))]
    for y, (_, row) in zip(ys, s.iterrows()):
        col = C_SPEC if row.smd >= 0.2 else (C_SLE if row.smd >= 0.1 else GREY)
        ax.plot([0, row.smd], [y, y], color=col, lw=1.6)
        ax.plot([row.smd], [y], marker="o", ms=5.5, color=col, mec="white",
                mew=0.7)
        ax.text(row.smd + 0.008, y, "%.3f" % row.smd, va="center", fontsize=6.8,
                color=col)
    ax.axvline(0.2, color="#333", lw=1.0, ls=":")
    ax.text(0.205, ys[0] + 0.6, "0.2", fontsize=6.9, color="#333")
    ax.set_yticks(ys)
    ax.set_yticklabels([v.replace(", %", "").replace(", median", "")
                        for v in s.variable], fontsize=7.2)
    ax.set_xlim(0, 0.34)
    ax.set_xlabel("|SMD|, note available vs not", fontsize=8)
    ax.set_title("A  the discharge-note restriction\n     is only mildly selective",
                 fontsize=8.8, loc="left")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    # --- B: anchor
    ax = fig.add_subplot(gs[0, 1])
    vals = [(1636, 1.57, 1.05, 2.35, C_SLE),
            (1250, 1.67, 1.07, 2.62, C_SPEC)]
    for i, (n, o, lo, hi, col) in enumerate(vals):
        y = -i
        ax.plot([lo, hi], [y, y], color=col, lw=1.6)
        ax.plot([o], [y], marker="o", ms=7, color=col, mec="white", mew=0.8)
        ax.text(2.72, y, "%.2f (%.2f-%.2f)" % (o, lo, hi), fontsize=7.2,
                va="center", ha="left", color=col, clip_on=False)
        ax.text(1.0, y + 0.42, "n = %d" % n, fontsize=7.2, ha="center",
                color="#333")
    ax.axvline(1.0, color="#888", lw=0.9, ls="--")
    plain_log(ax, [1, 1.5, 2, 3])
    ax.set_xlim(0.9, 2.9)
    ax.set_ylim(-1.65, 0.85)
    ax.set_yticks([])
    ax.set_xlabel("RA/SLE ratio of dose-trend ORs for death", fontsize=8)
    ax.set_title("B  removing the restriction does not\n     change the interaction",
                 fontsize=8.8, loc="left")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    # --- C: operationalisation sensitivity
    ax = fig.add_subplot(gs[0, 2])
    items = [("activity: narrative\n(spec+)", 1.39, 0.72, 2.69, 0.076),
             ("activity: narrative\n(spec-)", 2.08, 1.07, 4.03, 0.020),
             ("activity: discharge dx\n(dxspec+)", 0.71, 0.13, 4.03, 0.805),
             ("activity: discharge dx\n(dxspec-)", 1.69, 1.07, 2.66, 0.021)]
    for i, (lab, o, lo, hi, p) in enumerate(items):
        y = -i
        hot = "spec-" in lab
        col = C_SPEC if hot else (C_SLE if "+" in lab else C_RA)
        ax.plot([lo, hi], [y, y], color=col, lw=1.7)
        ax.plot([o], [y], marker="o", ms=6.5, color=col, mec="white", mew=0.8)
        ax.text(4.30, y, "%.2f (%.2f-%.2f)" % (o, lo, hi), fontsize=6.7,
                va="center", ha="left", color=col, clip_on=False)
        ax.text(6.40, y, "%.3f" % p, fontsize=6.7, va="center", ha="left",
                color="#555", clip_on=False)
        ax.text(0.295, y, lab, fontsize=7.0, va="center", ha="right",
                clip_on=False)
    ax.axvline(1.0, color="#888", lw=0.9, ls="--")
    plain_log(ax, [0.5, 1, 2, 4])
    ax.set_xlim(0.30, 4.2)
    ax.set_ylim(-3.6, 0.85)
    ax.set_yticks([])
    ax.set_xlabel("RA/SLE ratio of dose-trend ORs for death", fontsize=8)
    ax.text(4.30, 0.72, "RA/SLE (95% CI)", fontsize=6.9, va="center", ha="left",
            clip_on=False, fontweight="bold")
    ax.text(6.40, 0.72, "P (LRT)", fontsize=6.9, va="center", ha="left",
            clip_on=False, fontweight="bold")
    ax.set_title("C  the mechanism is not an artefact of\n     how activity was extracted",
                 fontsize=8.8, loc="left")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    fig.suptitle("Figure 27  Could the mechanism be an artefact?  Selection, "
                 "restriction, and operationalisation", fontsize=10.2, x=0.008,
                 ha="left", y=1.03)
    save(fig, "Figure27_mechanism_sensitivity")


if __name__ == "__main__":
    for f in (fig23, fig24, fig25, fig26, fig27):
        f()
    print("done")
