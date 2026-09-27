# -*- coding: utf-8 -*-
"""Figures for the v7 report (decomposing the "no documented activity" stratum).

  Figure 28  the three channels: composition, clinical profile, dose-stratum mix
  Figure 29  is "under-documentation" a recording artefact?  note length vs death
  Figure 30  where the death interaction lives, after the decomposition
  Figure 31  robustness: length adjustment, landmark, IPTW, D3 self-check

Layout rules inherited from 170/180/190 (they prevented every collision seen):
  * explicit y = -row index, never a running counter
  * right-hand text goes OUTSIDE the axes with clip_on=False
  * log axes get explicit ticks + NullLocator/NullFormatter
"""
import json
import os
import textwrap

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
C_INACT, C_DEFER, C_UDOC = "#2F7A4F", "#B7791F", "#B03A2E"
C_POS, C_NEG = "#B03A2E", "#2F7A4F"
GREY = "#9AA0A6"

J = json.load(open(os.path.join(OUT, "_v7_results.json"), encoding="utf-8"))
PR = pd.DataFrame(J["profile"])
NL = pd.DataFrame(J["notelen"])
ST = pd.DataFrame(J["strata"])
UD = pd.DataFrame(J["underdoc"])
G0 = pd.DataFrame(J["g0"])
D3 = pd.DataFrame(J["d3"])
IW = pd.DataFrame(J["iptw"])
LA = pd.DataFrame(J["len_adj"])
DT = pd.DataFrame(J["deathtab"])
DTX = J["deathtab_extra"]
G0X = pd.DataFrame(J["g0_dx"])

CH = ["INACTIVE", "INF_DEFER", "UNDERDOC"]
CHLAB = {"INACTIVE": "truly inactive", "INF_DEFER": "infection-deferred",
         "UNDERDOC": "undocumented"}
CHCOL = {"INACTIVE": C_INACT, "INF_DEFER": C_DEFER, "UNDERDOC": C_UDOC}
STRAT = ["none", ">0-<10", "10-<50", ">=50"]


def prow(lab):
    r = PR[PR.variable == lab]
    assert len(r), lab
    return r.iloc[0]


def pct(s):
    return float(str(s).replace("%", ""))


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


def forest(ax, rows, xlim, xticks, xlabel, ref=1.0, ylim=None):
    if ylim is None:
        ylim = (-max(1, len(rows)) + 0.4, 0.6)
    ax.set_ylim(*ylim)
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


def rt(ax, xf, y, s, color="#333", size=6.7, bold=False, ha="left"):
    """Text positioned by (axes-fraction x, data y) -- never escapes the axes."""
    ax.annotate(str(s), xy=(xf, y), xycoords=("axes fraction", "data"),
                fontsize=size, va="center", ha=ha, color=color,
                clip_on=False, annotation_clip=False,
                fontweight="bold" if bold else None)


def cap(fig, text, y=-0.015, width=165, size=7.4):
    """Figure caption, hard-wrapped.

    A single 400-character line silently widens the tight bbox to ~22 in and
    squeezes every panel into the left third of the canvas.  Wrap it.
    """
    fig.text(0.008, y, "\n".join(textwrap.wrap(" ".join(text.split()), width)),
             fontsize=size, color="#444", ha="left", va="top")


def srow(name):
    r = ST[ST.stratum == name]
    assert len(r), name
    return r.iloc[0]


# ===================================================================== Fig 28
def fig28():
    fig = plt.figure(figsize=(11.2, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.62, 1.0], hspace=0.72,
                          wspace=0.30)

    # --- A: composition of the "no documented activity" stratum
    ax = fig.add_subplot(gs[0, :])
    ns = {c: int(PR[PR.variable == "n"][c].iloc[0]) for c in CH}
    tot = sum(ns.values())
    left = 0.0
    for c in CH:
        ax.barh(0, ns[c], left=left, height=0.5, color=CHCOL[c], alpha=0.9)
        ax.text(left + ns[c] / 2, 0, "%d\n%.1f%%" % (ns[c], 100 * ns[c] / tot),
                ha="center", va="center", fontsize=8.4, color="white",
                fontweight="bold")
        left += ns[c]
    ax.set_xlim(0, tot * 1.005)
    ax.set_ylim(-0.75, 0.62)
    ax.set_yticks([])
    ax.set_xlabel("patients", fontsize=8)
    ax.set_title("A  what the 874 'no documented disease-specific activity' "
                 "admissions actually consist of", fontsize=8.8, loc="left")
    ax.legend(handles=[Line2D([], [], marker="s", ls="", color=CHCOL[c],
                              label="%s  (%s)" % (CHLAB[c], c)) for c in CH],
              frameon=False, fontsize=7.6, ncol=3, loc="lower left",
              bbox_to_anchor=(0, -0.55))
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    # --- B: clinical profile
    ax = fig.add_subplot(gs[1, 0])
    items = ["HOME GC on admission %", "GC given in first 24 h %",
             "structural GC user %", "any activity-assay mentioned %",
             "GC unmentioned in narrative %"]
    labs = ["home GC\non admit list", "GC given\nfirst 24 h",
            "structural\nGC user", "activity assay\nmentioned",
            "GC never\nmentioned"]
    x = np.arange(len(items))
    w = 0.26
    for k, c in enumerate(CH):
        vals = [pct(prow(it)[c]) for it in items]
        ax.bar(x + (k - 1) * w, vals, w, color=CHCOL[c], alpha=0.9,
               label=CHLAB[c])
        for xx, v in zip(x + (k - 1) * w, vals):
            ax.text(xx, v + 1.6, "%.0f" % v, ha="center", fontsize=6.2,
                    color=CHCOL[c])
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=6.3)
    ax.set_ylim(0, 108)
    ax.set_ylabel("% of patients", fontsize=8)
    ax.set_title("B  the three channels are three different populations",
                 fontsize=8.8, loc="left")
    ax.legend(frameon=False, fontsize=7.0, ncol=3, loc="upper center")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # --- C: mortality
    ax = fig.add_subplot(gs[1, 1])
    items = ["30-day mortality %", "in-hospital mortality %",
             "blood culture + (24h-7d) %", "new antimicrobial %"]
    labs = ["30-day\ndeath", "in-hospital\ndeath", "blood cx\n+ (24 h-7 d)",
            "new\nantimicrobial"]
    x = np.arange(len(items))
    for k, c in enumerate(CH):
        vals = [pct(prow(it)[c]) for it in items]
        ax.bar(x + (k - 1) * w, vals, w, color=CHCOL[c], alpha=0.9,
               label=CHLAB[c])
        for xx, v in zip(x + (k - 1) * w, vals):
            ax.text(xx, v + 0.9, "%.1f" % v, ha="center", fontsize=6.2,
                    color=CHCOL[c])
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=6.3)
    ax.set_ylim(0, 57)
    ax.set_ylabel("% of patients", fontsize=8)
    ax.set_title("C  deaths concentrate in the undocumented,\n"
                 "     not in the documented-inactive",
                 fontsize=8.8, loc="left")
    ax.legend(frameon=False, fontsize=7.0, ncol=3, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    cap(fig,
        "Channels are mutually exclusive (evidence-priority: explicit "
        "inactivity > explicit infection-related hold > otherwise).  "
        "'infection-deferred' does NOT mean 'should have been given but was "
        "not': %.0f%% of them received GC in the first 24 h."
        % pct(prow("GC given in first 24 h %")["INF_DEFER"]), y=-0.03)
    fig.suptitle("Figure 28  The 'no documented activity' group is not one "
                 "population but three", fontsize=10.2, x=0.008, ha="left",
                 y=1.02)
    save(fig, "Figure28_three_channels")


# ===================================================================== Fig 29
def fig29():
    fig = plt.figure(figsize=(11.2, 5.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.15], wspace=0.46)

    # --- A: note-length quartile vs mortality
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(NL))
    ax.bar(x - 0.19, NL.death_30d, 0.38, color=C_UDOC, alpha=0.9,
           label="30-day death")
    ax.bar(x + 0.19, NL.death_hosp, 0.38, color=C_SLE, alpha=0.9,
           label="in-hospital death")
    for xx, v in zip(x - 0.19, NL.death_30d):
        ax.text(xx, v + 0.4, "%.1f" % v, ha="center", fontsize=6.6, color=C_UDOC)
    for xx, v in zip(x + 0.19, NL.death_hosp):
        ax.text(xx, v + 0.4, "%.1f" % v, ha="center", fontsize=6.6, color=C_SLE)
    ax.set_xticks(x)
    ax.set_xticklabels(["Q1\nshortest", "Q2", "Q3", "Q4\nlongest"], fontsize=6.8)
    ax.set_ylim(0, 21)
    ax.set_ylabel("mortality (%)", fontsize=8)
    ax.set_xlabel("discharge-note length quartile", fontsize=8)
    ax.set_title("A  shorter note, higher mortality\n     (OR %.2f per +1000 chars)"
                 % (J["notelen_cont"]["OR"]), fontsize=8.8, loc="left")
    ax.legend(frameon=False, fontsize=7.2, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # --- B: died vs survived, characters
    ax = fig.add_subplot(gs[0, 1])
    two = DT[DT.variable.isin(["note chars, median",
                               "hospital-course chars, median"])]
    x = np.arange(len(two))
    ax.bar(x - 0.19, two.died_n, 0.38, color=C_UDOC, alpha=0.9, label="died")
    ax.bar(x + 0.19, two.survived_n, 0.38, color=GREY, alpha=0.9,
           label="survived")
    for xx, v in zip(x - 0.19, two.died_n):
        ax.text(xx, v + 260, "%.0f" % v, ha="center", fontsize=6.6, color=C_UDOC)
    for xx, v in zip(x + 0.19, two.survived_n):
        ax.text(xx, v + 260, "%.0f" % v, ha="center", fontsize=6.6, color="#555")
    ax.set_xticks(x)
    ax.set_xticklabels(["whole\ndischarge note", "hospital-course\nsection only"],
                       fontsize=6.8)
    ax.set_ylim(0, 15500)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: "%.0fk" % (v / 1000.0)))
    ax.set_ylabel("characters, median", fontsize=8)
    ax.set_title("B  the deficit is in the whole note,\n"
                 "     not in the clinical narrative", fontsize=8.8, loc="left")
    ax.legend(frameon=False, fontsize=7.2, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # --- C: died vs survived, physiology
    ax = fig.add_subplot(gs[0, 2])
    two = DT[DT.variable.isin(["ICU LOS days, median", "SOFA, median"])]
    x = np.arange(len(two))
    ax.bar(x - 0.19, two.died_n, 0.38, color=C_UDOC, alpha=0.9, label="died")
    ax.bar(x + 0.19, two.survived_n, 0.38, color=GREY, alpha=0.9,
           label="survived")
    for xx, v in zip(x - 0.19, two.died_n):
        ax.text(xx, v + 0.15, "%.1f" % v, ha="center", fontsize=7.0, color=C_UDOC)
    for xx, v in zip(x + 0.19, two.survived_n):
        ax.text(xx, v + 0.15, "%.1f" % v, ha="center", fontsize=7.0, color="#555")
    ax.set_xticks(x)
    ax.set_xticklabels(["ICU length of stay\n(days)", "SOFA\n(first 24 h)"],
                       fontsize=6.8)
    ax.set_ylim(0, 10.5)
    ax.set_ylabel("median", fontsize=8)
    ax.set_title("C  and the patients who died were\n"
                 "     sicker and stayed longer", fontsize=8.8, loc="left")
    ax.legend(frameon=False, fontsize=7.2, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    cap(fig,
        "A: the association is real and strong.  B/C: but it is NOT "
        "'the doctor failed to describe disease activity' -- the "
        "clinical-narrative section is the same length in both groups, "
        "while the patients who died had higher SOFA and LONGER ICU stay "
        "(Spearman note length vs ICU LOS = %.2f).  The deficit is in the "
        "templated post-discharge material (discharge medications, follow-up), "
        "i.e. informative missingness, not measurement failure."
        % DTX["sp_note_los"], y=-0.03)
    fig.suptitle("Figure 29  Is 'under-documentation' a recording artefact?",
                 fontsize=10.2, x=0.008, ha="left", y=1.02)
    save(fig, "Figure29_note_length_artefact")


# ===================================================================== Fig 30
KSR = {"spec-  (neg-fixed)": "spec- after negation fix (n=874)",
       "  . INACTIVE": "  . INACTIVE (true inactive)",
       "  . INF_DEFER": "  . INF_DEFER (infection)",
       "  . UNDERDOC": "  . UNDERDOC (undocumented)"}
KDIS = {"spec-  (neg-fixed)": "all spec-", "  . INACTIVE": ". INACTIVE",
        "  . INF_DEFER": ". INF_DEFER", "  . UNDERDOC": ". UNDERDOC"}


def fig30():
    S = {
        "v6 spec-  (raw)": srow("v6 spec- (raw, n=856)"),
        "spec-  (neg-fixed)": srow("spec- after negation fix (n=874)"),
        "v6 spec+  (raw)": srow("v6 spec+ (raw)"),
        "spec+  (neg-fixed)": srow("spec+ after negation fix"),
        "  . INACTIVE": srow("  . INACTIVE (true inactive)"),
        "  . INF_DEFER": srow("  . INF_DEFER (infection)"),
        "  . UNDERDOC": srow("  . UNDERDOC (undocumented)"),
    }
    ORDER = list(S.keys())

    def chcol(k):
        if k.startswith("  ."):
            return (C_UDOC if "UNDERDOC" in k else
                    C_DEFER if "DEFER" in k else C_INACT)
        return C_SLE if "+" not in k else GREY

    fig = plt.figure(figsize=(13.4, 5.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 1.10, 1.02], wspace=1.00)

    # --- A: interaction forest
    ax = fig.add_subplot(gs[0, 0])
    rows = []
    nrow = 0.0
    ymin = 0.0
    for k in ORDER:
        r = S[k]
        if r.ix_or is not None and np.isfinite(r.ix_or):
            rows.append(dict(y=nrow, OR=r.ix_or, lo=max(r.ix_lo, 0.03),
                             hi=min(r.ix_hi, 60.0), col=chcol(k), ms=6.5))
            ymin = min(ymin, nrow)
        nrow -= 1
    forest(ax, rows, (0.03, 60.0), [0.05, 0.1, 0.5, 1, 2, 10, 50],
           "RA/SLE ratio of dose-trend ORs for 30-day death",
           ylim=(ymin - 0.65, 1.15))
    for i, k in enumerate(ORDER):
        r, y, col = S[k], -i, chcol(k)
        if r.ix_or is not None and np.isfinite(r.ix_or):
            rt(ax, 1.03, y, "%.2f (%.2f\u2013%.2f)"
               % (r.ix_or, r.ix_lo, r.ix_hi), col)
        else:
            rt(ax, 1.03, y, "not estimable", "#999")
        rt(ax, -0.025, y, k, "#111", 7.2, bold=k.startswith("  ."), ha="right")
    rt(ax, 1.03, 0.90, "RA/SLE (95% CI)", "#333", 6.8, bold=True)
    ax.set_title("A  splitting spec- moves the interaction\n"
                 "     into the undocumented channel", fontsize=8.8, loc="left")

    # --- B: per-disease trends
    ax = fig.add_subplot(gs[0, 1])
    ks = list(KSR.keys())
    rows2 = []
    nrow = 0.0
    ymin = 0.0
    for k in ks:
        r = S[k]
        for g, col, mk, o, lo, hi in [
                ("SLE", C_SLE, "o", r.sle_or, r.sle_lo, r.sle_hi),
                ("RA", C_RA, "s", r.ra_or, r.ra_lo, r.ra_hi)]:
            if o is not None and np.isfinite(o):
                rows2.append(dict(y=nrow, OR=o, lo=max(lo, 0.02),
                                  hi=min(hi, 40.0), col=col, marker=mk, ms=6))
                ymin = min(ymin, nrow)
            nrow -= 1
        nrow -= 0.7
    forest(ax, rows2, (0.02, 40.0), [0.05, 0.1, 0.5, 1, 2, 10, 40],
           "dose-trend OR for 30-day death", ylim=(ymin - 0.65, 1.15))
    nrow = 0.0
    for k in ks:
        r = S[k]
        for g, col, o, lo, hi in [("SLE", C_SLE, r.sle_or, r.sle_lo, r.sle_hi),
                                  ("RA", C_RA, r.ra_or, r.ra_lo, r.ra_hi)]:
            if o is not None and np.isfinite(o):
                rt(ax, 1.04, nrow, "%.2f (%.2f\u2013%.2f)" % (o, lo, hi), col)
            else:
                rt(ax, 1.04, nrow, "%s not estimable" % g, col)
            nrow -= 1
        rt(ax, -0.02, nrow + 2.62, KDIS[k], "#111", 7.0, bold=True, ha="right")
        nrow -= 0.7
    rt(ax, 1.04, 0.90, "OR (95% CI)", "#333", 6.8, bold=True)
    ax.set_title("B  RA trends up, SLE trends down\n"
                 "     in the undocumented channel too", fontsize=8.8, loc="left")

    # --- C: positive control in the same channels
    ax = fig.add_subplot(gs[0, 2])
    rows3 = []
    nrow = 0.0
    ymin = 0.0
    for k in ks:
        for g in ["SLE", "RA"]:
            sub = ST[(ST.stratum == KSR[k]) & (ST.outcome == "hyper_48h") &
                     (ST.disease == g)]
            if len(sub):
                sub = sub.iloc[0]
                rows3.append(dict(y=nrow, OR=sub.or_, lo=max(sub.lo, 0.1),
                                  hi=min(sub.hi, 12.0),
                                  col=C_SLE if g == "SLE" else C_RA, ms=6,
                                  marker="o" if g == "SLE" else "s"))
                ymin = min(ymin, nrow)
            nrow -= 1
        nrow -= 0.7
    forest(ax, rows3, (0.1, 12.0), [0.2, 0.5, 1, 2, 4, 10],
           "dose-trend OR for glucose \u2265180 (POSITIVE CONTROL)",
           ylim=(ymin - 0.65, 1.15))
    nrow = 0.0
    for k in ks:
        for g, col in [("SLE", C_SLE), ("RA", C_RA)]:
            sub = ST[(ST.stratum == KSR[k]) & (ST.outcome == "hyper_48h") &
                     (ST.disease == g)]
            if len(sub):
                sub = sub.iloc[0]
                rt(ax, 1.04, nrow, "%.2f (%.2f\u2013%.2f)" % (sub.or_, sub.lo,
                                                              sub.hi), col)
            nrow -= 1
        rt(ax, -0.02, nrow + 2.62, KDIS[k], "#111", 7.0, bold=True, ha="right")
        nrow -= 0.7
    rt(ax, 1.04, 0.90, "OR (95% CI)", "#333", 6.8, bold=True)
    ax.set_title("C  and the pharmacological control\n"
                 "     still works in every channel", fontsize=8.8, loc="left")

    cap(fig,
        "Markers: circles = SLE, squares = RA.  INACTIVE (%.0f events) and "
        "INF_DEFER (%.0f event) cannot support an interaction estimate; this is "
        "reported, not hidden.  The negation-fixed and raw spec definitions "
        "give the same answer (%d of %d 'activity-positive' admissions were "
        "purely negated, e.g. 'no evidence of active lupus')."
        % (srow("  . INACTIVE (true inactive)").n_ev,
           srow("  . INF_DEFER (infection)").n_ev,
           J["n_neg_falsepos"], J["n_rawpos"]), y=-0.03)
    fig.suptitle("Figure 30  Where the death interaction actually lives",
                 fontsize=10.2, x=0.008, ha="left", y=1.02)
    save(fig, "Figure30_interaction_location")


# ===================================================================== Fig 31
def fig31():
    LAB = {"CORE (v6 baseline)": "CORE",
           "CORE + note length (/1000 char)": "+ whole-note len",
           "CORE + HC-section length (/1000)": "+ narrative len",
           "CORE + note len + structural GC user": "+ note len + GC user",
           "UNDERDOC: CORE": "UD: CORE",
           "UNDERDOC: CORE + note length": "UD: + note len",
           "UNDERDOC: CORE + HC-section length": "UD: + narrative",
           "UNDERDOC: CORE + note len + structural GC user": "UD: + note + GC"}

    fig = plt.figure(figsize=(13.4, 5.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.05, 1.25], wspace=1.15)

    # --- A: length-adjusted interaction
    ax = fig.add_subplot(gs[0, 0])
    rows = []
    nrow = 0.0
    ymin = 0.0
    for _, r in LA.iterrows():
        if r.adjustment not in LAB:
            continue
        col = C_UDOC if r.adjustment.startswith("UNDERDOC") else C_SLE
        if r.adjustment == "CORE + note length (/1000 char)":
            col = C_DEFER
        rows.append(dict(y=nrow, OR=r.or_, lo=max(r.lo, 0.1), hi=min(r.hi, 20.0),
                         col=col, ms=6.5))
        ymin = min(ymin, nrow)
        nrow -= 1
        if r.adjustment == "CORE + note len + structural GC user":
            nrow -= 0.55
    forest(ax, rows, (0.1, 20.0), [0.2, 0.5, 1, 2, 5, 10, 20],
           "RA/SLE ratio of dose-trend ORs for death",
           ylim=(ymin - 0.65, 1.15))
    nrow = 0.0
    for _, r in LA.iterrows():
        if r.adjustment not in LAB:
            continue
        col = C_UDOC if r.adjustment.startswith("UNDERDOC") else C_SLE
        if r.adjustment == "CORE + note length (/1000 char)":
            col = C_DEFER
        rt(ax, 1.03, nrow, "%.2f (%.2f\u2013%.2f)"
           % (r.or_, r.lo, r.hi), col)
        rt(ax, -0.025, nrow, LAB[r.adjustment], "#111", 7.0, ha="right")
        nrow -= 1
        if r.adjustment == "CORE + note len + structural GC user":
            nrow -= 0.55
    rt(ax, 1.03, 0.90, "RA/SLE (95% CI)", "#333", 6.8, bold=True)
    ax.set_title("A  adjusting for recording intensity\n"
                 "     strengthens, never explains, the interaction",
                 fontsize=8.8, loc="left")

    # --- B: landmark
    ax = fig.add_subplot(gs[0, 1])
    lm = LA[LA.adjustment.str.startswith("landmark")]
    items = []
    for _, r in lm.iterrows():
        lab = r.adjustment.replace("landmark: ", "").split(" (")[0]
        lab = lab.replace("ICU LOS >= 3 d", "LOS \u2265 3 d")
        lab = lab.replace(" + note length adj", " + note len")
        lab = lab.replace(" + note len adj", " + note len")
        lab = "%s  (%d/%d)" % (lab, r.n_ev, r.n)
        items.append((lab, r.or_, r.lo, r.hi, r.p_lrt, r.n_ev, r.n,
                      C_UDOC if ">= 0" in lab else C_SLE))
    rows2 = []
    nrow = 0.0
    for lab, o, lo, hi, p, ev, n, col in items:
        if o is not None and np.isfinite(o):
            rows2.append(dict(y=nrow, OR=o, lo=max(lo, 0.2), hi=min(hi, 25.0),
                              col=col, ms=6.5))
        nrow -= 1
    forest(ax, rows2, (0.2, 25.0), [0.2, 0.5, 1, 2, 5, 10, 20],
           "RA/SLE ratio, 30-day death", ylim=(-len(items) - 0.35, 1.15))
    nrow = 0.0
    for lab, o, lo, hi, p, ev, n, col in items:
        if o is not None and np.isfinite(o):
            rt(ax, 1.04, nrow, "%.2f (%.2f\u2013%.2f)" % (o, lo, hi), col)
            rt(ax, 1.58, nrow, "%.3f" % p, "#555")
        rt(ax, -0.02, nrow, lab, "#111", 6.9, ha="right")
        nrow -= 1
    rt(ax, 1.04, 0.90, "RA/SLE (95% CI)", "#333", 6.8, bold=True)
    rt(ax, 1.58, 0.90, "P (LRT)", "#333", 6.8, bold=True)
    ax.set_title("B  landmark: restricting to LOS \u2265 3 d\n"
                 "     halves the events and blurs it", fontsize=8.8, loc="left")

    # --- C: D3 self-check
    ax = fig.add_subplot(gs[0, 2])
    OSHORT = {"30-day death": "30-d death",
              "in-hospital death": "in-hosp",
              "blood culture +": "blood cx",
              "new antimicrobial": "new abx"}
    SSHORT = {"spec- (fixed)": "spec-", "all with note": "all"}
    xmax = 0.12
    for _, r in D3.iterrows():
        xmax = max(xmax, r.rel_diff / 100.0 + 0.045)
    ax.set_xlim(-0.02, xmax)
    ax.set_ylim(-len(D3) + 0.35, 1.45)
    yt, yl = [], []
    for i, (_, r) in enumerate(D3.iterrows()):
        y = -i
        rel = r.rel_diff / 100.0
        col = C_NEG if r.agree else C_DEFER
        ax.plot([0, rel], [y, y], color=col, lw=1.5)
        ax.plot([rel], [y], marker="o", ms=5.5, color=col, mec="white",
                mew=0.7)
        ax.annotate("%.1f%%%s" % (r.rel_diff, "" if r.agree else "  CHECK"),
                    xy=(rel + 0.004, y), fontsize=6.3, va="center", ha="left",
                    color=col, annotation_clip=False)
        yt.append(y)
        yl.append("%s | %s" % (OSHORT.get(r.outcome, r.outcome),
                               SSHORT.get(r.stratum, r.stratum)))
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=6.3)
    ax.axvline(0.10, color="#888", lw=0.9, ls=":")
    ax.annotate("10% = noise tolerance", xy=(0.104, 0.85), fontsize=6.5,
                va="center", ha="left", color="#666", annotation_clip=False)
    ax.set_xlabel("relative difference, interaction vs empirical trend ratio",
                  fontsize=7.6)
    ax.set_title("C  D3 self-check: every non-zero\n"
                 "     effect agrees in direction", fontsize=8.8, loc="left")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.annotate("the single CHECK is a zero effect (1.01 vs 0.98)",
                xy=(0.0, -0.30), xycoords="axes fraction", fontsize=6.5,
                color="#666", annotation_clip=False)

    _lm = LA[LA.adjustment.str.contains("landmark") &
             LA.adjustment.str.contains("note len")]
    _lmv = _lm.iloc[0] if len(_lm) else None
    cap(fig,
        "A+B: the interaction resists adjustment for how much was written, "
        "but does not survive the loss of power from a 3-day landmark "
        "(the landmark model that also adjusts for note length is back at "
        "%s, P=%s, though with a very wide CI).  "
        "C: the direction self-check passes for every non-zero effect."
        % (("%.2f" % _lmv.or_) if _lmv is not None else "n/a",
           ("%.3f" % _lmv.p_lrt) if _lmv is not None else "n/a"), y=-0.03)
    fig.suptitle("Figure 31  Robustness of the decomposed interaction",
                 fontsize=10.2, x=0.008, ha="left", y=1.02)
    save(fig, "Figure31_robustness")


# ===================================================================== Fig 32
def fig32():
    fig = plt.figure(figsize=(12.4, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.60)

    GLAB = {"INACTIVE": "truly\ninactive", "INF_DEFER": "infection\ndeferred",
            "UNDERDOC": "undocumented"}
    # --- A: G0 composition
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(G0))
    ax.bar(x - 0.19, G0.n_sle, 0.38, color=C_SLE, alpha=0.9, label="SLE")
    ax.bar(x + 0.19, G0.n_ra, 0.38, color=C_RA, alpha=0.9, label="RA")
    for xx, v in zip(x - 0.19, G0.n_sle):
        ax.text(xx, v + 8, "%d" % v, ha="center", fontsize=6.8, color=C_SLE)
    for xx, v in zip(x + 0.19, G0.n_ra):
        ax.text(xx, v + 8, "%d" % v, ha="center", fontsize=6.8, color=C_RA)
    ax.set_xticks(x)
    ax.set_xticklabels([GLAB.get(c, CHLAB[c]) for c in G0.channel], fontsize=6.5)
    ax.set_ylim(0, 500)
    ax.set_ylabel("patients (no GC in first 24 h)", fontsize=8)
    ax.set_title("A  the reference group, decomposed\n     (n = %d)"
                 % int(G0.n.sum()), fontsize=8.8, loc="left")
    ax.legend(frameon=False, fontsize=7.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # --- B: G0 mortality
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(x, G0.death_30d, 0.5, color=[CHCOL[c] for c in G0.channel], alpha=0.9)
    for xx, v in zip(x, G0.death_30d):
        ax.text(xx, v + 0.2, "%.1f%%" % v, ha="center", fontsize=7.0, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels([GLAB.get(c, CHLAB[c]) for c in G0.channel], fontsize=6.5)
    ax.set_ylim(0, 11)
    ax.set_ylabel("30-day mortality (%)", fontsize=8)
    ax.set_title("B  and where the deaths in the\n"
                 "     reference group come from", fontsize=8.8, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # --- C: IPTW
    ax = fig.add_subplot(gs[0, 2])
    items = [("spec- all", "SLE", C_SLE, "o"), ("spec- all", "RA", C_RA, "s"),
             ("  UNDERDOC", "SLE", C_SLE, "o"), ("  UNDERDOC", "RA", C_RA, "s")]
    rows = []
    nrow = 0.0
    for st, g, col, mk in items:
        r = IW[(IW.stratum == st) & (IW.disease == g)]
        if not len(r):
            nrow -= 1
            continue
        r = r.iloc[0]
        rows.append(dict(y=nrow, OR=r.or_, lo=max(r.lo, 0.2),
                         hi=min(r.hi, 10.0), col=col, marker=mk, ms=6.5))
        nrow -= 1
    full = {"spec- all": "spec- all  (all 874)", "  UNDERDOC": "UNDERDOC  (%d)"
            % int(IW[IW.stratum == "  UNDERDOC"].n.max())}
    forest(ax, rows, (0.2, 10.0), [0.2, 0.5, 1, 2, 5, 10],
           "IPTW dose-trend OR (per stratum increment)", ylim=(-4.5, 1.15))
    nrow = 0.0
    for st, g, col, mk in items:
        r = IW[(IW.stratum == st) & (IW.disease == g)]
        if not len(r):
            nrow -= 1
            continue
        r = r.iloc[0]
        rt(ax, 1.04, nrow, "%.2f (%.2f\u2013%.2f)  P=%.3f" % (r.or_, r.lo, r.hi,
                                                              r.p), col)
        if g == "SLE":
            rt(ax, -0.025, nrow - 0.45, full[st], "#111", 7.2, bold=True,
               ha="right")
        nrow -= 1
    rt(ax, 1.04, 0.90, "OR (95% CI), P", "#333", 6.8, bold=True)
    ax.set_title("C  weighted model: the RA trend persists\n"
                 "     inside the undocumented channel",
                 fontsize=8.8, loc="left")

    _ud = G0[G0.channel == "UNDERDOC"].iloc[0]
    _in = G0[G0.channel == "INACTIVE"].iloc[0]
    _udx = G0X[G0X.channel == "UNDERDOC"].iloc[0]
    cap(fig,
        "The reference group is dominated by %d undocumented admissions "
        "(%.1f%% mortality; %d RA deaths vs %d SLE deaths); only %d patients "
        "are documented as truly inactive and they carry a %.1f%% mortality.  "
        "Markers: circles = SLE, squares = RA."
        % (_ud.n, _ud.death_30d, _udx.ra_died, _udx.sle_died, _in.n,
           _in.death_30d), y=-0.03)
    fig.suptitle("Figure 32  The reference group after decomposition, and the "
                 "weighted check", fontsize=10.2, x=0.008, ha="left", y=1.02)
    save(fig, "Figure32_g0_and_iptw")


if __name__ == "__main__":
    for f in (fig28, fig29, fig30, fig31, fig32):
        f()
    print("done")
