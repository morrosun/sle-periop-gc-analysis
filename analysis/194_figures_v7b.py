# -*- coding: utf-8 -*-
"""Figures for the v7b report (blinded double-coding audit of the v7 channels).

  Figure 33  how well the algorithm agrees with blinded human coding
  Figure 34  what the audit does to the v7 numbers once fed back in

Layout rules inherited from 190/192 (they prevented every collision seen):
  * explicit data coordinates, never a running counter
  * labels that must sit outside the axes are placed by axes-fraction x
  * captions are hard-wrapped (a 400-char single line blows up the tight bbox)
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
C_ALG, C_HUM = "#8E44AD", "#1F7A8C"

J = json.load(open(os.path.join(OUT, "_v7b_results.json"), encoding="utf-8"))
E = json.load(open(os.path.join(OUT, "_v7b_est.json"), encoding="utf-8"))

KAP = pd.DataFrame(J["kappa"])
PER = pd.DataFrame(J["perclass"])
CNF = pd.DataFrame(J["confusion"])
CONF = pd.DataFrame(J["confirm"])
TRU = pd.DataFrame(E["truth"])
TRG = pd.DataFrame(E["triggers"])
MOR = pd.DataFrame(E["mortality"])

CH = ["INACTIVE", "INF_DEFER", "UNDERDOC"]
CHLAB = {"INACTIVE": "truly inactive", "INF_DEFER": "infection-deferred",
         "UNDERDOC": "undocumented"}
CHCOL = {"INACTIVE": C_INACT, "INF_DEFER": C_DEFER, "UNDERDOC": C_UDOC}
TICK = {"INACTIVE": "truly\ninactive", "INF_DEFER": "infection-\ndeferred",
        "UNDERDOC": "undocumented"}
BLAB = {"ACTIVITY-POSITIVE": "activity-positive", "PURELY-NEGATED": "purely negated"}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, "%s.%s" % (name, ext)), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def rt(ax, xf, y, s, color="#333", size=6.8, bold=False, ha="left"):
    ax.annotate(str(s), xy=(xf, y), xycoords=("axes fraction", "data"),
                fontsize=size, va="center", ha=ha, color=color,
                clip_on=False, annotation_clip=False,
                fontweight="bold" if bold else None)


def cap(fig, text, y=-0.015, width=165, size=7.4):
    fig.text(0.008, y, "\n".join(textwrap.wrap(" ".join(text.split()), width)),
             fontsize=size, color="#444", ha="left", va="top")


def clean(ax, sides=("top", "right")):
    for s in sides:
        ax.spines[s].set_visible(False)


# ===================================================================== Fig 33
def fig33():
    fig = plt.figure(figsize=(12.8, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.00, 1.05, 1.18], wspace=0.52)

    # ---------------------------------------------------------- A: confusion
    ax = fig.add_subplot(gs[0, 0])
    ctab = (CNF[CNF.task == "A"].pivot(index="algo", columns="ref", values="n")
            .reindex(index=CH, columns=CH).fillna(0).astype(int))
    cm = ctab.values.astype(float)
    vmax = cm.max()
    ax.imshow(np.zeros_like(cm), cmap="Greys", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            v = cm[i, j]
            frac = v / vmax
            col = "#B03A2E" if i != j else "#2F7A4F"
            ax.add_patch(plt.Rectangle((j - .48, i - .48), .96, .96,
                                       facecolor=col,
                                       alpha=0.10 + 0.55 * frac, lw=0))
            ax.text(j, i, "%d" % v, ha="center", va="center", fontsize=10.5,
                    color="#111" if frac < .62 else "white",
                    fontweight="bold" if i == j else None)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([TICK[c] for c in CH], fontsize=7.2)
    ax.set_yticklabels([TICK[c] for c in CH], fontsize=7.2)
    ax.set_xlabel("blinded human reference (task A)", fontsize=8.2)
    ax.set_ylabel("algorithm channel", fontsize=8.2)
    ax.set_xlim(-.5, 2.5)
    ax.set_ylim(2.5, -.5)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    PPVi = {r.channel: r.ppv for _, r in TRU.iterrows()}
    KA = KAP[(KAP.task == "A") & (KAP.pair == "Alg vs R1")].iloc[0]
    ax.set_title("A  only the diagonal should be occupied\n"
                 "     accuracy %.2f vs chance %.2f  (\u03ba %.2f)"
                 % (KA.agreement, KA.expected, KA.kappa),
                 fontsize=8.9, loc="left")

    # ---------------------------------------------------------- B: kappa
    ax = fig.add_subplot(gs[0, 1])
    pairs = [("R1 vs R2", "human vs human", C_HUM, "o"),
             ("Alg vs R1", "algorithm vs R1", C_ALG, "s"),
             ("Alg vs R2", "algorithm vs R2", C_ALG, "s")]
    y = 0.0
    for task, tcol in [("A", C_SLE), ("B", C_RA)]:
        rt(ax, -0.03, y + 0.62, "task %s" % task, "#111", 8.0, bold=True,
           ha="right")
        y -= 0.95
        for pr, lab, col, mk in pairs:
            r = KAP[(KAP.task == task) & (KAP.pair == pr)]
            if not len(r):
                continue
            r = r.iloc[0]
            ax.barh(y, r.kappa, height=0.62, color=col, alpha=0.88)
            lo = max(r.k_lo, -0.30)
            ax.plot([lo, r.k_hi], [y, y], color="#333", lw=0.9)
            ax.plot([lo, r.k_hi], [y, y], "|", color="#333", ms=3.4)
            ax.text(min(r.k_hi, 1.06) + 0.03, y,
                    "%.2f (%.2f\u2013%.2f)" % (r.kappa, r.k_lo, r.k_hi),
                    fontsize=6.7, va="center",
                    color="#111" if r.kappa > 0.4 else "#B03A2E")
            y -= 1.0
        y -= 0.45
    ax.axvline(0.0, color="#888", lw=0.9, ls="--")
    ax.axvspan(-0.30, 0.20, color="#B03A2E", alpha=0.06, lw=0)
    ax.text(0.005, 1.32, "\u2264 0.20  poor", fontsize=6.6, color="#8B2E24")
    ax.text(0.635, 1.32, "\u2265 0.61  substantial", fontsize=6.6, color="#2F7A4F")
    ax.set_xlim(-0.30, 1.32)
    ax.set_ylim(y + 0.55, 1.85)
    ax.set_yticks([])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Cohen's \u03ba (95% bootstrap CI)", fontsize=8.2)
    clean(ax, ("top", "right", "left"))
    ax.set_title("B  humans agree with each other far better\n"
                 "     than the algorithm agrees with either", fontsize=8.9,
                 loc="left")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C_HUM,
                              label="human vs human", mec="white"),
                       Line2D([], [], marker="s", ls="", color=C_ALG,
                              label="algorithm vs human", mec="white")],
              frameon=False, fontsize=7.2, loc="lower right")

    # ---------------------------------------------------------- C: PPV
    ax = fig.add_subplot(gs[0, 2])
    items = ([("A", c) for c in CH] +
             [("B", c) for c in ["ACTIVITY-POSITIVE", "PURELY-NEGATED"]])
    y = 0.0
    for task, lab in items:
        r = CONF[(CONF.task == task) & (CONF.algo_label == lab)]
        if not len(r):
            continue
        r = r.iloc[0]
        col = (CHCOL.get(lab, C_ALG) if task == "A" else
               C_POS if lab == "ACTIVITY-POSITIVE" else C_NEG)
        ax.barh(y, 100 * r.rate, height=0.62, color=col, alpha=0.88)
        ax.plot([100 * r.lo, 100 * r.hi], [y, y], color="#333", lw=0.9)
        ax.plot([100 * r.lo, 100 * r.hi], [y, y], "|", color="#333", ms=3.4)
        ax.text(103.5, y, "%.1f%%  (%d/%d)" % (100 * r.rate, r.confirmed, r.n),
                fontsize=6.9, va="center",
                color="#B03A2E" if r.rate < 0.5 else "#2F7A4F")
        lab2 = CHLAB[lab] if lab in CHLAB else BLAB[lab]
        rt(ax, -0.03, y, lab2, "#111", 7.3, ha="right")
        y -= 1.0
        if lab == "UNDERDOC":
            y -= 0.55
    ax.set_xlim(0, 100)
    ax.set_ylim(y + 0.6, 1.5)
    ax.set_yticks([])
    ax.set_xlabel("confirmed by blinded reference (%)", fontsize=8.2)
    clean(ax, ("top", "right", "left"))
    ax.set_title("C  what each algorithm label actually means\n"
                 "     (positive predictive value of the label)",
                 fontsize=8.9, loc="left")

    cap(fig,
        "Task A is the 3-way split of the 'no specific activity' stratum "
        "(n=98: 35 algorithm-INACTIVE, 23 algorithm-INF_DEFER, 40 "
        "algorithm-UNDERDOC). Task B is the negation rule (n=50). Raters were "
        "two independent blinded coders working from the same codebook; the "
        "single disagreement was resolved by a third adjudicator. The "
        "algorithm agrees with the blinded reference on 0.46 of admissions "
        "against a chance agreement of 0.40 (kappa 0.10) -- it is perfectly "
        "reproducible, but it is not measuring what the coders measured.",
        y=-0.03)
    fig.suptitle("Figure 33  Blinded double-coding audit of the v7 channels",
                 fontsize=10.4, x=0.008, ha="left", y=1.015)
    save(fig, "Figure33_blinded_audit")


# ===================================================================== Fig 34
def fig34():
    fig = plt.figure(figsize=(12.8, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.00, 1.15], wspace=0.46)

    # ---------------------------------------------------------- A: sizes
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(3)
    nom = TRU.nominal.values.astype(float)
    est = TRU.est.values.astype(float)
    lo = TRU.lo.values.astype(float)
    hi = TRU.hi.values.astype(float)
    ax.bar(x - 0.20, nom, 0.38, color=GREY, alpha=0.85, label="v7 nominal")
    ax.bar(x + 0.20, est, 0.38, color=[CHCOL[c] for c in CH], alpha=0.9,
           label="audit-corrected")
    ax.errorbar(x + 0.20, est, yerr=[est - np.clip(lo, 0, None),
                                     np.clip(hi, 0, 900) - est],
                fmt="none", ecolor="#333", elinewidth=0.9, capsize=2.4)
    for xx, v in zip(x - 0.20, nom):
        ax.text(xx, v + 16, "%d" % v, ha="center", fontsize=7.0, color="#555",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    for xx, v in zip(x + 0.20, est):
        ax.text(xx, v + 16, "%.0f" % v, ha="center", fontsize=7.0,
                color="#111", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.set_yscale("log")
    ax.set_ylim(1, 2600)
    ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(mticker.FixedFormatter(["1", "10", "100",
                                                        "1000"]))
    ax.set_xticks(x)
    ax.set_xticklabels([TICK[c] for c in CH], fontsize=7.4)
    ax.set_ylabel("admissions (log scale)", fontsize=8.2)
    ax.set_title("A  the two small channels shrink by half\n"
                 "     to three quarters once corrected", fontsize=8.9,
                 loc="left")
    ax.legend(frameon=False, fontsize=7.2, loc="upper right")
    clean(ax)

    # ---------------------------------------------------------- B: mortality
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(3)
    m7 = MOR.mort_v7.values.astype(float)
    me = MOR.mort_est.values.astype(float)
    ax.bar(x - 0.20, m7, 0.38, color=GREY, alpha=0.85, label="v7 nominal")
    ax.bar(x + 0.20, me, 0.38, color=[CHCOL[c] for c in CH], alpha=0.9,
           label="audit-corrected")
    mlo = MOR.lo.values.astype(float)
    mhi = MOR.hi.values.astype(float)
    ax.errorbar(x + 0.20, me, yerr=[me - np.clip(mlo, 0, None),
                                    np.clip(mhi, 0, 60) - me],
                fmt="none", ecolor="#333", elinewidth=0.9, capsize=2.4)
    for xx, v in zip(x - 0.20, m7):
        ax.text(xx, v + 0.35, "%.1f" % v, ha="center", fontsize=7.0, color="#555",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    for xx, v in zip(x + 0.20, me):
        ax.text(xx, v + 0.35, "%.1f" % v, ha="center", fontsize=7.0,
                color="#111", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.set_xticks(x)
    ax.set_xticklabels([TICK[c] for c in CH], fontsize=7.4)
    ax.set_ylabel("30-day mortality (%)", fontsize=8.2)
    ax.set_ylim(0, 22)
    ax.set_title("B  and the mortality gradient between the\n"
                 "     channels largely dissolves", fontsize=8.9, loc="left")
    ax.legend(frameon=False, fontsize=7.2, loc="upper right")
    clean(ax)
    ax.text(1.20, 1.4, "no deaths among\n~6 people", fontsize=6.4,
            color="#8B2E24", ha="center", va="bottom")

    # ---------------------------------------------------------- C: triggers
    ax = fig.add_subplot(gs[0, 2])
    t = TRG[TRG.scope == "all 856"].copy()
    t = t.sort_values("n", ascending=True)
    y = 0.0
    for _, r in t.iterrows():
        ax.barh(y, r.n, height=0.62, color=C_UDOC, alpha=0.85)
        ax.text(r.n + 0.9, y, "%d" % r.n, fontsize=7.0, va="center",
                color="#111")
        rt(ax, -0.03, y, r.trigger, "#111", 7.4, ha="right")
        y -= 1.0
    ax.set_xlim(0, 52)
    ax.set_ylim(y + 0.6, 1.5)
    ax.set_yticks([])
    ax.set_xlabel("algorithm-INACTIVE calls\nin the whole stratum (n=54)",
                  fontsize=7.8)
    clean(ax, ("top", "right", "left"))
    ax.set_title("C  one bare word drives almost all of it\n"
                 "     (81% of 'inactive' calls = 'asymptomatic')",
                 fontsize=8.9, loc="left")

    cap(fig,
        "Panel A/B replay the v7 decomposition after replacing the algorithm "
        "labels with the blinded reference. The 'truly inactive' arm falls "
        "from 57 to about 24 admissions (95% CI 1.5-66) and its mortality "
        "rises from 3.5% to 6.4%, while 'undocumented' is essentially "
        "unchanged at 9.6%. Panel C shows the mechanism: one bare keyword, "
        "applied with no negation and no domain check, generated four fifths "
        "of the smallest channel. 'asymptomatic' fires on non-rheumatologic "
        "findings (asymptomatic AFib, asymptomatic bacteriuria, an "
        "asymptomatic ovarian cyst); 'remission' fires on lung and bladder "
        "cancer; 'free of disease' fires on a coronary angiogram report "
        "('the RCA was free of disease').",
        y=-0.03)
    fig.suptitle("Figure 34  Feeding the audit back into the v7 decomposition",
                 fontsize=10.4, x=0.008, ha="left", y=1.015)
    save(fig, "Figure34_audit_corrected_v7")


if __name__ == "__main__":
    print("figures v7b")
    fig33()
    fig34()
