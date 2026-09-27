# -*- coding: utf-8 -*-
"""Figures for the v8 report (three-rule keyword engine + continuous density).

  Figure 35  the three rules take the classifier from near-chance to usable
  Figure 36  the continuous activity-information density replaces the 3-way split

Layout rules inherited from 190/192/194 (they prevented every collision seen):
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
from matplotlib.ticker import FixedLocator, NullLocator
from sklearn.metrics import roc_curve, roc_auc_score

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

C_INACT, C_DEFER, C_UDOC = "#2F7A4F", "#B7791F", "#B03A2E"
C_V7, C_V8 = "#9AA0A6", "#2B6CB0"
C_RULE = "#8E44AD"
GREY = "#9AA0A6"

J = json.load(open(os.path.join(OUT, "_v8.json"), encoding="utf-8"))
PPV = pd.DataFrame(J["ppv"])
PPVB = pd.DataFrame(J["ppv_B"])
AUC = pd.DataFrame(J["auc"])
IX = pd.DataFrame(J["ix"])

CH = ["INACTIVE", "INF_DEFER", "UNDERDOC"]
CHLAB = {"INACTIVE": "truly inactive", "INF_DEFER": "infection-deferred",
         "UNDERDOC": "undocumented"}
CHCOL = {"INACTIVE": C_INACT, "INF_DEFER": C_DEFER, "UNDERDOC": C_UDOC}

gold = pd.read_csv(os.path.join(OUT, "adjud_merged.csv"))
gold["stay_key"] = gold.stay_key.astype(int)
ad = pd.read_csv(os.path.join(DATA, "gc_activity_density.csv"))
gold = gold.merge(ad, on="stay_key", how="left")
gold["v8_activity_pos"] = (gold.v8_act_pos.fillna(0) > 0).astype(int)
A = gold[gold.task == "A"].copy()
B = gold[gold.task == "B"].copy()
A["human_documented"] = A.ref_std.isin(["INACTIVE", "INF_DEFER"]).astype(int)
B["human_act"] = (B.ref_std == "ACTIVITY-POSITIVE").astype(int)


def cap(fig, s, y=-0.02, width=168):
    fig.text(0.5, y, "\n".join(textwrap.wrap(s, width)), ha="center",
             va="top", fontsize=7.2, color="#444444")


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, "%s.%s" % (name, ext)), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def rt(ax, xf, y, s, color="#333", size=6.8, bold=False, ha="left"):
    ax.annotate(str(s), xy=(xf, y), xycoords=("axes fraction", "data"),
                fontsize=size, va="center", ha=ha, color=color,
                fontweight="bold" if bold else "normal",
                clip_on=False, annotation_clip=False)


def rtx(ax, x, yf, s, color="#333", size=6.8, bold=False, ha="center"):
    ax.annotate(str(s), xy=(x, yf), xycoords=("data", "axes fraction"),
                fontsize=size, va="center", ha=ha, color=color,
                fontweight="bold" if bold else "normal",
                clip_on=False, annotation_clip=False)


# ==================================================================== Figure 35
def fig35():
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5),
                             gridspec_kw=dict(wspace=0.42, left=0.06,
                                              right=0.99, top=0.84, bottom=0.24))

    # ---- A: PPV by class, v7 vs v8
    ax = axes[0]
    x = np.arange(3)
    wdt = 0.34
    for i, (eng, col, off) in enumerate([("v7", C_V7, -wdt / 2),
                                         ("v8", C_V8, +wdt / 2)]):
        vals, los, his = [], [], []
        for c in CH:
            r = PPV[(PPV.engine == eng) & (PPV.task == "A") & (PPV.cls == c)]
            if len(r) == 0 or not np.isfinite(r.iloc[0].ppv):
                vals.append(0.0)
                los.append(0.0)
                his.append(0.0)
            else:
                v = float(r.iloc[0].ppv)
                vals.append(v * 100)
                los.append(max(0.0, v - float(r.iloc[0].ppv_lo)) * 100)
                his.append(max(0.0, float(r.iloc[0].ppv_hi) - v) * 100)
        ax.bar(x + off, vals, wdt, color=col, edgecolor="white", linewidth=0.6,
               label=eng.upper(), yerr=[los, his],
               error_kw=dict(ecolor="#444", lw=0.8, capsize=2.2))
        for xi, v, c in zip(x + off, vals, CH):
            n = int(PPV[(PPV.engine == eng) & (PPV.task == "A") &
                        (PPV.cls == c)].iloc[0].n_algo_pos) if len(
                PPV[(PPV.engine == eng) & (PPV.task == "A") &
                    (PPV.cls == c)]) else 0
            ax.text(xi, v + 4, "%d%%" % round(v), ha="center", va="bottom",
                    fontsize=6.4, color="#222")
            ax.annotate("n=%d" % n, xy=(xi, -0.19),
                        xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=5.8, color="#777",
                        clip_on=False, annotation_clip=False)
    ax.set_xticks(x)
    ax.set_xticklabels([CHLAB[c].replace(" ", "\n") for c in CH], fontsize=7.2)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("positive predictive value (%)", fontsize=7.6)
    ax.legend(frameon=False, fontsize=7, loc="upper left",
              bbox_to_anchor=(0.0, 1.10), ncol=2)
    ax.set_title("A  per-class PPV (task A, n=98)", fontsize=8.2, loc="left",
                 pad=22)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.axhline(100, color="#DDD", lw=0.7, zorder=0)

    # ---- B: kappa
    ax = axes[1]
    KAP = pd.DataFrame(J["kappa"])
    lab = ["task A\n(channels)", "task B\n(activity)"]
    v7 = [float(KAP[(KAP.task == "A") & (KAP.engine == "v7")].iloc[0].kappa),
          float(KAP[(KAP.task == "B") & (KAP.engine == "v7")].iloc[0].kappa)]
    v8 = [float(KAP[(KAP.task == "A") & (KAP.engine == "v8")].iloc[0].kappa),
          float(KAP[(KAP.task == "B") & (KAP.engine == "v8(narrow)")].iloc[0].kappa)]
    x = np.arange(2)
    for vals, col, off, nm in [(v7, C_V7, -0.17, "v7"),
                               (v8, C_V8, +0.17, "v8")]:
        ax.bar(x + off, vals, 0.32, color=col, edgecolor="white", linewidth=0.6,
               label=nm.upper())
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.02, "%.2f" % v, ha="center", va="bottom",
                    fontsize=6.6, color="#222")
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=7.2)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Cohen's $\\kappa$ (algorithm vs human)", fontsize=7.6)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title("B  agreement with the blinded reference", fontsize=8.2,
                 loc="left", pad=22)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- C: what each rule blocked
    ax = axes[2]
    blk = pd.read_csv(os.path.join(DATA, "gc_activity_density.csv"))
    rules = ["non-rheum domain\n(inactivity cue)", "bare word not in a\nrheum sentence",
             "inverting negation", "non-rheum domain\n(activity cue)",
             "hypothesis / differential", "history / sequela marker",
             "no rheum entity within\nco-occurrence radius"]
    vals = [int(blk.d_inact_domain.sum()), int(blk.d_inact_sent.sum()),
            int(blk.d_inact_invert.sum()), int(blk.d_act_domain.sum()),
            int(blk.d_act_hypoth.sum()), int(blk.d_act_past.sum()),
            int(blk.d_act_radius.sum())]
    y = np.arange(len(rules))[::-1]
    ax.barh(y, vals, 0.62, color=C_RULE, edgecolor="white", linewidth=0.6)
    for yi, v in zip(y, vals):
        ax.text(v * 1.25 if v > 0 else 4, yi, "%d" % v, va="center",
                ha="left", fontsize=6.4, color="#333")
    ax.set_yticks(y)
    ax.set_yticklabels(rules, fontsize=6.2)
    ax.set_xscale("log")
    ax.set_xlim(0.6, 4200)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1000"], fontsize=6.6)
    ax.set_xlabel("cue occurrences blocked (whole cohort, n=1250 notes)",
                  fontsize=7.2)
    ax.set_title("C  what the three rules block", fontsize=8.2, loc="left",
                 pad=22)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    cap(fig, "Figure 35. Three rules (entity restriction, negation, and domain "
             "judgement) applied to the same keyword lists that produced the v7 "
             "channels. Left: positive predictive value of each channel against "
             "the blinded human reference; error bars are 95% Wilson intervals. "
             "Middle: Cohen's kappa between the algorithm and the human "
             "reference for both coding tasks. Right: number of cue occurrences "
             "each rule removed across the cohort (log scale). The classifier "
             "moves from near-chance agreement (kappa 0.10 / 0.08) to "
             "substantial agreement (0.94 / 0.74) without any change to the "
             "underlying word lists.", y=-0.06)
    save(fig, "Figure35_three_rule_engine")


# ==================================================================== Figure 36
def fig36():
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6),
                             gridspec_kw=dict(wspace=0.40, left=0.06,
                                              right=0.99, top=0.84, bottom=0.26))

    # ---- A: density by human reference class
    ax = axes[0]
    classes = [("UNDERDOC", A[A.ref_std == "UNDERDOC"].aid_doc.values, C_UDOC),
               ("INF_DEFER", A[A.ref_std == "INF_DEFER"].aid_doc.values, C_DEFER),
               ("INACTIVE", A[A.ref_std == "INACTIVE"].aid_doc.values, C_INACT)]
    for i, (nm, v, col) in enumerate(classes):
        v = v[np.isfinite(v)]
        jit = (np.random.RandomState(7 + i).rand(len(v)) - 0.5) * 0.34
        ax.scatter(v, np.full(len(v), i) + jit, s=13, color=col, alpha=0.55,
                   edgecolor="none", zorder=3)
        if len(v):
            ax.plot([np.median(v)], [i], marker="|", ms=16, mew=2.0,
                    color="#111", zorder=4)
            ax.text(np.median(v), i + 0.30, "med %.2f" % np.median(v),
                    ha="center", va="bottom", fontsize=6.2, color="#333")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([c[0] for c in classes], fontsize=7.0)
    ax.set_xlabel("activity-information density\nper 1000 narrative characters",
                  fontsize=7.4)
    ax.set_title("A  density by blinded human label", fontsize=8.2,
                 loc="left", pad=22)
    ax.set_xlim(-0.12, max(1.2, float(np.nanmax(A.aid_doc)) * 1.05))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- B: ROC
    ax = axes[1]
    specs = [("documented vs not (task A)", A.human_documented.values,
              A.aid_doc.values, "#2B6CB0"),
             ("activity vs negated (task B)", B.human_act.values,
              B.aid_doc.values, "#D85A30")]
    for nm, y, x, col in specs:
        m = np.isfinite(y) & np.isfinite(x)
        fpr, tpr, _ = roc_curve(y[m], x[m])
        a = roc_auc_score(y[m], x[m])
        ax.plot(fpr, tpr, color=col, lw=1.6, label="%s  AUC=%.3f" % (nm, a))
    ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.8, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("1 - specificity", fontsize=7.4)
    ax.set_ylabel("sensitivity", fontsize=7.4)
    ax.legend(frameon=False, fontsize=6.4, loc="lower right")
    ax.set_title("B  discrimination of the density", fontsize=8.2, loc="left",
                 pad=22)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- C: interaction forest
    ax = axes[2]
    SRC = [("v7: truly inactive", "v7: INACTIVE", GREY),
           ("v7: infection-deferred", "v7: INF_DEFER", GREY),
           ("v7: undocumented", "v7: UNDERDOC", C_V7),
           ("v8: has information", "v8: 有活动度信息 (aid_broad>0)", "#B08BC7"),
           ("v8: no information", "v8: 无活动度信息 (aid_broad=0)", C_V8),
           ("whole spec- layer", "参考: 全 spec- 层", "#1B4F8A")]
    rows = []
    for lab_, g, col in SRC:
        r = IX[IX.group == g]
        if len(r) == 0:
            continue
        r = r.iloc[0]
        rows.append((lab_, r.OR, r.lo, r.hi, int(r.n), int(r.n_ev), col))
    y = np.arange(len(rows))[::-1]
    for yi, (nm, orr, lo, hi, n, ev, col) in zip(y, rows):
        if not np.isfinite(orr):
            ax.text(3.0, yi, "not estimable (%d event)" % ev, va="center",
                    ha="center", fontsize=6.2, color="#999", style="italic")
            continue
        ax.plot([lo, hi], [yi, yi], color=col, lw=1.5, solid_capstyle="butt")
        ax.plot([orr], [yi], marker="s", ms=4.4, color=col, zorder=3)
        ax.text(hi * 1.10, yi, "%.2f (%.2f-%.2f)" % (orr, lo, hi), va="center",
                ha="left", fontsize=6.0, color="#333")
    ax.axvline(1.0, color="#888", lw=0.9, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(0.04, 42)
    ax.set_xticks([0.1, 0.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0.1", "0.5", "1", "2", "5", "10", "20"], fontsize=6.6)
    ax.set_yticks(y)
    ax.set_yticklabels(["%s\n(n=%d, ev=%d)" % (r[0], r[4], r[5]) for r in rows],
                       fontsize=5.9)
    ax.set_xlabel("RA/SLE ratio of the GC dose slope", fontsize=7.2)
    ax.set_title("C  the interaction becomes estimable", fontsize=8.2,
                 loc="left", pad=22)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    cap(fig, "Figure 36. Replacing the three-way split with a continuous "
             "activity-information density. Left: density per human reference "
             "label (in the blinded sample); all undocumented notes sit at "
             "exactly zero. Middle: ROC curves for the primary density "
             "(aid_doc) against the two human references. Right: the RA/SLE "
             "ratio of the glucocorticoid "
             "dose slope (i.e. the disease-by-dose interaction) estimated inside "
             "each stratum; using the density lets the interaction be estimated "
             "on the whole layer instead of only on the undocumented stratum, "
             "and the estimate is unchanged.", y=-0.06)
    save(fig, "Figure36_density_replaces_channels")


if __name__ == "__main__":
    fig35()
    fig36()
