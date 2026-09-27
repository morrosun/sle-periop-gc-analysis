# -*- coding: utf-8 -*-
"""Figures for the SLE-vs-RA head-to-head (v4 report).

  Figure 13  study design and cohort flow (rheumatic / systemic autoimmune)
  Figure 14  who reaches the ICU: SLE vs RA, and the indication-structure gap
  Figure 15  dose-response by disease -- the core figure
  Figure 16  interaction robustness: forest across specifications + bootstrap
  Figure 17  GC dependence gradient, and what the external databases could see

Layout rules kept from v3 (they prevented every collision seen earlier):
  * every row has an explicit index y = -row, never a running counter
  * right-hand text goes OUTSIDE the axes with clip_on=False, and the panel
    holding it occupies the full figure width so the overflow cannot land on
    a neighbouring panel (bbox_inches="tight" then absorbs it)
  * log axes get explicit ticks plus NullLocator/NullFormatter, never
    matplotlib's own minor labelling
  * free-floating labels are avoided: if a label set can collide, use a legend
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import NullLocator, NullFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
FIG = os.path.join(OUT, "fig")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.edgecolor": "#555555",
    "axes.linewidth": 0.8,
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

C_SLE, C_RA, C_OTH = "#2B6CB0", "#D85A30", "#7F77DD"
C_INF, GREY, RED, AMBER, GREEN = "#2F7A4F", "#9AA0A6", "#C0392B", "#D9A227", "#2F7A4F"
DBCOL = {"mimiciv": "#2B6CB0", "eicu": "#D85A30", "nwicu": "#7F77DD"}
DBLAB = {"MIMIC-IV": "#2B6CB0", "eICU-CRD": "#D85A30", "NWICU": "#7F77DD"}
DISEASE_COL = {
    "IIM": "#1D9E75", "MCTD": "#1D9E75", "Vasculitis": "#1D9E75",
    "PMR_GCA": "#1D9E75", "Behcet": "#1D9E75",
    "SLE": C_SLE, "RA": C_RA, "Sarcoidosis": C_OTH, "SSc": C_OTH,
    "RP": C_OTH, "PsA": C_OTH, "pSS": C_OTH, "axSpA": C_OTH, "APS": C_OTH,
}

LB = {"G0_none": "none", "G1_low": ">0-<10", "G2_mod": "10-<50", "G3_high": ">=50"}
ORDER = ["G0_none", "G1_low", "G2_mod", "G3_high"]

# the analysis keys the JSON uses, and the short form the IPTW table uses
FULL = {
    "blood culture + >24 h (specific)": "blood culture + >24 h",
    "infection ICD code (wide)": "infection ICD code",
    "hospital death": "hospital death",
    "glucose >=180 (POSITIVE CONTROL)": "glucose >=180",
    "GI bleeding (NEGATIVE CONTROL)": "GI bleeding",
}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, "%s.%s" % (name, ext)), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def plain_log(ax, ticks):
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(["%g" % t for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


# ------------------------------------------------------------------ load
co = pd.read_csv(os.path.join(DATA, "cohort_rheum_icu.csv")).rename(columns={"sofa24": "sofa"})
with open(os.path.join(OUT, "_head2head_results.json"), encoding="utf-8") as f:
    R = json.load(f)
J = {k: pd.DataFrame(v) for k, v in R.items() if isinstance(v, list)}
G = R.get("gc_dependence_gradient")
rob_absorb = pd.read_csv(os.path.join(OUT, "table_h2h_robust_absorb.csv"))
rob_age = pd.read_csv(os.path.join(OUT, "table_h2h_robust_age.csv"))
rob_surg = pd.read_csv(os.path.join(OUT, "table_h2h_robust_surg.csv"))
rob_expo = pd.read_csv(os.path.join(OUT, "table_h2h_robust_expo.csv"))
rob_boot = pd.read_csv(os.path.join(OUT, "table_h2h_robust_boot.csv"))
mde = pd.read_csv(os.path.join(OUT, "table_ext_h2h_mde.csv"))
pc_ext = pd.read_csv(os.path.join(OUT, "table_ext_h2h_positive_control.csv"))

DOSE = J["dose"].set_index(["outcome", "disease"])


# ==================================================== Figure 13 -- design
def fig13():
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.88, 6.90)
    ax.axis("off")

    def box(x, y, w, h, title, sub, fc, ec, fs=8.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10,rounding_size=0.10",
                                    fc=fc, ec=ec, lw=1.0, zorder=2))
        ax.text(x + w / 2, y + h - 0.24, title, ha="center", va="top", fontsize=fs,
                weight="bold", color="#1A1A1A", zorder=3)
        if sub:
            ax.text(x + w / 2, y + h - 0.60, sub, ha="center", va="top", fontsize=7.0,
                    color="#333333", zorder=3, linespacing=1.40)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=11, lw=1.0, color="#666666", zorder=1))

    ax.text(0.0, 6.72, "A   cohort construction", fontsize=8.6, weight="bold", color="#1A1A1A")
    box(0.2, 5.22, 3.0, 1.45, "MIMIC-IV admissions",
        "ICD-coded rheumatic /\nsystemic autoimmune disease\n22 123 admissions", "#EAF1F8", C_SLE)
    box(3.9, 5.22, 2.7, 1.45, "first ICU stay, adult",
        "4 195 stays", "#EAF1F8", C_SLE)
    box(7.3, 5.22, 2.5, 1.45, "24 h landmark",
        "ICU LOS >= 24 h\n3 361 stays analysed", "#EAF1F8", C_SLE)
    arrow(3.2, 5.94, 3.9, 5.94)
    arrow(6.6, 5.94, 7.3, 5.94)

    # -- B : 14 disease groups, two rows so the names fit at full length
    ax.text(0.0, 4.98, "B   the 14 mutually exclusive disease groups", fontsize=8.6,
            weight="bold", color="#1A1A1A")
    NAME = {"PMR_GCA": "PMR/GCA"}
    grp = co.primary_grp.value_counts()
    xs, w, half = 0.2, 1.355, 7
    for i, (k, v) in enumerate(grp.items()):
        r, c = divmod(i, half)
        x = xs + c * w
        y = 4.20 - r * 0.76
        col = DISEASE_COL.get(k, GREY)
        ax.add_patch(FancyBboxPatch((x, y), w - 0.09, 0.70,
                                    boxstyle="round,pad=0.02,rounding_size=0.05",
                                    fc="white", ec=col, lw=0.9))
        ax.text(x + (w - 0.09) / 2, y + 0.52, NAME.get(k, k), ha="center", va="top",
                fontsize=6.2, color="#222222")
        ax.text(x + (w - 0.09) / 2, y + 0.11, "%d" % v, ha="center", va="bottom",
                fontsize=7.0, weight="bold", color=col)
    ax.text(0.2, 3.32, "note   SLE and RA together contribute 1 636 of the 3 361 stays "
                       "(48.7%); the other 12 groups are the context in which the "
                       "head-to-head is read.",
            fontsize=6.6, color="#777777", style="italic", va="top")

    ax.text(0.0, 2.76, "C   the head-to-head", fontsize=8.6, weight="bold", color="#1A1A1A")
    box(0.2, 1.20, 4.3, 1.50, "SLE   n = 433",
        "any GC first 24 h  52.7%\nage 57.1 y   61.0% renal failure\n"
        "bacteraemia >24 h  36    hospital death  45",
        "#EAF1F8", C_SLE, fs=8.2)
    box(5.3, 1.20, 4.5, 1.50, "RA   n = 1203",
        "any GC first 24 h  38.9%\nage 72.0 y   42.6% renal failure\n"
        "bacteraemia >24 h  51    hospital death  134",
        "#FBEDE7", C_RA, fs=8.2)

    ax.text(0.0, 0.84, "D   what is measured", fontsize=8.6, weight="bold", color="#1A1A1A")
    ax.text(0.2, 0.22,
            "exposure   glucocorticoid DOSE INTENSITY in the first ICU day (prednisone-equivalent mg/day;\n"
            "                window = min(24 h, ICU LOS)) -- not cumulative dose; strata 0 / >0-<10 / 10-<50 / >=50\n"
            "outcomes  blood culture positive >24 h (specific) | infection ICD code (wide) | hospital death\n"
            "controls   GC -> glucose >=180 mg/dL (POSITIVE, a known pharmacological effect) | GI bleeding (NEGATIVE)",
            fontsize=7.3, va="top", color="#333333", linespacing=1.75)
    save(fig, "Figure13_cohort_design")


# ============================================ Figure 14 -- baseline + indication
BASENAME = {"GC dose, median mg/day*": "GC dose intensity, mg/day",
            "ANY GC first 24 h, %": "any GC first 24 h, %"}


def fig14():
    b = J["baseline"]
    fig = plt.figure(figsize=(7.6, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.62)

    ax = fig.add_subplot(gs[0, 0])
    d = b.iloc[::-1].reset_index(drop=True)
    n = len(d)
    for i, r in d.iterrows():
        y = -i
        v = float(r.smd)
        col = RED if v > 0.3 else (AMBER if v > 0.1 else GREY)
        ax.plot([0, v], [y, y], color=col, lw=1.5, zorder=2, solid_capstyle="butt")
        ax.plot(v, y, "s", color=col, ms=4.0, zorder=3)
        ax.text(v + 0.035, y, "%.2f" % v, fontsize=6.4, va="center", color=col, zorder=3)
    ax.axvline(0.1, color="#888888", lw=0.8, ls="--", zorder=1)
    ax.set_yticks([-i for i in range(n)])
    ax.set_yticklabels([BASENAME.get(s, s) if len(s) < 34 else s[:32] + ".."
                        for s in d.variable], fontsize=6.8)
    ax.set_xlim(0, 1.20)
    ax.set_ylim(-n + 0.4, 0.75)
    ax.set_xlabel("|standardised mean difference|, SLE vs RA", fontsize=7.8)
    ax.set_title("A   SLE and RA do not reach the ICU the same way",
                 fontsize=8.6, weight="bold", loc="left")
    ax.text(0.60, -6.0, "age alone\ndiffers by 15 years", fontsize=6.8, color=RED,
            va="center", ha="left", linespacing=1.5)
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    ind = J["indication"].copy()
    piv = ind.pivot_table(index="disease", columns="variable", values="smd")
    dd = ["SLE", "RA", "Vasculitis", "PMR_GCA", "Sarcoidosis"]
    dd = [x for x in dd if x in piv.index]
    piv = piv.loc[dd]
    vars_ = ["age", "SOFA", "renal failure", "vasopressor", "immunosuppressant",
             "shock", "surgical"]
    vars_ = [v for v in vars_ if v in piv.columns]
    h, step = 0.105, 1.30
    for j, dis in enumerate(dd):
        for i, v in enumerate(vars_):
            val = piv.loc[dis, v]
            y = -j * step - i * h
            col = RED if val > 0.25 else (GREEN if val < -0.25 else GREY)
            ax2.barh(y, val, height=h * 0.80, color=col, zorder=2)
    ax2.axvline(0, color="#333333", lw=0.9, zorder=3)
    ax2.set_yticks([-j * step - 3 * h for j in range(len(dd))])
    ax2.set_yticklabels(dd, fontsize=7.2)
    bottom = -(len(dd) - 1) * step - (len(vars_) - 1) * h - 0.30
    ax2.set_ylim(bottom, 0.85)
    ax2.set_xlim(-0.80, 0.98)
    ax2.set_xlabel("signed SMD  (>=50 mg/day  minus  no glucocorticoid)", fontsize=7.2)
    ax2.set_title("B   what high-dose glucocorticoid signals", fontsize=8.6,
                  weight="bold", loc="left")
    ax2.legend(handles=[Line2D([], [], color=RED, lw=4, label="sicker at high dose"),
                        Line2D([], [], color=GREEN, lw=4, label="healthier at high dose"),
                        Line2D([], [], color=GREY, lw=4, label="|SMD| <= 0.25")],
               loc="upper left", bbox_to_anchor=(0.0, -0.125), fontsize=6.2,
               frameon=False, ncol=3, handlelength=1.3, columnspacing=1.0,
               handletextpad=0.4)
    ax2.spines[["top", "right"]].set_visible(False)
    save(fig, "Figure14_baseline_and_indication")


# ===================================== Figure 15 -- dose response by disease
def fig15():
    panels = [("blood culture + >24 h (specific)", "blood culture positive >24 h", "specific"),
              ("infection ICD code (wide)", "infection ICD code", "wide"),
              ("hospital death", "hospital death", "safety"),
              ("glucose >=180 (POSITIVE CONTROL)", "glucose >=180 mg/dL", "POSITIVE CONTROL")]
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.6))
    dose_rows = J["strata"]

    def trend_txt(outc):
        try:
            s = DOSE.loc[(outc, "SLE")]
            r = DOSE.loc[(outc, "RA")]
        except KeyError:
            return ""
        return ("dose-trend OR  SLE %.2f (%.2f-%.2f)     RA %.2f (%.2f-%.2f)"
                % (s.adj_or, s.adj_lo, s.adj_hi, r.adj_or, r.adj_lo, r.adj_hi))

    for ax, (y, lab, tag) in zip(axes.ravel(), panels):
        dr = dose_rows[dose_rows.outcome == y]
        for dis, col in [("SLE", C_SLE), ("RA", C_RA)]:
            xs, ys, los, his = [0], [1.0], [1.0], [1.0]
            for k, gs in enumerate(ORDER[1:], start=1):
                row = dr[(dr.disease == dis) & (dr.stratum == LB[gs])]
                if len(row) == 0:
                    continue
                r = row.iloc[0]
                xs.append(k)
                ys.append(float(r.OR))
                los.append(float(r.lo))
                his.append(float(r.hi))
            ax.fill_between(xs, los, his, color=col, alpha=0.13, zorder=1)
            ax.plot(xs, ys, "-o", color=col, lw=1.6, ms=4.4, label=dis, zorder=3)
        ax.axhline(1.0, color="#333333", lw=0.9, ls="--", zorder=2)
        ax.set_yscale("log")
        ax.set_yticks([0.25, 0.5, 1, 2, 4, 8])
        ax.set_yticklabels(["0.25", "0.5", "1", "2", "4", "8"])
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(range(4))
        ax.set_xticklabels([LB[g] for g in ORDER], fontsize=7.0, rotation=18, ha="right")
        ax.set_xlim(-0.35, 3.35)
        ax.set_ylim(0.17, 13)
        ax.set_xlabel("prednisone-equivalent mg/day, first 24 h", fontsize=7.2)
        ax.set_title("%s   (%s)" % (lab, tag), fontsize=8.4, weight="bold", loc="left",
                     pad=15)
        ax.text(0.0, 1.012, trend_txt(y), transform=ax.transAxes, fontsize=6.4,
                color="#555555", va="bottom")
        ax.spines[["top", "right"]].set_visible(False)
    axes.ravel()[0].set_ylabel("OR vs no glucocorticoid", fontsize=7.6)
    axes.ravel()[0].legend(fontsize=7.4, frameon=False, loc="upper left",
                           bbox_to_anchor=(0.02, 1.0))
    fig.suptitle("Dose-response of the same exposure in two diseases",
                 fontsize=9.4, weight="bold", x=0.012, ha="left", y=1.005)
    fig.tight_layout()
    save(fig, "Figure15_dose_response_by_disease")


# ============================================ Figure 16 -- interaction forest
def fig16():
    prim = J["interaction"].set_index("outcome")
    ipi = J["iptw_interaction"].set_index("outcome") if len(J["iptw_interaction"]) else None
    if ipi is not None:
        ipi.index = [FULL.get(i, i) for i in ipi.index]

    KEY = {
        "bacteraemia >24 h": "blood culture + >24 h (specific)",
        "infection code": "infection ICD code (wide)",
        "hospital death": "hospital death",
        "glucose >=180 (control)": "glucose >=180 (POSITIVE CONTROL)",
        "GI bleeding (control)": "GI bleeding (NEGATIVE CONTROL)",
    }

    def age_row(outc, win="50-80"):
        m = rob_age[(rob_age.outcome == outc) & (rob_age.window == win)]
        return (float(m.iloc[0].adj_or), float(m.iloc[0].adj_p)) if len(m) else (np.nan, np.nan)

    def surg_row(outc):
        m = rob_surg[rob_surg.outcome == outc]
        return (float(m.iloc[0].nonsurg_or), float(m.iloc[0].nonsurg_p)) if len(m) else (np.nan, np.nan)

    rows = []           # (label, spec, OR, lo, hi, P, block, is_key)

    def add(nm, spec, OR, lo, hi, P, block, key=False):
        rows.append((nm, spec, OR, lo, hi, P, block, key))

    for nm in ["bacteraemia >24 h", "hospital death", "infection code",
               "glucose >=180 (control)", "GI bleeding (control)"]:
        ok = KEY[nm]
        if ok in prim.index:
            r = prim.loc[ok]
            add(nm, "primary adjusted", r.OR, r.lo, r.hi, r.p_wald, nm, True)
        if ipi is not None and FULL.get(ok, ok) in ipi.index:
            r = ipi.loc[FULL[ok]]
            add(nm, "IPTW weighted", r.OR, r.lo, r.hi, r["p"], nm)
        if nm in ("bacteraemia >24 h", "hospital death"):
            v, p = age_row(FULL[ok])
            add(nm, "age 50-80 only", v, np.nan, np.nan, p, nm)
            v, p = surg_row(FULL[ok])
            add(nm, "non-surgical only", v, np.nan, np.nan, p, nm)

    BLOCKCOL = {"bacteraemia >24 h": C_INF, "hospital death": RED, "infection code": C_SLE,
                "glucose >=180 (control)": AMBER, "GI bleeding (control)": GREY}

    # ---- panel A : full-width so the numeric column has room to spill right
    fig = plt.figure(figsize=(7.6, 6.1))
    gs = fig.add_gridspec(2, 1, height_ratios=[4.35, 1.0], hspace=0.52)

    ax = fig.add_subplot(gs[0, 0])
    y = 0.0
    yticks, ylabs = [], []
    for nm, spec, OR, lo, hi, P, block, key in rows:
        col = BLOCKCOL[block]
        ax.plot([1, OR], [y, y], color=col, lw=1.2, zorder=2, alpha=0.70)
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [y, y], color=col, lw=2.0, zorder=3, solid_capstyle="round")
        ax.plot(OR, y, "s", color=col, ms=5.0 if key else 3.9, zorder=4)
        txt = "%.2f (%.2f-%.2f)" % (OR, lo, hi) if np.isfinite(lo) else "%.2f" % OR
        star = "*" if (P is not None and np.isfinite(P) and P < 0.05) else ""
        ax.text(4.72, y, "%s  P=%.3f%s" % (txt, P, star), fontsize=6.5, va="center",
                ha="left", clip_on=False,
                color=col if star else "#333333",
                fontweight="bold" if key else "normal")
        yticks.append(y)
        ylabs.append(nm if spec == "primary adjusted" else "     " + spec)
        y -= 1.0
    # alternating block shading
    i = 0
    blk = 0
    while i < len(rows):
        b = rows[i][6]
        j = i
        while j + 1 < len(rows) and rows[j + 1][6] == b:
            j += 1
        if blk % 2 == 0:
            ax.axhspan(-j + 0.5, -i + 0.5, color="#F5F5F2", zorder=0)
        blk += 1
        i = j + 1
    ax.axvline(1.0, color="#333333", lw=1.0, ls="--", zorder=1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabs, fontsize=6.6)
    ax.set_ylim(y + 0.55, 0.75)
    ax.set_xlim(0.25, 4.6)
    plain_log(ax, [0.3, 0.5, 1, 2, 4])
    ax.set_xlabel("RA / SLE ratio of the dose-trend odds ratios", fontsize=7.6)
    ax.set_title("A   is the dose-response the same in both diseases?", fontsize=8.6,
                 weight="bold", loc="left")
    ax.text(1.06, 0.72, "steeper in RA", fontsize=6.2, color="#666666", va="center")
    ax.text(0.94, 0.72, "steeper in SLE", fontsize=6.2, color="#666666", va="center",
            ha="right")
    ax.spines[["top", "right"]].set_visible(False)

    # ---- panel B : resampling, full width, same left edge
    ax2 = fig.add_subplot(gs[1, 0])
    names = {"blood culture + >24 h": ("bacteraemia >24 h", C_INF),
             "hospital death": ("hospital death", RED),
             "glucose >=180 (POS CTRL)": ("glucose >=180\n(positive control)", AMBER)}
    y = 0.0
    yt, yl = [], []
    for _, r in rob_boot.iterrows():
        nm, col = names.get(r.outcome, (r.outcome[:18], GREY))
        ax2.plot([r.lo, r.hi], [y, y], color=col, lw=2.3, solid_capstyle="round", zorder=3)
        ax2.plot(r["median"], y, "s", color=col, ms=5.2, zorder=4)
        frac = r.frac_gt1
        side = 100 * (frac if frac > 0.5 else 1 - frac)
        who = "RA steeper" if frac > 0.5 else "SLE steeper"
        ax2.text(2.92, y, "median %.2f   %.1f%% of 2 000 resamples %s"
                 % (r["median"], side, who), fontsize=6.3, va="center", ha="left",
                 color="#555555")
        yt.append(y)
        yl.append(nm)
        y -= 1.0
    ax2.axvline(1.0, color="#333333", lw=1.0, ls="--", zorder=2)
    ax2.set_xlim(0.40, 2.75)
    plain_log(ax2, [0.5, 1, 2])
    ax2.set_yticks(yt)
    ax2.set_yticklabels(yl, fontsize=6.8)
    ax2.set_ylim(y + 0.5, 0.62)
    ax2.set_xlabel("bootstrap distribution of the RA / SLE ratio", fontsize=7.4)
    ax2.set_title("B   stratified bootstrap, 2 000 replicates", fontsize=8.6,
                  weight="bold", loc="left")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.text(0.012, -0.035,
             "The interaction is expressed as the RA / SLE ratio, so a value below 1 means the dose-response is steeper in SLE.\n"
             "* P < 0.05.  Non-surgical and age-restricted rows have a point estimate only, because the restricted model was\n"
             "fitted separately rather than through the interaction term.  No control outcome shows a significant interaction.",
             fontsize=6.4, color="#555555", va="top", linespacing=1.6)
    save(fig, "Figure16_interaction_robustness")


# ============================ Figure 17 -- GC dependence + external feasibility
def fig17():
    fig = plt.figure(figsize=(7.6, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.22], wspace=0.40)

    # ---- A : does the infection association track GC dependence?
    ax = fig.add_subplot(gs[0, 0])
    sp = J["spectrum"].dropna(subset=["trend_or", "trend_lo"]).copy()
    sp = sp[(sp.trend_lo > 0) & sp.trend_hi.notna()].sort_values("gc_dep")
    for _, r in sp.iterrows():
        col = DISEASE_COL.get(r.disease, GREY)
        big = r.disease in ("SLE", "RA")
        ax.plot([100 * r.gc_dep] * 2, [r.trend_lo, r.trend_hi], color=col, lw=1.0,
                alpha=0.7, zorder=2, solid_capstyle="round")
        ax.plot([100 * r.gc_dep], [r.trend_or], "o", color=col,
                ms=8.0 if big else 5.2, zorder=3,
                markeredgecolor="white", markeredgewidth=0.7)
    ax.axhline(1.0, color="#333333", lw=0.9, ls="--", zorder=1)
    ax.set_yscale("log")
    ax.set_yticks([0.7, 1, 1.5, 2])
    ax.set_yticklabels(["0.7", "1", "1.5", "2"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("% of all admissions for that disease\ncarrying a systemic GC order",
                  fontsize=7.0)
    ax.set_ylabel("dose-trend OR\n(infection ICD code)", fontsize=7.0)
    ax.set_xlim(12, 80)
    ax.set_ylim(0.60, 2.35)
    ax.set_title("A   does the association track GC dependence?", fontsize=8.4,
                 weight="bold", loc="left")
    handles = []
    for d in sp.disease:
        handles.append(Line2D([], [], color=DISEASE_COL.get(d, GREY), marker="o", ls="",
                              ms=5.0 if d not in ("SLE", "RA") else 7.0, label=d))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.01), fontsize=5.7,
              frameon=False, ncol=3, handlelength=1.0, columnspacing=0.9,
              handletextpad=0.35)
    if G:
        ax.text(0.98, 0.98,
                "Spearman rho = %.2f (P=%.2f)\nslope per +10 pp: OR %.3f (%.3f-%.3f), P=%.3f"
                % (G["spearman"], G["spearman_p"], G["slope_per_10pp"], G["lo"], G["hi"], G["p"]),
                transform=ax.transAxes, fontsize=6.1, color="#444444", va="top", ha="right",
                linespacing=1.55)
    ax.spines[["top", "right"]].set_visible(False)

    # ---- B : what the external databases could have detected
    ax2 = fig.add_subplot(gs[0, 1])
    DBNAME = {"mimiciv": "MIMIC-IV", "eicu": "eICU-CRD", "nwicu": "NWICU"}
    OUTN = {"infection ICD code": "infection code", "hospital death": "hospital death"}
    m = mde[mde.outcome.isin(["infection ICD code", "hospital death"])].copy()
    y = 0.0
    yt, yl = [], []
    for occ in ["infection ICD code", "hospital death"]:
        y -= 0.55
        ax2.axhspan(y - 0.50, y + 0.50, color="#F2F2EF", zorder=0)
        ax2.text(0.024, y, OUTN[occ], fontsize=7.6, fontweight="bold", va="center",
                 zorder=3, transform=ax2.get_yaxis_transform())
        for db in ["mimiciv", "eicu", "nwicu"]:
            for dis in ["SLE", "RA"]:
                r = m[(m.db == db) & (m.disease == dis) & (m.outcome == occ)]
                if len(r) == 0:
                    continue
                y -= 1.0
                v = r.iloc[0].mde_or
                col = DBCOL[db]
                if v is None or not np.isfinite(v):
                    ax2.barh(y, 8.1, height=0.56, color=col, alpha=0.12, zorder=1)
                    ax2.text(0.024, y, "no event in the reference group", fontsize=6.3,
                             va="center", color=col, zorder=3,
                             transform=ax2.get_yaxis_transform())
                else:
                    ax2.barh(y, float(v), height=0.56, color=col,
                             alpha=0.85 if db == "mimiciv" else 0.58, zorder=1)
                    ax2.text(float(v) + 0.16, y, "%.1f" % v, fontsize=6.4, va="center",
                             color=col, zorder=3)
                yt.append(y)
                yl.append("%s  %s" % (DBNAME[db], dis))
        y -= 0.30
    ax2.axvline(1.30, color=RED, lw=1.2, ls="--", zorder=4)
    ax2.text(1.46, 1.02, "MIMIC-IV observed RA\nmortality trend 1.30", fontsize=6.2,
             color=RED, va="top", linespacing=1.45)
    ax2.set_yticks(yt)
    ax2.set_yticklabels(yl, fontsize=6.4)
    ax2.set_ylim(y - 0.25, 1.05)
    ax2.set_xlim(0, 8.6)
    ax2.set_xlabel("minimum detectable OR (80% power, alpha = 0.05)", fontsize=7.2)
    ax2.set_title("B   what the external databases could have detected", fontsize=8.4,
                  weight="bold", loc="left")
    ax2.spines[["top", "right", "left"]].set_visible(False)
    fig.text(0.012, -0.045,
             "eICU-CRD and NWICU have too few events to reproduce the mortality interaction (about a third of the power), and neither can test the\n"
             "bacteraemia outcome at all: eICU recorded no positive blood culture after the landmark in either disease, and NWICU has no microbiology table.",
             fontsize=6.4, color="#555555", va="top", linespacing=1.6)
    save(fig, "Figure17_gc_dependence_and_mde")


if __name__ == "__main__":
    print("=== v4 figures ===")
    fig13()
    fig14()
    fig15()
    fig16()
    fig17()
    print("done")
