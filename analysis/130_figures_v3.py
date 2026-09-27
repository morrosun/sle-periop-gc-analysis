# -*- coding: utf-8 -*-
"""Figures for the external-validation (v3) report.

  Figure 9   outcome availability across databases
  Figure 10  positive control (GC -> hyperglycaemia) dose response, three databases
  Figure 11  forest plot: any GC vs none, per outcome, per database + pooled
  Figure 12  minimum detectable effect in each database vs the effect MIMIC-IV saw

Layout note: every figure uses an explicit row index (y = -row) and a right-hand
text margin placed OUTSIDE the axes with clip_on=False, so nothing can collide
with a title, legend or tick label.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, NullFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

C_MIM, C_EIC, C_NW = "#2B6CB0", "#D85A30", "#7F77DD"
GREY, RED, AMBER, GREEN = "#9AA0A6", "#C0392B", "#D9A227", "#2F7A4F"
DBCOL = {"MIMIC-IV": C_MIM, "eICU-CRD": C_EIC, "NWICU": C_NW}
DBKEY = {"mimiciv": "MIMIC-IV", "eicu": "eICU-CRD", "nwicu": "NWICU"}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, "%s.%s" % (name, ext)), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def load():
    d = {}
    for k, f in [("mimiciv", "cohort_mimic_harmonized.csv"),
                 ("eicu", "cohort_eicu.csv"), ("nwicu", "cohort_nwicu.csv")]:
        d[k] = pd.read_csv(os.path.join(ROOT, "data", f))
    with open(os.path.join(OUT, "_external_validation_results.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    return d, res


def plain_log(ax, ticks):
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(["%g" % t for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


# ====================================================================== Fig 9
def fig9(res):
    rows = res["feasibility"]
    nrow, ncol = len(rows), 3
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(-nrow - 0.4, 1.5)
    ax.axis("off")

    xlab0, xdata0, cw = 0.0, 0.445, 0.178
    ax.text(xlab0, 0.9, "outcome", fontsize=7.6, fontweight="bold", color="#444", va="center")
    for j, k in enumerate(["mimiciv", "eicu", "nwicu"]):
        ax.text(xdata0 + j * cw + cw / 2 - 0.012, 0.9, DBKEY[k], fontsize=7.6,
                color=DBCOL[DBKEY[k]], fontweight="bold", ha="center", va="center")
    ax.plot([0, 1], [0.55, 0.55], color="#BBBBBB", lw=0.8)

    for i, r in enumerate(rows):
        y = -i - 0.45
        if i % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#F7F7F5", zorder=0)
        ax.text(xlab0, y, r["outcome"], fontsize=7.3, va="center", color="#222")
        for j, k in enumerate(["mimiciv", "eicu", "nwicu"]):
            v = r[k]
            xc = xdata0 + j * cw + cw / 2 - 0.012
            if v in ("N/A", "no micro table"):
                ax.text(xc, y, "no such data" if v == "no micro table" else "n/a",
                        fontsize=6.8, color=GREY, style="italic", ha="center", va="center")
                continue
            ev, tot = [int(z) for z in v.split("/")]
            col = RED if ev == 0 else (AMBER if ev < 15 else GREEN)
            fc = {"#C0392B": "#FBE9E7", "#D9A227": "#FDF4E1", "#2F7A4F": "#E7F3EC"}[col]
            ax.text(xc, y, "%d / %d" % (ev, tot), fontsize=7.4, ha="center", va="center",
                    color=col, fontweight="bold" if ev < 15 else "normal",
                    bbox=dict(boxstyle="round,pad=0.30", fc=fc, ec=col, lw=0.6))

    ax.text(0, -nrow - 0.05,
            "Cells give events / analysable stays.  "
            "Red = no event at all;  amber = fewer than 15 events;  grey = the database holds no such data.",
            fontsize=6.5, color="#555", va="top")
    ax.text(0, -nrow - 0.45,
            "NWICU ships no microbiology table, so the timed bloodstream-infection outcome cannot be built there at all.",
            fontsize=6.5, color="#555", va="top")
    ax.set_title("Figure 9  Outcome availability in the three databases", fontsize=9.5, pad=14)
    save(fig, "Figure9_outcome_availability")


# ===================================================================== Fig 10
def fig10(d, res):
    fig = plt.figure(figsize=(7.6, 4.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.0], height_ratios=[4.2, 1.75],
                          wspace=0.42, hspace=0.55)
    ax = fig.add_subplot(gs[0, 0])
    axt = fig.add_subplot(gs[1, 0])
    axb = fig.add_subplot(gs[:, 1])

    order = ["G0_none", "G1_low", "G2_mod", "G3_high"]
    xl = ["none", ">0–<10", "10–<50", "\u2265 50"]
    tab = []
    for k in ["mimiciv", "eicu", "nwicu"]:
        x = d[k]
        pts, ns = [], []
        for g in order:
            s = x.loc[x.gc24_str == g, "hyper_48h"]
            pts.append(100 * s.mean(skipna=True) if len(s) else np.nan)
            ns.append(int(s.notna().sum()))
        ax.plot(range(4), pts, "-o", color=DBCOL[DBKEY[k]], ms=4.2, lw=1.5,
                label=DBKEY[k], clip_on=False)
        tab.append(ns)
    ax.set_xticks(range(4))
    ax.set_xticklabels(xl, fontsize=7.2)
    ax.set_xlim(-0.28, 3.28)
    ax.set_ylim(0, 78)
    ax.set_ylabel("glucose \u2265 180 mg/dL within 48 h (%)", fontsize=7.6)
    ax.set_xlabel("glucocorticoid dose, first 24 h (mg PE/day)", fontsize=7.4, labelpad=4)
    ax.set_title("Positive control: GC \u2192 hyperglycaemia", fontsize=9, pad=18)
    ax.legend(fontsize=6.9, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=3, handlelength=1.4, columnspacing=1.2)
    ax.grid(axis="y", color="#EFEFEF", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    axt.set_xlim(-0.28, 3.28)
    axt.set_ylim(0.0, 4.6)
    axt.axis("off")
    axt.text(-0.03, 4.5, "stays with a glucose value", fontsize=6.1, color="#777",
             ha="left", va="center", transform=axt.get_yaxis_transform(), clip_on=False)
    for j, k in enumerate(["mimiciv", "eicu", "nwicu"]):
        y = 3.5 - j * 1.35
        axt.text(-0.03, y, "%s (n)" % DBKEY[k], fontsize=6.4, color=DBCOL[DBKEY[k]],
                 ha="right", va="center", transform=axt.get_yaxis_transform(), clip_on=False)
        for i in range(4):
            axt.text(i, y, "%d" % tab[j][i], fontsize=6.7, ha="center", va="center",
                     color="#555" if tab[j][i] >= 10 else RED)

    labs, keys = ["MIMIC-IV", "eICU-CRD", "NWICU"], ["mimiciv", "eicu", "nwicu"]
    pc = {r["db"]: r for r in res["positive_control"]["rows"]}
    mt = res.get("pc_trend_meta")
    ypos = []
    y = 0
    for i, lab in enumerate(labs):
        y -= 1.0
        ypos.append(y)
        r = pc.get(lab)
        s = (r or {}).get("OR_trend_per_stratum", "")
        if "n.e." in s or not s:
            axb.text(0.02, y, "not estimable", fontsize=6.9, color=GREY, va="center",
                     transform=axb.get_yaxis_transform())
            continue
        lo, hi = [float(z) for z in s[s.index("(") + 1:s.index(")")].split("-")]
        orr = float(s.split(" ")[0])
        col = DBCOL[lab]
        axb.plot([lo, hi], [y, y], color=col, lw=1.5)
        axb.plot(orr, y, "s", color=col, ms=5)
        axb.text(1.03, y + 0.16, s, fontsize=6.8, ha="left", va="center",
                 transform=axb.get_yaxis_transform(), clip_on=False)
        axb.text(1.03, y - 0.30, "P = %s" % r["p_trend"], fontsize=6.2, ha="left",
                 va="center", transform=axb.get_yaxis_transform(), clip_on=False, color="#666")
    if mt:
        y -= 1.15
        ypos.append(y)
        axb.plot([mt["lo"], mt["hi"]], [y, y], color="#222222", lw=2.0)
        axb.plot(mt["OR"], y, "D", color="#222222", ms=5.5)
        axb.text(1.03, y + 0.16, "pooled %.2f (%.2f\u2013%.2f)" % (mt["OR"], mt["lo"], mt["hi"]),
                 fontsize=6.8, ha="left", va="center", transform=axb.get_yaxis_transform(),
                 clip_on=False, fontweight="bold")
        axb.text(1.03, y - 0.30, "P = %.3f   I\u00b2 = %.0f%%" % (mt["p"], mt["I2"]),
                 fontsize=6.2, ha="left", va="center", transform=axb.get_yaxis_transform(),
                 clip_on=False, color="#666")
    axb.axvline(1.0, color="#999999", lw=0.9, ls="--")
    plain_log(axb, [0.5, 1, 2, 3, 4])
    axb.set_xlim(0.42, 4.2)
    axb.set_ylim(y - 0.75, 0.35)
    axb.set_yticks(ypos)
    axb.set_yticklabels(labs + (["pooled"] if mt else []), fontsize=7.4)
    axb.set_xlabel("OR per dose-stratum increment (95% CI)", fontsize=7.4)
    axb.set_title("Trend across the four dose strata", fontsize=9, pad=10)
    axb.grid(axis="x", color="#EFEFEF", lw=0.6)
    for s in ("top", "right", "left"):
        axb.spines[s].set_visible(False)
    save(fig, "Figure10_positive_control_external")


# ===================================================================== Fig 11
def fig11(res):
    oc = res["outcomes"]
    panels = [("culture_pos_after24", "Blood culture positive >24 h"),
              ("infect_icd", "Infection diagnosis code (untimed)"),
              ("abx_new_after48", "New antibiotic >48 h"),
              ("gi_bleed", "GI bleeding (negative outcome)"),
              ("death_hosp", "Hospital death")]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    rows = []          # (kind, y, payload)
    y = 0
    for var, lab in panels:
        blk = oc.get(var) or {}
        y -= 1.15
        rows.append(("head", y, lab))
        for e in blk.get("ests", []):
            y -= 1.0
            rows.append(("est", y, e))
        m = blk.get("meta")
        if m and m["k"] >= 2:
            y -= 1.0
            rows.append(("pool", y, m))
        elif not blk.get("ests"):
            y -= 1.0
            rows.append(("none", y, "no database could estimate this contrast"))
        y -= 0.35

    for kind, yy, pay in rows:
        if kind == "head":
            ax.axhspan(yy - 0.95, yy + 0.5, color="#F4F4F2", zorder=0)
            ax.text(0.24, yy, pay, fontsize=7.6, fontweight="bold", va="center", zorder=3)
    for kind, yy, pay in rows:
        if kind == "est":
            col = DBCOL[pay["db"]]
            lo, hi = max(float(pay["lo"]), 0.20), min(float(pay["hi"]), 55.0)
            ax.plot([lo, hi], [yy, yy], color=col, lw=1.3, zorder=2)
            ax.plot(float(pay["OR"]), yy, "s", color=col, ms=4.0, zorder=3)
            txt = "%s  %.2f (%.2f\u2013%.2f)" % (pay["db"], float(pay["OR"]),
                                                 float(pay["lo"]), float(pay["hi"]))
            if pay["n_ev"] < 10:
                txt += "   (!)"
            ax.text(64, yy, txt, fontsize=6.4, va="center", ha="left", clip_on=False,
                    color=RED if pay["n_ev"] < 10 else "#222")
        elif kind == "pool":
            ax.plot([max(float(pay["lo"]), 0.20), min(float(pay["hi"]), 55.0)], [yy, yy],
                    color="#222222", lw=1.8, zorder=2)
            ax.plot(float(pay["OR"]), yy, "D", color="#222222", ms=4.6, zorder=3)
            ax.text(64, yy, "pooled %.2f (%.2f\u2013%.2f)   I\u00b2 = %.0f%%" % (
                float(pay["OR"]), float(pay["lo"]), float(pay["hi"]), float(pay["I2"])),
                fontsize=6.6, va="center", ha="left", clip_on=False, fontweight="bold")
        elif kind == "none":
            ax.text(0.24, yy, pay, fontsize=6.8, color=GREY, style="italic", va="center")
    ax.axvline(1.0, color="#888888", lw=0.9, ls="--", zorder=1)
    plain_log(ax, [0.2, 0.5, 1, 2, 5, 10, 30])
    ax.set_xlim(0.18, 50)
    ax.set_ylim(y - 0.9, 0.75)
    ax.set_yticks([])
    ax.set_xlabel("odds ratio, any glucocorticoid vs none (95% CI)", fontsize=7.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Line2D([], [], color=C_MIM, marker="s", ls="-", label="MIMIC-IV"),
                       Line2D([], [], color=C_EIC, marker="s", ls="-", label="eICU-CRD"),
                       Line2D([], [], color=C_NW, marker="s", ls="-", label="NWICU"),
                       Line2D([], [], color="#222222", marker="D", ls="-",
                              label="random-effects pooled")],
              loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=6.9, frameon=False,
              ncol=4, columnspacing=1.6, handlelength=1.5)
    ax.set_title("Figure 11  External validation: any glucocorticoid vs none",
                 fontsize=9.5, pad=22)
    fig.text(0.02, -0.02,
             "(!) fewer than 10 events in the whole database \u2014 the estimate is driven by a handful of patients.\n"
             "For the bloodstream-infection outcome only MIMIC-IV contributes: eICU-CRD recorded no positive blood\n"
             "culture after the landmark (2 blood cultures in the entire SLE cohort) and NWICU has no microbiology table.",
             fontsize=6.4, color="#555", va="top")
    save(fig, "Figure11_forest_external")


# ===================================================================== Fig 12
def fig12(res):
    mde = pd.DataFrame(res["mde"])
    order = ["culture_pos_after24", "abx_new_after48", "infect_icd"]
    nice = {"culture_pos_after24": "Blood culture positive >24 h",
            "abx_new_after48": "New antibiotic >48 h",
            "infect_icd": "Infection diagnosis code"}
    keys = ["MIMIC-IV", "eICU-CRD", "NWICU"]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    rows, y = [], 0
    for var in order:
        sub = mde[mde.outcome == var]
        y -= 1.15
        rows.append(("head", y, nice[var]))
        for k in keys:
            r = sub[sub.db == k]
            if len(r) == 0:
                continue
            y -= 1.0
            rows.append(("bar", y, (k, r.iloc[0]["MDE_OR"])))
        y -= 0.30

    for kind, yy, pay in rows:
        if kind == "head":
            ax.axhspan(yy - 1.0, yy + 0.5, color="#F4F4F2", zorder=0)
            ax.text(0.014, yy, pay, fontsize=7.6, fontweight="bold", va="center", zorder=3,
                    transform=ax.get_yaxis_transform())
    xmax = np.log10(60)
    for kind, yy, pay in rows:
        if kind != "bar":
            continue
        k, v = pay
        col = DBCOL[k]
        try:
            vv = float(v)                      # JSON stores a few as text
        except (TypeError, ValueError):
            ax.barh(yy, xmax, left=0, height=0.62, color=col, alpha=0.13, zorder=1)
            ax.text(0.05, yy, "%s \u2014 not computable (no event in the reference group)" % k,
                    fontsize=6.5, va="center", zorder=3, color=col,
                    transform=ax.get_yaxis_transform())
            continue
        ax.barh(yy, np.log10(vv), left=0, height=0.62, color=col, alpha=0.85, zorder=1)
        ax.text(0.05, yy, k, fontsize=6.9, va="center", zorder=3)
        xlab = max(np.log10(vv) + 0.045, np.log10(2.93) + 0.10)   # never sit on the marker line
        ax.text(xlab, yy, "OR %.1f" % vv, fontsize=6.8, va="center", color=col, zorder=3)
    ax.axvline(np.log10(2.93), color=RED, lw=1.1, ls="--", zorder=2)
    ax.set_xlim(0, xmax)
    ax.set_ylim(y - 0.8, 2.6)
    ax.text(np.log10(2.93) - 0.07, 2.45,
            "MIMIC-IV observed OR 2.93 (bloodstream infection)", fontsize=6.2, color=RED,
            va="top", ha="right", zorder=4,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=RED, lw=0.6))
    ax.set_yticks([])
    ax.set_xticks([np.log10(v) for v in [1, 2, 3, 5, 10, 20, 40]])
    ax.set_xticklabels(["1", "2", "3", "5", "10", "20", "40"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("smallest odds ratio detectable at 80% power (log scale)", fontsize=7.8)
    ax.set_title("Figure 12  What each database could have detected", fontsize=9.5, pad=10)
    ax.grid(axis="x", color="#EFEFEF", lw=0.6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    save(fig, "Figure12_mde")


def main():
    d, res = load()
    print("generating figures")
    fig9(res)
    fig10(d, res)
    fig11(res)
    fig12(res)


if __name__ == "__main__":
    main()
