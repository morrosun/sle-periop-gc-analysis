# -*- coding: utf-8 -*-
"""
202_figures_v10.py —— Figure 38：R1 / R2 / R3 规则消融

面板
  A  加权 PPV(任务 A / INACTIVE)：8 个配置 × 样本内/样本外（含 bootstrap 95% CI）
  B  加权 PPV(任务 B / ACTIVITY-POSITIVE)：同上
  C  配对检验的 b 计数（FULL 判对而该配置判错的例数，样本外）——
     这条直接回答「某条规则被拿掉会损失多少」，是归因的核心图
  D  FULL − NONE 的加权 PPV 差（含 95% CI）：样本内 vs 样本外

风格与 v7–v9 各版图一致（白底、DejaVu Sans、图注 textwrap、图内不出现中文）。
"""
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

S_IN, S_OUT = "样本内(v7b)", "样本外(v9)"
VARIANTS = ["FULL", "-R1", "-R2", "-R3", "R1only", "R2only", "R3only", "NONE"]
VARTXT = ["FULL\nR1R2R3", "-R1", "-R2", "-R3", "R1\nonly", "R2\nonly",
          "R3\nonly", "NONE"]

C_IN, C_OUT = "#9AA0A6", "#B03A2E"
C_B = "#2B6CB0"

BC = pd.read_csv(os.path.join(OUT, "table_v10_boot_ci.csv"))
MC = pd.read_csv(os.path.join(OUT, "table_v10_mcnemar.csv"))
DF = pd.read_csv(os.path.join(OUT, "table_v10_diff_ci.csv"))


def cap(fig, s, y=-0.03, width=172):
    fig.text(0.5, y, "\n".join(textwrap.wrap(s, width)), ha="center",
             va="top", fontsize=7.2, color="#444444")


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, "%s.%s" % (name, ext)), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def ppv_panel(ax, task, cls, title):
    x = np.arange(len(VARIANTS))
    w = 0.36
    for off, sname, col, lab in [(-w / 2, S_IN, C_IN, "in-sample (v7b, n=148)"),
                                 (w / 2, S_OUT, C_OUT,
                                  "out-of-sample (v9, n=131)")]:
        vals, los, his = [], [], []
        for v in VARIANTS:
            r = BC[(BC["sample"] == sname) & (BC.task == task) &
                   (BC.variant == v)]
            if len(r) and np.isfinite(r.iloc[0].ppv_w):
                vals.append(float(r.iloc[0].ppv_w))
                los.append(float(r.iloc[0].ppv_w) - float(r.iloc[0].ppv_lo))
                his.append(float(r.iloc[0].ppv_hi) - float(r.iloc[0].ppv_w))
            else:
                vals.append(0.0)
                los.append(0.0)
                his.append(0.0)
        ax.bar(x + off, vals, w, color=col, alpha=0.9, edgecolor="white",
               linewidth=0.6, label=lab,
               yerr=[los, his], error_kw=dict(ecolor="#666666", lw=0.7,
                                              capsize=1.6))
        for xi, v in zip(x + off, vals):
            if v > 0:
                ax.text(xi, v + 0.035, "%.2f" % v, ha="center", va="bottom",
                        fontsize=6.2, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(VARTXT, fontsize=6.6)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("design-weighted PPV")
    ax.set_title(title, fontsize=8.6, pad=4)
    ax.legend(fontsize=6.4, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", lw=0.5)
    ax.set_axisbelow(True)


def fig38():
    fig = plt.figure(figsize=(7.4, 7.0))
    gs = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.26,
                          left=0.09, right=0.98, top=0.93, bottom=0.16)

    axA = fig.add_subplot(gs[0, 0])
    ppv_panel(axA, "A", "INACTIVE", "A  Task A: PPV of INACTIVE (three-channel)")

    axB = fig.add_subplot(gs[0, 1])
    ppv_panel(axB, "B", "ACTIVITY-POSITIVE",
              "B  Task B: PPV of ACTIVITY-POSITIVE")

    # ---- C: McNemar b counts, out-of-sample
    axC = fig.add_subplot(gs[1, 0])
    m = MC[(MC["sample"] == S_OUT) & (MC.cls == "(exact match)")]
    pairs = ["FULL vs " + v for v in ["-R1", "-R2", "-R3", "R1only", "R2only",
                                      "R3only", "NONE"]]
    lbl = ["-R1", "-R2", "-R3", "R1only", "R2only", "R3only", "NONE"]
    assert len(m), "McNemar 表缺少「FULL vs ...」行"
    x = np.arange(len(pairs))
    w = 0.36
    ymax = 1.0
    for off, t, col, lab in [(-w / 2, "A", C_B, "task A"),
                             (w / 2, "B", C_OUT, "task B")]:
        vals = []
        for p in pairs:
            r = m[(m.task == t) & (m.pair == p)]
            assert len(r) == 1, "配对行缺失：%s / %s" % (t, p)
            vals.append(int(r.iloc[0].b))
        ymax = max(ymax, max(vals) * 1.22)
        axC.bar(x + off, vals, w, color=col, alpha=0.9, edgecolor="white",
                linewidth=0.6, label=lab)
        for xi, v in zip(x + off, vals):
            axC.text(xi, v + 0.3, str(v), ha="center", va="bottom",
                     fontsize=6.4, color=col)
    axC.set_xticks(x)
    axC.set_xticklabels(lbl, fontsize=6.6)
    axC.set_ylim(0, ymax)
    axC.set_ylabel("cases FULL gets right\nand the variant gets wrong")
    axC.set_title("C  Rule attribution, out-of-sample (paired)", fontsize=8.6,
                  pad=4)
    axC.legend(fontsize=6.4, frameon=False, loc="upper right")
    axC.spines[["top", "right"]].set_visible(False)
    axC.grid(axis="y", color="#DDDDDD", lw=0.5)
    axC.set_axisbelow(True)
    axC.annotate("R1 drops\nnothing", xy=(0.0, 0.6), xytext=(0.13, 0.52),
                 textcoords="axes fraction", fontsize=6.4, color="#777777",
                 ha="left", arrowprops=dict(arrowstyle="->", lw=0.6,
                                            color="#999999"))

    # ---- D: FULL − NONE
    axD = fig.add_subplot(gs[1, 1])
    labs, diffs, los, his, cols = [], [], [], [], []
    for sname, tag, col in [(S_IN, "in-sample", C_IN),
                            (S_OUT, "out-of-sample", C_OUT)]:
        for task in ["A", "B"]:
            r = DF[(DF["sample"] == sname) & (DF.task == task)]
            if not len(r):
                continue
            labs.append("%s\ntask %s" % (tag, task))
            diffs.append(float(r.iloc[0]["diff"]))
            los.append(float(r.iloc[0]["diff"]) - float(r.iloc[0].lo))
            his.append(float(r.iloc[0].hi) - float(r.iloc[0]["diff"]))
            cols.append(col)
    y = np.arange(len(labs))[::-1]
    axD.barh(y, diffs, 0.55, color=cols, alpha=0.9, edgecolor="white",
             linewidth=0.6, xerr=[los, his],
             error_kw=dict(ecolor="#555555", lw=0.8, capsize=2.2))
    for yi, v, lo in zip(y, diffs, los):
        axD.text(v + (0.03 if v >= 0 else -0.03), yi + 0.30,
                 "%+.3f" % v, va="center",
                 ha="left" if v >= 0 else "right", fontsize=6.5, color="#333")
        if abs(v - lo) < 1e-9:
            axD.text(v, yi - 0.34, "CI lower bound = 0", ha="center",
                     va="top", fontsize=6.0, color="#B03A2E")
    axD.axvline(0, color="#888888", lw=0.8)
    axD.set_yticks(y)
    axD.set_yticklabels(labs, fontsize=6.6)
    axD.set_xlim(-0.05, 1.15)
    axD.set_xlabel("design-weighted PPV: FULL − NONE")
    axD.set_title("D  Whole engine vs bare word lists", fontsize=8.6, pad=4)
    axD.spines[["top", "right"]].set_visible(False)
    axD.grid(axis="x", color="#DDDDDD", lw=0.5)
    axD.set_axisbelow(True)

    cap(fig, "Figure 38. Rule ablation of the three-rule keyword engine. "
             "(A, B) Design-weighted PPV of each rule configuration, in-sample "
             "(v7b, n=148) and out-of-sample (v9, n=131); error bars are "
             "stratified bootstrap 95% CIs. (C) Paired McNemar attribution: "
             "number of cases the full engine classifies correctly while the "
             "ablated configuration does not, out-of-sample. R1 changes no "
             "case at all; R2 changes 3 on task B; R3 accounts for 21 of the "
             "25 task-B gains. (D) Difference between the full engine and bare "
             "word lists; the task-A out-of-sample interval includes zero, "
             "i.e. the engine adds nothing there.", y=-0.035)
    save(fig, "Figure38_rule_ablation")


if __name__ == "__main__":
    fig38()
