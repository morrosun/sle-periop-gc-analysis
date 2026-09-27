# -*- coding: utf-8 -*-
"""
209_figures_v12.py —— Figure 40：v2 手册修订的量与后果

面板
  A  标签迁移：哪些 v1 阳性在 v2 下不再成立（按任务×样本）
  B  v8 引擎的 PPV：v1 金标准 vs v2 金标准 —— 修手册并没有救活分类标签
  C  连续密度 aid_doc 的 AUC：v1 vs v2 —— 修手册反而把它推高
  D  11 例翻转的规则归因（研究者据仲裁/评分引文归类）

风格与 v7–v11 各版图一致（白底、DejaVu Sans、图注 textwrap、图内不出现中文）。
"""
import json
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

C_MAIN = "#B03A2E"      # 阳性/翻转
C_SUB = "#2B6CB0"       # 对照口径
C_GREY = "#9AA0A6"
C_OK = "#1E8449"

# ------------------------------------------------------------------ 翻转归因
# 研究者据两名评分者/第三仲裁的引文，把每一例归到「决定它的那一条 v2 规则」。
FLIP_REASON = {
    "N-A-024": ("A. INF_DEFER clause removed",
                "concern for malignancy vs. infection / held until pathology"),
    "N-B-008": ("B. Rule 1 domain", "Crohn's flare"),
    "N-B-009": ("B. Rule 1 domain", "active flare of gout"),
    "N-B-037": ("B. Rule 1 domain", "ILD flare, not attributed to a rheumatic disease"),
    "N-B-038": ("B. Rule 1 domain", "PNA / Bronchiectasis flare"),
    "B-022": ("C. Rule 2 assertion", "believed to be drug-induced SLE"),
    "B-028": ("C. Rule 2 assertion", "Likely ___ SLE (diagnostic impression)"),
    "N-B-035": ("C. Rule 2 assertion", "likely a lupus flair"),
    "N-B-048": ("C. Rule 2 assertion", "presumed Stills Flare"),
    "N-B-056": ("C. Rule 2 assertion", "possible lupus cerebritis"),
    "N-B-036": ("D. Rule 3 temporal", "pt states he is in a flare, no escalation"),
}

M = pd.read_csv(os.path.join(OUT, "table_v12_metrics.csv"))
AU = pd.read_csv(os.path.join(OUT, "table_v12_auc.csv"))
J = json.load(open(os.path.join(OUT, "_v12.json"), encoding="utf-8"))


def cap(fig, s, y=-0.035, width=178):
    fig.text(0.5, y, "\n".join(textwrap.wrap(s, width)), ha="center",
             va="top", fontsize=7.2, color="#444444")


def f2(v):
    return "%.2f" % v


fig = plt.figure(figsize=(7.2, 5.3))
gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.30,
                      left=0.085, right=0.975, top=0.90, bottom=0.19)

# ---------------------------------------------------------------- A 迁移
axA = fig.add_subplot(gs[0, 0])
MG = pd.read_csv(os.path.join(OUT, "table_v12_migration.csv"))
labs, stay, flip = [], [], []
for r in MG.itertuples():
    labs.append("task %s\n%s" % (r.task, r.batch))
    stay.append(int(r.n_stay))
    flip.append(int(r.n_flip))
x = np.arange(len(labs))
axA.bar(x, stay, 0.58, color=C_GREY, label="stays positive under v2")
axA.bar(x, flip, 0.58, bottom=stay, color=C_MAIN, label="flips to non-positive")
for xi, (s, f_) in zip(x, zip(stay, flip)):
    axA.text(xi, s + f_ + 0.45, "%d/%d" % (f_, s + f_), ha="center",
             va="bottom", fontsize=7.0, color=C_MAIN if f_ else C_GREY)
axA.set_xticks(x)
axA.set_xticklabels(labs, fontsize=7.0)
axA.set_ylabel("previously positive cases")
axA.set_ylim(0, max(np.array(stay) + np.array(flip)) * 1.55)
axA.set_title("A  Which v1 positives survive the v2 manual", fontsize=8.8, pad=4)
axA.legend(fontsize=6.1, frameon=False, loc="upper center", ncol=2,
           handlelength=1.1, columnspacing=1.0)
axA.spines[["top", "right"]].set_visible(False)
axA.grid(axis="y", color="#DDDDDD", lw=0.5)
axA.set_axisbelow(True)

# ---------------------------------------------------------------- B 引擎 PPV
axB = fig.add_subplot(gs[0, 1])
picks = [("v7b", "B", "ACTIVITY-POSITIVE", "v7b\nB"),
         ("v9", "A", "INACTIVE", "v9\nA:INACT"),
         ("v9", "A", "UNDERDOC", "v9\nA:UNDOC"),
         ("v9", "B", "ACTIVITY-POSITIVE", "v9\nB")]
labs2, v1, v2 = [], [], []
for b, t, c, lab in picks:
    q = M[(M.batch == b) & (M.task == t) & (M.cls == c)]
    p1 = q[q.gold == "gold_v1"].iloc[0]
    p2 = q[q.gold == "gold_v2"].iloc[0]
    labs2.append(lab)
    v1.append(float(p1.ppv_w) if np.isfinite(p1.ppv_w) else 0.0)
    v2.append(float(p2.ppv_w) if np.isfinite(p2.ppv_w) else 0.0)
x2 = np.arange(len(labs2))
w = 0.36
axB.bar(x2 - w / 2, v1, w, color=C_SUB, alpha=0.9, edgecolor="white",
        linewidth=0.6, label="v1 gold")
axB.bar(x2 + w / 2, v2, w, color=C_MAIN, alpha=0.9, edgecolor="white",
        linewidth=0.6, label="v2 gold (revised)")
for xi, a, b_ in zip(x2, v1, v2):
    axB.text(xi - w / 2, a + 0.02, f2(a), ha="center", va="bottom",
             fontsize=6.4, color=C_SUB)
    axB.text(xi + w / 2, b_ + 0.02, f2(b_), ha="center", va="bottom",
             fontsize=6.4, color=C_MAIN)
axB.set_xticks(x2)
axB.set_xticklabels(labs2, fontsize=6.8)
axB.set_ylabel("weighted PPV of the v8 engine")
axB.set_ylim(0, 1.34)
axB.set_title("B  Fixing the manual does not rescue the labels", fontsize=8.8,
              pad=4)
axB.legend(fontsize=6.1, frameon=False, loc="upper left", ncol=1,
           handlelength=1.1)
axB.spines[["top", "right"]].set_visible(False)
axB.grid(axis="y", color="#DDDDDD", lw=0.5)
axB.set_axisbelow(True)
axB.annotate("0.343 -> 0.114", xy=(3 + w / 2, 0.13), xytext=(2.30, 0.72),
             fontsize=6.3, color=C_MAIN,
             arrowprops=dict(arrowstyle="->", lw=0.6, color=C_MAIN))
axB.text(2.62, 0.60, "", fontsize=5.9, color="#8A8A8A", ha="center", va="top")

# ---------------------------------------------------------------- C 密度 AUC
axC = fig.add_subplot(gs[1, 0])
picksC = [("v7b", "A"), ("v7b", "B"), ("v9", "A"), ("v9", "B")]
labsC, a1, a2, lo1, hi1, lo2, hi2, nc = [], [], [], [], [], [], [], []
for b, t in picksC:
    def get(gv, var):
        q = AU[(AU.batch == b) & (AU.task == t) & (AU.gold == gv) &
               (AU["var"] == var)]
        return q.iloc[0]
    A1, A2 = get("gold_v1", "aid_doc"), get("gold_v2", "aid_doc")
    N2 = get("gold_v2", "narr_char")
    labsC.append("%s\n%s" % (b, "task %s" % t))
    a1.append(A1.auc); a2.append(A2.auc)
    lo1.append(A1.lo); hi1.append(A1.hi)
    lo2.append(A2.lo); hi2.append(A2.hi)
    nc.append(N2.auc)
x3 = np.arange(len(labsC))
axC.bar(x3 - w / 2, a1, w, color=C_SUB, alpha=0.9, edgecolor="white",
        linewidth=0.6, label="v1 gold")
axC.bar(x3 + w / 2, a2, w, color=C_MAIN, alpha=0.9, edgecolor="white",
        linewidth=0.6, label="v2 gold (revised)")
for xi, (lo_, hi_) in zip(x3 - w / 2, zip(lo1, hi1)):
    axC.plot([xi, xi], [lo_, hi_], color="#1B3B57", lw=0.9)
for xi, (lo_, hi_) in zip(x3 + w / 2, zip(lo2, hi2)):
    axC.plot([xi, xi], [lo_, hi_], color="#6E2118", lw=0.9)
for xi, v in zip(x3 + w / 2, a2):
    axC.text(xi, v + 0.014, f2(v), ha="center", va="bottom", fontsize=6.4,
             color=C_MAIN)
axC.axhline(0.5, color=C_GREY, lw=0.7, ls=":")
axC.text(-0.46, 0.515, "chance", fontsize=6.0, color=C_GREY)
axC.set_xticks(x3)
axC.set_xticklabels(labsC, fontsize=6.8)
axC.set_ylabel("AUC of activity-information density")
axC.set_ylim(0, 1.34)
axC.set_title("C  The continuous density gets stronger", fontsize=8.8, pad=4)
axC.legend(fontsize=6.1, frameon=False, loc="upper center", ncol=2,
           handlelength=1.1, columnspacing=1.0, bbox_to_anchor=(0.5, 1.005))
axC.spines[["top", "right"]].set_visible(False)
axC.grid(axis="y", color="#DDDDDD", lw=0.5)
axC.set_axisbelow(True)

# ---------------------------------------------------------------- D 归因
axD = fig.add_subplot(gs[1, 1])
order = ["A. INF_DEFER clause removed", "B. Rule 1 domain",
         "C. Rule 2 assertion", "D. Rule 3 temporal"]
cnt = {k: 0 for k in order}
for cid, (rule, _) in FLIP_REASON.items():
    cnt[rule] += 1
y4 = np.arange(len(order))
vals = [cnt[k] for k in order]
cols = [C_GREY, C_OK, C_MAIN, C_SUB]
axD.barh(y4, vals, 0.55, color=cols, alpha=0.9, edgecolor="white",
         linewidth=0.6)
for yi, v in zip(y4, vals):
    axD.text(v + 0.09, yi, "%d" % v, ha="left", va="center", fontsize=7.2,
             color="#333333")
axD.set_yticks(y4)
axD.set_yticklabels(["INF_DEFER\nclause removed", "Rule 1\ndomain",
                     "Rule 2\nassertion", "Rule 3\ntemporal"], fontsize=6.9)
axD.set_xlabel("cases flipped by that rule")
axD.set_xlim(0, max(vals) + 1.1)
axD.set_title("D  Which rule did the flips (11 cases)", fontsize=8.8, pad=4)
axD.spines[["top", "right"]].set_visible(False)
axD.grid(axis="x", color="#DDDDDD", lw=0.5)
axD.set_axisbelow(True)
axD.invert_yaxis()
axD.text(0.97, 0.94, "Rule 2 (assertion) alone\naccounts for 5 of 11",
         transform=axD.transAxes, fontsize=6.2, color="#777777", ha="right",
         va="top")

cap(fig, "Figure 40. Revising the coding manual.  (A) Of the 24 cases labelled positive under the "
         "v1 manual, %d no longer qualify once the v2 rules are applied; all %d v7b INF_DEFER cases "
         "survive but the single v9 INF_DEFER case does not.  (B) Weighted PPV of the v8 keyword "
         "engine against the v1 and v2 reference standards: correcting the gold standard makes the "
         "three-class labels fail harder (task B: 0.343 -> 0.114), because the engine's apparent true "
         "positives were largely the manual's false positives.  (C) Out-of-sample AUC of the "
         "continuous activity-information density rises from 0.911 to 0.968 (task A) and from 0.870 to "
         "0.911 (task B); bars show point estimates, whiskers the stratified-bootstrap 95%% CI.  "
         "(D) Attribution of the 11 flips to the individual manual rules, assigned by the authors "
         "from the raters' and arbiter's verbatim quotes."
         % (J["n_flip"], J["mig_A_v7b_n"]), y=-0.04, width=182)

fig.savefig(os.path.join(FIG, "Figure40_manual_v2.png"), dpi=320)
fig.savefig(os.path.join(FIG, "Figure40_manual_v2.pdf"))
plt.close(fig)
print("saved Figure40_manual_v2.png / .pdf")

# 归因表落盘
rows = [dict(case_id=c, rule=FLIP_REASON[c][0], decisive_quote=FLIP_REASON[c][1])
        for c in sorted(FLIP_REASON)]
pd.DataFrame(rows).to_csv(os.path.join(OUT, "table_v12_flip_reason.csv"),
                          index=False)
print("saved table_v12_flip_reason.csv (%d rows)" % len(rows))
