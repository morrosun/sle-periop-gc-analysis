# -*- coding: utf-8 -*-
"""
205_figures_v11.py —— Figure 39：连续密度作为文本侧主输出后的交互分析

面板
  A  密度轴的「分辨力」：各候选密度变量的正密度例数 / 正侧事件数 / 零值率
     —— 直接说明为什么密度轴做不了剂量-反应
  B  交互森林图：整层 vs 各密度分层（对数轴，超宽 CI 截断并标注）
  C  三项交互的最小可检 OR（MDE）与实测修饰 OR —— 「阴性 ≠ 无修饰」的可视化
  D  记录长度五分位的交互 + 跨层趋势（唯一有分辨力的对照轴）

风格与 v7–v10 各版图一致（白底、DejaVu Sans、图注 textwrap、图内不出现中文）。
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

C_MAIN = "#B03A2E"      # 主估计（红，中国习惯：风险方向）
C_SUB = "#2B6CB0"       # 分层
C_GREY = "#9AA0A6"
C_OK = "#1E8449"

J = json.load(open(os.path.join(OUT, "_v11.json"), encoding="utf-8"))
AX = pd.DataFrame(J["axis"])
ST = pd.DataFrame(J["strata"])
TW = pd.DataFrame(J["threeway"])
LN = pd.DataFrame(J["length"])
AN = J["anchor_ci"]


def cap(fig, s, y=-0.035, width=176):
    fig.text(0.5, y, "\n".join(textwrap.wrap(s, width)), ha="center",
             va="top", fontsize=7.2, color="#444444")


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, "%s.%s" % (name, ext)), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


fig = plt.figure(figsize=(11.2, 8.4))
gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.30,
                      left=0.085, right=0.975, top=0.925, bottom=0.115)

# ---------------------------------------------------------------- A 分辨力
axA = fig.add_subplot(gs[0, 0])
a = AX[AX["var"] != "narr_char"].reset_index(drop=True)
x = np.arange(len(a))
w = 0.38
b1 = axA.bar(x - w / 2, a["n_pos"], w, color=C_SUB, alpha=0.85,
             edgecolor="white", linewidth=0.6, label="cases with density > 0")
b2 = axA.bar(x + w / 2, a["ev_pos"], w, color=C_MAIN, alpha=0.9,
             edgecolor="white", linewidth=0.6, label="events in that side")
for xi, v in zip(x - w / 2, a["n_pos"]):
    axA.text(xi, v * 1.12, "%d" % int(v), ha="center", va="bottom",
             fontsize=6.6, color=C_SUB)
for xi, v in zip(x + w / 2, a["ev_pos"]):
    axA.text(xi, max(v, 0.35) * 1.12, "%d" % int(v), ha="center", va="bottom",
             fontsize=6.6, color=C_MAIN)
axA.set_yscale("log")
axA.set_ylim(0.3, 1400)
axA.set_xticks(x)
axA.set_xticklabels([s.replace("（主）", "\n(primary)") for s in a["var"]],
                    fontsize=6.6)
axA.set_ylabel("count (log scale)")
axA.set_title("A  The density axis has almost no events to work with",
              fontsize=8.8, pad=4)
axA.legend(fontsize=6.3, frameon=False, loc="upper right",
           borderaxespad=0.3)
axA.spines[["top", "right"]].set_visible(False)
axA.grid(axis="y", color="#E4E4E4", lw=0.5)
axA.set_axisbelow(True)
axA.annotate("the whole layer holds %d events" % int(AN["n_ev"]),
             xy=(0, 0), xytext=(0.03, 0.06), textcoords="axes fraction",
             fontsize=6.4, color="#666666")
axA.annotate("zero-inflation: %.0f%% of cases sit at exactly 0"
             % float(a.iloc[0]["zero_pct"]),
             xy=(0, 0), xytext=(0.03, 0.155), textcoords="axes fraction",
             fontsize=6.4, color="#666666")

# ---------------------------------------------------------------- B 森林
axB = fig.add_subplot(gs[0, 1])
rows = []
rows.append(("whole layer (primary)", AN["OR"], AN["lo"], AN["hi"], C_MAIN, 1.4))
for _, r in ST.iterrows():
    if r["stratum"] == "全 spec− 层":
        continue
    col = C_GREY if not np.isfinite(r["OR"]) else C_SUB
    rows.append((r["stratum"], r["OR"], r["lo"], r["hi"], col, 1.0))
LABB = {"aid_doc = 0": "aid_doc = 0 (undocumented)",
        "aid_doc > 0": "aid_doc > 0 (documented)",
        "  0 < aid_doc ≤ 正侧中位": "  0 < aid_doc ≤ median(+)",
        "  aid_doc > 正侧中位": "  aid_doc > median(+)",
        "aid_broad = 0": "aid_broad = 0",
        "aid_broad > 0": "aid_broad > 0",
        "info_n = 0": "info_n = 0",
        "info_n = 1": "info_n = 1",
        "info_n ≥ 2": "info_n ≥ 2"}
ypos = np.arange(len(rows))[::-1]
CLIP = 60.0
for (lab, orr, lo, hi, col, lw), yy in zip(rows, ypos):
    if np.isfinite(orr):
        lo_c, hi_c = max(lo, 1 / CLIP), min(hi, CLIP)
        axB.plot([lo_c, hi_c], [yy, yy], color=col, lw=lw, solid_capstyle="butt")
        axB.plot([orr], [yy], "o", color=col, ms=4.4 if lw > 1 else 3.6)
        if hi > CLIP or lo < 1 / CLIP:
            axB.text(min(hi_c * 1.25, CLIP), yy, "↗", fontsize=7,
                     color=col, va="center")
            axB.text(max(1 / CLIP * 0.8, lo_c * 0.8), yy, "↖", fontsize=7,
                     color=col, va="center", ha="right")
    else:
        axB.text(1.0, yy, "not estimable", fontsize=6.4, color=col,
                 va="center", ha="center")
axB.axvline(1.0, color="#888888", lw=0.9, ls="--")
axB.set_xscale("log")
axB.set_xlim(1 / CLIP, CLIP)
axB.set_yticks(ypos)
axB.set_yticklabels([LABB.get(l, l) for l, *_ in rows], fontsize=6.6)
axB.set_xlabel("RA/SLE dose-slope ratio  (95% CI)")
axB.set_title("B  The whole-layer estimate IS the undocumented-layer estimate",
              fontsize=8.8, pad=4)
axB.spines[["top", "right"]].set_visible(False)
axB.grid(axis="x", color="#E4E4E4", lw=0.5)
axB.set_axisbelow(True)

# ---------------------------------------------------------------- C 最小可检
axC = fig.add_subplot(gs[1, 0])
t = TW[TW["status"] == "ok"].reset_index(drop=True)
lab = ["%s\n[%s]" % (m.replace("（主）", "(primary)"),
                     c.replace("+长度", "+length").replace("CORE", "CORE"))
       for m, c in zip(t["mod"], t["covs"])]
y = np.arange(len(t))[::-1]
for (orr, lo, hi, mde), yy, lb in zip(t[["OR", "lo", "hi", "mde"]].values, y, lab):
    axC.plot([lo, hi], [yy, yy], color=C_SUB, lw=1.1, alpha=0.55)
    axC.plot([orr], [yy], "o", color=C_SUB, ms=4.2)
    axC.plot([mde], [yy], "|", color=C_MAIN, ms=11, mew=1.8)
    axC.text(mde * 1.06, yy, "MDE %.2f" % mde, fontsize=6.2, color=C_MAIN,
             va="center",
             bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                       alpha=0.85))
axC.axvline(1.0, color="#888888", lw=0.9, ls="--")
axC.set_yticks(y)
axC.set_yticklabels(lab, fontsize=6.3)
axC.set_xscale("log")
axC.set_xlim(0.35, 12)
axC.set_xlabel("three-way interaction OR  (• estimate, — 95% CI,  | minimum detectable)")
axC.set_title("C  A 'null' three-way term cannot exclude small modification",
              fontsize=8.8, pad=4)
axC.spines[["top", "right"]].set_visible(False)
axC.grid(axis="x", color="#E4E4E4", lw=0.5)
axC.set_axisbelow(True)
axC.annotate("only modifiers ≥ MDE would have been seen", xy=(0, 0),
             xytext=(0.42, 0.03), textcoords="axes fraction", fontsize=6.3,
             color="#666666")

# ---------------------------------------------------------------- D 长度
axD = fig.add_subplot(gs[1, 1])
y = np.arange(len(LN))[::-1]
for (lab_, orr, lo, hi, ne), yy in zip(
        LN[["band", "OR", "lo", "hi", "n_ev"]].values, y):
    if np.isfinite(orr):
        lo_c, hi_c = max(lo, 0.05), min(hi, 20)
        axD.plot([lo_c, hi_c], [yy, yy], color=C_SUB, lw=1.2)
        axD.plot([orr], [yy], "o", color=C_SUB, ms=4.4)
        axD.text(hi_c * 1.1, yy, "ev=%d" % int(ne), fontsize=6.2,
                 color="#666666", va="center")
    else:
        axD.text(1.0, yy, "not estimable", fontsize=6.4, color=C_GREY,
                 va="center", ha="center")
axD.axvline(AN["OR"], color=C_MAIN, lw=1.2, ls="-")
axD.text(AN["OR"] * 1.04, y[-1] + 0.42, "whole layer %.2f" % AN["OR"],
         fontsize=6.3, color=C_MAIN)
axD.axvline(1.0, color="#888888", lw=0.9, ls="--")
axD.set_yticks(y)
axD.set_yticklabels(LN["band"], fontsize=7.0)
axD.set_xscale("log")
axD.set_xlim(0.05, 40)
axD.set_xlabel("RA/SLE dose-slope ratio  (95% CI)")
axD.set_title("D  The only axis with resolution (record length) does NOT overturn it",
              fontsize=8.8, pad=4)
axD.spines[["top", "right"]].set_visible(False)
axD.grid(axis="x", color="#E4E4E4", lw=0.5)
axD.set_axisbelow(True)
if J.get("length_trend"):
    lt = J["length_trend"]
    axD.annotate("trend across quintiles:  %.2f× per quintile (P=%.3f)"
                 % (lt["mult"], lt["p"]), xy=(0, 0),
                 xytext=(0.03, 0.04), textcoords="axes fraction",
                 fontsize=6.4, color="#666666")

fig.suptitle("Figure 39. Continuous activity-information density as the text-side "
             "primary output: the disease x dose interaction re-estimated",
             fontsize=10.4, y=0.978)
cap(fig, "Panel A: in the spec− layer the density variables are zero-inflated "
         "(aid_doc = 0 in %.1f%% of cases); the density-positive side holds only "
         "%d events of %d, so no dose-response along the density axis is "
         "identifiable. Panel B: the whole-layer RA/SLE dose-slope ratio (%.2f, "
         "%.2f-%.2f) is numerically the undocumented-layer estimate, which carries "
         "%d of %d events. Panel C: three-way s x SLE x density terms are all "
         "non-significant, but the minimum detectable modifier OR is %.1f-%.1f, so "
         "only large modification is excluded. Panel D: record-length quintiles "
         "(the one axis with events in every stratum) show no monotone trend and "
         "adjusting for length does not attenuate the interaction."
         % (float(AX[AX["var"] == "aid_doc（主）"].iloc[0]["zero_pct"]),
            int(AX[AX["var"] == "aid_doc（主）"].iloc[0]["ev_pos"]),
            int(AN["n_ev"]), AN["OR"], AN["lo"], AN["hi"],
            int(ST[ST["stratum"] == "aid_doc = 0"].iloc[0]["n_ev"]),
            int(AN["n_ev"]),
            float(t["mde"].min()), float(t["mde"].max())), y=-0.005)

save(fig, "Figure39_density_interaction")
print("[saved] Figure39_density_interaction.png/pdf")
