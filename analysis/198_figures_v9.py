# -*- coding: utf-8 -*-
"""
198_figures_v9.py —— Figure 37：样本内 → 样本外复现（v8 判定器）与连续密度的存活

面板
  A 逐类 PPV：样本内（v7b，n=148）vs 样本外（v9，n=131），点旁标 Fisher 精确 P
  B 算法-人类一致性 κ：样本内 vs 样本外（v8 与 v7 并列）
  C 连续变量 AUC：样本内 vs 样本外（含样本外的分层 bootstrap 95% CI），附长度对照
  D 失效模式：算法阳性判定率 vs 人类真阳性率（过度判定而非漏判）

风格与 v7/v8 各版图一致（白底、DejaVu Sans、图注 textwrap）。
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
DATA = os.path.join(ROOT, "data")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": "#555555", "axes.linewidth": 0.8,
    "axes.labelcolor": "#222222", "text.color": "#222222",
    "xtick.color": "#333333", "ytick.color": "#333333",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

C_IN, C_OUT = "#9AA0A6", "#B03A2E"
C_V8, C_V7 = "#2B6CB0", "#B7791F"
C_DEN = "#2F7A4F"

J8 = json.load(open(os.path.join(OUT, "_v8.json"), encoding="utf-8"))
CMP = pd.read_csv(os.path.join(OUT, "table_v9_compare.csv"))
FIS = pd.read_csv(os.path.join(OUT, "table_v9_compare_ci.csv"))
K8 = pd.read_csv(os.path.join(OUT, "table_v8_kappa.csv"))
K9 = pd.read_csv(os.path.join(OUT, "table_v9_kappa.csv"))
AUC9 = pd.read_csv(os.path.join(OUT, "table_v9_auc.csv"))
MODE = pd.read_csv(os.path.join(OUT, "table_v9_mode.csv"))


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
                fontweight="bold" if bold else "normal", clip_on=False)


def fig37():
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.7),
                             gridspec_kw=dict(wspace=0.62, left=0.05,
                                              right=0.985, top=0.78,
                                              bottom=0.32))

    # ---------------- A: PPV in vs out
    ax = axes[0]
    seq = ["INACTIVE", "INF_DEFER", "UNDERDOC", "ACTIVITY-POSITIVE"]
    labs = ["truly inactive\n(task A)", "infection-deferred\n(task A)",
            "undocumented\n(task A)", "activity-positive\n(task B)"]
    y = np.arange(len(seq))[::-1]
    for yi, cls in zip(y, seq):
        r = CMP[CMP.cls == cls]
        if not len(r):
            continue
        r = r.iloc[0]
        a, b = float(r.ppv_in), float(r.ppv_out)
        ax.plot([a, b], [yi, yi], color="#CCCCCC", lw=2.2, zorder=1,
                solid_capstyle="round")
        ax.plot([a], [yi], "o", ms=6.2, color=C_IN, zorder=3)
        ax.plot([b], [yi], "o", ms=6.2, color=C_OUT, zorder=3)
        p = FIS[FIS.cls == cls]
        ptxt = ("P=%.3f" % p.iloc[0].fisher_p) if len(p) else ""
        ax.annotate("%.2f (%d/%d)" % (b, int(r.n_out * b), int(r.n_out)),
                    xy=(b, yi), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=6.2, color=C_OUT)
        ax.annotate("%.2f (%d/%d)" % (a, int(r.n_in * a), int(r.n_in)),
                    xy=(a, yi), xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=6.2, color="#666666")
        ax.annotate(ptxt, xy=(1.10, yi), ha="left", va="center",
                    fontsize=6.0, color="#333", style="italic",
                    annotation_clip=False)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=6.8)
    ax.set_xlim(-0.06, 1.04)
    ax.set_ylim(-0.75, len(seq) - 0.25)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("positive predictive value", fontsize=7.4)
    ax.set_title("A  PPV: in-sample vs out-of-sample", fontsize=8.0,
                 loc="left", pad=20)
    ax.annotate("Fisher P", xy=(1.10, len(seq) - 0.42), ha="left", va="center",
                fontsize=5.8, color="#666", annotation_clip=False)
    ax.plot([], [], "o", color=C_IN, ms=6, label="in-sample (v7b, n=148)")
    ax.plot([], [], "o", color=C_OUT, ms=6, label="out-of-sample (v9, n=131)")
    ax.legend(frameon=False, fontsize=5.9, loc="lower center", ncol=2,
              bbox_to_anchor=(0.5, 1.005))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---------------- B: kappa in vs out
    ax = axes[1]
    groups = [("v8 · task A", "A", "v8"), ("v8 · task B", "B", "v8(narrow)"),
              ("v7 · task A", "A", "v7"), ("v7 · task B", "B", "v7")]
    x = np.arange(len(groups))
    wdt = 0.36
    for i, (nm, t, eng) in enumerate(groups):
        a = K8[(K8.task == t) & (K8.engine == eng)]
        b = K9[(K9.task == t) & (K9.engine == ("v8" if eng != "v7" else "v7"))]
        va = float(a.iloc[0].kappa) if len(a) else np.nan
        vb = float(b.iloc[0].kappa) if len(b) else np.nan
        col = C_V8 if eng != "v7" else C_V7
        ax.bar(i - wdt / 2, va, wdt, color=col, alpha=0.42,
               edgecolor="white", label="in-sample" if i == 0 else None)
        ax.bar(i + wdt / 2, vb, wdt, color=col, alpha=0.95,
               edgecolor="white", label="out-of-sample" if i == 0 else None)
        ax.annotate("%.2f" % va, xy=(i - wdt / 2, va), xytext=(0, 2),
                    textcoords="offset points", ha="center", fontsize=6.0,
                    color="#555")
        ax.annotate("%.2f" % vb, xy=(i + wdt / 2, vb), xytext=(0, 2),
                    textcoords="offset points", ha="center", fontsize=6.0,
                    color="#222")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=6.2, rotation=20,
                       ha="right")
    ax.set_ylim(-0.12, 1.05)
    ax.set_ylabel("Cohen's $\\kappa$ (algorithm vs human)", fontsize=7.0)
    ax.set_title("B  agreement: in-sample vs out-of-sample", fontsize=8.0,
                 loc="left", pad=20)
    ax.legend(frameon=False, fontsize=6.1, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---------------- C: AUC in vs out
    ax = axes[2]
    ins_auc = pd.DataFrame(J8["auc"])
    pairs = [("aid_doc", "A_documented", "A（有 vs 无）", "aid_doc · documented (task A)"),
             ("aid_doc", "B_activity", "B（活动 vs 纯否定）", "aid_doc · affirmative (task B)"),
             ("aid_broad", "A_documented", "A（有 vs 无）", "aid_broad · documented (task A)"),
             ("aid_broad", "B_activity", "B（活动 vs 纯否定）", "aid_broad · affirmative (task B)"),
             ("narr_char", "A_documented", "A（有 vs 无）", "narr_char · length control (task A)")]
    y = np.arange(len(pairs))[::-1]
    for yi, (var, task, tag, disp) in zip(y, pairs):
        a = ins_auc[(ins_auc["var"] == var) & (ins_auc.task == task)]
        b = AUC9[(AUC9["var"] == var) & (AUC9.task == tag)]
        if not len(a) or not len(b):
            continue
        va = float(a.iloc[0].auc)
        vb = float(b.iloc[0].auc)
        lo, hi = lo_hi(var, tag)
        col = C_DEN if var != "narr_char" else "#888888"
        ax.plot([lo, hi], [yi, yi], color=col, lw=1.4, alpha=0.75)
        ax.plot([va], [yi], "o", ms=5.6, color=col, alpha=0.42, zorder=3)
        ax.plot([vb], [yi], "o", ms=5.6, color=col, zorder=4)
        ax.annotate("%.3f" % va, xy=(va, yi), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=6.0,
                    color="#666")
        ax.annotate("%.3f" % vb, xy=(vb, yi), xytext=(0, -13),
                    textcoords="offset points", ha="center", fontsize=6.0,
                    color=col)
    ax.axvline(0.5, color="#BBBBBB", lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(["aid_doc — task A", "aid_doc — task B",
                        "aid_broad — task A", "aid_broad — task B",
                        "narr_char (length) — task A"], fontsize=6.2)
    ax.set_xlim(0.25, 1.06)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-0.75, len(pairs) - 0.25)
    ax.set_xlabel("AUC (95% CI out-of-sample)", fontsize=7.4)
    ax.set_title("C  the continuous density survives", fontsize=8.0,
                 loc="left", pad=20)
    ax.text(0.52, len(pairs) - 0.55, "chance", fontsize=5.6, color="#999",
            ha="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---------------- D: over-calling
    ax = axes[3]
    rows = ["INACTIVE", "UNDERDOC", "ACTIVITY-POSITIVE"]
    labs = ["truly inactive", "undocumented", "activity-positive\n(task B)"]
    x = np.arange(len(rows))
    algo = [float(MODE[MODE.cls == c].iloc[0].algo_pos_rate) for c in rows]
    true = [float(MODE[MODE.cls == c].iloc[0].true_pos_rate) for c in rows]
    ax.bar(x - 0.19, algo, 0.36, color=C_OUT, alpha=0.85, edgecolor="white",
           label="algorithm calls positive")
    ax.bar(x + 0.19, true, 0.36, color="#4C78A8", alpha=0.85, edgecolor="white",
           label="human reference positive")
    for i, (a, t) in enumerate(zip(algo, true)):
        ax.annotate("%.1f%%" % a, xy=(i - 0.19, a), xytext=(0, 2),
                    textcoords="offset points", ha="center", fontsize=6.0,
                    color=C_OUT)
        ax.annotate("%.1f%%" % t, xy=(i + 0.19, t), xytext=(0, 2),
                    textcoords="offset points", ha="center", fontsize=6.0,
                    color="#2E5A87")
        ax.annotate("sens %.2f\nPPV %.2f" % (
            MODE[MODE.cls == rows[i]].iloc[0].sens,
            MODE[MODE.cls == rows[i]].iloc[0].ppv),
            xy=(i, max(a, t) + 18), ha="center", fontsize=5.9, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=6.6)
    ax.set_ylim(0, 132)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("% of sampled cases", fontsize=7.4)
    ax.set_title("D  failure mode: over-calling, not missing", fontsize=8.0,
                 loc="left", pad=20)
    ax.legend(frameon=False, fontsize=5.9, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.02))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    cap(fig, "Figure 37. Out-of-sample replication of the v8 keyword engine on a "
             "completely fresh blinded sample (131 cases, zero overlap with the "
             "148 cases used to build the rules). A: per-class positive "
             "predictive value in-sample vs out-of-sample (Fisher exact P for "
             "the difference). B: Cohen's kappa against the human reference. "
             "C: discriminative performance of the continuous activity-"
             "information density (bootstrap 95% CI out-of-sample; narr_char is "
             "a length-only control). D: the engine calls far more cases "
             "positive than the reference does, while sensitivity stays at "
             "1.00 -- the failure mode is over-calling, not missing.", y=-0.10)
    save(fig, "Figure37_out_of_sample_replication")


_CI = {}


def set_ci():
    """从 172 的文本里取样本外 AUC 的 bootstrap CI（保证与报告同源）。"""
    p = os.path.join(OUT, "172_replication_tests.txt")
    if not os.path.exists(p):
        return
    for ln in open(p, encoding="utf-8"):
        for tag, var in [("任务 A aid_doc AUC", ("aid_doc", "A（有 vs 无）")),
                         ("任务 B aid_doc AUC", ("aid_doc", "B（活动 vs 纯否定）")),
                         ("任务 A aid_broad AUC", ("aid_broad", "A（有 vs 无）")),
                         ("任务 B aid_broad AUC", ("aid_broad", "B（活动 vs 纯否定）"))]:
            if ln.strip().startswith(tag):
                ci = ln.split()[-1]
                lo, hi = ci.split("–")
                _CI[var] = (float(lo), float(hi))


def lo_hi(var, tag):
    return _CI.get((var, tag), (np.nan, np.nan))


if __name__ == "__main__":
    set_ci()
    print("  CI loaded:", len(_CI))
    fig37()
