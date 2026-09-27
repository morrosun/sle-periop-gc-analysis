# -*- coding: utf-8 -*-
"""Publication figures for the SLE ICU glucocorticoid dose-intensity study (all labels in English)."""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, OUT = os.path.join(BASE, "data"), os.path.join(BASE, "out")
FIG = os.path.join(OUT, "fig")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

ORDER = ["G0_none", "G1_low", "G2_mod", "G3_high"]
LAB = {"G0_none": "0 (none)", "G1_low": ">0-<10", "G2_mod": "10-<50", "G3_high": ">=50"}

df = pd.read_csv(os.path.join(DATA, "cohort_sle_icu_v3.csv"))
nc = pd.read_csv(os.path.join(DATA, "negctrl_vars.csv"))
df = df.merge(nc, on=["hadm_id", "stay_id"], how="left")
df["imm_any"] = (df[["imm_cyc", "imm_mmf", "imm_aza", "imm_cni", "imm_mtx", "imm_bio"]].max(axis=1) == 1).astype(int)
df["emerg"] = df["admission_type"].astype(str).str.contains("EMER", case=False, na=False).astype(int)
df["female"] = (df["gender"] == "F").astype(int)
for v in ["sofa", "charlson", "apsiii", "oasis", "wbc_min", "platelets_min", "creatinine_max"]:
    df[v] = df[v].fillna(df[v].median())
df["gc_str"] = pd.Categorical(np.where(df.gc_daily_pe_mg <= 0, "G0_none",
                              np.where(df.gc_daily_pe_mg < 10, "G1_low",
                              np.where(df.gc_daily_pe_mg < 50, "G2_mod", "G3_high"))),
                              categories=ORDER)

COV = ["age", "female", "charlson", "sofa", "apsiii", "on_vent", "on_vaso", "on_rrt",
       "lupus_nephritis", "imm_any", "emerg", "wbc_min", "platelets_min", "creatinine_max"]
BASE_COV = "age + female + charlson + sofa + on_vent + on_vaso + on_rrt + lupus_nephritis + imm_any + emerg"
REF = "C(gc_str, Treatment(reference='G0_none'))"

# ============================================================ Figure 1: flow diagram
fig, ax = plt.subplots(figsize=(6.4, 7.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")
def box(x, y, w, h, title, sub="", fc="#EAF3FB", ec="#2B6CB0"):
    ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fc, edgecolor=ec, lw=1.0, zorder=2))
    ax.text(x, y + (0.18 if sub else 0), title, ha="center", va="center", fontsize=10, zorder=3)
    if sub:
        ax.text(x, y - 0.28, sub, ha="center", va="center", fontsize=8.5, color="#444", zorder=3)
def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#555"), zorder=1)

box(5, 11.0, 7.2, 1.1, "MIMIC-IV hospital admissions with an SLE diagnosis code",
    "ICD-10 M32.x or ICD-9 710.0 - 3,069 admissions / 1,158 patients", "#F1EFE8", "#888780")
arrow(5, 10.4, 5, 9.7)
box(5, 9.0, 7.2, 1.1, "Admissions with an ICU stay", "550 admissions (first ICU stay per admission)", "#F1EFE8", "#888780")
arrow(5, 8.4, 5, 7.7)
box(5, 7.0, 7.2, 1.1, "Adult (>=18 y) first ICU stay", "550 stays analysed", "#E6F1FB", "#2B6CB0")
arrow(5, 6.45, 2.4, 5.85)
arrow(5, 6.45, 6.6, 5.85)
box(2.4, 5.2, 2.9, 1.1, "No glucocorticoid", "n = 255 (46.4%)", "#E6F1FB", "#2B6CB0")
box(6.6, 5.2, 4.4, 1.1, "Glucocorticoid exposure", "n = 295 (53.6%)", "#E6F1FB", "#2B6CB0")
for xb in (2.2, 5.0, 7.8):
    arrow(6.6, 4.65, xb, 3.55)
box(2.2, 3.0, 2.4, 1.0, ">0-<10 mg/day", "n = 128", "#FAEEDA", "#B7791F")
box(5.0, 3.0, 2.4, 1.0, "10-<50 mg/day", "n = 98", "#FAEEDA", "#B7791F")
box(7.8, 3.0, 2.4, 1.0, ">=50 mg/day", "n = 69", "#FAEEDA", "#B7791F")
ax.text(5, 1.6, "Outcomes: in-hospital infection (n = 301); 30-day death (n = 54)",
        ha="center", fontsize=9, style="italic", color="#333")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "Figure1_flow.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(FIG, "Figure1_flow.pdf"), bbox_inches="tight")
plt.close()
print("Figure 1 done")

# ============================================================ Figure 2: spline dose-response
def rcs_basis(x, knots):
    x = np.asarray(x, float); k = np.asarray(knots, float); n = len(k); out = [x]
    for j in range(n - 2):
        d = (np.maximum(x - k[j], 0) ** 3
             - np.maximum(x - k[-2], 0) ** 3 * (k[n - 1] - k[j]) / (k[n - 1] - k[n - 2])
             + np.maximum(x - k[n - 1], 0) ** 3 * (k[n - 2] - k[j]) / (k[n - 1] - k[n - 2]))
        out.append(d / (k[n - 1] - k[0]) ** 2)
    return np.column_stack(out)

z = df["gc_daily_pe_mg"].values
exp_ = z > 0
knots = np.percentile(z[exp_], [5, 35, 65, 95])
B = rcs_basis(np.log(z[exp_] + 1), np.log(knots + 1))
Xr = np.zeros((len(df), B.shape[1])); Xr[exp_] = B
for i in range(B.shape[1]):
    df["rcs%d" % i] = Xr[:, i]
names = ["rcs%d" % i for i in range(B.shape[1])]
m = smf.logit("infection ~ " + " + ".join(names) + " + " + BASE_COV, data=df).fit(disp=0)
beta = np.array([m.params[n] for n in names])
V = m.cov_params().loc[names, names].values

grid = np.exp(np.linspace(np.log(1), np.log(300), 220))
Bg = rcs_basis(np.log(grid + 1), np.log(knots + 1))
lp = Bg @ beta
se = np.sqrt(np.einsum("ij,jk,ik->i", Bg, V, Bg))
or_, lo, hi = np.exp(lp), np.exp(lp - 1.96 * se), np.exp(lp + 1.96 * se)

fig, ax = plt.subplots(figsize=(6.6, 4.3))
ax.axhline(1.0, color="#888", lw=0.8, ls="--")
ax.fill_between(grid, lo, hi, color="#B5D4F4", alpha=0.55, zorder=1)
ax.plot(grid, or_, color="#185FA5", lw=2.0, zorder=2)
for d, lab in [(10, "10"), (50, "50")]:
    ax.axvline(d, color="#BBB", lw=0.8, ls=":", zorder=0)
ax.set_xscale("log")
ax.set_xticks([1, 5, 10, 25, 50, 100, 300])
ax.set_xticklabels(["1", "5", "10", "25", "50", "100", "300"])
ax.set_xlabel("Glucocorticoid dose intensity (mg/day prednisone-equivalent)")
ax.set_ylabel("Adjusted odds ratio for in-hospital infection")
ax.set_title("Dose-response vs zero exposure (reference, OR = 1)", fontsize=10)
ax.set_ylim(0.3, 4.0)
i50 = (np.abs(grid - 50)).argmin()
ax.plot([grid[i50]], [or_[i50]], "o", color="#A32D2D", ms=5, zorder=3)
ax.annotate("peak OR %.2f\n(%.2f-%.2f) at 50 mg/day" % (or_[i50], lo[i50], hi[i50]),
            xy=(grid[i50], or_[i50]), xytext=(grid[i50] * 1.5, or_[i50] + 0.9),
            fontsize=8.5, color="#A32D2D",
            arrowprops=dict(arrowstyle="->", color="#A32D2D", lw=0.9))
plt.tight_layout()
plt.savefig(os.path.join(FIG, "Figure2_dose_response.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(FIG, "Figure2_dose_response.pdf"), bbox_inches="tight")
plt.close()
print("Figure 2 done")

# ============================================================ Figure 3: forest plot
X = df[COV].values; X = (X - X.mean(0)) / X.std(0)
T = pd.Categorical(df["gc_str"], categories=ORDER).codes
ps = LogisticRegression(solver="lbfgs", max_iter=5000, C=1.0).fit(X, T)
PS = ps.predict_proba(X); marg = np.array([(T == g).mean() for g in range(4)])
df["sw"] = marg[T] / np.clip(PS[np.arange(len(T)), T], 1e-6, 1)

def get_or(m, pref, ref_str="C(gc_str, Treatment(reference='G0_none'))[T."):
    out = {}
    for nm in m.params.index:
        if nm.startswith(pref):
            k = nm.split("T.")[-1].rstrip("]")
            b, se = m.params[nm], m.bse[nm]
            out[k] = (np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), m.pvalues[nm])
    return out

m_crude = smf.logit("infection ~ " + REF, data=df).fit(disp=0)
m_adj = smf.logit("infection ~ " + REF + " + " + BASE_COV, data=df).fit(disp=0)
m_iptw = smf.glm("infection ~ " + REF, data=df, family=sm.families.Binomial(),
                 freq_weights=df["sw"]).fit()
m_cult = smf.logit("culture_after_icu ~ " + REF + " + " + BASE_COV, data=df).fit(disp=0)
m_abx = smf.logit("abx_new_after48 ~ " + REF + " + " + BASE_COV, data=df).fit(disp=0)

panels = [("Unadjusted", m_crude), ("Adjusted", m_adj), ("IPTW-weighted", m_iptw),
          ("Positive blood culture", m_cult), ("New antibiotic >48 h", m_abx)]

fig, ax = plt.subplots(figsize=(7.2, 5.0))
y = 0
for name, mm in panels:
    y -= 1
    ax.axhspan(y - 0.55, y + 0.55, color="#F7F7F5" if (y % 2 == 0) else "white", zorder=0)
    for g in ["G1_low", "G2_mod", "G3_high"]:
        r = get_or(mm, "C(gc_str")[g]
        yy = y + (0.3 if g == "G1_low" else (0.0 if g == "G2_mod" else -0.3))
        col = "#1D9E75" if g == "G1_low" else ("#D85A30" if g == "G2_mod" else "#7F77DD")
        ax.plot([r[1], r[2]], [yy, yy], color=col, lw=1.4, zorder=2)
        ax.plot(r[0], yy, "s", color=col, ms=4.5, zorder=3)
        ax.text(9.6, yy, "%.2f (%.2f-%.2f)" % (r[0], r[1], r[2]), fontsize=7.6, va="center", ha="left")
ax.set_xscale("log"); ax.set_xlim(0.28, 26)
ax.set_xticks([0.3, 0.5, 1, 2, 4, 8])
ax.set_xticklabels(["0.3", "0.5", "1", "2", "4", "8"])
from matplotlib.ticker import NullFormatter
ax.xaxis.set_minor_formatter(NullFormatter())
ax.axvline(1.0, color="#333", lw=0.9, ls="--", zorder=1)
ax.set_yticks([-1, -2, -3, -4, -5]); ax.set_yticklabels([p[0] for p in panels])
ax.set_xlabel("Odds ratio vs no glucocorticoid (95% CI)")
ax.set_title("In-hospital infection across glucocorticoid dose strata", fontsize=10)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color="#1D9E75", marker="s", ls="-", label=">0-<10 mg/day"),
                   Line2D([], [], color="#D85A30", marker="s", ls="-", label="10-<50 mg/day"),
                   Line2D([], [], color="#7F77DD", marker="s", ls="-", label=">=50 mg/day")],
          loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "Figure3_forest.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(FIG, "Figure3_forest.pdf"), bbox_inches="tight")
plt.close()
print("Figure 3 done")

# ============================================================ Figure 4: balance + weights
def max_smd(d, wcol=None):
    w = np.ones(len(d)) if wcol is None else d[wcol].values
    out = {}
    for v in COV:
        mm_ = d[v].values.astype(float)
        sd = np.sqrt(np.average((mm_ - np.average(mm_, weights=w)) ** 2, weights=w))
        if sd == 0:
            out[v] = 0.0; continue
        worst = 0
        for g in ["G1_low", "G2_mod", "G3_high"]:
            i1 = (d.gc_str == g).values; i0 = (d.gc_str == "G0_none").values
            worst = max(worst, abs((np.average(mm_[i1], weights=w[i1]) -
                                    np.average(mm_[i0], weights=w[i0])) / sd))
        out[v] = worst
    return out

su, sw_ = max_smd(df, None), max_smd(df, "sw")
fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
ax = axes[0]
yv = np.arange(len(COV))
ax.scatter([su[v] for v in COV], yv, s=26, color="#A32D2D", label="Unweighted", zorder=3)
ax.scatter([sw_[v] for v in COV], yv, s=26, color="#185FA5", label="IPTW-weighted", zorder=3)
for i, v in enumerate(COV):
    ax.plot([su[v], sw_[v]], [i, i], color="#BBB", lw=0.9, zorder=1)
ax.axvline(0.1, color="#555", ls="--", lw=0.9)
ax.set_yticks(yv)
ax.set_yticklabels([v.replace("_", " ") for v in COV], fontsize=7.8)
ax.set_xlabel("Maximum standardised mean difference vs no-glucocorticoid group")
ax.set_title("Covariate balance", fontsize=10)
ax.legend(fontsize=8, frameon=False)

ax = axes[1]
for g, c in zip(ORDER, ["#888780", "#1D9E75", "#D85A30", "#7F77DD"]):
    s = df.loc[df.gc_str == g, "sw"]
    ax.hist(s, bins=np.linspace(0, 6.2, 34), alpha=0.55, label=LAB[g], color=c)
ax.axvline(1.0, color="#333", ls="--", lw=0.9)
ax.set_xlabel("Stabilised inverse-probability weight")
ax.set_ylabel("Number of stays")
ax.set_title("Weight distribution (mean %.2f, range %.2f-%.2f)" % (df.sw.mean(), df.sw.min(), df.sw.max()), fontsize=10)
ax.legend(fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "Figure4_balance.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(FIG, "Figure4_balance.pdf"), bbox_inches="tight")
plt.close()
print("Figure 4 done")

# ============================================================ summary tables
mfull = m_adj
rows = []
for g in ["G1_low", "G2_mod", "G3_high"]:
    r = get_or(mfull, "C(gc_str")[g]
    ri = get_or(m_iptw, "C(gc_str")[g]
    rows.append(dict(stratum=LAB[g],
                     n=int((df.gc_str == g).sum()),
                     events=int(df.loc[df.gc_str == g, "infection"].sum()),
                     adj_OR=round(r[0], 2), adj_CI="%.2f-%.2f" % (r[1], r[2]), adj_P=round(r[3], 3),
                     iptw_OR=round(ri[0], 2), iptw_CI="%.2f-%.2f" % (ri[1], ri[2]), iptw_P=round(ri[3], 3)))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "table_dose_response.csv"), index=False)
print("\nsummary table saved -> out/table_dose_response.csv")
print(pd.DataFrame(rows).to_string(index=False))
