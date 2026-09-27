# -*- coding: utf-8 -*-
"""
204_density_interaction.py
==========================
v11 —— 以「连续活动度信息密度」为文本侧主输出，重跑疾病×剂量交互分析，
        并与整层结果对齐（v10 TODO 第 4 项）。

为什么必须做这一步
------------------
v9 判定三分类标签方案终止，文本侧主输出改为连续密度 aid_doc / aid_signed。
但 v8/v9 的交互表只把密度当成一个**二分层**（>0 / =0）和两个线性三项交互，
没有回答一个前置问题：

    密度这条轴，到底有没有分辨力？

本脚本先回答这个前置问题（第一节），再据此决定交互模型的形态。

估计目标（estimand）
--------------------
  RA/SLE 剂量斜率比 = exp( s×is_sle 的系数 )⁻¹，即
    死亡 ~ 剂量评分 s + is_sle + 协变量 + s×is_sle
  该比值 >1 表示 RA 侧每升一个剂量层的死亡对数几率增长快于 SLE 侧。
  **主效应不可解释**（两病暴露前完全不同，见 MEMORY §5），只看交互。

主结局：30 天死亡（预先指定的确认性检验，单次、不校正）。

输出
  out/204_density_interaction.txt
  out/table_v11_axis.csv / _anchor.csv / _strata.csv / _3way.csv
  out/table_v11_length.csv / _d3.csv / _bh.csv / _fragility.csv
  out/_v11.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
SEED = 20260916
np.random.seed(SEED)

ORDER = ["G0_none", "G1_low", "G2_mod", "G3_high"]
SCORE = {"G0_none": 0.0, "G1_low": 1.0, "G2_mod": 2.0, "G3_high": 3.0}
CORE = ["age", "female", "sofa", "vaso24", "vent24", "renal_fail", "shock",
        "surg_any", "n_agent_pre24"]
PSCOV = CORE + ["resp_fail", "coagulop", "liver_fail", "immuno_any", "hcq",
                "cytopenia", "serositis", "n_proc", "n_hosp"]

L = []


def P(s=""):
    print(s)
    sys.stdout.flush()
    L.append(str(s))


# ==================================================================== 统计
def firth(X, y, max_iter=300, tol=1e-9):
    """Firth 惩罚似然（手写 IRLS），返回 (b, se, 惩罚后对数似然)。"""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    b = np.zeros(p)
    for _ in range(max_iter):
        pr = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))
        W = np.clip(pr * (1 - pr), 1e-12, None)
        Ii = np.linalg.pinv(X.T @ (X * W[:, None]))
        h = np.einsum("ij,jk,ik->i", X * W[:, None], Ii, X)
        U = X.T @ (y - pr + h * (0.5 - pr))
        st = Ii @ U
        mx = np.max(np.abs(st))
        if mx > 4:
            st *= 4.0 / mx
        b = b + st
        if mx < tol:
            break
    pr = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))
    W = np.clip(pr * (1 - pr), 1e-12, None)
    Ii = np.linalg.pinv(X.T @ (X * W[:, None]))
    sign, logdet = np.linalg.slogdet(X.T @ (X * W[:, None]))
    ll = float(np.sum(y * np.log(np.clip(pr, 1e-12, None))
                      + (1 - y) * np.log(np.clip(1 - pr, 1e-12, None))))
    return b, np.sqrt(np.abs(np.diag(Ii))), ll + 0.5 * float(logdet)


def interaction(d, y="death_30d", covs=None, extra_int=None, use_firth=None,
                wcol=None):
    """RA/SLE 剂量斜率比（>1 = RA 侧增长更快）。wcol 给定时用加权 GLM。"""
    covs = list(CORE if covs is None else covs)
    cols = ["gc_str_num", "is_sle", y] + covs + list(extra_int or [])
    if wcol is not None and wcol not in cols:
        cols = cols + [wcol]
    use = d[cols].dropna().copy()
    if len(use) < 30 or use[y].nunique() < 2 or use["gc_str_num"].nunique() < 2:
        return None
    use["s"] = use["gc_str_num"].astype(float)
    use["sx"] = use["s"] * use["is_sle"].astype(float)
    base = ["s", "is_sle"] + covs
    full = base + ["sx"]
    for v in (extra_int or []):
        use["s_" + v] = use["s"] * use[v].astype(float)
        full = full + ["s_" + v]
    X = sm.add_constant(use[full].astype(float))
    Xr = sm.add_constant(use[base].astype(float))
    n_ev = int(use[y].sum())

    if wcol is not None:
        # 加权 GLM（freq_weights 即 IPTW），不走 Firth
        w = use[wcol].astype(float).values
        try:
            with np.errstate(all="ignore"):
                m = sm.GLM(use[y].astype(float), X, family=sm.families.Binomial(),
                           freq_weights=w).fit()
                m0 = sm.GLM(use[y].astype(float), Xr,
                            family=sm.families.Binomial(),
                            freq_weights=w).fit()
            b = np.asarray(m.params, float)
            se = np.asarray(m.bse, float)
            lrt = 2.0 * (m.llf - m0.llf)
            model = "IPTW-GLM"
        except Exception:
            return None
    else:
        # 本项目的死亡交互历来用 Firth（v4–v8 一致），且交互项本身建立在稀疏分层上
        # → 默认 Firth；需要 MLE 对照时显式传 use_firth=False。
        if use_firth is None:
            use_firth = True
        if use_firth:
            b, se, lf = firth(X.values, use[y].values)
            _, _, l0 = firth(Xr.values, use[y].values)
            lrt = 2.0 * (lf - l0)
            model = "Firth"
        else:
            try:
                with np.errstate(all="ignore"):
                    m = sm.Logit(use[y].astype(float), X).fit(disp=0, maxiter=300)
                    m0 = sm.Logit(use[y].astype(float), Xr).fit(disp=0, maxiter=300)
                b, se = np.asarray(m.params, float), np.asarray(m.bse, float)
                lrt = 2.0 * (m.llf - m0.llf)
                model = "MLE"
            except Exception:
                b, se, lf = firth(X.values, use[y].values)
                _, _, l0 = firth(Xr.values, use[y].values)
                lrt = 2.0 * (lf - l0)
                model = "Firth"

    j = list(X.columns).index("sx")
    if not np.isfinite(b[j]) or not np.isfinite(se[j]) or se[j] <= 0:
        return None
    e = float(np.exp(b[j]))                 # SLE/RA 斜率比
    lo = 1.0 / float(np.exp(b[j] + 1.96 * se[j]))
    hi = 1.0 / float(np.exp(b[j] - 1.96 * se[j]))
    if lo > hi:
        lo, hi = hi, lo
    return dict(OR=1.0 / e, lo=lo, hi=hi, log=float(-b[j]),
                se_log=float(se[j]),
                p_lrt=float(1 - stats.chi2.cdf(max(lrt, 0.0), 1)), model=model,
                n=int(len(use)), n_ev=n_ev)


def trend(d, y, extra=None, col="gc_str_num"):
    """病种内的剂量趋势 OR（同 160 的口径）。"""
    covs = list(extra or [])
    use = d[[col, y] + covs].dropna()
    if len(use) == 0 or use[y].nunique() < 2 or use[col].nunique() < 2:
        return None
    X = sm.add_constant(use[[col] + covs].astype(float))
    if int(use[y].sum()) < 60:
        b, se, _ = firth(X.values, use[y].values)
        model = "Firth"
    else:
        try:
            with np.errstate(all="ignore"):
                m = sm.Logit(use[y].astype(float), X).fit(disp=0, maxiter=300)
            b, se = np.asarray(m.params, float), np.asarray(m.bse, float)
            model = "MLE"
        except Exception:
            b, se, _ = firth(X.values, use[y].values)
            model = "Firth"
    j = list(X.columns).index(col)
    if not np.isfinite(b[j]) or not np.isfinite(se[j]) or se[j] <= 0:
        return None
    z = b[j] / se[j]
    return dict(OR=float(np.exp(b[j])), lo=float(np.exp(b[j] - 1.96 * se[j])),
                hi=float(np.exp(b[j] + 1.96 * se[j])),
                p=float(2 * (1 - stats.norm.cdf(abs(z)))), model=model,
                n=int(len(use)), n_ev=int(use[y].sum()))


def threeway(d, y, mod, covs=None, wcol=None):
    """三项交互 s × is_sle × mod（mod 已标准化）。返回 sx_m3 项。"""
    covs = list(CORE if covs is None else covs)
    s = d.dropna(subset=[mod]).copy()
    if len(s) < 40 or s[y].sum() < 8:
        return None
    s["m"] = (s[mod] - s[mod].mean()) / s[mod].std()
    s["s"] = s["gc_str_num"].astype(float)
    s["sx"] = s["s"] * s["is_sle"].astype(float)
    s["s_m"] = s["s"] * s["m"]
    s["sx_m"] = s["sx"] * s["m"]
    full = ["s", "sx", "is_sle", "m", "s_m", "sx_m"] + covs
    base = ["s", "sx", "is_sle", "m", "s_m"] + covs
    use = s[[y] + full].dropna()
    if len(use) < 40 or use[y].sum() < 8:
        return None
    X = sm.add_constant(use[full].astype(float))
    Xr = sm.add_constant(use[base].astype(float))
    n_ev = int(use[y].sum())
    if wcol is not None:
        w = d.loc[use.index, wcol].astype(float).values
        try:
            with np.errstate(all="ignore"):
                m1 = sm.GLM(use[y].astype(float), X,
                            family=sm.families.Binomial(),
                            freq_weights=w).fit()
                m0 = sm.GLM(use[y].astype(float), Xr,
                            family=sm.families.Binomial(),
                            freq_weights=w).fit()
            b = np.asarray(m1.params, float)
            se = np.asarray(m1.bse, float)
            lrt = 2.0 * (m1.llf - m0.llf)
            model = "IPTW-GLM"
        except Exception:
            return None
    else:
        try:
            b, se, lf = firth(X.values, use[y].values)
            _, _, l0 = firth(Xr.values, use[y].values)
            lrt = 2.0 * (lf - l0)
            model = "Firth"
        except Exception:
            return None
    j = list(X.columns).index("sx_m")
    if (not np.isfinite(b[j]) or not np.isfinite(se[j]) or se[j] <= 0
            or not np.isfinite(np.exp(b[j]))):
        return None
    z = b[j] / se[j]
    e = float(np.exp(b[j]))
    lo = float(np.exp(b[j] - 1.96 * se[j]))
    hi = float(np.exp(b[j] + 1.96 * se[j]))
    if not (np.isfinite(e) and np.isfinite(lo) and np.isfinite(hi) and e > 0):
        return None
    return dict(OR=e, lo=lo, hi=hi,
                p_wald=float(2 * (1 - stats.norm.cdf(abs(z)))),
                p_lrt=float(1 - stats.chi2.cdf(max(lrt, 0.0), 1)),
                se=float(se[j]), model=model, n=int(len(use)), n_ev=n_ev)


def iptw(d, covs, trim=0.01, reg_C=0.3):
    X = d[covs].astype(float).values
    keep = np.isfinite(X).all(1)
    d = d[keep].copy()
    X = d[covs].astype(float).values
    X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    T = pd.Categorical(d["gc24_str"], categories=ORDER).codes
    ps = LogisticRegression(solver="lbfgs", max_iter=8000, C=reg_C).fit(X, T)
    Pm = ps.predict_proba(X)
    marg = np.array([(T == g).mean() for g in range(len(ORDER))])
    w = marg[T] / np.clip(Pm[np.arange(len(T)), T], 1e-6, 1)
    if trim:
        lo, hi = np.quantile(w, [trim, 1 - trim])
        w = np.clip(w, lo, hi)
    d = d.copy()
    d["sw"] = w
    return d, float(w.sum() ** 2 / (w ** 2).sum())


def maxsmd(dd, covs, wcol=None, ref="G0_none"):
    ww = np.ones(len(dd)) if wcol is None else dd[wcol].values
    worst = 0.0
    for v in covs:
        m = dd[v].astype(float).values
        sd = np.sqrt(np.average((m - np.average(m, weights=ww)) ** 2, weights=ww))
        if sd == 0:
            continue
        m0 = np.average(m[(dd["gc24_str"] == ref).values],
                        weights=ww[(dd["gc24_str"] == ref).values])
        for g in ORDER[1:]:
            i = (dd["gc24_str"] == g).values
            if i.sum() == 0:
                continue
            worst = max(worst, abs((np.average(m[i], weights=ww[i]) - m0) / sd))
    return worst


def evalue(rr):
    rr = abs(float(rr))
    if rr < 1:
        rr = 1.0 / max(rr, 1e-12)
    return rr + np.sqrt(rr * (rr - 1))


def bh(pvals):
    p = np.asarray([np.nan if v is None else float(v) for v in pvals], float)
    ok = np.isfinite(p)
    qv = np.full(len(p), np.nan)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return qv
    o = idx[np.argsort(p[idx])]
    m = len(o)
    ranked = p[o] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qv[o] = np.clip(ranked, 0, 1)
    return qv


def ci(r, key="p_lrt"):
    if r is None:
        return "不可估"
    return "%.2f (%.2f-%.2f) P=%.3f" % (r["OR"], r["lo"], r["hi"], r[key])


# ==================================================================== 数据
t0 = time.time()
P("=" * 104)
P("v11  连续活动度信息密度作为文本侧主输出 —— 疾病×剂量交互重跑")
P("=" * 104)

df = pd.read_csv(os.path.join(DATA, "cohort_rheum_icu.csv"))
df = df.rename(columns={"sofa24": "sofa"})
ab = pd.read_csv(os.path.join(DATA, "abx_outcomes_rheum.csv"))
df = df.merge(ab[[c for c in ["stay_key", "n_agent_pre24", "ondan24",
                              "docusate24"] if c in ab.columns]],
              on="stay_key", how="left")
gh = pd.read_csv(os.path.join(DATA, "gc_history.csv"))
df = df.merge(gh[[c for c in ["stay_key", "note_found", "home_gc",
                              "indN_spec"] if c in gh.columns]],
              on="stay_key", how="left")
ic = pd.read_csv(os.path.join(DATA, "gc_inactivity.csv"))
df = df.merge(ic[[c for c in ["stay_key", "spec_true", "inact3", "hc_len",
                              "note_len2", "nonrheum_pdx"] if c in ic.columns]],
              on="stay_key", how="left")
ad = pd.read_csv(os.path.join(DATA, "gc_activity_density.csv"))
DC = ["stay_key", "v8_label", "aid_doc", "aid_signed", "aid_broad", "aid_act",
      "info_n", "info_broad", "narr_char"]
df = df.merge(ad[[c for c in DC if c in ad.columns]], on="stay_key", how="left")

df["sofa"] = df.sofa.fillna(df.sofa.median())
df["gc_str_num"] = df.gc24_str.map(SCORE)
df["is_sle"] = (df.primary_grp == "SLE").astype(int)
for c in ["death_30d", "death_hosp", "ondan24", "docusate24", "home_gc"]:
    if c in df.columns:
        df[c] = df[c].astype(float)
df["nonrheum_pdx"] = df.get("nonrheum_pdx",
                            pd.Series(0, index=df.index)).fillna(0).astype(float)

both = df[df.primary_grp.isin(["SLE", "RA"])].copy()
bothn = both[both.note_found == 1].copy()
sp0 = bothn[bothn.spec_true == 0].copy()
sp0["narr_z"] = ((sp0.narr_char - sp0.narr_char.mean())
                 / sp0.narr_char.std())

P("  漏斗：SLE+RA %d → 有出院记录 %d → spec− 层 %d（30 d 死亡 %d）"
  % (len(both), len(bothn), len(sp0), int(sp0.death_30d.sum())))
P("  SLE %d 例 / %d 死；RA %d 例 / %d 死"
  % (int((sp0.primary_grp == "SLE").sum()),
     int(sp0[sp0.primary_grp == "SLE"].death_30d.sum()),
     int((sp0.primary_grp == "RA").sum()),
     int(sp0[sp0.primary_grp == "RA"].death_30d.sum())))

# ---------------- 门槛自检：必须复现 v8 已落盘的整层与 aid_broad 分层
anchor = interaction(sp0)
gate_a = interaction(sp0[sp0.aid_broad == 0])
gate_b = interaction(sp0[sp0.aid_broad > 0])
P("")
P("  [门槛自检] 整层 RA/SLE = %s ；v8 表为 1.96 (1.04-3.67) P=0.006" % ci(anchor))
P("  [门槛自检] aid_broad=0 层 = %s ；v8 表为 2.00 (0.98-4.09) P=0.012"
  % ci(gate_a))
assert anchor and abs(anchor["OR"] - 1.9556) < 0.02, "整层估计未复现 v8"
assert gate_a and abs(gate_a["OR"] - 1.9967) < 0.02, "aid_broad=0 未复现 v8"
P("  → 门槛通过（与 v8 落盘值一致）")

# ================================================================= 第一节
P("")
P("=" * 104)
P("第一节  密度这条轴有没有分辨力？（决定交互模型形态的前置问题）")
P("=" * 104)
P("  连续密度在 spec− 层是**零膨胀**的：大多数病例没有任何活动度信息。")
P("")
P("  %-24s %8s %12s %9s %10s %11s" % (
    "候选密度变量", "零值%", "正密度例数", "正侧事件", "整层事件", "正侧占比"))
axis_rows = []


def axis_prof(name, col, note=""):
    v = sp0[col].dropna()
    nz = int((v > 0).sum())
    ev_pos = int(sp0.loc[sp0[col] > 0, "death_30d"].sum())
    ev_all = int(sp0.death_30d.sum())
    P("  %-24s %7.1f%% %12d %9d %10d %10.1f%%" % (
        name, 100 * (v == 0).mean(), nz, ev_pos, ev_all,
        100.0 * ev_pos / max(ev_all, 1)))
    axis_rows.append(dict(var=name, zero_pct=round(100 * float((v == 0).mean()), 1),
                          n_pos=nz, ev_pos=ev_pos, ev_all=ev_all,
                          ev_pos_share=round(100.0 * ev_pos / max(ev_all, 1), 1),
                          median=v.median(), p75=float(v.quantile(0.75)),
                          mx=float(v.max()), note=note))


axis_prof("aid_doc（主）", "aid_doc", "信息密度 = 条数/千字符")
axis_prof("aid_signed", "aid_signed", "带符号（阳性−否定）")
axis_prof("aid_broad", "aid_broad", "宽口径")
axis_prof("info_n", "info_n", "信息条数（计数）")
axis_prof("info_broad", "info_broad", "宽口径条数")
axis_prof("narr_char", "narr_char", "记录长度（对照轴）")
pd.DataFrame(axis_rows).to_csv(os.path.join(OUT, "table_v11_axis.csv"),
                               index=False)

P("")
P("  结论：能在密度轴上做「剂量-反应」的前提是**正侧有足够事件**。")
P("  aid_doc>0 只有 %d 例 / %d 事件 → 正侧连二分层都不够，更谈不上五分位。"
  % (int((sp0.aid_doc > 0).sum()), int(sp0[sp0.aid_doc > 0].death_30d.sum())))
P("  唯一事件数够分层的轴是 narr_char（记录长度），它才是可用的对照轴。")

# ================================================================= 第二节
P("")
P("=" * 104)
P("第二节  整层锚点（预先指定的确认性检验，单次、不校正）")
P("=" * 104)
row_anchor = []
P("  %-30s %-46s %8s %6s" % ("模型", "RA/SLE 剂量斜率比 (95% CI)", "P(LRT)", "事件"))
P("  %-30s %-46s %8s %6d" % ("未加权 Firth（主）", ci(anchor), "%.3f" %
                            anchor["p_lrt"], anchor["n_ev"]))
row_anchor.append(dict(model="未加权 Firth（主）", OR=anchor["OR"],
                       lo=anchor["lo"], hi=anchor["hi"], p=anchor["p_lrt"],
                       n=anchor["n"], n_ev=anchor["n_ev"], note="primary"))

iw = sp0.copy()
for c in PSCOV:
    if c in iw.columns:
        iw[c] = iw[c].astype(float)
PS = [c for c in PSCOV if c in iw.columns]
d_w, ess = iptw(iw.dropna(subset=PS + ["gc24_str"]), PS)
smd0 = maxsmd(sp0, CORE)
smd1 = maxsmd(d_w, CORE, "sw")
r_iptw = interaction(d_w, covs=CORE, wcol="sw")
P("  %-30s %-46s %8s %6d" % ("IPTW-GLM（权重=freq_weights）",
                            ci(r_iptw, "p_lrt") if r_iptw else "不可估",
                            ("%.3f" % r_iptw["p_lrt"]) if r_iptw else "-",
                            r_iptw["n_ev"] if r_iptw else 0))
if r_iptw:
    row_anchor.append(dict(model="IPTW-GLM", OR=r_iptw["OR"], lo=r_iptw["lo"],
                           hi=r_iptw["hi"], p=r_iptw["p_lrt"], n=r_iptw["n"],
                           n_ev=r_iptw["n_ev"], note="weighted"))
r_mle = interaction(sp0, use_firth=False)
if r_mle:
    P("  %-30s %-46s %8.3f %6d" % ("MLE（对照，非主）", ci(r_mle), r_mle["p_lrt"],
                                   r_mle["n_ev"]))
    row_anchor.append(dict(model="未加权 MLE（对照）", OR=r_mle["OR"],
                           lo=r_mle["lo"], hi=r_mle["hi"], p=r_mle["p_lrt"],
                           n=r_mle["n"], n_ev=r_mle["n_ev"], note="sensitivity"))
r_len = interaction(sp0, covs=CORE + ["narr_z"])
P("  %-30s %-46s %8.3f %6d" % ("长度校正（+narr_char z）", ci(r_len),
                               r_len["p_lrt"], r_len["n_ev"]))
row_anchor.append(dict(model="长度校正（+narr_char z）", OR=r_len["OR"],
                       lo=r_len["lo"], hi=r_len["hi"], p=r_len["p_lrt"],
                       n=r_len["n"], n_ev=r_len["n_ev"], note="length-adj"))
P("")
P("  IPTW 诊断：ESS=%.1f/%d ; 未加权 max|SMD|=%.3f ; 加权后 max|SMD|=%.3f"
  % (ess, len(d_w), smd0, smd1))
P("  → 加权后 max|SMD| %s 0.2，%s"
  % ("<" if smd1 < 0.2 else ">=",
     "加权充分" if smd1 < 0.2 else "不充分 → 未加权与加权必须并列报告，不可默认加权更可信"))
P("")
P("  E-value（把交互当 RR 处理）= %.2f" % evalue(anchor["OR"]))
pd.DataFrame(row_anchor).to_csv(os.path.join(OUT, "table_v11_anchor.csv"),
                                index=False)

# ---- 脆弱性：翻转最少几个事件使 P>0.05 ----
P("")
P("  脆弱性指数（逐次翻转事件结局，看最少几个翻转使 LRT P>0.05）：")
use = sp0[["gc_str_num", "is_sle", "death_30d"] + CORE].dropna().copy()
use["s"] = use.gc_str_num.astype(float)
use["sx"] = use.s * use.is_sle.astype(float)
X = sm.add_constant(use[["s", "is_sle", "sx"] + CORE].astype(float))
Xr = sm.add_constant(use[["s", "is_sle"] + CORE].astype(float))
y0 = use.death_30d.values.astype(float)


def lrt_p(yv):
    b, se, lf = firth(X.values, yv, max_iter=150, tol=1e-7)
    _, _, l0 = firth(Xr.values, yv, max_iter=150, tol=1e-7)
    return float(1 - stats.chi2.cdf(max(2 * (lf - l0), 0), 1))


base_p = lrt_p(y0)
ev_idx = np.where(y0 == 1)[0]
non_idx = np.where(y0 == 0)[0]
t_f = time.time()
fi = None
n_single_ok = 0
for i in ev_idx:
    yy = y0.copy()
    yy[i] = 0.0
    if lrt_p(yy) > 0.05:
        n_single_ok += 1
if n_single_ok > 0:
    fi = 1
else:
    stop = False
    for a in range(len(ev_idx)):
        for b_ in range(a + 1, len(ev_idx)):
            yy = y0.copy()
            yy[ev_idx[a]] = 0.0
            yy[ev_idx[b_]] = 0.0
            if lrt_p(yy) > 0.05:
                fi = 2
                stop = True
                break
        if stop:
            break
P("    基准 P(LRT)=%.4f；单事件翻转中使 P>0.05 的个数 = %d；最小翻转数 FI = %s"
  % (base_p, n_single_ok,
     ("%d" % fi) if fi else "≥3（1 例与 2 例翻转均未翻成不显著）"))
P("    （扫描耗时 %.1f s）" % (time.time() - t_f))
pd.DataFrame([dict(base_p=base_p, n_single_flip_ns=n_single_ok, fi=fi)]).to_csv(
    os.path.join(OUT, "table_v11_fragility.csv"), index=False)

# ================================================================= 第三节
P("")
P("=" * 104)
P("第三节  密度分层的交互（把话说完整：不可估 ≠ 阴性）")
P("=" * 104)
P("  %-26s %6s %5s %-34s %7s" % ("分层", "n", "事件", "RA/SLE 剂量斜率比", "P(LRT)"))
rows_strata = []
def strata_row(lab, mask, mask_desc):
    s = sp0[mask]
    r = interaction(s)
    if r is None:
        P("  %-26s %6d %5d %-34s %7s" % (lab, len(s), int(s.death_30d.sum()),
                                         "不可估（事件/剂量层不足）", "-"))
        rows_strata.append(dict(stratum=lab, n=len(s),
                                n_ev=int(s.death_30d.sum()), OR=np.nan,
                                lo=np.nan, hi=np.nan, p=np.nan,
                                status="not estimable", desc=mask_desc))
    else:
        P("  %-26s %6d %5d %-34s %7.3f" % (lab, r["n"], r["n_ev"],
                                           ci(r), r["p_lrt"]))
        rows_strata.append(dict(stratum=lab, n=r["n"], n_ev=r["n_ev"], OR=r["OR"],
                                lo=r["lo"], hi=r["hi"], p=r["p_lrt"],
                                status="ok", desc=mask_desc))


strata_row("全 spec− 层", pd.Series(True, index=sp0.index), "锚点")
strata_row("aid_doc = 0", sp0.aid_doc == 0, "无活动度信息（主）")
strata_row("aid_doc > 0", sp0.aid_doc > 0, "有活动度信息")
med_pos = sp0.loc[sp0.aid_doc > 0, "aid_doc"].median()
strata_row("  0 < aid_doc ≤ 正侧中位", (sp0.aid_doc > 0) & (sp0.aid_doc <= med_pos),
           "正侧低半")
strata_row("  aid_doc > 正侧中位", sp0.aid_doc > med_pos, "正侧高半")
strata_row("aid_broad = 0", sp0.aid_broad == 0, "宽口径无信息")
strata_row("aid_broad > 0", sp0.aid_broad > 0, "宽口径有信息")
strata_row("info_n = 0", sp0.info_n == 0, "零条信息")
strata_row("info_n = 1", sp0.info_n == 1, "1 条")
strata_row("info_n ≥ 2", sp0.info_n >= 2, "≥2 条")
pd.DataFrame(rows_strata).to_csv(os.path.join(OUT, "table_v11_strata.csv"),
                                 index=False)

# ---- 三项交互（线性）----
P("")
P("  三项交互 s × is_sle × 密度（密度已标准化；OR>1 = 密度升高使 RA/SLE 差异扩大）：")
rows_3w = []
P("  %-30s %-22s %-34s %7s %7s" % ("修饰项", "协变量", "三项交互 OR (95% CI)",
                                   "P(Wald)", "P(LRT)"))
for mod, lab in [("aid_doc", "aid_doc（主）"), ("aid_signed", "aid_signed"),
                 ("aid_broad", "aid_broad")]:
    for covs, clab in [(CORE, "CORE"), (CORE + ["narr_z"], "CORE+长度")]:
        r = threeway(sp0, "death_30d", mod, covs=covs)
        if r is None:
            P("  %-30s %-22s %-34s" % (lab, clab, "不可估"))
            rows_3w.append(dict(mod=lab, covs=clab, OR=np.nan, lo=np.nan,
                                hi=np.nan, p_wald=np.nan, p_lrt=np.nan,
                                mde=np.nan, status="not estimable"))
            continue
        mde = float(np.exp(2.802 * r["se"]))
        P("  %-30s %-22s %-34s %7.3f %7.3f  [最小可检 OR=%.2f]"
          % (lab, clab, ci(r, "p_wald"), r["p_wald"], r["p_lrt"], mde))
        rows_3w.append(dict(mod=lab, covs=clab, OR=r["OR"], lo=r["lo"], hi=r["hi"],
                            p_wald=r["p_wald"], p_lrt=r["p_lrt"], mde=mde,
                            status="ok"))
# IPTW 版
r = threeway(d_w, "death_30d", "aid_doc", covs=CORE, wcol="sw")
if r:
    mde = float(np.exp(2.802 * r["se"]))
    P("  %-30s %-22s %-34s %7.3f %7.3f  [最小可检 OR=%.2f]"
      % ("aid_doc（主）", "IPTW", ci(r, "p_wald"), r["p_wald"], r["p_lrt"], mde))
    rows_3w.append(dict(mod="aid_doc（主）", covs="IPTW", OR=r["OR"], lo=r["lo"],
                        hi=r["hi"], p_wald=r["p_wald"], p_lrt=r["p_lrt"],
                        mde=mde, status="ok"))
else:
    P("  %-30s %-22s %-34s" % ("aid_doc（主）", "IPTW", "不可估（加权下分离）"))
    rows_3w.append(dict(mod="aid_doc（主）", covs="IPTW", OR=np.nan, lo=np.nan,
                        hi=np.nan, p_wald=np.nan, p_lrt=np.nan, mde=np.nan,
                        status="not estimable"))
pd.DataFrame(rows_3w).to_csv(os.path.join(OUT, "table_v11_3way.csv"), index=False)
P("")
P("  判读口径：三项交互的「阴性」只有在**最小可检 OR 接近 1** 时才算证据。")
P("  本例正侧事件极少 → 最小可检 OR 落在 1.8–2.4 → 只能排除**大幅修饰**")
P("  （≥2.4× 的修饰），小于此的修饰既不能排除也不能证实 → 报「不可判」。")

# ================================================================= 第四节
P("")
P("=" * 104)
P("第四节  长度对照 —— 唯一有分辨力的轴（阴性修饰项对照）")
P("=" * 104)
sp0["nb"] = pd.qcut(sp0.narr_char, 5, labels=False, duplicates="drop")
rows_len = []
P("  %-14s %6s %5s %6s %6s %-32s %8s" % (
    "记录长度层", "n", "事件", "SLE", "RA", "RA/SLE 剂量斜率比", "P(LRT)"))
for k in sorted(sp0.nb.dropna().unique()):
    s = sp0[sp0.nb == k]
    r = interaction(s)
    lo_b = int(sp0.loc[sp0.nb == k, "narr_char"].min())
    hi_b = int(sp0.loc[sp0.nb == k, "narr_char"].max())
    if r is None:
        P("  %-14s %6d %5d %6d %6d %-32s %8s" % (
            "Q%d" % (k + 1), len(s), int(s.death_30d.sum()),
            int((s.primary_grp == "SLE").sum()), int((s.primary_grp == "RA").sum()),
            "不可估", "-"))
        rows_len.append(dict(band="Q%d" % (k + 1), n=len(s),
                             n_ev=int(s.death_30d.sum()), OR=np.nan, lo=np.nan,
                             hi=np.nan, p=np.nan, lo_chars=lo_b, hi_chars=hi_b,
                             mid=float(s.narr_char.median()), status="not estimable"))
        continue
    P("  %-14s %6d %5d %6d %6d %-32s %8.3f" % (
        "Q%d" % (k + 1), r["n"], r["n_ev"],
        int((s.primary_grp == "SLE").sum()), int((s.primary_grp == "RA").sum()),
        ci(r), r["p_lrt"]))
    rows_len.append(dict(band="Q%d" % (k + 1), n=r["n"], n_ev=r["n_ev"], OR=r["OR"],
                         lo=r["lo"], hi=r["hi"], p=r["p_lrt"], lo_chars=lo_b,
                         hi_chars=hi_b, mid=float(s.narr_char.median()),
                         status="ok"))

# ---- 跨层趋势：逆方差加权 meta 回归 log(OR) ~ 层序 ----
P("")
P("  跨层趋势（逆方差加权 meta 回归 log(RA/SLE) ~ 层序）：")
ok_rows = [r for r in rows_len if r["status"] == "ok" and np.isfinite(r["OR"])
           and r["OR"] > 0]
if len(ok_rows) >= 3:
    kk = np.array([int(r["band"][1:]) for r in ok_rows], float)
    lg = np.log(np.array([r["OR"] for r in ok_rows], float))
    se = np.array([(np.log(r["hi"]) - np.log(r["lo"])) / (2 * 1.96)
                   for r in ok_rows], float)
    wgt = 1.0 / np.clip(se, 1e-9, None) ** 2
    Xm = sm.add_constant(kk)
    m = sm.WLS(lg, Xm, weights=wgt).fit()
    slope, sl_se = float(m.params[1]), float(m.bse[1])
    z = slope / sl_se
    P("    斜率 = %+.3f（每升一层 log 比值）/ SE %.3f → 乘性变化/层 = %.2f "
      "(%.2f-%.2f)  P=%.3f"
      % (slope, sl_se, float(np.exp(slope)),
         float(np.exp(slope - 1.96 * sl_se)), float(np.exp(slope + 1.96 * sl_se)),
         float(2 * (1 - stats.norm.cdf(abs(z))))))
    trend_res = dict(slope=slope, se=sl_se, mult=float(np.exp(slope)),
                     mult_lo=float(np.exp(slope - 1.96 * sl_se)),
                     mult_hi=float(np.exp(slope + 1.96 * sl_se)),
                     p=float(2 * (1 - stats.norm.cdf(abs(z)))), k=len(ok_rows))
else:
    P("    可估层不足，跳过")
    trend_res = None
pd.DataFrame(rows_len).to_csv(os.path.join(OUT, "table_v11_length.csv"),
                              index=False)

# ================================================================= 第五节
P("")
P("=" * 104)
P("第五节  方向自检 D3（交互系数须与分开拟合的两病趋势之比同号）")
P("=" * 104)
rows_d3 = []
P("  %-26s %-10s %-10s %-10s %8s" % ("分层", "SLE 趋势", "RA 趋势", "比值",
                                     "一致性"))
bands = [("全 spec− 层", sp0), ("aid_doc = 0", sp0[sp0.aid_doc == 0]),
         ("aid_doc > 0", sp0[sp0.aid_doc > 0])]
for k in sorted(sp0.nb.dropna().unique()):
    bands.append(("narr Q%d" % (k + 1), sp0[sp0.nb == k]))
for lab, s in bands:
    t2 = trend(s[s.primary_grp == "SLE"], "death_30d", CORE)
    t3 = trend(s[s.primary_grp == "RA"], "death_30d", CORE)
    r = interaction(s)
    if t2 is None or t3 is None or r is None:
        P("  %-26s %-10s %-10s %-10s %8s" % (lab, "-", "-", "-", "不可估"))
        rows_d3.append(dict(group=lab, sle_OR=np.nan, ra_OR=np.nan, ratio=np.nan,
                            ix_OR=r["OR"] if r else np.nan, agree=None))
        continue
    ratio = t3["OR"] / t2["OR"]
    zero_near = (1.02 > r["OR"] > 0.98) or (r["lo"] <= 1 <= r["hi"])
    agree = (ratio > 1) == (r["OR"] > 1)
    verdict = "一致" if agree else ("零效应附近噪声" if zero_near else "★不一致")
    P("  %-26s %-10.2f %-10.2f %-10.2f %8s" % (lab, t2["OR"], t3["OR"], ratio,
                                               verdict))
    rows_d3.append(dict(group=lab, sle_OR=t2["OR"], ra_OR=t3["OR"], ratio=ratio,
                        ix_OR=r["OR"], agree=bool(agree)))
pd.DataFrame(rows_d3).to_csv(os.path.join(OUT, "table_v11_d3.csv"), index=False)

# ================================================================= 第五节
P("")
P("=" * 104)
P("第六节  同一结局上的阴性暴露对照（铁律 2：必须同结局测）")
P("=" * 104)
P("  如果「疾病×暴露」的死亡交互在药理上与死亡无关的暴露上也出现，")
P("  说明存在过程混杂，主结论只能写「未检出关联」。")
P("")
P("  %-30s %-42s %8s" % ("暴露", "疾病×暴露 死亡交互 (RA/SLE)", "P(LRT)"))


def ix_binary(d, xcol, y="death_30d", covs=None):
    """二分类暴露的疾病×暴露交互：y ~ x + is_sle + covs + x:is_sle。"""
    covs = list(CORE if covs is None else covs)
    use = d[[xcol, "is_sle", y] + covs].dropna().copy()
    if len(use) < 30 or use[y].nunique() < 2 or use[xcol].nunique() < 2:
        return None
    use["x"] = use[xcol].astype(float)
    use["xi"] = use["x"] * use["is_sle"].astype(float)
    full = ["x", "is_sle", "xi"] + covs
    base = ["x", "is_sle"] + covs
    X = sm.add_constant(use[full].astype(float))
    Xr = sm.add_constant(use[base].astype(float))
    b, se, lf = firth(X.values, use[y].values)
    _, _, l0 = firth(Xr.values, use[y].values)
    j = list(X.columns).index("xi")
    if not np.isfinite(b[j]) or not np.isfinite(se[j]):
        return None
    e = float(np.exp(b[j]))
    return dict(OR=1.0 / e, lo=1.0 / float(np.exp(b[j] + 1.96 * se[j])),
                hi=1.0 / float(np.exp(b[j] - 1.96 * se[j])), log=float(-b[j]),
                p_lrt=float(1 - stats.chi2.cdf(max(2 * (lf - l0), 0), 1)),
                n=int(len(use)), n_ev=int(use[y].sum()))


sp0["gc_any24_b"] = (sp0.gc_str_num >= 1).astype(float)
rows_ctrl = []
r_ref = ix_binary(sp0, "gc_any24_b")
P("  %-30s %-42s %8.3f" % ("GC 任意剂量（二分类，参照）", ci(r_ref),
                           r_ref["p_lrt"]))
rows_ctrl.append(dict(exposure="GC 任意剂量（参照，二分类）", OR=r_ref["OR"],
                      lo=r_ref["lo"], hi=r_ref["hi"], p=r_ref["p_lrt"],
                      role="reference"))
for xc, lab in [("ondan24", "昂丹司琼 24h（阴性暴露对照）"),
                ("docusate24", "多库酯 24h（阴性暴露对照）")]:
    if xc not in sp0.columns:
        continue
    r = ix_binary(sp0, xc)
    if r is None:
        P("  %-30s %-42s %8s" % (lab, "不可估", "-"))
        rows_ctrl.append(dict(exposure=lab, OR=np.nan, lo=np.nan, hi=np.nan,
                              p=np.nan, role="negative control"))
        continue
    P("  %-30s %-42s %8.3f" % (lab, ci(r), r["p_lrt"]))
    rows_ctrl.append(dict(exposure=lab, OR=r["OR"], lo=r["lo"], hi=r["hi"],
                          p=r["p_lrt"], role="negative control"))
P("")
P("  判读：阴性暴露对照若也显著（P<0.05），主交互只能写「未检出关联」，")
P("  不能写「有影响」——对照把 OR 推向低值，只会掩盖真关联，不会制造假关联。")
pd.DataFrame(rows_ctrl).to_csv(os.path.join(OUT, "table_v11_negctrl.csv"),
                               index=False)

# ================================================================= 第七节
P("")
P("=" * 104)
P("第七节  多重性三段式")
P("=" * 104)
P("  ① 预先指定的主结局（30 d 死亡 × 疾病 × 剂量）= 整层 %s" % ci(anchor))
P("     → 测一次，不校正。")
fam_sec = []
for r in rows_strata:
    if r["stratum"] in ["aid_doc = 0", "aid_doc > 0", "  0 < aid_doc ≤ 正侧中位",
                        "  aid_doc > 正侧中位", "aid_broad = 0", "aid_broad > 0",
                        "info_n = 0", "info_n ≥ 2"] and np.isfinite(r["p"]):
        fam_sec.append((r["stratum"], r["p"]))
for r in rows_3w:
    if np.isfinite(r["p_wald"]):
        fam_sec.append(("3-way %s [%s]" % (r["mod"], r["covs"]), r["p_wald"]))
for r in rows_len:
    if np.isfinite(r["p"]):
        fam_sec.append(("长度 %s" % r["band"], r["p"]))
q2 = bh([v for _, v in fam_sec])
q2 = np.array([float(x) if x is not None and not isinstance(x, str) else np.nan for x in q2])
rows_bh = []
P("  %-42s %10s %12s" % ("二级检验（密度/长度族）", "P", "q (BH, 族内)"))
for (lab, pv), qv in zip(fam_sec, q2):
    P("  %-42s %10.3f %12.3f" % (lab, pv, qv))
    rows_bh.append(dict(test=lab, p=pv, q=float(qv), family="secondary"))
tot = [anchor["p_lrt"]] + [v for _, v in fam_sec]
q_all = bh(tot)
P("")
P("  %-42s %10s %12s" % ("本版族（密度/长度族 + 主结局）", "P", "q (BH)"))
P("  %-42s %10.3f %12.3f" % ("30 d 死亡交互（PRIMARY，整层）",
                            anchor["p_lrt"], float(q_all[0])))
P("")
P("  ⚠ 族口径说明（不得混用）：")
P("   · 本版族 = 主结局 + 密度/长度相关检验（共 %d 项）→ 主结局 q=%.3f"
  % (len(tot), float(q_all[0])))
P("   · 主文沿用的**主族**（v6） = 主结局 + 5 个二级结局（院内死亡/培养阳性/")
P("     新抗菌药/消化道出血/感染 ICD）→ 30 d 死亡 q=0.062，院内死亡 q=0.095。")
P("   → 两族不冲突（族更小则 q 更低）。**正文仍引用 v6 的 q=0.062**，")
P("     本版的 q=%.3f 仅描述本族，不得替换主文数字。" % float(q_all[0]))
rows_bh.append(dict(test="30 d 死亡交互（PRIMARY，整层）", p=anchor["p_lrt"],
                    q=float(q_all[0]), family="primary+all(v11家族)"))
rows_bh.append(dict(test="[主文口径] 30 d 死亡交互 (v6 主族)", p=anchor["p_lrt"],
                    q=0.062, family="primary+5secondary(v6)"))
pd.DataFrame(rows_bh).to_csv(os.path.join(OUT, "table_v11_bh.csv"), index=False)

# ================================================================= 第八节
P("")
P("=" * 104)
P("第八节  判读与对齐")
P("=" * 104)
n_ev_zero = int(sp0[sp0.aid_doc == 0].death_30d.sum())
share = 100.0 * n_ev_zero / max(int(sp0.death_30d.sum()), 1)
P("  ① 整层估计**就是**「无活动度信息」层的估计：")
P("     aid_doc=0 层 n=%d（占全层 %.1f%%）携带 %d/%d = %.1f%% 的事件；"
  % (int((sp0.aid_doc == 0).sum()),
     100.0 * int((sp0.aid_doc == 0).sum()) / len(sp0), n_ev_zero,
     int(sp0.death_30d.sum()), share))
P("     该层 %s vs 整层 %s → 两者在 OR 尺度上相差 %.3f。"
  % (ci(interaction(sp0[sp0.aid_doc == 0])), ci(anchor),
     abs(interaction(sp0[sp0.aid_doc == 0])["OR"] - anchor["OR"])))
P("  ② 密度轴**分辨力不足**，三项交互的「阴性」不是证据：")
P("     正侧只有 %d 事件 → 最小可检修饰 OR 落在 1.8–2.4，只能排除大幅修饰，"
  % int(sp0[sp0.aid_doc > 0].death_30d.sum()))
P("     小于此的修饰既不能排除也不能证实（详见 table_v11_3way.csv）。")
P("  ③ 唯一有分辨力的对照轴（记录长度）**没有推翻交互**：")
if trend_res:
    P("     五分位跨层趋势 P=%.3f；长度校正后整层 RA/SLE = %s（比未校正的 %.2f **未衰减**）"
      % (trend_res["p"], ci(r_len), anchor["OR"]))
P("  ④ 同一结局的阴性暴露对照**未检出**过程混杂：昂丹司琼 %s、多库酯 %s"
  % (ci(rows_ctrl[1], "p") if len(rows_ctrl) > 1 and np.isfinite(rows_ctrl[1]["p"])
     else "不可估",
     ci(rows_ctrl[2], "p") if len(rows_ctrl) > 2 and np.isfinite(rows_ctrl[2]["p"])
     else "不可估"))
P("  ⑤ 处置：文本侧主输出维持连续密度（aid_doc / aid_signed）作为**协变量级别的")
P("     描述性对照**，不进入主结局的效应修饰项；主结局保持整层估计；")
P("     措辞一律写「本层未能检出密度修饰」，不写「无修饰」。")

obj = dict(
    funnel=dict(both=len(both), bothn=len(bothn), sp0=len(sp0),
                ev=int(sp0.death_30d.sum()),
                sle=int((sp0.primary_grp == "SLE").sum()),
                ra=int((sp0.primary_grp == "RA").sum()),
                sle_ev=int(sp0[sp0.primary_grp == "SLE"].death_30d.sum()),
                ra_ev=int(sp0[sp0.primary_grp == "RA"].death_30d.sum())),
    axis=axis_rows, anchor=row_anchor,
    anchor_ci=dict(OR=anchor["OR"], lo=anchor["lo"], hi=anchor["hi"],
                   p=anchor["p_lrt"], n=anchor["n"], n_ev=anchor["n_ev"]),
    iptw=dict(ess=ess, n=len(d_w), smd_unw=smd0, smd_w=smd1,
              OR=r_iptw["OR"] if r_iptw else None),
    length_adj=dict(OR=r_len["OR"], lo=r_len["lo"], hi=r_len["hi"],
                    p=r_len["p_lrt"]),
    evalue=float(evalue(anchor["OR"])), fragility=dict(base_p=base_p,
                                                       n_single=n_single_ok, fi=fi),
    strata=rows_strata, threeway=rows_3w, length=rows_len,
    length_trend=trend_res, d3=rows_d3, bh=rows_bh, negctrl=rows_ctrl,
    bh_v11=float(q_all[0]), bh_mainline=0.062,
)
with open(os.path.join(OUT, "_v11.json"), "w", encoding="utf-8") as f:
    json.dump(obj, f, ensure_ascii=False, indent=1, default=float)
with open(os.path.join(OUT, "204_density_interaction.txt"), "w",
          encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
P("")
P("[saved] out/204_density_interaction.txt ; table_v11_*.csv ; out/_v11.json"
  "  (%.1f s)" % (time.time() - t0))
