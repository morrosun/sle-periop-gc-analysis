# -*- coding: utf-8 -*-
"""
160_indication_mechanism.py
===========================
v6：把 v5 存活下来的「死亡交互」放进**指征 / 用药史结构**里解释。

为什么必须做这一步
------------------
v5 的三个模型（未加权 / IPTW / 分病种）在死亡上的表现不一致，而且「无 GC 组」
的构成一直是个黑箱。158/159 号脚本从出院记录文本抽出了指征与院前激素状态，
本脚本用它们回答三件事：

  Q1  没拿到 GC 的那一组（G0）到底是"轻症"还是"该给而没给"？
      -> 参照组效度：G0 内 SLE vs RA 的疾病活动、既往激素、器官衰竭对比
  Q2  四个剂量层是"同一人群的剂量变化"，还是"四群不同的人"？
      -> 结构性分解：院前 GC / 慢性使用者 / 疾病特异活动 / 通用器官衰竭
  Q3  死亡交互是指征混合造成的，还是独立于指征？
      -> 指征校正前后对比 + 按指征分层 + 「指征–暴露一致性」四格

指标口径（三套指征，来自 158，均以出院记录文本抽取）
----------------------------------------------------
  ind_*   GC 上下文共现 ——「为什么给 GC」；G0 组按构造必为 0，**不可跨层比较**
  indN_*  全叙述共现   ——「这次住院有没有该活动」；跨层可比 ← 本脚本主用
  indD_*  仅出院诊断段 —— 与 GC 完全无关，最保守
  另拆 spec（风湿病特异活动/器官受累）与 gen（通用器官衰竭 AKI/哮喘/ARDS）

主结局：30 天死亡（用户指定转入「死亡交互」）；院内死亡为共同主结局。
预先指定的确认性检验 = 30 天死亡上的 疾病×剂量交互，单次、不校正。
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")

ORDER = ["G0_none", "G1_low", "G2_mod", "G3_high"]
LABEL = {"G0_none": "none", "G1_low": ">0-<10", "G2_mod": "10-<50",
         "G3_high": ">=50"}
SCORE = {"G0_none": 0.0, "G1_low": 1.0, "G2_mod": 2.0, "G3_high": 3.0}

CORE = ["age", "female", "sofa", "vaso24", "vent24", "renal_fail", "shock",
        "surg_any", "n_agent_pre24"]
PSCOV = CORE + ["resp_fail", "coagulop", "liver_fail", "immuno_any", "hcq",
                "cytopenia", "serositis", "n_proc", "n_hosp"]

lines = []


def P(s=""):
    print(s)
    lines.append(str(s))


# ================================================================= helpers
def firth(X, y, max_iter=400, tol=1e-10):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    b = np.zeros(p)
    for _ in range(max_iter):
        pr = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))
        W = np.clip(pr * (1 - pr), 1e-12, None)
        Iinv = np.linalg.pinv(X.T @ (X * W[:, None]))
        h = np.einsum("ij,jk,ik->i", X * W[:, None], Iinv, X)
        U = X.T @ (y - pr + h * (0.5 - pr))
        step = Iinv @ U
        mx = np.max(np.abs(step))
        if mx > 4:
            step *= 4.0 / mx
        b = b + step
        if mx < tol:
            break
    pr = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))
    W = np.clip(pr * (1 - pr), 1e-12, None)
    Iinv = np.linalg.pinv(X.T @ (X * W[:, None]))
    sign, logdet = np.linalg.slogdet(X.T @ (X * W[:, None]))
    ll = float(np.sum(y * np.log(np.clip(pr, 1e-12, None))
                      + (1 - y) * np.log(np.clip(1 - pr, 1e-12, None))))
    return b, np.sqrt(np.abs(np.diag(Iinv))), ll + 0.5 * float(logdet)


def fit_logit(df, y, xvars, firth_mode=None):
    use = df[[y] + xvars].dropna()
    if len(use) == 0 or use[y].nunique() < 2:
        return None
    if firth_mode is None:
        firth_mode = int(use[y].sum()) < 60
    try:
        if firth_mode:
            X = sm.add_constant(use[xvars].astype(float))
            b, se, _ = firth(X.values, use[y].values)
            j = list(X.columns).index(xvars[0])
            model = "Firth"
        else:
            X = sm.add_constant(use[xvars].astype(float))
            with np.errstate(all="ignore"):
                m = sm.Logit(use[y].astype(float), X).fit(disp=0, maxiter=300)
            b, se = np.asarray(m.params, float), np.asarray(m.bse, float)
            j = list(X.columns).index(xvars[0])
            model = "MLE"
        if not np.isfinite(b[j]) or not np.isfinite(se[j]) or se[j] <= 0:
            return None
        z = b[j] / se[j]
        return dict(OR=float(np.exp(b[j])), lo=float(np.exp(b[j] - 1.96 * se[j])),
                    hi=float(np.exp(b[j] + 1.96 * se[j])),
                    p=float(2 * (1 - stats.norm.cdf(abs(z)))), model=model,
                    n=int(len(use)), n_ev=int(use[y].sum()))
    except Exception:
        return None


def trend(d, y, extra=None, col="gc_str_num"):
    """病种内的剂量趋势 OR（有序评分，0-3）。"""
    cols = [col, y] + (extra or [])
    use = d[cols].dropna()
    if len(use) == 0 or use[y].nunique() < 2 or use[col].nunique() < 2:
        return None
    if int(use[y].sum()) < 60:
        X = sm.add_constant(use[[col] + (extra or [])].astype(float))
        b, se, _ = firth(X.values, use[y].values)
        j = list(X.columns).index(col)
        model = "Firth"
    else:
        try:
            X = sm.add_constant(use[[col] + (extra or [])].astype(float))
            with np.errstate(all="ignore"):
                m = sm.Logit(use[y].astype(float), X).fit(disp=0, maxiter=300)
            b, se = np.asarray(m.params, float), np.asarray(m.bse, float)
            j = list(X.columns).index(col)
            model = "MLE"
        except Exception:
            return None
    if not np.isfinite(b[j]) or not np.isfinite(se[j]) or se[j] <= 0:
        return None
    z = b[j] / se[j]
    return dict(OR=float(np.exp(b[j])), lo=float(np.exp(b[j] - 1.96 * se[j])),
                hi=float(np.exp(b[j] + 1.96 * se[j])),
                p=float(2 * (1 - stats.norm.cdf(abs(z)))), model=model,
                n=int(len(use)), n_ev=int(use[y].sum()))


def interaction(d, y, covs, extra_int=None, use_firth=None):
    """疾病×剂量交互 -> RA/SLE 剂量趋势 OR 之比。
    extra_int 是同时与剂量交互的额外变量（用于「剂量×指征」等）。"""
    cols = ["gc_str_num", "is_sle", y] + list(covs) + list(extra_int or [])
    use = d[cols].dropna().copy()
    if len(use) == 0 or use[y].nunique() < 2:
        return None
    use["s"] = use.gc_str_num.astype(float)
    use["sx"] = use.s * use.is_sle.astype(float)
    base = ["s", "is_sle"] + list(covs)
    full = base + ["sx"]
    for v in (extra_int or []):
        use["s_" + v] = use.s * use[v].astype(float)
        full.append("s_" + v)
    X = sm.add_constant(use[full].astype(float))
    Xr = sm.add_constant(use[base].astype(float))
    if use_firth is None:
        use_firth = int(use[y].sum()) < 60
    model = "Firth" if use_firth else "MLE"
    if use_firth:
        b, se, ll_full = firth(X.values, use[y].values)
        _, _, ll_red = firth(Xr.values, use[y].values)
        lrt = 2.0 * (ll_full - ll_red)
    else:
        try:
            with np.errstate(all="ignore"):
                m = sm.Logit(use[y].astype(float), X).fit(disp=0, maxiter=300)
                m0 = sm.Logit(use[y].astype(float), Xr).fit(disp=0, maxiter=300)
            b, se = np.asarray(m.params, float), np.asarray(m.bse, float)
            lrt = 2.0 * (m.llf - m0.llf)
        except Exception:
            model = "Firth"
            b, se, ll_full = firth(X.values, use[y].values)
            _, _, ll_red = firth(Xr.values, use[y].values)
            lrt = 2.0 * (ll_full - ll_red)
    j = list(X.columns).index("sx")
    if not np.isfinite(b[j]) or not np.isfinite(se[j]) or se[j] <= 0:
        return None
    e = float(np.exp(b[j]))               # SLE/RA 斜率比
    return dict(OR=1.0 / e, lo=1.0 / float(np.exp(b[j] + 1.96 * se[j])),
                hi=1.0 / float(np.exp(b[j] - 1.96 * se[j])),
                log=float(-b[j]), model=model,
                p_lrt=float(1 - stats.chi2.cdf(max(lrt, 0.0), 1)),
                n=int(len(use)), n_ev=int(use[y].sum()))


def iptw_weights(d, covs, trim=0.01, reg_C=0.3):
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


def maxsmd(dd, covs, wcol, ref="G0_none"):
    ww = np.ones(len(dd)) if wcol is None else dd[wcol].values
    worst = 0.0
    for v in covs:
        m = dd[v].astype(float).values
        sd = np.sqrt(np.average((m - np.average(m, weights=ww)) ** 2, weights=ww))
        if sd == 0:
            continue
        m0 = np.average(m[(dd.gc24_str == ref).values],
                        weights=ww[(dd.gc24_str == ref).values])
        for g in ORDER[1:]:
            i = (dd.gc24_str == g).values
            if i.sum() == 0:
                continue
            worst = max(worst, abs((np.average(m[i], weights=ww[i]) - m0) / sd))
    return worst


def ci(r, key="p"):
    if r is None:
        return "not estimable"
    return "%.2f (%.2f-%.2f) P=%.3f %s" % (r["OR"], r["lo"], r["hi"], r[key],
                                           r.get("model", ""))


def p3(v):
    if v is None or not np.isfinite(v):
        return "NA"
    return "%.3f" % v


def evalue(rr):
    rr = max(float(rr), 1e-9)
    if rr < 1:
        rr = 1.0 / rr
    return rr + np.sqrt(rr * (rr - 1))


# ================================================================= load
df = pd.read_csv(os.path.join(DATA, "cohort_rheum_icu.csv"))
df = df.rename(columns={"sofa24": "sofa"})
ab = pd.read_csv(os.path.join(DATA, "abx_outcomes_rheum.csv"))
NEW = ["stay_key", "new_agent_24_7d", "incident_strict", "legacy_any48",
       "culture_pos_24_7d", "n_agent_pre24",
       "abx_naive_at24", "window_complete", "ondan24", "docusate24"]
df = df.merge(ab[NEW], on="stay_key", how="left")

gh = pd.read_csv(os.path.join(DATA, "gc_history.csv"))
GCOLS = [c for c in gh.columns if c != "stay_key"]
df = df.merge(gh[["stay_key"] + GCOLS], on="stay_key", how="left")

df["sofa"] = df.sofa.fillna(df.sofa.median())
df["gc_str_num"] = df.gc24_str.map(SCORE)
df["is_sle"] = (df.primary_grp == "SLE").astype(int)
df["is_ra"] = (df.primary_grp == "RA").astype(int)
df["hyper_48h"] = df.hyper_48h.astype(float)

for c in ["death_30d", "death_hosp", "gi_bleed", "infect_icd", "new_agent_24_7d",
          "culture_pos_24_7d", "incident_strict",
          "legacy_any48", "window_complete", "ondan24", "docusate24",
          "gc_user_struct", "home_gc", "gc_prior_hadm", "gc_pre_icu",
          "gc_any_hosp", "cxh" if "cxh" in df.columns else "home_gc"]:
    if c in df.columns:
        df[c] = df[c].astype(float)

# ---- 结构性 GC 使用者（院前家庭用药 / 既往住院用过 / 窗口前本次住院已用）
df["gc_user_struct"] = ((df.home_gc == 1) | (df.gc_prior_hadm == 1) |
                        (df.gc_pre_icu == 1)).astype(float)

# ---- 指征衍生（note_found=0 记为缺失，绝不当作「无活动」）
nf = df.get("note_found", pd.Series(0, index=df.index)).fillna(0).astype(int)
for src, dst in [("indN_spec", "spec"), ("indN_gen", "gen"),
                 ("indN_shock", "shockdoc"), ("indD_spec", "dxspec"),
                 ("ind_spec", "gcspec")]:
    v = df[src].fillna(0).astype(float)
    df[dst] = np.where(nf == 1, v, np.nan)
df["cxhold"] = np.where(nf == 1,
                        df.get("ctx_hold_gc", pd.Series(0, index=df.index))
                        .fillna(0).astype(float), np.nan)
df["cxinf"] = np.where(nf == 1,
                       df.get("ctx_infection", pd.Series(0, index=df.index))
                       .fillna(0).astype(float), np.nan)

# ---- 指征-暴露一致性四格（仅在 note_found=1 的人群里）
df["gcpos"] = (df.gc_str_num >= 1).astype(int)
CONC = ["C_noGC_spec", "C_noGC_nospec", "C_gc_spec", "C_gc_nospec"]
df["conc"] = np.where(nf != 1, "no_note",
                      np.where((df.gcpos == 1) & (df.spec == 1), "C_gc_spec",
                      np.where((df.gcpos == 1) & (df.spec == 0), "C_gc_nospec",
                      np.where((df.gcpos == 0) & (df.spec == 1), "C_noGC_spec",
                               "C_noGC_nospec"))))

both = df[df.primary_grp.isin(["SLE", "RA"])].copy()
bothn = both[both.note_found == 1].copy()
sle = both[both.primary_grp == "SLE"]
ra = both[both.primary_grp == "RA"]

P("=" * 104)
P("v6  SLE vs RA HEAD-TO-HEAD : DEATH INTERACTION + INDICATION MECHANISM")
P("=" * 104)
P("  cohort n=%d  SLE n=%d  RA n=%d" % (len(df), len(sle), len(ra)))
P("  出院记录可得: 全队列 %.1f%% ; SLE+RA %.1f%% (n=%d)" % (
    100 * df.note_found.fillna(0).mean(),
    100 * both.note_found.mean(), len(bothn)))
P("  PRE-SPECIFIED CONFIRMATORY TEST = disease x dose interaction on 30-day death")
P("  outcomes: death_30d (primary) / death_hosp (co-primary) / "
  "culture_pos_24_7d / new_agent_24_7d (secondary)")
P("  controls : hyper_48h (positive) / gi_bleed / infect_icd")
P("")

# ================================================================= 0b
P("=" * 104)
P("0b. ANCHOR -- the note-availability restriction is a SELECTION; does it move the")
P("    death interaction?  (v5 used ALL SLE+RA; v6 needs a discharge note)")
P("=" * 104)
r_full = interaction(both, "death_30d", CORE)
r_note = interaction(bothn, "death_30d", CORE)
P("  full SLE+RA (n=%d, events=%d) : %s" % (
    len(both), int(both.death_30d.sum()),
    "%.2f (%.2f-%.2f) P=%.3f" % (r_full["OR"], r_full["lo"], r_full["hi"],
                                 r_full["p_lrt"]) if r_full else "NE"))
P("  with discharge note (n=%d, events=%d): %s" % (
    len(bothn), int(bothn.death_30d.sum()),
    "%.2f (%.2f-%.2f) P=%.3f" % (r_note["OR"], r_note["lo"], r_note["hi"],
                                 r_note["p_lrt"]) if r_note else "NE"))
P("")
P("  选择性的来源（全 SLE+RA 内，有 vs 无出院记录）：")
sel_a = both[both.note_found == 1]
sel_b = both[both.note_found == 0]
P("  %-42s %14s %14s %10s" % ("", "note (n=%d)" % len(sel_a),
                             "no note (n=%d)" % len(sel_b), "|SMD|"))
sel_rows = []
for v, lab, kind in [("age", "age, median", "med"),
                     ("sofa", "SOFA, median", "med"),
                     ("death_30d", "30-day mortality, %", "pct"),
                     ("death_hosp", "in-hospital mortality, %", "pct"),
                     ("gc24_daily_pe_mg", "GC dose mg/day, median", "med"),
                     ("gc_user_struct", "structural GC user, %", "pct"),
                     ("hcq", "hydroxychloroquine, %", "pct")]:
    a, b = sel_a[v].astype(float), sel_b[v].astype(float)
    s = ("%.1f%%" % (100 * a.mean()), "%.1f%%" % (100 * b.mean())) if kind == "pct" \
        else ("%.1f" % a.median(), "%.1f" % b.median())
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    smd = abs(a.mean() - b.mean()) / sd if sd > 0 else np.nan
    P("  %-42s %14s %14s %10.3f" % (lab, s[0], s[1], smd))
    sel_rows.append(dict(variable=lab, note=s[0], no_note=s[1],
                         smd=round(float(smd), 4) if np.isfinite(smd) else None))
P("")
P("  => 若两者交互估计接近，则文本分析在 1250 例子集内的结论可外推到全队列；")
P("     若差异明显，则必须把「有出院记录」本身当成一个选择变量来处理。")


# ================================================================= 1
P("=" * 104)
P("1. Q1 -- IS THE REFERENCE GROUP (no GC in first 24 h) THE SAME IN BOTH DISEASES?")
P("=" * 104)
P("  只在 G0_none 内比较，SLE vs RA：")
g0 = bothn[bothn.gc24_str == "G0_none"]
g0s, g0r = g0[g0.primary_grp == "SLE"], g0[g0.primary_grp == "RA"]
VARS = [("age", "age, median", "med"),
        ("sofa", "SOFA, median", "med"),
        ("renal_fail", "renal failure, %", "pct"),
        ("shock", "shock, %", "pct"),
        ("resp_fail", "respiratory failure, %", "pct"),
        ("vaso24", "vasopressor <=24 h, %", "pct"),
        ("vent24", "ventilation <=24 h, %", "pct"),
        ("immuno_any", "non-GC immunosuppressant, %", "pct"),
        ("hcq", "hydroxychloroquine, %", "pct"),
        ("home_gc", "HOME GC on admission list, %", "pct"),
        ("gc_prior_hadm", "GC in a PRIOR admission, %", "pct"),
        ("gc_user_struct", "structural GC user, %", "pct"),
        ("spec", "documented disease-specific activity, %", "pct"),
        ("gen", "documented generic organ failure, %", "pct"),
        ("shockdoc", "documented shock / stress-dose, %", "pct"),
        ("cxinf", "\"infection\" discussed, %", "pct"),
        ("cxhold", "GC explicitly held/tapered, %", "pct"),
        ("death_30d", "30-day mortality, %", "pct"),
        ("death_hosp", "in-hospital mortality, %", "pct")]
P("  %-42s %14s %14s %10s" % ("", "SLE (n=%d)" % len(g0s),
                             "RA (n=%d)" % len(g0r), "|SMD|"))
ref_rows = []
for v, lab, kind in VARS:
    a, b = g0s[v].astype(float), g0r[v].astype(float)
    if kind == "pct":
        s = ("%.1f%%" % (100 * a.mean()), "%.1f%%" % (100 * b.mean()))
    else:
        s = ("%.1f" % a.median(), "%.1f" % b.median())
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    smd = abs(a.mean() - b.mean()) / sd if sd > 0 else np.nan
    P("  %-42s %14s %14s %10.3f" % (lab, s[0], s[1], smd))
    ref_rows.append(dict(variable=lab, sle=s[0], ra=s[1],
                         smd=round(float(smd), 4) if np.isfinite(smd) else None))
P("")
P("  G0 组的「真 GC-free」比例：")
for g, sub in [("SLE", g0s), ("RA", g0r)]:
    P("    %s  真 GC-free(home_gc=0 & 全程无 GC) %.1f%% ; 住院期内后来用过 GC %.1f%%"
      % (g, 100 * ((sub.home_gc == 0) & (sub.gc_any_hosp == 0)).mean(),
         100 * (sub.gc_any_hosp == 1).mean()))

# ================================================================= 2
P("")
P("=" * 104)
P("2. Q2 -- ARE THE FOUR DOSE STRATA ONE POPULATION, OR FOUR DIFFERENT POPULATIONS?")
P("=" * 104)
P("  %-9s %6s %9s %9s %9s %9s %9s %9s" % (
    "stratum", "n", "homeGC%", "gcu%", "spec%", "gen%", "shock%", "未提GC%"))
struct_rows = []
for g in ["SLE", "RA"]:
    s = bothn[bothn.primary_grp == g]
    P("  -- %s (n=%d) --" % (g, len(s)))
    for st in ORDER:
        sub = s[s.gc24_str == st]
        if not len(sub):
            continue
        P("  %-9s %6d %8.1f%% %8.1f%% %8.1f%% %8.1f%% %8.1f%% %8.1f%%" % (
            LABEL[st], len(sub), 100 * sub.home_gc.mean(),
            100 * sub.gc_user_struct.mean(), 100 * sub.spec.mean(),
            100 * sub.gen.mean(), 100 * sub.shockdoc.mean(),
            100 * (sub.n_gc_mentions.fillna(0) == 0).mean()))
        struct_rows.append(dict(
            disease=g, stratum=LABEL[st], n=len(sub),
            home_gc=round(100 * sub.home_gc.mean(), 2),
            gcu=round(100 * sub.gc_user_struct.mean(), 2),
            spec=round(100 * sub.spec.mean(), 2),
            gen=round(100 * sub.gen.mean(), 2),
            shock=round(100 * sub.shockdoc.mean(), 2),
            gc_unmentioned=round(100 * (sub.n_gc_mentions.fillna(0) == 0).mean(), 2)))
P("")
P("  读法：G1_low 层在 SLE/RA 都是 %s 的院前 GC 携带者 —— 低剂量层不是"
  "「少量用激素」，" % "85-88%")
P("        而是**继续家庭剂量**；真正的新起始集中在 G2/G3。")

# ================================================================= 3
P("")
P("=" * 104)
P("3. Q3a -- INDICATION x EXPOSURE CONCORDANCE (the 2x2 that the mechanism lives in)")
P("=" * 104)
P("  spec = 出院记录中出现风湿病特异的疾病活动/器官受累")
P("  四格:  A GC+ & spec+  |  B GC+ & spec-  |  C noGC & spec+  |  D noGC & spec-")
P("  %-16s %6s %6s %8s %8s %10s" % (
    "concordance", "SLE n", "RA n", "SLE 死亡%", "RA 死亡%", "RA/SLE 死亡比"))
conc_rows = []
for c, lab in [("C_gc_spec", "A GC+ spec+"), ("C_gc_nospec", "B GC+ spec-"),
               ("C_noGC_spec", "C noGC spec+"), ("C_noGC_nospec", "D noGC spec-")]:
    a = bothn[(bothn.conc == c) & (bothn.primary_grp == "SLE")]
    b = bothn[(bothn.conc == c) & (bothn.primary_grp == "RA")]
    da = 100 * a.death_30d.mean() if len(a) else np.nan
    db = 100 * b.death_30d.mean() if len(b) else np.nan
    ratio = (db / da) if (len(a) and len(b) and da > 0 and db > 0) else np.nan
    P("  %-16s %6d %6d %8s %8s %10s" % (
        lab, len(a), len(b),
        "%.1f" % da if np.isfinite(da) else "NA",
        "%.1f" % db if np.isfinite(db) else "NA",
        "%.2f" % ratio if np.isfinite(ratio) else "NA"))
    conc_rows.append(dict(group=lab, code=c, sle_n=len(a), ra_n=len(b),
                          sle_death=round(da, 2) if np.isfinite(da) else None,
                          ra_death=round(db, 2) if np.isfinite(db) else None,
                          sle_ev=int(a.death_30d.sum()), ra_ev=int(b.death_30d.sum())))
P("")
P("  判读：C 格（有活动却没用 GC）是最可疑的效度缺口 —— 人数与死亡率决定结论。")

# ================================================================= 4
P("")
P("=" * 104)
P("4. DEATH DOSE-RESPONSE WITHIN DISEASE, STRATIFIED BY INDICATION / GC HISTORY")
P("=" * 104)
OUTS = [("death_30d", "30-day death"), ("death_hosp", "in-hospital death"),
        ("culture_pos_24_7d", "blood culture + (spec.)"),
        ("new_agent_24_7d", "new antimicrobial"),
        ("hyper_48h", "glucose >=180 (POS CTRL)")]
STRATA = [("ALL (SLE+RA, with note)", lambda d: d),
          ("spec+ (documented activity)", lambda d: d[d.spec == 1]),
          ("spec- (no documented activity)", lambda d: d[d.spec == 0]),
          ("GC-naive (gc_user_struct=0)", lambda d: d[d.gc_user_struct == 0]),
          ("prior/home GC user", lambda d: d[d.gc_user_struct == 1])]
P("  %-34s %-24s %-24s" % ("stratum", "SLE trend OR", "RA trend OR"))
doserep = []
for slab, fn in STRATA:
    sub = fn(bothn)
    cells = []
    for g in ["SLE", "RA"]:
        r = trend(sub[sub.primary_grp == g], "death_30d", CORE)
        cells.append(r)
    P("  %-34s %-24s %-24s" % (
        slab, ci(cells[0]) if cells[0] else "NE",
        ci(cells[1]) if cells[1] else "NE"))
    for g, r in zip(["SLE", "RA"], cells):
        if r:
            doserep.append(dict(stratum=slab, disease=g, outcome="death_30d",
                                OR=round(r["OR"], 3), lo=round(r["lo"], 3),
                                hi=round(r["hi"], 3), p=r["p"], n=r["n"],
                                n_ev=r["n_ev"], model=r["model"]))
P("")
P("  同分层下的阳性对照（确保分层没有毁掉暴露测量）：")
P("  %-34s %-24s %-24s" % ("stratum", "SLE glucose OR", "RA glucose OR"))
for slab, fn in STRATA:
    sub = fn(bothn)
    cells = [trend(sub[sub.primary_grp == g], "hyper_48h", CORE)
             for g in ["SLE", "RA"]]
    P("  %-34s %-24s %-24s" % (
        slab, ci(cells[0]) if cells[0] else "NE",
        ci(cells[1]) if cells[1] else "NE"))
    for g, r in zip(["SLE", "RA"], cells):
        if r:
            doserep.append(dict(stratum=slab, disease=g, outcome="hyper_48h",
                                OR=round(r["OR"], 3), lo=round(r["lo"], 3),
                                hi=round(r["hi"], 3), p=r["p"], n=r["n"],
                                n_ev=r["n_ev"], model=r["model"]))
P("")
P("  其余结局（全队列 spec 分层）：")
for y, lab in OUTS[2:]:
    P("  -- %s --" % lab)
    for slab, fn in STRATA[:3]:
        sub = fn(bothn)
        cells = [trend(sub[sub.primary_grp == g], y, CORE) for g in ["SLE", "RA"]]
        P("    %-32s %-24s %-24s" % (
            slab, ci(cells[0]) if cells[0] else "NE",
            ci(cells[1]) if cells[1] else "NE"))
        for g, r in zip(["SLE", "RA"], cells):
            if r:
                doserep.append(dict(stratum=slab, disease=g, outcome=y,
                                    OR=round(r["OR"], 3), lo=round(r["lo"], 3),
                                    hi=round(r["hi"], 3), p=r["p"], n=r["n"],
                                    n_ev=r["n_ev"], model=r["model"]))

# ================================================================= 5
P("")
P("=" * 104)
P("5. THE DEATH INTERACTION -- WHAT HAPPENS WHEN INDICATION STRUCTURE IS ADDED")
P("=" * 104)
P("  RA/SLE ratio of dose-trend ORs for DEATH; each adjustment set adds structure.")
ADJ = [
    ("A  base (v5 adjustment set)", [], CORE, "base"),
    ("B  + structural GC user", [], CORE + ["gc_user_struct"], "gcu"),
    ("C  + documented activity (spec)", ["spec"], CORE, "spec_ix"),
    ("D  + generic organ failure (gen)", ["gen"], CORE, "gen_ix"),
    ("E  + shock / stress-dose", ["shockdoc"], CORE, "shock_ix"),
    ("F  + spec & gen & shock (mechanism)", ["spec", "gen", "shockdoc"],
     CORE, "mech"),
    ("G  + all of the above", ["spec", "gen", "shockdoc"],
     CORE + ["gc_user_struct"], "all"),
]
P("  %-38s %-26s %8s %8s %8s" % ("adjustment set", "RA/SLE (RA vs SLE)", "P(LRT)",
                                 "n_ev", "attn%"))
ixrep = []
base_or = None
for lab, ei, cvs, key in ADJ:
    r = interaction(bothn, "death_30d", cvs, extra_int=ei)
    if r is None:
        P("  %-38s %s" % (lab, "not estimable"))
        continue
    if base_or is None:
        base_or = r["OR"]
    attn = 100 * (1 - np.log(r["OR"]) / np.log(base_or)) if base_or else np.nan
    P("  %-38s %-26s %8s %8d %7.1f%%" % (
        lab, "%.2f (%.2f-%.2f)" % (r["OR"], r["lo"], r["hi"]),
        p3(r["p_lrt"]), r["n_ev"], attn))
    ixrep.append(dict(label=lab, key=key, OR=round(r["OR"], 3),
                      lo=round(r["lo"], 3), hi=round(r["hi"], 3),
                      p_lrt=r["p_lrt"], n=r["n"], n_ev=r["n_ev"],
                      atten_pct=round(float(attn), 2) if np.isfinite(attn) else None))
P("")
P("  同一组调整下，主结局与其余结局（看指征是否「只解释死亡」）：")
for y, lab in [("death_hosp", "in-hospital death"),
               ("culture_pos_24_7d", "blood culture + (spec.)"),
               ("new_agent_24_7d", "new antimicrobial"),
               ("hyper_48h", "glucose >=180 (POS CTRL)")]:
    r0 = interaction(bothn, y, CORE)
    r1 = interaction(bothn, y, CORE + ["gc_user_struct"],
                     extra_int=["spec", "gen", "shockdoc"])
    P("  %-30s base %-22s +mechanism %-22s" % (
        lab, "%.2f (%.2f-%.2f)" % (r0["OR"], r0["lo"], r0["hi"]) if r0 else "NE",
        "%.2f (%.2f-%.2f)" % (r1["OR"], r1["lo"], r1["hi"]) if r1 else "NE"))

# ================================================================= 6
P("")
P("=" * 104)
P("6. DEATH INTERACTION *WITHIN* EACH INDICATION STRATUM (does it survive?)")
P("=" * 104)
P("  %-36s %-26s %8s %6s %6s" % ("stratum", "RA/SLE (RA vs SLE)", "P(LRT)",
                                 "n_SLE", "n_RA"))
ixstrat = []
for slab, fn in [("ALL with note", lambda d: d),
                 ("spec+ documented activity", lambda d: d[d.spec == 1]),
                 ("spec- no activity", lambda d: d[d.spec == 0]),
                 ("generic organ failure +", lambda d: d[d.gen == 1]),
                 ("shock / stress-dose +", lambda d: d[d.shockdoc == 1]),
                 ("GC-naive (gc_user_struct=0)", lambda d: d[d.gc_user_struct == 0]),
                 ("GC user (prior/home/ward)", lambda d: d[d.gc_user_struct == 1]),
                 ("concordant GC+ spec+", lambda d: d[(d.gcpos == 1) & (d.spec == 1)]),
                 ("discordant GC+ spec-", lambda d: d[(d.gcpos == 1) & (d.spec == 0)]),
                 ("discordant noGC spec+", lambda d: d[(d.gcpos == 0) & (d.spec == 1)])]:
    sub = fn(bothn)
    r = interaction(sub, "death_30d", CORE)
    ns = int((sub.primary_grp == "SLE").sum())
    nr = int((sub.primary_grp == "RA").sum())
    P("  %-36s %-26s %8s %6d %6d" % (
        slab, "%.2f (%.2f-%.2f)" % (r["OR"], r["lo"], r["hi"]) if r
        else "not estimable",
        p3(r["p_lrt"]) if r else "-", ns, nr))
    if r:
        ixstrat.append(dict(stratum=slab, OR=round(r["OR"], 3),
                            lo=round(r["lo"], 3), hi=round(r["hi"], 3),
                            p_lrt=r["p_lrt"], p_wald=r.get("p_wald"),
                            n=r["n"], n_ev=r["n_ev"], n_sle=ns, n_ra=nr,
                            model=r["model"]))
P("")
P("  阳/阴性对照在同一分层下的交互（检验分层本身是否制造信号）：")
P("  %-36s %-26s %-26s" % ("stratum", "glucose (POS)", "GI bleed"))
for slab, fn in [("ALL with note", lambda d: d),
                 ("spec+ documented activity", lambda d: d[d.spec == 1]),
                 ("spec- no activity", lambda d: d[d.spec == 0])]:
    sub = fn(bothn)
    a = interaction(sub, "hyper_48h", CORE)
    b = interaction(sub, "gi_bleed", CORE)
    P("  %-36s %-26s %-26s" % (
        slab,
        "%.2f (%.2f-%.2f)" % (a["OR"], a["lo"], a["hi"]) if a else "NE",
        "%.2f (%.2f-%.2f)" % (b["OR"], b["lo"], b["hi"]) if b else "NE"))

# ================================================================= 7
P("")
P("=" * 104)
P("7. D3 SELF-CHECK -- interaction must agree in DIRECTION with the ratio of the")
P("   two separately fitted per-disease trends (an inverted term still gives a P)")
P("=" * 104)
P("  %-30s %10s %10s %10s %6s" % ("outcome", "ix RA/SLE", "emp RA/SLE",
                                 "rel.diff", "OK"))
d3rows = []
for y, lab in [("death_30d", "30-day death"), ("death_hosp", "in-hospital death"),
               ("culture_pos_24_7d", "blood culture +"), ("new_agent_24_7d", "new antimicrobial"),
               ("hyper_48h", "glucose >=180"), ("gi_bleed", "GI bleeding"),
               ("infect_icd", "infection ICD")]:
    r = interaction(bothn, y, CORE)
    t1 = trend(bothn[bothn.primary_grp == "SLE"], y, CORE)
    t2 = trend(bothn[bothn.primary_grp == "RA"], y, CORE)
    if not (r and t1 and t2):
        continue
    emp = t2["OR"] / t1["OR"]
    agree = np.sign(r["OR"] - 1) == np.sign(emp - 1)
    rel = 100 * abs(r["OR"] - emp) / emp
    P("  %-30s %10.2f %10.2f %9.1f%% %6s" % (
        lab, r["OR"], emp, rel, "OK" if agree else "CHECK"))
    d3rows.append(dict(outcome=lab, var=y, ix=round(r["OR"], 3),
                       emp=round(emp, 3), rel_diff=round(rel, 1),
                       agree=bool(agree), p_lrt=r["p_lrt"], n_ev=r["n_ev"],
                       ix_lo=round(r["lo"], 3), ix_hi=round(r["hi"], 3)))
P("  规则：非零效应上出现方向不一致 = 必须停下修模型；零效应上（CI 跨 1 且")
P("        点估计距 1 很近）视为数值噪声，需在报告中如实标注。")

# ================================================================= 8
P("")
P("=" * 104)
P("8. FRAGILITY, E-VALUE AND MULTIPLICITY (30-day death, pre-specified)")
P("=" * 104)
rd = interaction(bothn, "death_30d", CORE)
if rd:
    P("  30-day death interaction RA/SLE = %.2f (%.2f-%.2f)  P(LRT)=%.3f  "
      "events=%d" % (rd["OR"], rd["lo"], rd["hi"], rd["p_lrt"], rd["n_ev"]))
    P("  E-value for the interaction (treating it as an RR) = %.2f" % evalue(rd["OR"]))
    # 脆弱性：翻转最少几个事件使 P>0.05
    use = bothn[["gc_str_num", "is_sle", "death_30d"] + CORE].dropna().copy()
    use["s"] = use.gc_str_num.astype(float)
    use["sx"] = use.s * use.is_sle.astype(float)
    X = sm.add_constant(use[["s", "is_sle", "sx"] + CORE].astype(float))
    Xr = sm.add_constant(use[["s", "is_sle"] + CORE].astype(float))
    b, se, ll = firth(X.values, use.death_30d.values)
    _, _, ll0 = firth(Xr.values, use.death_30d.values)
    base_lrt = 2 * (ll - ll0)
    P("  Firth LRT = %.3f (P=%.4f)" % (base_lrt,
                                       1 - stats.chi2.cdf(max(base_lrt, 0), 1)))
P("")
P("  multiplicity: pre-specified primary (death_30d interaction) tested ONCE,")
P("  not corrected. All other interaction tests BH-corrected within family.")
fam2 = []
for y in ["death_hosp", "culture_pos_24_7d", "new_agent_24_7d", "gi_bleed",
          "infect_icd"]:
    r = interaction(bothn, y, CORE)
    if r:
        fam2.append((y, r["p_lrt"]))
p2 = np.array([v for _, v in fam2])
o = np.argsort(p2)
m = len(p2)
q = np.empty(m)
prev = 1.0
for k in range(m - 1, -1, -1):
    prev = min(prev, p2[o[k]] * m / (k + 1))
    q[o[k]] = prev
bh_rows = []
P("  %-28s %10s %10s" % ("outcome", "P (LRT)", "q (BH, family)"))
for (y, pv), qv in zip(fam2, q):
    P("  %-28s %10.3f %10.3f" % (y, pv, qv))
    bh_rows.append(dict(outcome=y, p_lrt=pv, q=float(qv),
                        rank=int(np.where(o == list(p2).index(pv))[0][0]) + 1))
tot = np.array([pv for _, pv in fam2] + [rd["p_lrt"] if rd else np.nan])
tot = tot[np.isfinite(tot)]
o2 = np.argsort(tot)
q2 = np.empty(len(tot))
prev = 1.0
for k in range(len(tot) - 1, -1, -1):
    prev = min(prev, tot[o2[k]] * len(tot) / (k + 1))
    q2[o2[k]] = prev
P("")
P("  whole-family BH (incl. the pre-specified primary, for transparency):")
for y, pv in fam2 + [("death_30d (PRIMARY)", rd["p_lrt"] if rd else np.nan)]:
    if not np.isfinite(pv):
        continue
    P("    %-28s P=%.3f  q_whole=%.3f" % (y, pv, q2[list(tot).index(pv)]))

P("")
P("=" * 104)
P("9. OPERATIONALISATION SENSITIVITY -- does the mechanism depend on HOW activity")
P("   was extracted?  (narr=full narrative | dx=discharge diagnosis only)")
P("=" * 104)
P("  9a. 用最保守口径 dxspec（仅出院诊断段）重做 >G0 参照组对比：")
g0d = bothn[bothn.gc24_str == "G0_none"]
P("    %-38s %14s %14s" % ("", "SLE n=%d" % int((g0d.primary_grp == "SLE").sum()),
                             "RA n=%d" % int((g0d.primary_grp == "RA").sum())))
for col, lab in [("spec", "documented activity (narrative)"),
                 ("dxspec", "documented activity (discharge dx)"),
                 ("gen", "generic organ failure"),
                 ("shockdoc", "shock / stress-dose")]:
    a = g0d[g0d.primary_grp == "SLE"][col]
    b = g0d[g0d.primary_grp == "RA"][col]
    P("    %-38s %13.1f%% %13.1f%%" % (lab, 100 * a.mean(), 100 * b.mean()))
P("")
P("  9b. 一致性四格用 dxspec 重做（死亡 %）：")
P("    %-16s %6s %6s %9s %9s" % ("concordance(dx)", "SLE n", "RA n",
                                     "SLE死亡%", "RA死亡%"))
dxconc = []
for gcpos in [1, 0]:
    for sp in [1, 0]:
        a = bothn[(bothn.gcpos == gcpos) & (bothn.dxspec == sp) &
                  (bothn.primary_grp == "SLE")]
        b = bothn[(bothn.gcpos == gcpos) & (bothn.dxspec == sp) &
                  (bothn.primary_grp == "RA")]
        P("    %-16s %6d %6d %9s %9s" % (
            "GC%d/dxspec%d" % (gcpos, sp), len(a), len(b),
            "%.1f" % (100 * a.death_30d.mean()) if len(a) else "NA",
            "%.1f" % (100 * b.death_30d.mean()) if len(b) else "NA"))
        dxconc.append(dict(group="GC%d/dxspec%d" % (gcpos, sp), sle_n=len(a),
                           ra_n=len(b),
                           sle_death=round(100 * a.death_30d.mean(), 2) if len(a) else None,
                           ra_death=round(100 * b.death_30d.mean(), 2) if len(b) else None))
P("")
P("  9c. 交互分层用 dxspec 重做：")
P("    %-38s %-26s %8s" % ("stratum", "RA/SLE", "P(LRT)"))
dxsens = []
for lab, sub in [("activity: narrative (spec+)", bothn[bothn.spec == 1]),
                 ("activity: narrative (spec-)", bothn[bothn.spec == 0]),
                 ("activity: discharge dx (dxspec+)", bothn[bothn.dxspec == 1]),
                 ("activity: discharge dx (dxspec-)", bothn[bothn.dxspec == 0])]:
    r = interaction(sub, "death_30d", CORE)
    P("    %-38s %-26s %8s" % (
        lab, "%.2f (%.2f-%.2f)" % (r["OR"], r["lo"], r["hi"]) if r else "NE",
        p3(r["p_lrt"]) if r else "-"))
    if r:
        dxsens.append(dict(label=lab, OR=round(r["OR"], 3), lo=round(r["lo"], 3),
                           hi=round(r["hi"], 3), p_lrt=r["p_lrt"], n=r["n"],
                           n_ev=r["n_ev"]))
anchor = dict(full_n=int(len(both)), full_ev=int(both.death_30d.sum()),
              full_or=round(r_full["OR"], 3), full_lo=round(r_full["lo"], 3),
              full_hi=round(r_full["hi"], 3), full_p=r_full["p_lrt"],
              note_n=int(len(bothn)), note_ev=int(bothn.death_30d.sum()),
              note_or=round(r_note["OR"], 3), note_lo=round(r_note["lo"], 3),
              note_hi=round(r_note["hi"], 3), note_p=r_note["p_lrt"])
P("")
P("  9d. 把「无出院记录」当作独立第三层，在全队列上重估交互（避免用子集外推）：")
both_all = both.copy()
both_all["spec3"] = np.where(both_all.note_found == 1, both_all.spec, 2)
for lev, lab in [(1.0, "spec+ (activity documented)"), (0.0, "spec- (no activity)")]:
    sub = both_all[both_all.spec3 == lev]
    r = interaction(sub, "death_30d", CORE)
    P("    %-38s %-26s %8s" % (
        lab, "%.2f (%.2f-%.2f)" % (r["OR"], r["lo"], r["hi"]) if r else "NE",
        p3(r["p_lrt"]) if r else "-"))
P("    注：no-note 层无法判定活动度，故不进入上面两行；其存在本身即限制外推。")

# ================================================================= export
pd.DataFrame(ref_rows).to_csv(os.path.join(OUT, "table_v6_refgroup.csv"), index=False)
pd.DataFrame(sel_rows).to_csv(os.path.join(OUT, "table_v6_selection.csv"), index=False)
pd.DataFrame(struct_rows).to_csv(os.path.join(OUT, "table_v6_structure.csv"), index=False)
pd.DataFrame(conc_rows).to_csv(os.path.join(OUT, "table_v6_concordance.csv"), index=False)
pd.DataFrame(doserep).to_csv(os.path.join(OUT, "table_v6_dose.csv"), index=False)
pd.DataFrame(ixrep).to_csv(os.path.join(OUT, "table_v6_ix_adjust.csv"), index=False)
pd.DataFrame(ixstrat).to_csv(os.path.join(OUT, "table_v6_ix_strata.csv"), index=False)
pd.DataFrame(d3rows).to_csv(os.path.join(OUT, "table_v6_d3.csv"), index=False)
pd.DataFrame(bh_rows).to_csv(os.path.join(OUT, "table_v6_bh.csv"), index=False)

with open(os.path.join(OUT, "_v6_results.json"), "w", encoding="utf-8") as f:
    json.dump(dict(refgroup=ref_rows, structure=struct_rows, concordance=conc_rows,
                   dose=doserep, ix_adjust=ixrep, ix_strata=ixstrat, d3=d3rows,
                   bh=bh_rows, selection=sel_rows, dxconc=dxconc,
                   dxsens=dxsens, anchor=anchor), f, ensure_ascii=False, indent=1)

with open(os.path.join(OUT, "160_indication_mechanism.txt"), "w",
          encoding="utf-8") as f:
    f.write("\n".join(lines))
P("")
P("saved: out/160_indication_mechanism.txt + table_v6_*.csv + _v6_results.json")
