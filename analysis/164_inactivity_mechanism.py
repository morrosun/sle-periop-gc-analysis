# -*- coding: utf-8 -*-
"""
164_inactivity_mechanism.py
===========================
v7 —— 「无活动记录」三通道分解后，v6 的死亡交互结论还站得住吗？

v6 的结论是：**死亡交互只活在「无活动记录」(spec-) 层**（2.08 / 1.69）。
163 号脚本把这一层拆开后发现它不是同质的：
    INACTIVE  明确记载无活动        57  (6.5%)
    INF_DEFER 明确因感染暂缓/停用GC  25  (2.9%)
    UNDERDOC  记录里根本没提活动度  792 (90.6%)
而且 30 天死亡率为 3.5% / 4.0% / **9.7%** —— 死亡集中在「记录不足」层。

本脚本回答四个问题
  Q1  这三条通道的临床画像是否支持「真无活动 / 该给未给 / 记录不足」的命名？
  Q2  **记录强度本身是否预测死亡？**（若记录越少死亡越高，则 spec- 层的死亡信号
      是记录伪影，而不是疾病严重度 —— 这是本层结论的最大威胁）
  Q3  在否定校正后的 spec 下，v6「交互只活在 spec- 层」的结论是否改变？
  Q4  交互在 UNDERDOC 内部是否仍然存在（= 是否只是被记录不足驱动）？

同时复核 v6 的否定假阳性（18 例）对 spec+ 结论的影响。

输出 out/164_inactivity_mechanism.txt + table_v7_*.csv + _v7_results.json
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
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

lines = []


def P(s=""):
    print(s)
    lines.append(str(s))


# ================================================================= helpers (同 160)
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


def trend(d, y, extra=None, col="gc_str_num", use_firth=None, force_mle=False):
    cols = [col, y] + (extra or [])
    use = d[cols].dropna()
    if len(use) == 0 or use[y].nunique() < 2 or use[col].nunique() < 2:
        return None
    if use_firth is None:
        use_firth = int(use[y].sum()) < 60
    try:
        if use_firth and not force_mle:
            X = sm.add_constant(use[[col] + (extra or [])].astype(float))
            b, se, _ = firth(X.values, use[y].values)
            j = list(X.columns).index(col)
            model = "Firth"
        else:
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
    e = float(np.exp(b[j]))
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


def ci(r, key="p"):
    if r is None:
        return "not estimable"
    return "%.2f (%.2f-%.2f) P=%.3f %s" % (r["OR"], r["lo"], r["hi"], r[key],
                                           r.get("model", ""))


def p3(v):
    return "NA" if (v is None or not np.isfinite(v)) else "%.3f" % v


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
       "culture_pos_24_7d", "n_agent_pre24", "abx_naive_at24",
       "window_complete", "ondan24", "docusate24"]
df = df.merge(ab[NEW], on="stay_key", how="left")
gh = pd.read_csv(os.path.join(DATA, "gc_history.csv"))
KEEP = ["stay_key", "note_found", "note_len", "n_gc_mentions", "home_gc",
        "dc_gc", "home_gc_chronic_phrase", "gcctx_len", "ctx_infection",
        "ctx_hold_gc", "indN_spec", "indN_gen", "indN_shock", "indD_spec",
        "indD_gen", "indD_shock", "gc_prior_hadm", "gc_pre_icu",
        "gc_any_hosp"]
df = df.merge(gh[[c for c in KEEP if c in gh.columns]], on="stay_key",
              how="left")
ic = pd.read_csv(os.path.join(DATA, "gc_inactivity.csv"))
df = df.merge(ic[[c for c in ic.columns if c != "note_len2"]],
              on="stay_key", how="left")

df["sofa"] = df.sofa.fillna(df.sofa.median())
df["gc_str_num"] = df.gc24_str.map(SCORE)
df["is_sle"] = (df.primary_grp == "SLE").astype(int)
df["hyper_48h"] = df.hyper_48h.astype(float)
for c in ["death_30d", "death_hosp", "gi_bleed", "infect_icd", "new_agent_24_7d",
          "culture_pos_24_7d", "incident_strict", "legacy_any48",
          "window_complete", "ondan24", "docusate24", "home_gc", "gc_prior_hadm",
          "gc_pre_icu", "gc_any_hosp"]:
    if c in df.columns:
        df[c] = df[c].astype(float)
df["gc_user_struct"] = ((df.home_gc == 1) | (df.gc_prior_hadm == 1) |
                        (df.gc_pre_icu == 1)).astype(float)

both = df[df.primary_grp.isin(["SLE", "RA"])].copy()
bothn = both[both.note_found == 1].copy()
# spec- 层 = 否定校正后仍无疾病特异活动记录
sp0 = bothn[bothn.spec_true == 0].copy()

P("=" * 106)
P("v7  「无活动记录」三通道分解 —— v6 死亡交互结论的稳健性")
P("=" * 106)
P("  SLE+RA n=%d ; 有出院记录 n=%d ; 否定校正后 spec- n=%d (SLE %d / RA %d)"
  % (len(both), len(bothn), len(sp0),
     int((sp0.primary_grp == "SLE").sum()), int((sp0.primary_grp == "RA").sum())))
P("  30-day death: 全队列 %d ; spec- %d ; spec+ %d" % (
    int(bothn.death_30d.sum()), int(sp0.death_30d.sum()),
    int(bothn[bothn.spec_true == 1].death_30d.sum())))
P("")

# ================================================================= 1 通道画像
P("=" * 106)
P("1. Q1 -- 三条通道是不是三个不同的人群？（否定校正后的 spec- 层内）")
P("=" * 106)
PROF = [("n", "n", "i"),
        ("home_gc", "HOME GC on admission %", "p"),
        ("gc_user_struct", "structural GC user %", "p"),
        ("gc24_used", "GC given in first 24 h %", "p"),
        ("sofa", "SOFA, median", "m"),
        ("age", "age, median", "m"),
        ("icu_los_days", "ICU LOS days, median", "m"),
        ("spec_neg_hits", "activity terms NEGATED (n>0) %", "p2"),
        ("activ_any", "any activity-assay mentioned %", "p"),
        ("note_len", "discharge note chars, median", "m"),
        ("n_gc_mentions", "GC unmentioned in narrative %", "zu"),
        ("hcq", "hydroxychloroquine %", "p2"),
        ("immuno_any", "non-GC immunosuppressant %", "p"),
        ("death_30d", "30-day mortality %", "p"),
        ("death_hosp", "in-hospital mortality %", "p"),
        ("culture_pos_24_7d", "blood culture + (24h-7d) %", "p"),
        ("new_agent_24_7d", "new antimicrobial %", "p")]
sp0 = sp0.copy()
sp0["gc24_used"] = (sp0.gc_str_num >= 1).astype(float)
sp0["n_gc_mentions"] = sp0.n_gc_mentions.fillna(0)
GROUPS = ["INACTIVE", "INF_DEFER", "UNDERDOC"]
P("  %-40s %12s %12s %12s" % ("", "INACTIVE", "INF_DEFER", "UNDERDOC"))
prof_rows = []
for col, lab, kind in PROF:
    if col != "n" and col not in sp0.columns:
        P("  %-40s %12s %12s %12s" % (lab, "-", "-", "-"))
        continue
    cells = []
    for g in GROUPS:
        s = sp0[sp0.inact3 == g]
        if col == "n":
            cells.append("%d" % len(s))
        elif col == "n_gc_mentions":
            cells.append("%.1f%%" % (100 * (s[col] == 0).mean()))
        elif col == "spec_neg_hits":
            cells.append("%.1f%%" % (100 * (s[col] > 0).mean()))
        elif kind == "p":
            cells.append("%.1f%%" % (100 * s[col].mean()))
        elif kind == "p2":
            cells.append("%.1f%%" % (100 * s[col].dropna().mean()))
        elif kind == "m":
            cells.append("%.1f" % s[col].median())
        else:
            cells.append("%d" % len(s))
    P("  %-40s %12s %12s %12s" % (lab, cells[0], cells[1], cells[2]))
    prof_rows.append(dict(variable=lab, INACTIVE=cells[0], INF_DEFER=cells[1],
                          UNDERDOC=cells[2]))
P("")
P("  读法：INACTIVE 若真是「轻症」，SOFA / 器官衰竭 / 死亡率都应最低；")
P("        INF_DEFER 若真是「该给而未给」，其 GC 使用率应低于而非高于其他层。")

# ================================================================= 2 记录强度
P("")
P("=" * 106)
P("2. Q2 -- **记录强度本身是否预测死亡？**（本层结论的最大威胁）")
P("=" * 106)
P("  若「记得越少、死得越多」，则 spec- 层的死亡信号是**记录伪影**，")
P("  而不是疾病严重度 —— 这会让 v6 的整个机制解释失效。")
P("")
qq = sp0.note_len.quantile([0.25, 0.5, 0.75])
P("  note_len 四分位切点: %.0f / %.0f / %.0f" % tuple(qq))
sp0["nlq"] = pd.cut(sp0.note_len, [-1, qq.iloc[0], qq.iloc[1], qq.iloc[2], 1e12],
                    labels=["Q1 shortest", "Q2", "Q3", "Q4 longest"])
P("")
P("  %-14s %6s %8s %10s %10s %10s" % ("note length", "n", "SOFA", "30d death%",
                                       "hosp death%", "cult+%"))
nlq_rows = []
for lab, s in sp0.groupby("nlq", observed=True):
    P("  %-14s %6d %8.1f %9.1f%% %9.1f%% %9.1f%%" % (
        lab, len(s), s.sofa.median(), 100 * s.death_30d.mean(),
        100 * s.death_hosp.mean(), 100 * s.culture_pos_24_7d.mean()))
    nlq_rows.append(dict(bin=str(lab), n=len(s),
                         sofa=round(float(s.sofa.median()), 2),
                         death_30d=round(100 * float(s.death_30d.mean()), 2),
                         death_hosp=round(100 * float(s.death_hosp.mean()), 2),
                         culture_pos=round(100 * float(s.culture_pos_24_7d.mean()), 2)))
# 连续：每 +1000 字
sp0["nl_k"] = sp0.note_len / 1000.0
r_len = trend(sp0, "death_30d", CORE, col="nl_k", use_firth=False)
if r_len:
    P("")
    P("  连续检验（每 +1000 字符，调整 CORE）：OR %.3f (%.3f-%.3f) P=%.3f  n=%d ev=%d"
      % (r_len["OR"], r_len["lo"], r_len["hi"], r_len["p"], r_len["n"], r_len["n_ev"]))
P("  同法看「活动度评估证据」(activ_any) 与死亡：")
r_act = trend(sp0, "death_30d", CORE, col="activ_any", use_firth=False)
if r_act:
    P("    activ_any -> death OR %.3f (%.3f-%.3f) P=%.3f" % (
        r_act["OR"], r_act["lo"], r_act["hi"], r_act["p"]))
P("  以及 HC 段长度（hc_len）在三分内是否与死亡同向：")
for g in GROUPS:
    s = sp0[sp0.inact3 == g]
    if len(s) and int(s.death_30d.sum()) >= 5:
        r = trend(s.assign(hl_k=s.hc_len / 1000.0), "death_30d", CORE,
                  col="hl_k", use_firth=False)
        P("    %-10s n=%4d ev=%2d  OR/1000chars = %s" % (
            g, len(s), int(s.death_30d.sum()),
            "%.3f (%.3f-%.3f) P=%.3f" % (r["OR"], r["lo"], r["hi"], r["p"]) if r else "NE"))
    else:
        P("    %-10s n=%4d ev=%2d  too few events" % (
            g, len(s), int(s.death_30d.sum())))

# ------------------------------------------------------------ 2b/2c/2d
sp0["hl_k"] = sp0.hc_len / 1000.0
P("")
P("=" * 106)
P("2b. 记录长度短，是「记录不足」，还是「早死没来得及写完」？")
P("=" * 106)
d1 = sp0[sp0.death_30d == 1]
d0 = sp0[sp0.death_30d == 0]
P("  %-34s %14s %14s" % ("", "died (n=%d)" % len(d1), "survived (n=%d)" % len(d0)))
dt_rows = []
for v, lab, fmt in [("note_len", "note chars, median", "%.0f"),
                    ("hc_len", "hospital-course chars, median", "%.0f"),
                    ("icu_los_days", "ICU LOS days, median", "%.1f"),
                    ("sofa", "SOFA, median", "%.1f"),
                    ("n_culture_any", "any culture, median", "%.1f"),
                    ("n_gc_mentions", "GC mentions, median", "%.1f")]:
    if v not in sp0.columns:
        continue
    a, b = d1[v].astype(float), d0[v].astype(float)
    P("  %-34s %14s %14s" % (lab, fmt % a.median(), fmt % b.median()))
    dt_rows.append(dict(variable=lab, died=fmt % a.median(),
                        survived=fmt % b.median(),
                        died_n=float(a.median()), survived_n=float(b.median())))
sp = stats.spearmanr(sp0.note_len, sp0.icu_los_days, nan_policy="omit")
P("  Spearman(note_len, ICU LOS) = %.3f  P=%.3g  <- 若强正相关，则「短记录」≈「短住院」"
  % (sp.statistic, sp.pvalue))
sp2 = stats.spearmanr(sp0.hc_len, sp0.icu_los_days, nan_policy="omit")
P("  Spearman(hc_len,   ICU LOS) = %.3f  P=%.3g" % (sp2.statistic, sp2.pvalue))
dt_extra = dict(sp_note_los=round(float(sp.statistic), 3),
                sp_note_los_p=float(sp.pvalue),
                sp_hc_los=round(float(sp2.statistic), 3),
                sp_hc_los_p=float(sp2.pvalue),
                died_n=len(d1), surv_n=len(d0))
P("")
P("  => 若「整篇长度」预测死亡而「住院经过段长度」不预测，说明前面 2 节的关联")
P("     来自**住院时长/早期死亡**，而不是「医生没写活动度」。")
P("")
P("2c. 把记录长度放进交互模型：v6 的死亡交互还在吗？（spec- 层）")
P("  %-44s %-26s %8s %6s" % ("adjustment", "RA/SLE (RA vs SLE)", "P(LRT)", "ev"))
lenc_rows = []
for lab, cvs in [("CORE (v6 baseline)", CORE),
                 ("CORE + note length (/1000 char)", CORE + ["nl_k"]),
                 ("CORE + HC-section length (/1000)", CORE + ["hl_k"])]:
    r = interaction(sp0, "death_30d", cvs)
    if r is None:
        P("  %-44s %-26s %8s %6s" % (lab, "not estimable", "-", "-"))
        continue
    P("  %-44s %-26s %8s %6d" % (
        lab, "%.2f (%.2f-%.2f)" % (r["OR"], r["lo"], r["hi"]),
        p3(r["p_lrt"]), r["n_ev"]))
    lenc_rows.append(dict(adjustment=lab, or_=round(r["OR"], 3),
                          lo=round(r["lo"], 3), hi=round(r["hi"], 3),
                          p_lrt=r["p_lrt"], n=r["n"], n_ev=r["n_ev"]))
P("")
P("2d. landmark：只保留 ICU LOS >= 3 天者（排除早死造成的短记录）")
P("  %-34s %6s %6s %-26s %8s" % ("restriction", "n", "ev", "RA/SLE", "P(LRT)"))
for lab, sub in [("all spec- (LOS >= 0)", sp0),
                 ("ICU LOS >= 3 d", sp0[sp0.icu_los_days >= 3]),
                 ("ICU LOS >= 3 d  + note len adj",
                  sp0[sp0.icu_los_days >= 3])]:
    cvs = CORE if "adj" not in lab else CORE + ["nl_k"]
    r = interaction(sub, "death_30d", cvs)
    n_sle = int((sub.primary_grp == "SLE").sum())
    n_ra = int((sub.primary_grp == "RA").sum())
    P("  %-34s %6d %6d %-26s %8s" % (
        "%s (SLE %d/RA %d)" % (lab, n_sle, n_ra), len(sub),
        int(sub.death_30d.sum()),
        "%.2f (%.2f-%.2f)" % (r["OR"], r["lo"], r["hi"]) if r else "NE",
        p3(r["p_lrt"]) if r else "-"))
    if r:
        lenc_rows.append(dict(adjustment="landmark: " + lab, or_=round(r["OR"], 3),
                              lo=round(r["lo"], 3), hi=round(r["hi"], 3),
                              p_lrt=r["p_lrt"], n=len(sub), n_ev=int(sub.death_30d.sum())))

# ================================================================= 3 三分内分层
P("")
P("=" * 106)
P("3. Q3/Q4 -- 死亡剂量趋势与交互，在每条通道内分别估")
P("=" * 106)
P("  %-30s %-24s %-24s %-24s" % ("stratum", "SLE trend OR", "RA trend OR",
                                  "interaction RA/SLE"))
strat_rows = []
STRATA = [("v6 spec- (raw, n=%d)" % int((bothn.indN_spec == 0).sum()),
           bothn[bothn.indN_spec == 0]),
          ("v6 spec+ (raw)", bothn[bothn.indN_spec == 1]),
          ("spec- after negation fix (n=%d)" % len(sp0), sp0),
          ("spec+ after negation fix", bothn[bothn.spec_true == 1]),
          ("  . INACTIVE (true inactive)", sp0[sp0.inact3 == "INACTIVE"]),
          ("  . INF_DEFER (infection)", sp0[sp0.inact3 == "INF_DEFER"]),
          ("  . UNDERDOC (undocumented)", sp0[sp0.inact3 == "UNDERDOC"])]
for slab, sub in STRATA:
    t1 = trend(sub[sub.primary_grp == "SLE"], "death_30d", CORE)
    t2 = trend(sub[sub.primary_grp == "RA"], "death_30d", CORE)
    ix = interaction(sub, "death_30d", CORE)
    P("  %-30s %-24s %-24s %-24s" % (
        slab, ci(t1) if t1 else "NE", ci(t2) if t2 else "NE",
        "%.2f (%.2f-%.2f) P=%.3f" % (ix["OR"], ix["lo"], ix["hi"], ix["p_lrt"])
        if ix else "NE"))
    strat_rows.append(dict(
        stratum=slab, n=len(sub), n_ev=int(sub.death_30d.sum()),
        n_sle=int((sub.primary_grp == "SLE").sum()),
        n_ra=int((sub.primary_grp == "RA").sum()),
        sle_or=round(t1["OR"], 3) if t1 else None,
        sle_lo=round(t1["lo"], 3) if t1 else None,
        sle_hi=round(t1["hi"], 3) if t1 else None,
        sle_p=t1["p"] if t1 else None, sle_ev=t1["n_ev"] if t1 else None,
        ra_or=round(t2["OR"], 3) if t2 else None,
        ra_lo=round(t2["lo"], 3) if t2 else None,
        ra_hi=round(t2["hi"], 3) if t2 else None,
        ra_p=t2["p"] if t2 else None, ra_ev=t2["n_ev"] if t2 else None,
        ix_or=round(ix["OR"], 3) if ix else None,
        ix_lo=round(ix["lo"], 3) if ix else None,
        ix_hi=round(ix["hi"], 3) if ix else None,
        ix_p=ix["p_lrt"] if ix else None,
        ix_model=ix["model"] if ix else None, ix_ev=ix["n_ev"] if ix else None))
P("")
P("  阳性对照（葡萄糖 >=180）在同一批分层里，确认分层没有毁掉暴露测量：")
P("  %-30s %-24s %-24s" % ("stratum", "SLE trend OR", "RA trend OR"))
for slab, sub in STRATA:
    t1 = trend(sub[sub.primary_grp == "SLE"], "hyper_48h", CORE)
    t2 = trend(sub[sub.primary_grp == "RA"], "hyper_48h", CORE)
    P("  %-30s %-24s %-24s" % (slab, ci(t1) if t1 else "NE",
                                ci(t2) if t2 else "NE"))
    for gsm, r in [("SLE", t1), ("RA", t2)]:
        if r:
            strat_rows.append(dict(stratum=slab, disease=gsm,
                                   outcome="hyper_48h", n=r["n"], n_ev=r["n_ev"],
                                   or_=round(r["OR"], 3), lo=round(r["lo"], 3),
                                   hi=round(r["hi"], 3), p=r["p"],
                                   model=r["model"]))
P("")
P("  其余结局在三条通道内（仅 UNDERDOC 有足够事件）：")
for y, lab in [("death_hosp", "in-hospital death"),
               ("culture_pos_24_7d", "blood culture + (24h-7d)"),
               ("new_agent_24_7d", "new antimicrobial"),
               ("gi_bleed", "GI bleed (NEG CTRL)")]:
    P("  -- %s --" % lab)
    for g in GROUPS:
        s = sp0[sp0.inact3 == g]
        t1 = trend(s[s.primary_grp == "SLE"], y, CORE)
        t2 = trend(s[s.primary_grp == "RA"], y, CORE)
        P("    %-10s SLE %-24s RA %-24s" % (
            g, ci(t1) if t1 else "NE", ci(t2) if t2 else "NE"))
        for gsm, r in [("SLE", t1), ("RA", t2)]:
            if r:
                strat_rows.append(dict(stratum="%s|%s" % (g, lab), n=r["n"],
                                       n_ev=r["n_ev"], disease=gsm, outcome=y,
                                       or_=round(r["OR"], 3), lo=round(r["lo"], 3),
                                       hi=round(r["hi"], 3), p=r["p"],
                                       model=r["model"]))

# ================================================================= 4 UNDERDOC 细分
P("")
P("=" * 106)
P("4. UNDERDOC 内部再分解 —— 「记录不足」是一个标签还是三个？")
P("=" * 106)
ud = sp0[sp0.inact3 == "UNDERDOC"].copy()
P("  n=%d ; 30d death=%d (%.1f%%)" % (len(ud), int(ud.death_30d.sum()),
                                      100 * ud.death_30d.mean()))
P("  %-46s %6s %10s %10s" % ("undocumented sub-type", "n", "30d death%", "SOFA"))
uds = [("thin note (<8000 chars)", ud[ud.doc_thin == 1]),
       ("regular-length note", ud[ud.doc_thin == 0]),
       ("no activity assay mentioned", ud[ud.no_activ_eval == 1]),
       ("activity assay mentioned (but no activity)", ud[ud.no_activ_eval == 0]),
       ("GC never mentioned in narrative", ud[ud.n_gc_mentions.fillna(0) == 0]),
       ("GC mentioned but no activity", ud[(ud.n_gc_mentions.fillna(0) > 0)]),
       ("non-rheum primary dx", ud[ud.nonrheum_pdx == 1])]
ud_rows = []
for lab, s in uds:
    if not len(s):
        continue
    P("  %-46s %6d %9.1f%% %10.1f" % (lab, len(s), 100 * s.death_30d.mean(),
                                       s.sofa.median()))
    ud_rows.append(dict(subtype=lab, n=len(s),
                        death_30d=round(100 * float(s.death_30d.mean()), 2),
                        n_ev=int(s.death_30d.sum()),
                        sofa=round(float(s.sofa.median()), 2)))
P("")
P("  UNDERDOC 内的交互（RA/SLE）：")
for lab, s in [("all UNDERDOC", ud), ("thin note", ud[ud.doc_thin == 1]),
               ("regular note", ud[ud.doc_thin == 0]),
               ("no assay", ud[ud.no_activ_eval == 1]),
               ("assay mentioned", ud[ud.no_activ_eval == 0])]:
    if not len(s):
        continue
    r = interaction(s, "death_30d", CORE)
    P("    %-22s n=%4d ev=%2d  %s" % (
        lab, len(s), int(s.death_30d.sum()),
        "%.2f (%.2f-%.2f) P=%.3f" % (r["OR"], r["lo"], r["hi"], r["p_lrt"])
        if r else "NE"))
    if r:
        ud_rows.append(dict(subtype="ix: " + lab, n=len(s), n_ev=r["n_ev"],
                            ix_or=round(r["OR"], 3), ix_lo=round(r["lo"], 3),
                            ix_hi=round(r["hi"], 3), ix_p=r["p_lrt"]))
P("")
P("  UNDERDOC 内的交互，逐项调整记录长度（这是本层结论的最后一道防线）：")
P("    %-40s %-26s %8s %6s" % ("adjustment", "RA/SLE", "P(LRT)", "ev"))
for lab, cvs in [("CORE", CORE),
                 ("CORE + note length", CORE + ["nl_k"]),
                 ("CORE + HC-section length", CORE + ["hl_k"]),
                 ("CORE + note len + structural GC user",
                  CORE + ["nl_k", "gc_user_struct"])]:
    if "gc_user_struct" in cvs:
        ud2 = ud.assign(gc_user_struct=ud.gc_user_struct.astype(float))
    else:
        ud2 = ud
    if "nl_k" in cvs or "hl_k" in cvs:
        ud2 = ud2.dropna(subset=[c for c in cvs if c in ("nl_k", "hl_k")])
    r = interaction(ud2, "death_30d", cvs)
    if r is None:
        P("    %-40s %-26s %8s %6s" % (lab, "not estimable", "-", "-"))
        continue
    P("    %-40s %-26s %8s %6d" % (
        lab, "%.2f (%.2f-%.2f)" % (r["OR"], r["lo"], r["hi"]),
        p3(r["p_lrt"]), r["n_ev"]))
    lenc_rows.append(dict(adjustment="UNDERDOC: " + lab, or_=round(r["OR"], 3),
                          lo=round(r["lo"], 3), hi=round(r["hi"], 3),
                          p_lrt=r["p_lrt"], n=r["n"], n_ev=r["n_ev"]))

# ================================================================= 5 参照组
P("")
P("=" * 106)
P("5. 「没拿到 GC 的那一组」按三通道看是谁（G0_none 内，否定校正 spec-）")
P("=" * 106)
g0 = sp0[sp0.gc24_str == "G0_none"]
P("  n=%d ; 30d death=%d (%.1f%%)" % (len(g0), int(g0.death_30d.sum()),
                                      100 * g0.death_30d.mean()))
P("  %-14s %6s %8s %8s %9s %10s %10s" % (
    "channel", "n", "SLE n", "RA n", "SOFA", "30d death%", "homeGC%"))
g0_rows = []
for g in GROUPS:
    s = g0[g0.inact3 == g]
    if not len(s):
        continue
    P("  %-14s %6d %8d %8d %9.1f %9.1f%% %9.1f%%" % (
        g, len(s), int((s.primary_grp == "SLE").sum()),
        int((s.primary_grp == "RA").sum()), s.sofa.median(),
        100 * s.death_30d.mean(), 100 * s.home_gc.mean()))
    g0_rows.append(dict(channel=g, n=len(s),
                        n_sle=int((s.primary_grp == "SLE").sum()),
                        n_ra=int((s.primary_grp == "RA").sum()),
                        sofa=round(float(s.sofa.median()), 2),
                        death_30d=round(100 * float(s.death_30d.mean()), 2),
                        home_gc=round(100 * float(s.home_gc.mean()), 2)))
P("")
P("  三通道 × 病种 的 G0 出院去向（死亡/存活）：")
P("  %-14s %8s %8s %8s %8s" % ("channel", "SLE alive", "SLE died",
                                "RA alive", "RA died"))
g0dx_rows = []
for g in GROUPS:
    s = g0[g0.inact3 == g]
    sa = int(((s.primary_grp == "SLE") & (s.death_30d == 0)).sum())
    sd = int(((s.primary_grp == "SLE") & (s.death_30d == 1)).sum())
    ra_ = int(((s.primary_grp == "RA") & (s.death_30d == 0)).sum())
    rd = int(((s.primary_grp == "RA") & (s.death_30d == 1)).sum())
    P("  %-14s %8d %8d %8d %8d" % (g, sa, sd, ra_, rd))
    g0dx_rows.append(dict(channel=g, sle_alive=sa, sle_died=sd,
                          ra_alive=ra_, ra_died=rd))

# ================================================================= 6 D3 自检
P("")
P("=" * 106)
P("6. D3 SELF-CHECK（否定校正后的分层）")
P("=" * 106)
P("  %-34s %10s %10s %9s %6s" % ("outcome / stratum", "ix RA/SLE", "emp RA/SLE",
                                 "rel.diff", "OK"))
d3rows = []
for y, lab in [("death_30d", "30-day death"), ("death_hosp", "in-hospital death"),
               ("culture_pos_24_7d", "blood culture +"),
               ("new_agent_24_7d", "new antimicrobial"), ("hyper_48h", "glucose"),
               ("gi_bleed", "GI bleed")]:
    for slab, sub in [("spec- (fixed)", sp0), ("all with note", bothn)]:
        r = interaction(sub, y, CORE)
        t1 = trend(sub[sub.primary_grp == "SLE"], y, CORE)
        t2 = trend(sub[sub.primary_grp == "RA"], y, CORE)
        if not (r and t1 and t2):
            continue
        emp = t2["OR"] / t1["OR"]
        agree = np.sign(r["OR"] - 1) == np.sign(emp - 1)
        rel = 100 * abs(r["OR"] - emp) / emp
        P("  %-34s %10.2f %10.2f %8.1f%% %6s" % (
            "%s / %s" % (lab, slab), r["OR"], emp, rel,
            "OK" if agree else "CHECK"))
        d3rows.append(dict(outcome=lab, stratum=slab, var=y,
                           ix=round(r["OR"], 3), emp=round(emp, 3),
                           rel_diff=round(rel, 1), agree=bool(agree),
                           ix_lo=round(r["lo"], 3), ix_hi=round(r["hi"], 3),
                           p_lrt=r["p_lrt"], n_ev=r["n_ev"]))

# ================================================================= 7 IPTW 复核
P("")
P("=" * 106)
P("7. IPTW 复核 —— 三通道分解在加权模型下是否同向")
P("=" * 106)
PSCOV = CORE + ["resp_fail", "coagulop", "liver_fail", "immuno_any", "hcq",
                "cytopenia", "serositis", "n_proc", "n_hosp"]
PSCOV = [c for c in PSCOV if c in bothn.columns]
iw = sp0.copy()
for c in PSCOV:
    iw[c] = iw[c].astype(float)
d_w, ess = iptw_weights(iw.dropna(subset=PSCOV + ["gc24_str"]), PSCOV)
P("  ESS = %.1f (of n=%d)" % (ess, len(d_w)))
iw_rows = []
P("  %-30s %-26s %8s" % ("stratum × disease", "IPTW trend OR", "P"))
for slab, fn in [("spec- all", lambda d: d),
                 ("  INACTIVE", lambda d: d[d.inact3 == "INACTIVE"]),
                 ("  INF_DEFER", lambda d: d[d.inact3 == "INF_DEFER"]),
                 ("  UNDERDOC", lambda d: d[d.inact3 == "UNDERDOC"])]:
    for g in ["SLE", "RA"]:
        s = fn(d_w)
        s = s[s.primary_grp == g]
        if not len(s) or int(s.death_30d.sum()) < 5:
            P("  %-30s %-26s %8s" % ("%s | %s" % (slab, g), "too few events",
                                     "-"))
            continue
        r = trend(s, "death_30d", ["sw"], use_firth=False)
        P("  %-30s %-26s %8s" % ("%s | %s" % (slab, g), ci(r), p3(r["p"])))
        iw_rows.append(dict(stratum=slab, disease=g, or_=round(r["OR"], 3),
                            lo=round(r["lo"], 3), hi=round(r["hi"], 3),
                            p=r["p"], n=r["n"], n_ev=r["n_ev"]))
P("")
P("  阳性对照（葡萄糖）同表：")
nci_rows = []
for slab, fn in [("spec- all", lambda d: d), ("  UNDERDOC", lambda d: d[d.inact3 == "UNDERDOC"])]:
    for g in ["SLE", "RA"]:
        s = fn(d_w)
        s = s[s.primary_grp == g]
        r = trend(s, "hyper_48h", ["sw"], use_firth=False)
        P("  %-30s %-26s %8s" % ("%s | %s" % (slab, g), ci(r) if r else "NE",
                                 p3(r["p"]) if r else "-"))
        if r:
            nci_rows.append(dict(stratum=slab, disease=g, or_=round(r["OR"], 3),
                                 lo=round(r["lo"], 3), hi=round(r["hi"], 3),
                                 p=r["p"], n=r["n"], n_ev=r["n_ev"]))

# ============================================================ 8 通道 × 结构变量
P("")
P("=" * 106)
P("8. 通道 × 结构变量（供报告交叉表使用）")
P("=" * 106)
allnote = {}
for g in GROUPS:
    allnote[g] = int((bothn.inact3 == g).sum())
P("  有出院记录的全队列 n=%d 组成：%s" % (
    len(bothn), "  ".join("%s=%d(%.1f%%)" % (g, allnote[g],
                                             100.0 * allnote[g] / len(bothn))
                          for g in GROUPS)))

CHAN_DOSE = []
P("  通道 × 剂量层：")
P("  %-12s %8s %8s %8s %8s" % ("channel", "G0", "G1_low", "G2_mod", "G3_high"))
for g in GROUPS:
    s = sp0[sp0.inact3 == g]
    row = dict(channel=g, n=len(s))
    for ds in ["G0_none", "G1_low", "G2_mod", "G3_high"]:
        row[ds] = int((s.gc24_str == ds).sum())
    CHAN_DOSE.append(row)
    P("  %-12s %8d %8d %8d %8d" % (g, row["G0_none"], row["G1_low"],
                                    row["G2_mod"], row["G3_high"]))

PDX = []
P("  通道 × 主诊断类别：")
for g in GROUPS:
    s = sp0[sp0.inact3 == g]
    row = dict(channel=g, n=len(s))
    for c in sorted(s.pdx_class.dropna().unique()):
        row[c] = int((s.pdx_class == c).sum())
    row["nonrheum"] = int(s.nonrheum_pdx.fillna(0).sum())
    row["thin"] = int(s.doc_thin.fillna(0).sum())
    row["no_eval"] = int(s.no_activ_eval.fillna(0).sum())
    PDX.append(row)
    P("  %-12s n=%4d  non-rheum pdx %5.1f%%  thin note %5.1f%%  no assay eval %5.1f%%"
      % (g, len(s), 100.0 * row["nonrheum"] / max(len(s), 1),
         100.0 * row["thin"] / max(len(s), 1),
         100.0 * row["no_eval"] / max(len(s), 1)))

n_rawpos = int(((bothn.spec_true == 1) | (bothn.spec_falsepos == 1)).sum())
P("  否定校正：raw+ %d -> true+ %d（被否定 %d）"
  % (n_rawpos, int((bothn.spec_true == 1).sum()),
     int(bothn.spec_falsepos.sum())))

# ================================================================= export
def _num(d):
    if not d:
        return None
    return {k: (round(float(v), 4) if isinstance(v, (int, float))
                and not isinstance(v, bool) else v) for k, v in d.items()}


pd.DataFrame(prof_rows).to_csv(os.path.join(OUT, "table_v7_profile.csv"), index=False)
pd.DataFrame(nlq_rows).to_csv(os.path.join(OUT, "table_v7_notelen.csv"), index=False)
pd.DataFrame(lenc_rows).to_csv(os.path.join(OUT, "table_v7_lenadj.csv"), index=False)
pd.DataFrame(strat_rows).to_csv(os.path.join(OUT, "table_v7_strata.csv"), index=False)
pd.DataFrame(ud_rows).to_csv(os.path.join(OUT, "table_v7_underdoc.csv"), index=False)
pd.DataFrame(g0_rows).to_csv(os.path.join(OUT, "table_v7_g0.csv"), index=False)
pd.DataFrame(d3rows).to_csv(os.path.join(OUT, "table_v7_d3.csv"), index=False)
pd.DataFrame(iw_rows).to_csv(os.path.join(OUT, "table_v7_iptw.csv"), index=False)
with open(os.path.join(OUT, "_v7_results.json"), "w", encoding="utf-8") as f:
    json.dump(dict(profile=prof_rows, notelen=nlq_rows, strata=strat_rows,
                   underdoc=ud_rows, g0=g0_rows, g0_dx=g0dx_rows,
                   chan_dose=CHAN_DOSE, pdx=PDX, allnote=allnote,
                   iptw=iw_rows, iptw_negctrl=nci_rows, d3=d3rows,
                   len_adj=lenc_rows, deathtab=dt_rows, deathtab_extra=dt_extra,
                   notelen_cont=_num(r_len), activ_cont=_num(r_act),
                   n_spec0=len(sp0), n_spec1=int((bothn.spec_true == 1).sum()),
                   n_h2h=len(both), n_note=len(bothn),
                   n_sle_note=int((bothn.primary_grp == "SLE").sum()),
                   n_ra_note=int((bothn.primary_grp == "RA").sum()),
                   n_death_h2h=int(both.death_30d.sum()),
                   n_death_note=int(bothn.death_30d.sum()),
                   n_sle_spec0=int((sp0.primary_grp == "SLE").sum()),
                   n_ra_spec0=int((sp0.primary_grp == "RA").sum()),
                   n_death_spec0=int(sp0.death_30d.sum()),
                   n_rawpos=n_rawpos, n_neg_falsepos=int(bothn.spec_falsepos.sum()),
                   neg_falsepos=int(bothn.spec_falsepos.sum())),
              f, ensure_ascii=False, indent=1)
with open(os.path.join(OUT, "164_inactivity_mechanism.txt"), "w",
          encoding="utf-8") as f:
    f.write("\n".join(lines))
P("")
P("saved: out/164_inactivity_mechanism.txt + table_v7_*.csv + _v7_results.json")
