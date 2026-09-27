# -*- coding: utf-8 -*-
"""
156_head_to_head_v5_robustness.py

v5 稳健性：把「疾病×剂量交互」按 v5 的新主结局重打一遍，并补上 v4 没做的两件事：

  竞争风险 / 时间依赖
  -------------------
  主结局是 24 h-7 d 的计时事件，但 40.8% 的人在窗口结束前就出院了，他们被记为
  0 —— 这是 informative censoring。做法：
    (a) 完整观察窗子集（在院 >= 7 d 或 7 d 内未离院，n=1805）
    (b) Cox 比例风险模型，以"出院/死亡/7 d"三者最早者为删失
    (c) 严格 incident 子集（landmark 时抗菌药-naive）

  其余攻击（与 v4 一致，但换到新主结局）
  --------------------------------------
  1 年龄共同窗 / 让剂量效应随年龄变化（s:age, s:age+s:sofa）
  2 剔除外科入院
  3 暴露定义替换（序数 / any GC / >=50 vs none）
  4 bootstrap 交互系数的抽样分布
  5 脆弱性：30 d 死亡交互要翻转多少个事件才会失去 P<0.05
"""
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
SCORE = {"G0_none": 0.0, "G1_low": 1.0, "G2_mod": 2.0, "G3_high": 3.0}

CORE = ["age", "female", "sofa", "vaso24", "vent24", "renal_fail", "shock", "surg_any",
        "n_agent_pre24"]

PRIMARY = "new_agent_24_7d"
KSEC = "death_30d"
# (变量, 短标签) —— 用于所有稳健性表；PRIMARY 必须排第一
OUTS = [(PRIMARY, "new antimicrobial (PRIMARY)"),
        ("culture_pos_24_7d", "blood culture + (spec.)"),
        (KSEC, "30-day death"),
        ("incident_strict", "strict incident"),
        ("hyper_48h", "glucose >=180 (POS CTRL)"),
        ("gi_bleed", "GI bleeding (NEG CTRL)")]

lines = []


def P(s=""):
    print(s)
    lines.append(str(s))


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


def interaction(d, y, covs, extra_int=None, force_firth=False):
    """RA/SLE 剂量趋势 OR 比（每层增量）。extra_int 里的变量会额外与 s 交互。"""
    need = ["gc_str_num", "is_sle", y] + covs + (extra_int or [])
    need = list(dict.fromkeys(need))
    use = d[need].dropna().copy()
    if len(use) == 0 or use[y].nunique() < 2 or use.gc_str_num.nunique() < 2:
        return None
    use["s"] = use.gc_str_num.astype(float)
    use["sx"] = use.s * use.is_sle.astype(float)
    terms = ["s", "is_sle", "sx"] + covs
    for v in (extra_int or []):
        use["s_" + v] = use.s * use[v].astype(float)
        terms.append("s_" + v)
    X = sm.add_constant(use[terms].astype(float))
    Xr = sm.add_constant(use[["s", "is_sle"] + covs].astype(float))
    use_firth = force_firth or int(use[y].sum()) < 60
    if not use_firth:
        try:
            with np.errstate(all="ignore"):
                m = sm.Logit(use[y].astype(float), X).fit(disp=0, maxiter=300)
                m0 = sm.Logit(use[y].astype(float), Xr).fit(disp=0, maxiter=300)
            b, se = np.asarray(m.params, float), np.asarray(m.bse, float)
            lrt = 2.0 * (m.llf - m0.llf)
            model = "MLE"
        except Exception:
            use_firth = True
    if use_firth:
        b, se, llf = firth(X.values, use[y].values)
        _, _, llr = firth(Xr.values, use[y].values)
        lrt = 2.0 * (llf - llr)
        model = "Firth"
    j = list(X.columns).index("sx")
    z = b[j] / se[j]
    p_wald = float(2 * (1 - stats.norm.cdf(abs(z))))
    p_lrt = float(1 - stats.chi2.cdf(max(lrt, 0.0), 1)) if np.isfinite(lrt) else p_wald
    # 注意方向：exp(b[sx]) 是 SLE/RA，倒过来才是 RA/SLE（与 interaction() 一致）
    return dict(OR=float(np.exp(-b[j])), lo=float(np.exp(-b[j] - 1.96 * se[j])),
                hi=float(np.exp(-b[j] + 1.96 * se[j])),
                p=p_wald, p_lrt=p_lrt,
                n=int(len(use)), n_ev=int(use[y].sum()), model=model)


def trend(d, y, covs):
    use = d[["gc_str_num", y] + covs].dropna()
    if len(use) == 0 or use[y].nunique() < 2 or use.gc_str_num.nunique() < 2:
        return None
    X = sm.add_constant(use[["gc_str_num"] + covs].astype(float))
    b, se, ll = firth(X.values, use[y].values)
    j = list(X.columns).index("gc_str_num")
    z = b[j] / se[j]
    return dict(OR=float(np.exp(b[j])), lo=float(np.exp(b[j] - 1.96 * se[j])),
                hi=float(np.exp(b[j] + 1.96 * se[j])),
                p=float(2 * (1 - stats.norm.cdf(abs(z)))))


def ci(r, key="p"):
    if r is None:
        return "not estimable"
    return "%.2f (%.2f-%.2f) P=%.3f" % (r["OR"], r["lo"], r["hi"], r.get(key, r.get("p")))


def brief(r):
    if r is None:
        return "N/E"
    return "%.2f(%.2f-%.2f) P=%.3f" % (r["OR"], r["lo"], r["hi"], r.get("p", r.get("p_lrt")))


# ------------------------------------------------------------------ load
df = pd.read_csv(os.path.join(DATA, "cohort_rheum_icu.csv")).rename(columns={"sofa24": "sofa"})
ab = pd.read_csv(os.path.join(DATA, "abx_outcomes_rheum.csv"))
df = df.merge(ab[["stay_key", "legacy_any48", "n_agent_pre24", "abx_naive_at24",
                  "t_new_agent_h", "new_agent_24_7d", "incident_strict",
                  "new_agent_anytime", "culture_pos_24_7d", "h_censor",
                  "window_complete", "h_disch", "h_death"]],
              on="stay_key", how="left")
df["sofa"] = df.sofa.fillna(df.sofa.median())
df["gc_str_num"] = df.gc24_str.map(SCORE)
df["s"] = df.gc_str_num.astype(float)
for c in ["new_agent_24_7d", "incident_strict", "culture_pos_24_7d", "death_30d",
          "death_hosp", "hyper_48h", "gi_bleed", "n_agent_pre24", "window_complete"]:
    df[c] = df[c].astype(float)
both = df[df.primary_grp.isin(["SLE", "RA"])].copy()
both["is_ra"] = (both.primary_grp == "RA").astype(int)

P("=" * 108)
P("v5 ROBUSTNESS  --  does the disease x dose interaction survive?")
P("=" * 108)
P("  SLE n=%d (age %.1f)   RA n=%d (age %.1f)   gap %.1f y" % (
    (both.primary_grp == "SLE").sum(), both[both.primary_grp == "SLE"].age.median(),
    (both.primary_grp == "RA").sum(), both[both.primary_grp == "RA"].age.median(),
    both[both.primary_grp == "RA"].age.median() - both[both.primary_grp == "SLE"].age.median()))
P("  reference (full cohort, adjusted):")
for y, lab in OUTS[:4]:
    r = interaction(both, y, CORE)
    P("    %-28s %s" % (lab, ci(r, "p_lrt")))

# =====================================================================
P("")
P("=" * 108)
P("0. COMPETING RISK / INFORMATIVE CENSORING on the primary outcome")
P("=" * 108)
P("  40.8% of the cohort leaves hospital before day 7, so the 7-day window is")
P("  not fully observed for them and they are coded as 'no event'.")
P("  (a) restrict to a fully observed window;  (b) Cox with censoring at")
P("  discharge / death / day 7;  (c) strict-incident subgroup.")
P("")
cr_rows = []
comp = both[both.window_complete == 1].copy()
P("  (a) fully observed window only  n=%d (SLE %d / RA %d)" % (
    len(comp), (comp.primary_grp == "SLE").sum(), (comp.primary_grp == "RA").sum()))
for y, lab in OUTS:
    r = interaction(comp, y, CORE)
    P("      %-28s %s" % (lab, ci(r, "p_lrt")))
    cr_rows.append(dict(analysis="complete window", outcome=lab, var=y,
                        or_=(r or {}).get("OR"), lo=(r or {}).get("lo"),
                        hi=(r or {}).get("hi"), p=(r or {}).get("p"),
                        p_lrt=(r or {}).get("p_lrt"), n=(r or {}).get("n"),
                        n_ev=(r or {}).get("n_ev")))
P("")
P("  (c) strict-incident subgroup: landmark 时抗菌药-naive（新增用药才有意义）")
naive = both[both.abx_naive_at24 == 1].copy()
# n_agent_pre24 在该子集内恒为 0（naive 的定义），常数协变量会让设计矩阵奇异
CORE_NB = [c for c in CORE if c != "n_agent_pre24"]
P("      n=%d (SLE %d / RA %d), events=%d  [n_agent_pre24 已从协变量集中移除]" % (
    len(naive), (naive.primary_grp == "SLE").sum(), (naive.primary_grp == "RA").sum(),
    int(naive.new_agent_24_7d.sum())))
for y, lab in OUTS:
    r = interaction(naive, y, CORE_NB)
    P("      %-28s %s" % (lab, ci(r, "p_lrt")))
    cr_rows.append(dict(analysis="antimicrobial-naive", outcome=lab, var=y,
                        or_=(r or {}).get("OR"), lo=(r or {}).get("lo"),
                        hi=(r or {}).get("hi"), p=(r or {}).get("p"),
                        p_lrt=(r or {}).get("p_lrt"), n=(r or {}).get("n"),
                        n_ev=(r or {}).get("n_ev")))

P("")
P("  (b) Cox PH, time from landmark, censored at discharge / death / day 7")
P("      (an event at or after day 7 or after discharge is not counted)")
cox_rows = []
tmax = (both.h_censor.fillna(1e9).clip(upper=168) - 24).clip(lower=0.5)
ev = ((both.t_new_agent_h.notna()) & (both.t_new_agent_h < both.h_censor.fillna(1e9)) &
      (both.t_new_agent_h < 168)).astype(int)
tt = np.where(ev == 1, (both.t_new_agent_h - 24).clip(lower=0.5), tmax)
P("      events used = %d ; median follow-up %.1f h" % (int(ev.sum()), np.median(tt)))
mod = both[["s", "is_ra", "age", "female", "sofa", "vaso24", "vent24", "renal_fail",
            "shock", "surg_any", "n_agent_pre24"]].astype(float).copy()
mod["sx"] = mod.s * mod.is_ra
ok = np.isfinite(mod.values).all(1)          # ndarray, 不要再 .values
try:
    ph = sm.duration.PHReg(tt[ok], mod[ok], status=ev[ok].values, ties="breslow")
    res = ph.fit()
    names = list(mod.columns)
    j = names.index("sx")
    b, se = float(res.params[j]), float(res.bse[j])
    P("      disease x dose HR ratio (RA/SLE) = %.2f (%.2f-%.2f)  P=%.3f" % (
        np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se),
        2 * (1 - stats.norm.cdf(abs(b / se)))))
    j2 = names.index("s")
    b2, se2 = float(res.params[j2]), float(res.bse[j2])
    P("      dose trend (common across diseases) HR = %.2f (%.2f-%.2f) P=%.3f" % (
        np.exp(b2), np.exp(b2 - 1.96 * se2), np.exp(b2 + 1.96 * se2),
        2 * (1 - stats.norm.cdf(abs(b2 / se2)))))
    cox_rows.append(dict(term="disease x dose (RA/SLE)", hr=float(np.exp(b)),
                         lo=float(np.exp(b - 1.96 * se)), hi=float(np.exp(b + 1.96 * se)),
                         p=float(2 * (1 - stats.norm.cdf(abs(b / se))))))
    cox_rows.append(dict(term="dose trend", hr=float(np.exp(b2)),
                         lo=float(np.exp(b2 - 1.96 * se2)), hi=float(np.exp(b2 + 1.96 * se2)),
                         p=float(2 * (1 - stats.norm.cdf(abs(b2 / se2))))))
except Exception as e:
    P("      Cox failed: %s" % str(e)[:120])

# =====================================================================
P("")
P("=" * 108)
P("1a. AGE OVERLAP: restrict BOTH diseases to the same age window")
P("=" * 108)
WINDOWS = [("all ages", 0, 200), ("50-80 y", 50, 80), ("55-75 y", 55, 75),
           ("60-70 y", 60, 70)]
age_rows = []
P("  %-10s %7s %7s %10s %10s" % ("window", "SLE n", "RA n", "SLE ev*", "RA ev*"))
for lab, lo, hi in WINDOWS:
    w = both[(both.age >= lo) & (both.age <= hi)]
    a = w[w.primary_grp == "SLE"]
    b = w[w.primary_grp == "RA"]
    P("  %-10s %7d %7d %10d %10d" % (lab, len(a), len(b),
                                      int(a[PRIMARY].sum()), int(b[PRIMARY].sum())))
P("  * events are the PRIMARY outcome (new antimicrobial 24 h-7 d)")
P("")
P("  interaction under each age window (RA vs SLE trend ratio), BOTH outcomes:")
P("  %-10s %-28s %24s %8s   %7s %7s" % (
    "window", "outcome", "RA/SLE trend ratio", "P LRT", "SLE n", "RA n"))
for lab, lo, hi in WINDOWS:
    w = both[(both.age >= lo) & (both.age <= hi)]
    for y in [PRIMARY, KSEC, "culture_pos_24_7d", "hyper_48h", "gi_bleed"]:
        r = interaction(w, y, CORE)
        nm = dict(OUTS).get(y, y)
        P("  %-10s %-28s %24s %8s   %7d %7d" % (
            lab, nm, ci(r, "p_lrt"),
            ("%.3f" % r["p_lrt"]) if r else "",
            int((w.primary_grp == "SLE").sum()), int((w.primary_grp == "RA").sum())))
        age_rows.append(dict(window=lab, outcome=nm, var=y,
                             or_=(r or {}).get("OR"), lo=(r or {}).get("lo"),
                             hi=(r or {}).get("hi"), p=(r or {}).get("p"),
                             p_lrt=(r or {}).get("p_lrt"),
                             sle_n=int((w.primary_grp == "SLE").sum()),
                             ra_n=int((w.primary_grp == "RA").sum()),
                             n=(r or {}).get("n"), n_ev=(r or {}).get("n_ev")))
    P("")

P("=" * 108)
P("1b. ABSORB AGE: let the dose effect itself vary with age")
P("=" * 108)
P("  %-28s %24s %24s %24s" % ("outcome", "base adj", "+ s:age", "+ s:age + s:sofa"))
P("  (P = Wald; 加了额外交互项后 LRT 不再只针对交互项，因此此处统一用 Wald)")
abs_rows = []
for y, lab in OUTS:
    r0 = interaction(both, y, CORE)
    r1 = interaction(both, y, CORE, extra_int=["age"])
    r2 = interaction(both, y, CORE, extra_int=["age", "sofa"])
    P("  %-28s %24s %24s %24s" % (lab, ci(r0, "p"), ci(r1, "p"), ci(r2, "p")))
    abs_rows.append(dict(outcome=lab, var=y,
                         base=(r0 or {}).get("OR"), base_lo=(r0 or {}).get("lo"),
                         base_hi=(r0 or {}).get("hi"), base_p=(r0 or {}).get("p"),
                         s_age=(r1 or {}).get("OR"), s_age_lo=(r1 or {}).get("lo"),
                         s_age_hi=(r1 or {}).get("hi"), s_age_p=(r1 or {}).get("p"),
                         s_age_sofa=(r2 or {}).get("OR"), s_age_sofa_lo=(r2 or {}).get("lo"),
                         s_age_sofa_hi=(r2 or {}).get("hi"),
                         s_age_sofa_p=(r2 or {}).get("p")))

P("")
P("=" * 108)
P("2. INDICATION: drop surgical admissions")
P("=" * 108)
P("  %-28s %24s %24s %10s" % ("outcome", "all", "non-surgical only", "surg share"))
surg_rows = []
for y, lab in OUTS:
    r0 = interaction(both, y, CORE)
    ns = both[both.surg_any == 0]
    r1 = interaction(ns, y, CORE)
    P("  %-28s %24s %24s %9.1f%%" % (lab, ci(r0, "p_lrt"), ci(r1, "p_lrt"),
                                      100 * both.surg_any.mean()))
    surg_rows.append(dict(outcome=lab, var=y,
                          all_=(r0 or {}).get("OR"), all_lo=(r0 or {}).get("lo"),
                          all_hi=(r0 or {}).get("hi"), all_p=(r0 or {}).get("p_lrt"),
                          nonsurg=(r1 or {}).get("OR"), nonsurg_lo=(r1 or {}).get("lo"),
                          nonsurg_hi=(r1 or {}).get("hi"), nonsurg_p=(r1 or {}).get("p_lrt"),
                          nonsurg_n=(r1 or {}).get("n"), nonsurg_n_ev=(r1 or {}).get("n_ev")))

P("")
P("=" * 108)
P("3. EXPOSURE DEFINITION: is the ordinal dose score doing the work?")
P("=" * 108)
P("  %-28s %24s %24s %24s" % ("outcome", "ordinal score", "any GC >=24 h", ">=50 vs none"))
expo_rows = []


def interaction_binary(d, y, covs, expo_col):
    need = [expo_col, "is_sle", y] + covs
    use = d[need].dropna().copy()
    if len(use) == 0 or use[y].nunique() < 2:
        return None
    use[expo_col + "_sle"] = use[expo_col].astype(float) * use.is_sle.astype(float)
    terms = [expo_col, "is_sle", expo_col + "_sle"] + covs
    X = sm.add_constant(use[terms].astype(float))
    use_firth = int(use[y].sum()) < 60
    if not use_firth:
        try:
            with np.errstate(all="ignore"):
                m = sm.Logit(use[y].astype(float), X).fit(disp=0, maxiter=300)
            b, se = np.asarray(m.params, float), np.asarray(m.bse, float)
        except Exception:
            use_firth = True
    if use_firth:
        b, se, _ = firth(X.values, use[y].values)
    j = list(X.columns).index(expo_col + "_sle")
    z = b[j] / se[j]
    # 同样倒向：exp(b[sx]) = SLE/RA，报告 RA/SLE
    return dict(OR=float(np.exp(-b[j])), lo=float(np.exp(-b[j] - 1.96 * se[j])),
                hi=float(np.exp(-b[j] + 1.96 * se[j])),
                p=float(2 * (1 - stats.norm.cdf(abs(z)))),
                n=int(len(use)), n_ev=int(use[y].sum()))


for y, lab in OUTS:
    r0 = interaction(both, y, CORE)
    r1 = interaction_binary(both, y, CORE, "gc_any24")
    hi = both[both.gc24_str.isin(["G0_none", "G3_high"])].copy()
    hi["e_high"] = (hi.gc24_str == "G3_high").astype(float)
    r2 = interaction_binary(hi, y, CORE, "e_high")
    P("  %-28s %24s %24s %24s" % (lab, ci(r0, "p_lrt"), ci(r1), ci(r2)))
    expo_rows.append(dict(outcome=lab, var=y,
                          ordinal=(r0 or {}).get("OR"), ordinal_lo=(r0 or {}).get("lo"),
                          ordinal_hi=(r0 or {}).get("hi"), ordinal_p=(r0 or {}).get("p_lrt"),
                          any_gc=(r1 or {}).get("OR"), any_gc_lo=(r1 or {}).get("lo"),
                          any_gc_hi=(r1 or {}).get("hi"), any_gc_p=(r1 or {}).get("p"),
                          high=(r2 or {}).get("OR"), high_lo=(r2 or {}).get("lo"),
                          high_hi=(r2 or {}).get("hi"), high_p=(r2 or {}).get("p")))
P("  (for any-GC and >=50-vs-none the entries are the dose-effect ratio RA/SLE,")
P("   which is the same contrast as the ordinal interaction, on a binary exposure.)")

P("")
P("=" * 108)
P("4. BOOTSTRAP the interaction (2000 resamples, stratified by disease)")
P("=" * 108)
rng = np.random.RandomState(20260916)
boot_rows = []
for y, lab in [(PRIMARY, "new antimicrobial (PRIMARY)"), (KSEC, "30-day death"),
               ("culture_pos_24_7d", "blood culture + (spec.)"),
               ("hyper_48h", "glucose >=180 (POS CTRL)"),
               ("gi_bleed", "GI bleeding (NEG CTRL)")]:
    r0 = interaction(both, y, CORE)
    if r0 is None:
        continue
    n_ok = 0
    logs = []
    for _ in range(2000):
        parts = [both[both.primary_grp == g].sample(len(both[both.primary_grp == g]),
                                                    replace=True, random_state=rng.randint(1 << 30))
                 for g in ["SLE", "RA"]]
        bs = pd.concat(parts, ignore_index=True)
        rb = interaction(bs, y, CORE, force_firth=False)
        if rb is None:
            continue
        if np.isfinite(rb["OR"]) and rb["OR"] > 0:
            logs.append(np.log(rb["OR"]))
            n_ok += 1
    if n_ok < 100:
        P("  %-28s bootstrap unusable (%d replicates)" % (lab, n_ok))
        continue
    logs = np.array(logs)
    share_gt1 = float((logs > 0).mean())
    P("  %-28s point %.2f  boot median %.2f  95%% CI %.2f-%.2f  P(ratio>1)=%.3f  "
      "n=%d" % (lab, r0["OR"], np.exp(np.median(logs)),
                np.exp(np.percentile(logs, 2.5)), np.exp(np.percentile(logs, 97.5)),
                share_gt1, n_ok))
    boot_rows.append(dict(outcome=lab, var=y, point=r0["OR"],
                          boot_med=float(np.exp(np.median(logs))),
                          boot_lo=float(np.exp(np.percentile(logs, 2.5))),
                          boot_hi=float(np.exp(np.percentile(logs, 97.5))),
                          share_gt1=share_gt1, share_lt1=1 - share_gt1, n_ok=n_ok))

P("")
P("=" * 108)
P("5. FRAGILITY of the 30-day-death interaction")
P("=" * 108)
P("  How many events must be reversed (event -> non-event) before P crosses 0.05?")
frag_rows = []


def frag(d, y, n_max=60):
    r0 = interaction(d, y, CORE)
    if r0 is None or r0["p_lrt"] > 0.05:
        return None
    use = d[["gc_str_num", "is_sle", y] + CORE].dropna().copy()
    ev_idx = use.index[use[y] == 1].tolist()
    k = 0
    for _ in range(n_max):
        # 每次把"对交互贡献最大"的一个事件翻成非事件
        use2 = use.copy()
        best, bp = None, r0["p_lrt"]
        for i in ev_idx[:200]:
            use2.loc[i, y] = 0
            rb = interaction(use2, y, CORE)
            use2.loc[i, y] = 1
            if rb is not None and rb["p_lrt"] > bp:
                best, bp = i, rb["p_lrt"]
        if best is None:
            break
        use.loc[best, y] = 0
        ev_idx.remove(best)
        k += 1
        rb = interaction(use, y, CORE)
        if rb is None or rb["p_lrt"] > 0.05:
            return dict(flips=k, n_ev=int(use[y].sum()), p_after=float(rb["p_lrt"]) if rb else np.nan)
    return dict(flips=n_max, n_ev=int(use[y].sum()), p_after=np.nan)


for y, lab in [(KSEC, "30-day death"), ("culture_pos_24_7d", "blood culture + (spec.)")]:
    f = frag(both, y, n_max=40)
    if f is None:
        P("  %-28s interaction not significant to begin with" % lab)
    else:
        P("  %-28s %d flips out of %d events -> P=%.3f" % (
            lab, f["flips"], f["n_ev"], f["p_after"]))
        frag_rows.append(dict(outcome=lab, var=y, flips=f["flips"], n_ev=f["n_ev"],
                              p_after=f["p_after"]))

# ------------------------------------------------------------------ save
pd.DataFrame(cr_rows).to_csv(os.path.join(OUT, "table_v5_robust_competing.csv"), index=False)
pd.DataFrame(age_rows).to_csv(os.path.join(OUT, "table_v5_robust_age.csv"), index=False)
pd.DataFrame(abs_rows).to_csv(os.path.join(OUT, "table_v5_robust_absorb.csv"), index=False)
pd.DataFrame(surg_rows).to_csv(os.path.join(OUT, "table_v5_robust_surg.csv"), index=False)
pd.DataFrame(expo_rows).to_csv(os.path.join(OUT, "table_v5_robust_expo.csv"), index=False)
pd.DataFrame(boot_rows).to_csv(os.path.join(OUT, "table_v5_robust_boot.csv"), index=False)
pd.DataFrame(cox_rows).to_csv(os.path.join(OUT, "table_v5_robust_cox.csv"), index=False)
pd.DataFrame(frag_rows).to_csv(os.path.join(OUT, "table_v5_robust_fragility.csv"), index=False)
with open(os.path.join(OUT, "156_head2head_v5_robust.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nsaved -> out/156_head2head_v5_robust.txt")
