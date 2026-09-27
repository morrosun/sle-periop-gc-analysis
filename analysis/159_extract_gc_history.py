# -*- coding: utf-8 -*-
"""
159_extract_gc_history.py
=========================
从数据库侧构造 GC 用药史，用于两个目的：
  (1) **外部验证** 文本抽取得到的 `home_gc`（Medications on Admission）
  (2) 构造结构性的「慢性 GC 使用者」标记，区分
        · 本次住院全程都未用过 GC        -> gc_any_hosp = 0（真正的 GC-free 住院）
        · 入 ICU 前本次住院已用过 GC      -> gc_pre_icu = 1（窗口前暴露，含家庭剂量延续）
        · 既往住院用过 GC                 -> gc_prior_hadm = 1（疑似慢性使用者）

口径与本项目既有暴露定义完全一致（prescriptions / dose_unit_rx='mg' /
route IN IV,PO,PO/NG,IM / 泼尼松等效）。本脚本**不重算** gc24，只做加宽窗口的历史。

输出 data/gc_history.csv
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import psycopg2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(*a)
    sys.stdout.flush()


# 与 151 完全一致的药名 / 等效系数
DRUG_SQL = ("(pr.drug ILIKE '%%predni%%' OR pr.drug ILIKE '%%methylpred%%'"
            " OR pr.drug ILIKE '%%dexameth%%' OR pr.drug ILIKE '%%hydrocort%%'"
            " OR pr.drug ILIKE '%%betameth%%' OR pr.drug ILIKE '%%triamcin%%')")

PE = [(r"methylpred|medrol|solumedrol", 1.25), (r"prednisolone", 1.0),
      (r"prednisone|predni", 1.0), (r"hydrocort|cortef", 0.25),
      (r"dexameth", 6.67), (r"betameth", 6.67), (r"triamcin", 1.25)]


def pe_factor(name):
    n = (name or "").lower()
    for pat, f in PE:
        if re.search(pat, n):
            return f
    return None


def main():
    coh = pd.read_csv(os.path.join(DATA, "cohort_rheum_icu.csv"))
    coh["stay_key"] = coh.stay_key.astype(int)
    hset = set(coh.stay_key.tolist())
    log("cohort hadm:", len(hset))

    cc = psycopg2.connect(host=os.environ.get("MIMIC_DB_HOST", "localhost"),
                          port=int(os.environ.get("MIMIC_DB_PORT", "5432")),
                          user=os.environ.get("MIMIC_DB_USER", "postgres"),
                          password=os.environ.get("MIMIC_DB_PASSWORD", ""),
                          dbname="mimiciv")

    # 入 ICU 时间表（写进临时表，供 SQL 比较）
    cu = cc.cursor()
    cu.execute("DROP TABLE IF EXISTS tmp_icu")
    cu.execute("CREATE TEMP TABLE tmp_icu (hadm_id INT PRIMARY KEY, intime TIMESTAMP)")
    cu.executemany("INSERT INTO tmp_icu VALUES (%s,%s)",
                   [(int(r.stay_key), r.intime) for r in coh.itertuples()])
    cc.commit()
    log("tmp_icu built:", len(coh))

    # ---- 1) 本次住院内：入 ICU 前是否已用 GC
    q1 = ("SELECT pr.hadm_id AS stay_key, MIN(pr.starttime) AS t_first "
          "FROM mimiciv_hosp.prescriptions pr JOIN tmp_icu ti ON pr.hadm_id = ti.hadm_id "
          "WHERE pr.dose_unit_rx = 'mg' AND pr.dose_val_rx ~ '^[0-9]+(\.[0-9]+)?$' "
          "  AND pr.route IN ('IV','PO','PO/NG','IM') AND " + DRUG_SQL +
          "  AND pr.starttime IS NOT NULL AND pr.starttime < ti.intime "
          "GROUP BY pr.hadm_id")
    pre = pd.read_sql(q1, cc)
    log("gc_pre_icu:", len(pre))

    # ---- 2) 本次住院全程是否有 GC（不限窗口）
    q2 = ("SELECT pr.hadm_id AS stay_key, COUNT(*) AS n_gc_orders "
          "FROM mimiciv_hosp.prescriptions pr JOIN tmp_icu ti ON pr.hadm_id = ti.hadm_id "
          "WHERE pr.dose_unit_rx = 'mg' AND pr.dose_val_rx ~ '^[0-9]+(\.[0-9]+)?$' "
          "  AND pr.route IN ('IV','PO','PO/NG','IM') AND " + DRUG_SQL +
          "GROUP BY pr.hadm_id")
    anyh = pd.read_sql(q2, cc)
    log("gc_any_hosp:", len(anyh))

    # ---- 3) 既往住院是否用过 GC（admittime < 本次 ICU intime）
    q3 = ("SELECT DISTINCT pr.hadm_id AS idx_hadm, pr.subject_id, "
          "       MIN(pr.starttime) AS t_first, MAX(pr.starttime) AS t_last "
          "FROM mimiciv_hosp.prescriptions pr "
          "WHERE pr.dose_unit_rx = 'mg' AND pr.dose_val_rx ~ '^[0-9]+(\.[0-9]+)?$' "
          "  AND pr.route IN ('IV','PO','PO/NG','IM') AND " + DRUG_SQL +
          "GROUP BY pr.hadm_id, pr.subject_id")
    gcall = pd.read_sql(q3, cc)
    log("hadm with any GC (all time):", len(gcall))

    q4 = ("SELECT hadm_id, subject_id, admittime, dischtime "
          "FROM mimiciv_hosp.admissions")
    adm = pd.read_sql(q4, cc)
    cc.close()

    # ---- 既往 GC：按 subject 逐条判断 admittime < 本次 intime
    cur = coh[["stay_key", "subject_id", "intime"]].copy()
    cur["intime"] = pd.to_datetime(cur.intime)
    gcall["t_last"] = pd.to_datetime(gcall.t_last)
    adm = adm.drop_duplicates("hadm_id")
    ga = gcall.merge(adm[["hadm_id", "admittime"]], left_on="idx_hadm",
                     right_on="hadm_id", how="left")
    ga["admittime"] = pd.to_datetime(ga.admittime)

    prior_rows = []
    for r in cur.itertuples():
        sub = ga[(ga.subject_id == r.subject_id) & (ga.idx_hadm != r.stay_key)]
        past = sub[sub.admittime < r.intime]
        prior_rows.append(dict(
            stay_key=r.stay_key,
            gc_prior_hadm=int(len(past) > 0),
            n_prior_gc_hadm=int(len(past)),
            last_prior_gc=pd.to_datetime(past.t_last).max() if len(past) else pd.NaT))
    prior = pd.DataFrame(prior_rows)
    prior["days_since_last_gc"] = (
        (pd.to_datetime(cur.set_index("stay_key").intime).reindex(prior.stay_key).values
         - prior.last_prior_gc.values) / np.timedelta64(1, "D"))
    prior = prior.drop(columns=["last_prior_gc"])

    # 既往住院次数
    nprior = []
    for r in cur.itertuples():
        nprior.append(dict(
            stay_key=r.stay_key,
            n_prior_adm=int(((adm.subject_id == r.subject_id) &
                             (adm.admittime < r.intime)).sum())))
    nprior = pd.DataFrame(nprior)

    res = (coh[["stay_key"]]
           .merge(pre.rename(columns={"t_first": "t_gc_pre_icu"}), on="stay_key", how="left")
           .merge(anyh, on="stay_key", how="left")
           .merge(prior, on="stay_key", how="left")
           .merge(nprior, on="stay_key", how="left"))
    res["gc_pre_icu"] = res.t_gc_pre_icu.notna().astype(int)
    res["n_gc_orders"] = res.n_gc_orders.fillna(0).astype(int)
    res["gc_any_hosp"] = (res.n_gc_orders > 0).astype(int)
    res["gc_prior_hadm"] = res.gc_prior_hadm.fillna(0).astype(int)
    res["n_prior_gc_hadm"] = res.n_prior_gc_hadm.fillna(0).astype(int)
    res["n_prior_adm"] = res.n_prior_adm.fillna(0).astype(int)
    res = res.drop(columns=["t_gc_pre_icu"])

    # ---- 合并文本侧
    ind = pd.read_csv(os.path.join(DATA, "gc_indication.csv"))
    ind["stay_key"] = ind.stay_key.astype(int)
    m = res.merge(ind, on="stay_key", how="left")
    m.to_csv(os.path.join(DATA, "gc_history.csv"), index=False)
    log("saved data/gc_history.csv", m.shape)

    # ---------------------------------------------------------------- 验证
    coh2 = coh.merge(m, on="stay_key", how="left")
    coh2 = coh2[coh2.note_found == 1]
    L = []
    P = lambda *a: L.append(" ".join(str(x) for x in a))
    P("=" * 78)
    P("159 GC 用药史 + 文本 `home_gc` 的外部验证")
    P("=" * 78)
    P("n = %d (有出院记录)" % len(coh2))
    P("")
    P("--- 数据库侧 GC 用药史 ---")
    P("  本次住院全程用过 GC (gc_any_hosp) : %6d (%.1f%%)" % (
        coh2.gc_any_hosp.sum(), 100 * coh2.gc_any_hosp.mean()))
    P("  入 ICU 前本次住院已用 GC          : %6d (%.1f%%)" % (
        coh2.gc_pre_icu.sum(), 100 * coh2.gc_pre_icu.mean()))
    P("  既往住院用过 GC                   : %6d (%.1f%%)" % (
        coh2.gc_prior_hadm.sum(), 100 * coh2.gc_prior_hadm.mean()))
    P("  既往住院次数 中位 %.0f" % coh2.n_prior_adm.median())
    P("")
    P("=== 验证 1: 文本 home_gc  ×  数据库 gc_pre_icu（窗口前暴露）===")
    ct = pd.crosstab(coh2.home_gc, coh2.gc_pre_icu, margins=True)
    P(ct.to_string())
    a = ((coh2.home_gc == 1) & (coh2.gc_pre_icu == 1)).sum()
    b = ((coh2.home_gc == 0) & (coh2.gc_pre_icu == 0)).sum()
    P("  一致率 %.1f%%   敏感性 %.1f%%   特异性 %.1f%%" % (
        100 * (a + b) / len(coh2),
        100 * a / max(1, (coh2.gc_pre_icu == 1).sum()),
        100 * b / max(1, (coh2.gc_pre_icu == 0).sum())))
    P("")
    P("=== 验证 2: 文本 home_gc  ×  数据库 gc_prior_hadm（既往住院用过）===")
    ct2 = pd.crosstab(coh2.home_gc, coh2.gc_prior_hadm, margins=True)
    P(ct2.to_string())
    P("")
    P("=== 核心表: 剂量层 × GC 用药史（SLE / RA）===")
    for g in ["SLE", "RA"]:
        s = coh2[coh2.primary_grp == g]
        P("  %s n = %d" % (g, len(s)))
        P("    %-10s %6s %10s %10s %10s %10s" % (
            "24h剂量层", "n", "院前GC%", "全程GC%", "窗口前GC%", "既往GC%"))
        for st in ["G0_none", "G1_low", "G2_mod", "G3_high"]:
            sub = s[s.gc24_str == st]
            if not len(sub):
                continue
            P("    %-10s %6d %9.1f%% %9.1f%% %9.1f%% %9.1f%%" % (
                st, len(sub), 100 * sub.home_gc.mean(),
                100 * sub.gc_any_hosp.mean(),
                100 * sub.gc_pre_icu.mean(),
                100 * sub.gc_prior_hadm.mean()))
    P("")
    P("=== 关键问题: G0 组（24h 无 GC）里到底有多少是「真 GC-free」===")
    g0 = coh2[coh2.gc24_str == "G0_none"]
    P("  全 G0 n = %d" % len(g0))
    P("    home_gc=0 且 gc_any_hosp=0           : %5d (%.1f%%)  <- 真正 GC-free 住院" % (
        ((g0.home_gc == 0) & (g0.gc_any_hosp == 0)).sum(),
        100 * ((g0.home_gc == 0) & (g0.gc_any_hosp == 0)).mean()))
    P("    home_gc=1 或 gc_any_hosp=1           : %5d (%.1f%%)  <- 住院期内后来还是用了" % (
        ((g0.home_gc == 1) | (g0.gc_any_hosp == 1)).sum(),
        100 * ((g0.home_gc == 1) | (g0.gc_any_hosp == 1)).mean()))
    P("    既往住院用过 GC                      : %5d (%.1f%%)" % (
        g0.gc_prior_hadm.sum(), 100 * g0.gc_prior_hadm.mean()))
    P("")
    for g in ["SLE", "RA"]:
        s = g0[g0.primary_grp == g]
        P("  %s 的 G0 n = %d ; 真 GC-free %.1f%% ; 既往GC %.1f%% ; "
          "home_gc %.1f%%" % (
              g, len(s),
              100 * ((s.home_gc == 0) & (s.gc_any_hosp == 0)).mean(),
              100 * s.gc_prior_hadm.mean(), 100 * s.home_gc.mean()))
    P("")
    P("=== 慢性使用者定义候选（home_gc 或 既往GC 或 窗口前GC）===")
    coh2 = coh2.copy()
    coh2["gc_user_struct"] = ((coh2.home_gc == 1) | (coh2.gc_prior_hadm == 1) |
                              (coh2.gc_pre_icu == 1)).astype(int)
    P("  gc_user_struct 阳性 %d (%.1f%%)" % (
        coh2.gc_user_struct.sum(), 100 * coh2.gc_user_struct.mean()))
    P("  %-10s %10s %10s" % ("剂量层", "SLE gc_user%", "RA gc_user%"))
    for st in ["G0_none", "G1_low", "G2_mod", "G3_high"]:
        a = coh2[(coh2.gc24_str == st) & (coh2.primary_grp == "SLE")]
        b = coh2[(coh2.gc24_str == st) & (coh2.primary_grp == "RA")]
        P("  %-10s %9.1f%% %10.1f%%" % (
            st, 100 * a.gc_user_struct.mean(), 100 * b.gc_user_struct.mean()))

    txt = "\n".join(L)
    with open(os.path.join(OUT, "159_gc_history_qc.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
