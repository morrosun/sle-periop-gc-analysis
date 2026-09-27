# -*- coding: utf-8 -*-
"""
SLE ICU cohort extraction, v3
=============================
Changes vs v2
  * drop the ICU LOS >= 48 h restriction (it halved the cohort: 550 -> 279)
  * exposure window = min(48 h, ICU length of stay)
  * primary exposure metric = prednisone-equivalent DOSE INTENSITY (mg/day),
    so short and long stays are comparable; cumulative dose kept for sensitivity
  * keep a `los48` flag for the landmark sensitivity subset
"""
import psycopg2
import pandas as pd
import numpy as np
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root (data/ and out/ live here)

CONFIG = dict(
    host=os.environ.get("MIMIC_DB_HOST", "localhost"),
    port=int(os.environ.get("MIMIC_DB_PORT", "5432")),
    user=os.environ.get("MIMIC_DB_USER", "postgres"),
    password=os.environ.get("MIMIC_DB_PASSWORD", ""),
)
OUT = os.path.join(ROOT, "data")

SQL = r"""
WITH sle AS (
    SELECT DISTINCT d.hadm_id
    FROM mimiciv_hosp.diagnoses_icd d
    WHERE (d.icd_version = 10 AND d.icd_code LIKE 'M32%')
       OR (d.icd_version = 9  AND d.icd_code LIKE '7100%')
),
icu AS (
    SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime, i.los, i.first_careunit,
           ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.intime) AS rn
    FROM mimiciv_icu.icustays i JOIN sle ON i.hadm_id = sle.hadm_id
),
coh AS (SELECT * FROM icu WHERE rn = 1),
aged AS (
    SELECT c.*, a.admittime, a.dischtime, a.deathtime, a.hospital_expire_flag,
           a.admission_type, a.admission_location, a.insurance, a.race,
           p.gender, ag.age,
           LEAST(48.0, EXTRACT(EPOCH FROM (c.outtime - c.intime)) / 3600.0) AS win_hours
    FROM coh c
    JOIN mimiciv_hosp.admissions a ON c.hadm_id = a.hadm_id
    JOIN mimiciv_hosp.patients  p ON c.subject_id = p.subject_id
    LEFT JOIN mimiciv_derived.age ag ON c.hadm_id = ag.hadm_id
),
lm AS (SELECT * FROM aged WHERE age >= 18),
sofa_agg AS (
    SELECT s.stay_id,
           MAX(s.sofa_24hours) FILTER (WHERE s.hr BETWEEN 1 AND 48) AS sofa48_max,
           MAX(s.sofa_24hours) FILTER (WHERE s.hr BETWEEN 20 AND 28) AS sofa24
    FROM mimiciv_derived.sofa s JOIN lm ON s.stay_id = lm.stay_id GROUP BY s.stay_id
),
gc_rx AS (
    SELECT p.hadm_id, p.drug, p.route,
           CAST(p.dose_val_rx AS DOUBLE PRECISION) AS dose_mg,
           COALESCE(p.doses_per_24_hrs, 1.0)       AS freq24,
           p.starttime, p.stoptime,
           CASE
             WHEN p.drug ILIKE '%%methylpred%%'   THEN 1.25
             WHEN p.drug ILIKE '%%prednisolone%%' THEN 1.0
             WHEN p.drug ILIKE '%%predni%%'       THEN 1.0
             WHEN p.drug ILIKE '%%hydrocort%%'    THEN 0.25
             WHEN p.drug ILIKE '%%dexameth%%'     THEN 6.67
             WHEN p.drug ILIKE '%%betameth%%'     THEN 6.67
             WHEN p.drug ILIKE '%%triamcin%%'     THEN 1.25
             ELSE NULL
           END AS pe_factor
    FROM mimiciv_hosp.prescriptions p JOIN lm ON p.hadm_id = lm.hadm_id
    WHERE p.dose_unit_rx = 'mg'
      AND p.dose_val_rx ~ '^[0-9]+(\.[0-9]+)?$'
      AND p.route IN ('IV','PO','PO/NG','IM')
      AND p.starttime IS NOT NULL AND p.stoptime IS NOT NULL
      AND (p.drug ILIKE '%%predni%%' OR p.drug ILIKE '%%methylpred%%' OR p.drug ILIKE '%%dexameth%%'
           OR p.drug ILIKE '%%hydrocort%%' OR p.drug ILIKE '%%betameth%%' OR p.drug ILIKE '%%triamcin%%')
),
gc_win AS (
    SELECT g.hadm_id, g.pe_factor, g.dose_mg, g.freq24,
           GREATEST(0.0, LEAST(
               EXTRACT(EPOCH FROM (LEAST(lm.intime + (lm.win_hours * INTERVAL '1 hour'), g.stoptime)
                                   - GREATEST(lm.intime, g.starttime))) / 86400.0)) AS overlap_days,
           EXTRACT(EPOCH FROM (g.starttime - lm.intime)) / 3600.0                    AS start_hour
    FROM gc_rx g JOIN lm ON g.hadm_id = lm.hadm_id
    WHERE g.pe_factor IS NOT NULL
),
gc_agg AS (
    SELECT hadm_id,
           SUM(CASE WHEN overlap_days > 0 THEN dose_mg * freq24 * pe_factor * overlap_days ELSE 0 END) AS gc_cum_pe_mg,
           MAX(CASE WHEN overlap_days > 0 THEN dose_mg * freq24 * pe_factor END)                        AS gc_peak_daily_pe_mg,
           MIN(CASE WHEN overlap_days > 0 THEN start_hour END)                                          AS gc_first_hour,
           SUM(CASE WHEN overlap_days > 0 THEN 1 ELSE 0 END)                                            AS gc_n_orders
    FROM gc_win GROUP BY hadm_id
),
gc_pre AS (SELECT DISTINCT g.hadm_id FROM gc_rx g JOIN lm ON g.hadm_id = lm.hadm_id WHERE g.starttime < lm.intime),
inf AS (
    SELECT DISTINCT d.hadm_id FROM mimiciv_hosp.diagnoses_icd d JOIN lm ON d.hadm_id = lm.hadm_id
    WHERE (d.icd_version = 10 AND (
             LEFT(d.icd_code,2) = 'J1'
          OR LEFT(d.icd_code,3) IN ('A04','A40','A41','A49','J85','K61','K65','L02','L03','L08',
                                    'M86','N10','N39','R65','T79','T81','T82','T84')))
       OR (d.icd_version = 9  AND (
             LEFT(d.icd_code,3) IN ('038','320','322','324','420','421','481','482','483','484',
                                    '485','486','507','510','513','540','567','590','599','680',
                                    '681','682','683','684','685','686','711','730','998')
          OR d.icd_code LIKE '9959%'))
),
ln AS (
    SELECT DISTINCT hadm_id FROM mimiciv_hosp.diagnoses_icd
    WHERE (icd_version = 10 AND (icd_code LIKE 'M321%' OR icd_code LIKE 'N085%'))
       OR (icd_version = 9  AND icd_code LIKE '58381%')
),
imm AS (
    SELECT hadm_id,
           MAX(CASE WHEN drug ILIKE '%%cyclophosphamide%%' OR drug ILIKE '%%cytoxan%%' THEN 1 ELSE 0 END) AS imm_cyc,
           MAX(CASE WHEN drug ILIKE '%%mycophenolate%%' OR drug ILIKE '%%cellcept%%' THEN 1 ELSE 0 END) AS imm_mmf,
           MAX(CASE WHEN drug ILIKE '%%azathioprine%%' OR drug ILIKE '%%imuran%%' THEN 1 ELSE 0 END) AS imm_aza,
           MAX(CASE WHEN drug ILIKE '%%tacrolimus%%' OR drug ILIKE '%%cyclospor%%' THEN 1 ELSE 0 END) AS imm_cni,
           MAX(CASE WHEN drug ILIKE '%%methotrexate%%' OR drug ILIKE '%%trexall%%' THEN 1 ELSE 0 END) AS imm_mtx,
           MAX(CASE WHEN drug ILIKE '%%hydroxychloroquine%%' OR drug ILIKE '%%plaquenil%%' THEN 1 ELSE 0 END) AS imm_hcq,
           MAX(CASE WHEN drug ILIKE '%%rituximab%%' OR drug ILIKE '%%belimumab%%' THEN 1 ELSE 0 END) AS imm_bio
    FROM mimiciv_hosp.prescriptions WHERE hadm_id IN (SELECT hadm_id FROM lm) GROUP BY hadm_id
),
vent AS (
    SELECT v.stay_id, 1 AS on_vent FROM mimiciv_derived.ventilation v JOIN lm ON v.stay_id = lm.stay_id
    WHERE v.starttime < lm.intime + INTERVAL '48 hours' AND (v.endtime IS NULL OR v.endtime > lm.intime)
    GROUP BY v.stay_id
),
vaso AS (
    SELECT va.stay_id, 1 AS on_vaso FROM mimiciv_derived.vasoactive_agent va JOIN lm ON va.stay_id = lm.stay_id
    WHERE va.starttime < lm.intime + INTERVAL '48 hours' AND (va.endtime IS NULL OR va.endtime > lm.intime)
    GROUP BY va.stay_id
),
rrt AS (
    SELECT r.stay_id, 1 AS on_rrt FROM mimiciv_derived.first_day_rrt r JOIN lm ON r.stay_id = lm.stay_id
    WHERE r.dialysis_present = 1 OR r.dialysis_active = 1 GROUP BY r.stay_id
)
SELECT
    lm.subject_id, lm.hadm_id, lm.stay_id,
    lm.age, lm.gender, lm.race, lm.insurance, lm.admission_type, lm.admission_location,
    lm.intime, lm.outtime, lm.los AS icu_los, lm.first_careunit, lm.win_hours,
    CASE WHEN lm.los >= 2 THEN 1 ELSE 0 END AS los48,
    lm.admittime, lm.dischtime, lm.deathtime,
    EXTRACT(EPOCH FROM (lm.intime - lm.admittime)) / 3600.0 AS preicu_hours,
    CASE WHEN lm.deathtime IS NOT NULL AND lm.deathtime <= lm.intime + INTERVAL '30 days' THEN 1 ELSE 0 END AS death_30d,
    CASE WHEN lm.deathtime IS NOT NULL AND lm.deathtime <= lm.intime + INTERVAL '90 days' THEN 1 ELSE 0 END AS death_90d,
    lm.hospital_expire_flag AS death_inhosp,
    CASE WHEN i.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS infection,
    COALESCE(g.gc_cum_pe_mg, 0) AS gc_cum_pe_mg,
    g.gc_peak_daily_pe_mg, g.gc_first_hour,
    COALESCE(g.gc_n_orders, 0) AS gc_n_orders,
    CASE WHEN g.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS gc_any_hosp,
    CASE WHEN gp.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS gc_preicu,
    ch.charlson_comorbidity_index AS charlson,
    sa.sofa48_max AS sofa, sa.sofa24,
    ap.apsiii, oa.oasis, sp.sapsii,
    fl.albumin_min, fl.abs_lymphocytes_min, fl.wbc_min, fl.wbc_max,
    fl.platelets_min, fl.creatinine_max, fl.bilirubin_total_max,
    COALESCE(v.on_vent,0) AS on_vent, COALESCE(va.on_vaso,0) AS on_vaso, COALESCE(r.on_rrt,0) AS on_rrt,
    CASE WHEN lnf.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS lupus_nephritis,
    COALESCE(im.imm_cyc,0) AS imm_cyc, COALESCE(im.imm_mmf,0) AS imm_mmf,
    COALESCE(im.imm_aza,0) AS imm_aza, COALESCE(im.imm_cni,0) AS imm_cni,
    COALESCE(im.imm_mtx,0) AS imm_mtx, COALESCE(im.imm_hcq,0) AS imm_hcq,
    COALESCE(im.imm_bio,0) AS imm_bio
FROM lm
LEFT JOIN gc_agg g  ON lm.hadm_id = g.hadm_id
LEFT JOIN gc_pre gp ON lm.hadm_id = gp.hadm_id
LEFT JOIN inf    i  ON lm.hadm_id = i.hadm_id
LEFT JOIN mimiciv_derived.charlson      ch ON lm.hadm_id = ch.hadm_id
LEFT JOIN sofa_agg                      sa ON lm.stay_id = sa.stay_id
LEFT JOIN mimiciv_derived.apsiii        ap ON lm.stay_id = ap.stay_id
LEFT JOIN mimiciv_derived.oasis         oa ON lm.stay_id = oa.stay_id
LEFT JOIN mimiciv_derived.sapsii        sp ON lm.stay_id = sp.stay_id
LEFT JOIN mimiciv_derived.first_day_lab fl ON lm.stay_id = fl.stay_id
LEFT JOIN vent v  ON lm.stay_id = v.stay_id
LEFT JOIN vaso va ON lm.stay_id = va.stay_id
LEFT JOIN rrt  r  ON lm.stay_id = r.stay_id
LEFT JOIN ln   lnf ON lm.hadm_id = lnf.hadm_id
LEFT JOIN imm  im  ON lm.hadm_id = im.hadm_id
ORDER BY lm.subject_id, lm.intime
"""

def main():
    c = psycopg2.connect(dbname="mimiciv", **CONFIG)
    df = pd.read_sql_query(SQL, c)
    c.close()

    # dose intensity (mg/day prednisone-equivalent)
    df["win_days"] = np.maximum(df["win_hours"] / 24.0, 1.0 / 24.0)
    df["gc_daily_pe_mg"] = df["gc_cum_pe_mg"] / df["win_days"]
    df["gc_any"] = (df["gc_cum_pe_mg"] > 0).astype(int)

    print("rows:", len(df), "stay:", df.stay_id.nunique(), "hadm:", df.hadm_id.nunique(),
          "subject:", df.subject_id.nunique())
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "cohort_sle_icu_v3.csv"), index=False)
    print("saved cohort_sle_icu_v3.csv")

    print("\n--- outcome / exposure by cohort definition ---")
    print("  all                n=%3d  infection=%3d (%.1f%%)  d30=%3d (%.1f%%)" % (
        len(df), df.infection.sum(), 100 * df.infection.mean(), df.death_30d.sum(), 100 * df.death_30d.mean()))
    sub = df[df.los48 == 1]
    print("  ICU LOS >= 48 h    n=%3d  infection=%3d (%.1f%%)  d30=%3d (%.1f%%)" % (
        len(sub), sub.infection.sum(), 100 * sub.infection.mean(), sub.death_30d.sum(), 100 * sub.death_30d.mean()))

    print("\n--- GC exposure ---")
    print("  any GC order (hosp) : %d (%.1f%%)" % (df.gc_any_hosp.sum(), 100 * df.gc_any_hosp.mean()))
    print("  GC before ICU       : %d (%.1f%%)" % (df.gc_preicu.sum(), 100 * df.gc_preicu.mean()))
    print("  GC inside window >0 : %d (%.1f%%)" % (df.gc_any.sum(), 100 * df.gc_any.mean()))
    pos = df.loc[df.gc_daily_pe_mg > 0, "gc_daily_pe_mg"]
    print("  dose INTENSITY (mg/day prednisone-equiv) among exposed (n=%d):" % len(pos))
    for q in [0, .1, .25, .33, .5, .66, .75, .9, .95, 1.0]:
        print("     q%-5s %10.1f" % (q, pos.quantile(q)))

    print("\n--- candidate strata on dose intensity (mg/day) ---")
    cuts = [(0, 0, "0 (none)"), (0, 30, ">0-30"), (30, 60, "30-60"), (60, 1e18, ">60")]
    for lo, hi, lab in cuts:
        if lab == "0 (none)":
            m = df.gc_daily_pe_mg <= 0
        elif hi > 1e17:
            m = df.gc_daily_pe_mg > lo
        else:
            m = (df.gc_daily_pe_mg > lo) & (df.gc_daily_pe_mg <= hi)
        n = int(m.sum())
        if n == 0:
            print("  %-10s n=0" % lab); continue
        print("  %-10s n=%4d (%5.1f%%)  infection=%4d (%5.1f%%)  d30=%3d (%5.1f%%)  sofa=%s" % (
            lab, n, 100 * n / len(df), int(df.loc[m, 'infection'].sum()), 100 * df.loc[m, 'infection'].mean(),
            int(df.loc[m, 'death_30d'].sum()), 100 * df.loc[m, 'death_30d'].mean(),
            round(df.loc[m, 'sofa'].mean(), 1)))

    print("\n--- tertiles of dose intensity among exposed (data-driven) ---")
    try:
        t = pd.qcut(pos, 3, labels=["T1 low", "T2 mid", "T3 high"], duplicates="drop")
        tmp = df.loc[pos.index].assign(strata=t)
        for lab, grp in tmp.groupby("strata", observed=True):
            print("  %-8s n=%4d  med dose=%7.1f  infection=%4d (%5.1f%%)  d30=%3d (%5.1f%%)" % (
                lab, len(grp), grp.gc_daily_pe_mg.median(), int(grp.infection.sum()),
                100 * grp.infection.mean(), int(grp.death_30d.sum()), 100 * grp.death_30d.mean()))
    except Exception as e:
        print("  tertile failed:", e)

    print("\n--- missingness (%) ---")
    miss = (df.isna().mean() * 100).round(1).sort_values(ascending=False)
    print(miss[miss > 0].to_string())

if __name__ == "__main__":
    main()
