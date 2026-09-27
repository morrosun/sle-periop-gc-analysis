# -*- coding: utf-8 -*-
"""
SLE ICU cohort extraction (Plan A)
==================================
Target trial emulation components
  eligibility : >=18 y, ICD-10 M32% / ICD-9 7100% (SLE), first ICU stay, ICU LOS >= 48 h
  time zero   : ICU intime
  exposure    : prednisone-equivalent dose accumulated in the 48 h landmark window
                4 levels: none / low / moderate / high  (cut-points set from observed quantiles)
  outcome     : in-hospital infection (ICD code families), 30-day all-cause death
  follow-up   : ICU intime -> discharge, death or 30 d
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

# ----------------------------------------------------------------- SQL
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
    FROM mimiciv_icu.icustays i
    JOIN sle ON i.hadm_id = sle.hadm_id
),
coh AS (
    SELECT * FROM icu WHERE rn = 1
),
aged AS (
    SELECT c.*, a.admittime, a.dischtime, a.deathtime, a.hospital_expire_flag,
           a.admission_type, a.admission_location, a.insurance, a.race,
           p.gender, ag.age
    FROM coh c
    JOIN mimiciv_hosp.admissions a ON c.hadm_id = a.hadm_id
    JOIN mimiciv_hosp.patients  p ON c.subject_id = p.subject_id
    LEFT JOIN mimiciv_derived.age ag ON c.hadm_id = ag.hadm_id
),
-- 48h landmark cohort: ICU stay must cover the window
lm AS (
    SELECT * FROM aged
    WHERE age >= 18
      AND (outtime - intime) >= INTERVAL '48 hours'
),
-- ---------------- glucocorticoid prescriptions (systemic routes only) ----------------
gc_rx AS (
    SELECT p.hadm_id,
           p.drug,
           p.route,
           CAST(p.dose_val_rx AS DOUBLE PRECISION)                     AS dose_mg,
           COALESCE(p.doses_per_24_hrs, 1.0)                           AS freq24,
           p.starttime,
           p.stoptime,
           CASE
             WHEN p.drug ILIKE '%%methylpred%%' THEN 1.25
             WHEN p.drug ILIKE '%%predni%%'     THEN 1.0
             WHEN p.drug ILIKE '%%prednisolone%%' THEN 1.0
             WHEN p.drug ILIKE '%%hydrocort%%'  THEN 0.25
             WHEN p.drug ILIKE '%%dexameth%%'   THEN 6.67
             WHEN p.drug ILIKE '%%betameth%%'   THEN 6.67
             WHEN p.drug ILIKE '%%triamcin%%'   THEN 1.25
             ELSE NULL
           END                                                          AS pe_factor
    FROM mimiciv_hosp.prescriptions p
    JOIN lm ON p.hadm_id = lm.hadm_id
    WHERE p.dose_unit_rx = 'mg'
      AND p.dose_val_rx ~ '^[0-9]+(\.[0-9]+)?$'
      AND p.route IN ('IV','PO','PO/NG','IM')
      AND (p.drug ILIKE '%%predni%%' OR p.drug ILIKE '%%methylpred%%' OR p.drug ILIKE '%%dexameth%%'
           OR p.drug ILIKE '%%hydrocort%%' OR p.drug ILIKE '%%betameth%%' OR p.drug ILIKE '%%triamcin%%')
),
gc_win AS (
    SELECT g.hadm_id, g.drug, g.route, g.pe_factor, g.dose_mg, g.freq24,
           GREATEST(0.0, LEAST(EXTRACT(EPOCH FROM (LEAST(lm.intime + INTERVAL '48 hours',
                                                          COALESCE(g.stoptime, lm.intime + INTERVAL '48 hours')))
                                       - GREATEST(lm.intime, COALESCE(g.starttime, lm.intime)))) / 86400.0
                   ) AS overlap_days,
           (EXTRACT(EPOCH FROM (GREATEST(lm.intime, COALESCE(g.starttime, lm.intime)) - lm.intime)) / 3600.0)
                   AS start_hour
    FROM gc_rx g JOIN lm ON g.hadm_id = lm.hadm_id
    WHERE g.pe_factor IS NOT NULL
),
gc_agg AS (
    SELECT hadm_id,
           SUM(dose_mg * freq24 * pe_factor * overlap_days) AS gc48_pe_mg,
           MAX(dose_mg * freq24 * pe_factor)                AS gc_max_daily_pe_mg,
           MIN(start_hour)                                  AS gc_first_hour,
           COUNT(*)                                         AS gc_n_orders
    FROM gc_win
    GROUP BY hadm_id
),
-- ---------------- infection (ICD families, ICD-9 / ICD-10) ----------------
inf AS (
    SELECT DISTINCT d.hadm_id
    FROM mimiciv_hosp.diagnoses_icd d
    JOIN lm ON d.hadm_id = lm.hadm_id
    WHERE (d.icd_version = 10 AND (
             LEFT(d.icd_code,3) IN ('A04','A40','A41','A49','J85','K61','K65','L02','L03','L08','M86',
                                    'N10','N39','R65','T79','T81','T82','T84')
          OR LEFT(d.icd_code,2) = 'J1'))
       OR (d.icd_version = 9  AND (
             LEFT(d.icd_code,3) IN ('038','320','322','324','420','421','481','482','485','486','494',
                                    '507','510','513','540','567','590','599','680','681','682','683',
                                    '684','685','686','711','730','790','995','996','997','998')))
),
-- ---------------- SLE-specific proxies ----------------
ln AS (  -- lupus nephritis
    SELECT DISTINCT hadm_id FROM mimiciv_hosp.diagnoses_icd
    WHERE (icd_version = 10 AND (icd_code LIKE 'M321%' OR icd_code LIKE 'N085%'))
       OR (icd_version = 9  AND icd_code LIKE '58381%')
),
imm AS (  -- immunosuppressants / antimalarials (whole admission)
    SELECT hadm_id,
           MAX(CASE WHEN drug ILIKE '%%cyclophosphamide%%' OR drug ILIKE '%%cytoxan%%' THEN 1 ELSE 0 END) AS imm_cyc,
           MAX(CASE WHEN drug ILIKE '%%mycophenolate%%'    OR drug ILIKE '%%cellcept%%'  THEN 1 ELSE 0 END) AS imm_mmf,
           MAX(CASE WHEN drug ILIKE '%%azathioprine%%'     OR drug ILIKE '%%imuran%%'    THEN 1 ELSE 0 END) AS imm_aza,
           MAX(CASE WHEN drug ILIKE '%%tacrolimus%%'       OR drug ILIKE '%%cyclospor%%' OR drug ILIKE '%%cyclosporine%%' THEN 1 ELSE 0 END) AS imm_cni,
           MAX(CASE WHEN drug ILIKE '%%methotrexate%%'     OR drug ILIKE '%%trexall%%'   THEN 1 ELSE 0 END) AS imm_mtx,
           MAX(CASE WHEN drug ILIKE '%%hydroxychloroquine%%' OR drug ILIKE '%%plaquenil%%' THEN 1 ELSE 0 END) AS imm_hcq,
           MAX(CASE WHEN drug ILIKE '%%rituximab%%'        OR drug ILIKE '%%belimumab%%' THEN 1 ELSE 0 END) AS imm_bio
    FROM mimiciv_hosp.prescriptions
    WHERE hadm_id IN (SELECT hadm_id FROM lm)
    GROUP BY hadm_id
),
-- ---------------- organ support within 48 h ----------------
vent AS (
    SELECT v.stay_id, MAX(1) AS on_vent
    FROM mimiciv_derived.ventilation v JOIN lm ON v.stay_id = lm.stay_id
    WHERE v.starttime < lm.intime + INTERVAL '48 hours'
      AND (v.endtime IS NULL OR v.endtime > lm.intime)
    GROUP BY v.stay_id
),
vaso AS (
    SELECT va.stay_id, MAX(1) AS on_vaso
    FROM mimiciv_derived.vasoactive_agent va JOIN lm ON va.stay_id = lm.stay_id
    WHERE va.starttime < lm.intime + INTERVAL '48 hours'
      AND (va.endtime IS NULL OR va.endtime > lm.intime)
    GROUP BY va.stay_id
),
rrt AS (
    SELECT r.stay_id, MAX(1) AS on_rrt
    FROM mimiciv_derived.first_day_rrt r JOIN lm ON r.stay_id = lm.stay_id
    WHERE r.dialysis_present = 1 OR r.dialysis_active = 1
    GROUP BY r.stay_id
)
SELECT
    lm.subject_id, lm.hadm_id, lm.stay_id,
    lm.age, lm.gender, lm.race, lm.insurance,
    lm.admission_type, lm.admission_location,
    lm.intime, lm.outtime, lm.los AS icu_los, lm.first_careunit,
    lm.admittime, lm.dischtime, lm.deathtime, lm.hospital_expire_flag,
    EXTRACT(EPOCH FROM (lm.intime - lm.admittime)) / 3600.0            AS preicu_hours,
    CASE WHEN lm.deathtime IS NOT NULL AND lm.deathtime <= lm.intime + INTERVAL '30 days'
         THEN 1 ELSE 0 END                                             AS death_30d,
    CASE WHEN lm.deathtime IS NOT NULL AND lm.deathtime <= lm.intime + INTERVAL '90 days'
         THEN 1 ELSE 0 END                                             AS death_90d,
    lm.hospital_expire_flag                                            AS death_inhosp,
    CASE WHEN i.hadm_id IS NOT NULL THEN 1 ELSE 0 END                  AS infection,
    COALESCE(g.gc48_pe_mg, 0)                                          AS gc48_pe_mg,
    g.gc_max_daily_pe_mg,
    g.gc_first_hour,
    COALESCE(g.gc_n_orders, 0)                                         AS gc_n_orders,
    CASE WHEN g.hadm_id IS NOT NULL THEN 1 ELSE 0 END                  AS gc_any,
    ch.charlson_comorbidity_index                                      AS charlson,
    s.sofa_24hours                                                     AS sofa,
    ap.apsiii,
    oa.oasis,
    sp.sapsii,
    fl.albumin_min, fl.abs_lymphocytes_min, fl.wbc_min, fl.wbc_max,
    fl.platelets_min, fl.creatinine_max, fl.bilirubin_total_max,
    COALESCE(v.on_vent, 0)  AS on_vent,
    COALESCE(va.on_vaso, 0) AS on_vaso,
    COALESCE(r.on_rrt, 0)   AS on_rrt,
    CASE WHEN lnf.hadm_id IS NOT NULL THEN 1 ELSE 0 END                AS lupus_nephritis,
    COALESCE(im.imm_cyc, 0) AS imm_cyc, COALESCE(im.imm_mmf, 0) AS imm_mmf,
    COALESCE(im.imm_aza, 0) AS imm_aza, COALESCE(im.imm_cni, 0) AS imm_cni,
    COALESCE(im.imm_mtx, 0) AS imm_mtx, COALESCE(im.imm_hcq, 0) AS imm_hcq,
    COALESCE(im.imm_bio, 0) AS imm_bio
FROM lm
LEFT JOIN gc_agg  g  ON lm.hadm_id = g.hadm_id
LEFT JOIN inf     i  ON lm.hadm_id = i.hadm_id
LEFT JOIN mimiciv_derived.charlson ch ON lm.hadm_id = ch.hadm_id
LEFT JOIN mimiciv_derived.sofa     s  ON lm.stay_id = s.stay_id
LEFT JOIN mimiciv_derived.apsiii   ap ON lm.stay_id = ap.stay_id
LEFT JOIN mimiciv_derived.oasis    oa ON lm.stay_id = oa.stay_id
LEFT JOIN mimiciv_derived.sapsii   sp ON lm.stay_id = sp.stay_id
LEFT JOIN mimiciv_derived.first_day_lab fl ON lm.stay_id = fl.stay_id
LEFT JOIN vent v  ON lm.stay_id = v.stay_id
LEFT JOIN vaso va ON lm.stay_id = va.stay_id
LEFT JOIN rrt  r  ON lm.stay_id = r.stay_id
LEFT JOIN ln   lnf ON lm.hadm_id = lnf.hadm_id
LEFT JOIN imm  im  ON lm.hadm_id = im.hadm_id
ORDER BY lm.subject_id, lm.intime
"""

def main():
    print("connecting ...")
    c = psycopg2.connect(dbname="mimiciv", **CONFIG)
    df = pd.read_sql_query(SQL, c)
    c.close()
    print("rows:", len(df))

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "cohort_sle_icu_raw.csv")
    df.to_csv(p, index=False)
    print("saved:", p)

    # ---------- quick data-quality / distribution report ----------
    print("\n--- missingness (%) ---")
    miss = (df.isna().mean() * 100).round(1).sort_values(ascending=False)
    print(miss[miss > 0].to_string())

    print("\n--- GC exposure ---")
    print("any GC          :", int(df.gc_any.sum()), "(%s%%)" % round(100 * df.gc_any.mean(), 1))
    pos = df.loc[df.gc48_pe_mg > 0, "gc48_pe_mg"]
    print("n with dose > 0 :", len(pos))
    if len(pos):
        qs = [0, .1, .25, .33, .5, .66, .75, .9, .95, 1.0]
        print("prednisone-equiv 48h dose quantiles:")
        for q in qs:
            print("   q%-5s %10.1f" % (q, pos.quantile(q)))

    print("\n--- outcomes ---")
    for o in ["infection", "death_30d", "death_inhosp", "death_90d"]:
        if o in df:
            print("  %-14s %4d  (%.1f%%)" % (o, int(df[o].sum()), 100 * df[o].mean()))

    print("\n--- key covariates ---")
    for v in ["age", "charlson", "sofa", "apsiii", "oasis", "albumin_min", "abs_lymphocytes_min",
              "wbc_min", "platelets_min", "creatinine_max", "icu_los", "preicu_hours"]:
        if v in df:
            s = df[v].dropna()
            print("  %-22s n=%4d  mean=%8.2f  sd=%7.2f  median=%8.2f" %
                  (v, len(s), s.mean(), s.std(), s.median()))

    print("\n--- organ support / SLE proxies ---")
    for v in ["on_vent", "on_vaso", "on_rrt", "lupus_nephritis", "imm_cyc", "imm_mmf", "imm_aza",
              "imm_cni", "imm_mtx", "imm_hcq", "imm_bio"]:
        if v in df:
            print("  %-16s %4d (%.1f%%)" % (v, int(df[v].sum()), 100 * df[v].mean()))

if __name__ == "__main__":
    main()
