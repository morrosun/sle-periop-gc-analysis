# -*- coding: utf-8 -*-
"""
Before rebuilding anything: check whether the two proposed fixes are even feasible.

  (1) pre-ICU window  : how long is hospital admission -> ICU admission?
                        if most patients go straight to ICU, a "pre-ICU GC"
                        exposure has no window to be measured in
  (2) activity proxies: coverage of C3 / C4 / ESR / urine protein in *this* cohort
  (3) hyperglycaemia  : availability of post-ICU glucose for a real positive control
"""
import os
import psycopg2
import pandas as pd

CONFIG = dict(
    host=os.environ.get("MIMIC_DB_HOST", "localhost"),
    port=int(os.environ.get("MIMIC_DB_PORT", "5432")),
    user=os.environ.get("MIMIC_DB_USER", "postgres"),
    password=os.environ.get("MIMIC_DB_PASSWORD", ""),
)

Q = r"""
WITH sle AS (
    SELECT DISTINCT d.hadm_id FROM mimiciv_hosp.diagnoses_icd d
    WHERE (d.icd_version=10 AND d.icd_code LIKE 'M32%') OR (d.icd_version=9 AND d.icd_code LIKE '7100%')
),
icu AS (
    SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime,
           ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.intime) AS rn
    FROM mimiciv_icu.icustays i JOIN sle ON i.hadm_id = sle.hadm_id
),
lm AS (
    SELECT c.*, a.admittime, ag.age
    FROM icu c
    JOIN mimiciv_hosp.admissions a ON c.hadm_id = a.hadm_id
    LEFT JOIN mimiciv_derived.age ag ON c.hadm_id = ag.hadm_id
    WHERE c.rn = 1 AND ag.age >= 18
),
pre AS (
    SELECT lm.hadm_id,
           EXTRACT(EPOCH FROM (lm.intime - lm.admittime))/3600.0 AS pre_hours
    FROM lm
),
gcpre AS (
    SELECT p.hadm_id,
           MIN(p.starttime) AS first_gc,
           MAX(p.stoptime)  AS last_gc
    FROM mimiciv_hosp.prescriptions p JOIN lm ON p.hadm_id = lm.hadm_id
    WHERE (p.drug ILIKE '%predni%' OR p.drug ILIKE '%methylpred%' OR p.drug ILIKE '%dexameth%'
        OR p.drug ILIKE '%hydrocort%' OR p.drug ILIKE '%betameth%' OR p.drug ILIKE '%triamcin%')
      AND p.starttime IS NOT NULL
    GROUP BY p.hadm_id
),
gcin AS (   -- any GC order that overlaps [intime, intime+48h)
    SELECT DISTINCT p.hadm_id
    FROM mimiciv_hosp.prescriptions p JOIN lm ON p.hadm_id = lm.hadm_id
    WHERE (p.drug ILIKE '%predni%' OR p.drug ILIKE '%methylpred%' OR p.drug ILIKE '%dexameth%'
        OR p.drug ILIKE '%hydrocort%' OR p.drug ILIKE '%betameth%' OR p.drug ILIKE '%triamcin%')
      AND p.stoptime  > lm.intime
      AND p.starttime < lm.intime + INTERVAL '48 hours'
)
SELECT
    lm.subject_id, lm.hadm_id, lm.stay_id, lm.intime, lm.admittime,
    pre.pre_hours,
    CASE WHEN gp.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS gc_any_hosp,
    CASE WHEN gp.first_gc < lm.intime THEN 1 ELSE 0 END AS gc_before_icu,
    EXTRACT(EPOCH FROM (lm.intime - gp.first_gc))/3600.0 AS gc_first_to_icu_h,
    CASE WHEN gi.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS gc_in48
FROM lm
JOIN pre  ON lm.hadm_id = pre.hadm_id
LEFT JOIN gcpre gp ON lm.hadm_id = gp.hadm_id
LEFT JOIN gcin  gi ON lm.hadm_id = gi.hadm_id
"""

Q_LABCOV = r"""
WITH sle AS (
    SELECT DISTINCT d.hadm_id FROM mimiciv_hosp.diagnoses_icd d
    WHERE (d.icd_version=10 AND d.icd_code LIKE 'M32%') OR (d.icd_version=9 AND d.icd_code LIKE '7100%')
),
icu AS (
    SELECT i.hadm_id, i.stay_id, i.intime,
           ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.intime) AS rn
    FROM mimiciv_icu.icustays i JOIN sle ON i.hadm_id = sle.hadm_id
),
lm AS (SELECT * FROM icu WHERE rn=1),
lab AS (
    SELECT l.hadm_id, d.label,
           MIN(l.valuenum) AS vmin, MAX(l.valuenum) AS vmax,
           MIN(EXTRACT(EPOCH FROM (l.charttime - lm.intime))/3600.0) AS h_min,
           MAX(EXTRACT(EPOCH FROM (l.charttime - lm.intime))/3600.0) AS h_max
    FROM mimiciv_hosp.labevents l
    JOIN mimiciv_hosp.d_labitems d ON l.itemid = d.itemid
    JOIN lm ON l.hadm_id = lm.hadm_id
    WHERE d.label IN ('C3','C4','Sedimentation Rate','C-Reactive Protein',
                      'Protein/Creatinine Ratio','Glucose','Hemoglobin',
                      'Platelet Count','White Blood Cells')
      AND l.valuenum IS NOT NULL AND l.hadm_id IS NOT NULL
    GROUP BY l.hadm_id, d.label
)
SELECT label,
       COUNT(*) AS n_hadm_with_lab,
       ROUND(AVG(vmin)::numeric,2) AS mean_min,
       ROUND(AVG(vmax)::numeric,2) AS mean_max,
       ROUND(MIN(h_min)::numeric,1) AS earliest_h,
       ROUND(MAX(h_max)::numeric,1) AS latest_h
FROM lab GROUP BY label ORDER BY n_hadm_with_lab DESC
"""

Q_GLU = r"""
WITH sle AS (
    SELECT DISTINCT d.hadm_id FROM mimiciv_hosp.diagnoses_icd d
    WHERE (d.icd_version=10 AND d.icd_code LIKE 'M32%') OR (d.icd_version=9 AND d.icd_code LIKE '7100%')
),
icu AS (
    SELECT i.hadm_id, i.stay_id, i.intime,
           ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.intime) AS rn
    FROM mimiciv_icu.icustays i JOIN sle ON i.hadm_id = sle.hadm_id
),
lm AS (SELECT * FROM icu WHERE rn=1),
g AS (
    SELECT l.hadm_id,
           MAX(l.valuenum) FILTER (WHERE l.charttime >= lm.intime AND l.charttime < lm.intime + INTERVAL '48 hours') AS glu_max_post,
           MAX(l.valuenum) FILTER (WHERE l.charttime <  lm.intime) AS glu_max_pre,
           COUNT(*) FILTER (WHERE l.charttime >= lm.intime AND l.charttime < lm.intime + INTERVAL '48 hours') AS n_post
    FROM mimiciv_hosp.labevents l
    JOIN mimiciv_hosp.d_labitems d ON l.itemid = d.itemid
    JOIN lm ON l.hadm_id = lm.hadm_id
    WHERE d.label = 'Glucose' AND l.valuenum IS NOT NULL
    GROUP BY l.hadm_id
)
SELECT COUNT(*) AS n_with_post_glu,
       SUM(CASE WHEN glu_max_post IS NOT NULL THEN 1 ELSE 0 END) AS n_nonnull,
       SUM(CASE WHEN glu_max_post >= 180 THEN 1 ELSE 0 END) AS n_hyper180,
       SUM(CASE WHEN glu_max_post >= 200 THEN 1 ELSE 0 END) AS n_hyper200,
       ROUND(AVG(glu_max_post)::numeric,1) AS mean_max,
       ROUND(AVG(glu_max_pre)::numeric,1) AS mean_pre
FROM g
"""

def main():
    c = psycopg2.connect(dbname="mimiciv", **CONFIG)
    df = pd.read_sql_query(Q, c)
    print("=== cohort rows: %d ===" % len(df))
    print("\n--- pre-ICU window (admission -> ICU) in hours ---")
    for q in [0, .1, .25, .5, .75, .9, 1.0]:
        print("   q%-5s %10.1f" % (q, df.pre_hours.quantile(q)))
    print("   >=24 h : %d (%.1f%%)" % ((df.pre_hours >= 24).sum(), 100 * (df.pre_hours >= 24).mean()))
    print("   >=48 h : %d (%.1f%%)" % ((df.pre_hours >= 48).sum(), 100 * (df.pre_hours >= 48).mean()))
    print("   >=7 d  : %d (%.1f%%)" % ((df.pre_hours >= 168).sum(), 100 * (df.pre_hours >= 168).mean()))
    print("\n--- GC timing ---")
    print("   any GC order in hospitalisation : %d (%.1f%%)" % (df.gc_any_hosp.sum(), 100 * df.gc_any_hosp.mean()))
    print("   GC started BEFORE ICU           : %d (%.1f%%)" % (df.gc_before_icu.sum(), 100 * df.gc_before_icu.mean()))
    print("   GC overlapping [intime,+48h)    : %d (%.1f%%)" % (df.gc_in48.sum(), 100 * df.gc_in48.mean()))
    sub = df[df.pre_hours >= 24]
    print("\n   among the %d with >=24 h pre-ICU window:" % len(sub))
    print("     GC before ICU : %d (%.1f%%)" % (sub.gc_before_icu.sum(), 100 * sub.gc_before_icu.mean()))
    print("\n=== lab coverage in this cohort (n=%d hadm) ===" % df.hadm_id.nunique())
    lab = pd.read_sql_query(Q_LABCOV, c)
    print(lab.to_string(index=False))
    print("\n=== post-ICU glucose (positive control candidate) ===")
    print(pd.read_sql_query(Q_GLU, c).to_string(index=False))
    c.close()

if __name__ == "__main__":
    main()
