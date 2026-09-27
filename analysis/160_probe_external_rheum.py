# -*- coding: utf-8 -*-
"""
160_probe_external_rheum.py

Before writing any extraction code for the SLE-vs-RA head-to-head in the
external databases, count what is actually there.  The v3 lesson was
explicit: patient counts are not event counts, and an external validation
that cannot have events cannot say anything.

SLE and RA are defined to match MIMIC-IV as closely as each database
allows:
    SLE   ICD-10 M32*, ICD-9 710.0*, eICU diagnosisstring 'lupus'
    RA    ICD-10 M05*/M06*, ICD-9 714*, eICU diagnosisstring 'rheumatoid'

eICU stores diagnosis codes comma-separated inside `diagnosis.icd9code`
(despite the name it also holds ICD-10), so the codes are unnested before
matching -- a LIKE on the raw string would both miss and over-match.
"""
import os

import pandas as pd
import psycopg2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root (data/ and out/ live here)

CONFIG = dict(
    host=os.environ.get("MIMIC_DB_HOST", "localhost"),
    port=int(os.environ.get("MIMIC_DB_PORT", "5432")),
    user=os.environ.get("MIMIC_DB_USER", "postgres"),
    password=os.environ.get("MIMIC_DB_PASSWORD", ""),
)
OUT = os.path.join(ROOT, "out")

EICU = r"""
WITH dxs AS (
    SELECT d.patientunitstayid, TRIM(BOTH ' ' FROM u.code) AS code,
           LOWER(COALESCE(d.diagnosisstring,'')) AS ds
    FROM eicu_crd.diagnosis d,
         UNNEST(STRING_TO_ARRAY(COALESCE(d.icd9code,''), ',')) AS u(code)
),
cls AS (
    SELECT patientunitstayid,
      MAX(CASE WHEN code LIKE '7100%%' OR code LIKE 'M32%%' OR ds LIKE '%%lupus%%'
               THEN 1 ELSE 0 END) AS f_sle,
      MAX(CASE WHEN code LIKE '714%%' OR code LIKE 'M05%%' OR code LIKE 'M06%%'
               OR ds LIKE '%%rheumatoid%%' THEN 1 ELSE 0 END) AS f_ra,
      MAX(CASE WHEN code LIKE 'M45%%' OR code LIKE 'M46%%' OR code LIKE '720%%'
               OR ds LIKE '%%ankylosing%%' THEN 1 ELSE 0 END) AS f_axspa,
      MAX(CASE WHEN code LIKE 'M34%%' OR code LIKE '7101%%' THEN 1 ELSE 0 END) AS f_ssc,
      MAX(CASE WHEN code LIKE 'M31%%' OR code LIKE '446%%'
               OR ds LIKE '%%vasculitis%%' THEN 1 ELSE 0 END) AS f_vas
    FROM dxs GROUP BY patientunitstayid
),
fu AS (
    SELECT c.*, pt.unitdischargeoffset,
           ROW_NUMBER() OVER (PARTITION BY pt.patienthealthsystemstayid
                              ORDER BY pt.hospitaladmitoffset) AS rn
    FROM cls c JOIN eicu_crd.patient pt ON c.patientunitstayid = pt.patientunitstayid
    WHERE c.f_sle=1 OR c.f_ra=1 OR c.f_axspa=1 OR c.f_ssc=1 OR c.f_vas=1
)
SELECT CASE WHEN f_sle=1 THEN 'SLE' WHEN f_ra=1 THEN 'RA'
            WHEN f_vas=1 THEN 'Vasculitis' WHEN f_ssc=1 THEN 'SSc'
            ELSE 'axSpA' END AS primary_grp,
       COUNT(*) AS n_stay,
       SUM(CASE WHEN unitdischargeoffset >= 1440 THEN 1 ELSE 0 END) AS n_landmark,
       SUM(CASE WHEN rn=1 AND unitdischargeoffset >= 1440 THEN 1 ELSE 0 END) AS n_lm_first
FROM fu GROUP BY 1 ORDER BY 2 DESC
"""

# eICU outcomes, per disease
EICU_EV = r"""
WITH dxs AS (
    SELECT d.patientunitstayid, TRIM(BOTH ' ' FROM u.code) AS code,
           LOWER(COALESCE(d.diagnosisstring,'')) AS ds
    FROM eicu_crd.diagnosis d,
         UNNEST(STRING_TO_ARRAY(COALESCE(d.icd9code,''), ',')) AS u(code)
),
cls AS (
    SELECT patientunitstayid,
      MAX(CASE WHEN code LIKE '7100%%' OR code LIKE 'M32%%' OR ds LIKE '%%lupus%%'
               THEN 1 ELSE 0 END) AS f_sle,
      MAX(CASE WHEN code LIKE '714%%' OR code LIKE 'M05%%' OR code LIKE 'M06%%'
               OR ds LIKE '%%rheumatoid%%' THEN 1 ELSE 0 END) AS f_ra
    FROM dxs GROUP BY patientunitstayid
),
p0 AS (
    SELECT pt.patientunitstayid,
           CASE WHEN c.f_sle=1 THEN 'SLE' ELSE 'RA' END AS primary_grp,
           pt.unitdischargeoffset, pt.hospitaldischargestatus,
           ROW_NUMBER() OVER (PARTITION BY pt.patienthealthsystemstayid
                              ORDER BY pt.hospitaladmitoffset) AS rn
    FROM cls c JOIN eicu_crd.patient pt ON c.patientunitstayid = pt.patientunitstayid
    WHERE c.f_sle=1 OR c.f_ra=1
),
p AS (SELECT * FROM p0 WHERE rn=1 AND unitdischargeoffset >= 1440),
d2 AS (
    SELECT d.patientunitstayid,
      MAX(CASE WHEN TRIM(BOTH ' ' FROM u.code) LIKE '038%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'A41%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'J1%%'
               OR LOWER(COALESCE(d.diagnosisstring,'')) LIKE '%%sepsis%%'
               OR LOWER(COALESCE(d.diagnosisstring,'')) LIKE '%%pneumonia%%'
               OR LOWER(COALESCE(d.diagnosisstring,'')) LIKE '%%infection%%'
               THEN 1 ELSE 0 END) AS infect_icd
    FROM eicu_crd.diagnosis d,
         UNNEST(STRING_TO_ARRAY(COALESCE(d.icd9code,''), ',')) AS u(code)
    WHERE d.patientunitstayid IN (SELECT patientunitstayid FROM p)
    GROUP BY d.patientunitstayid
),
g AS (
    SELECT l.patientunitstayid, MAX(l.labresult) AS glu_max
    FROM eicu_crd.lab l JOIN p ON l.patientunitstayid = p.patientunitstayid
    WHERE l.labname IN ('glucose','bedside glucose') AND l.labresult IS NOT NULL
      AND l.labresultoffset >= 0 AND l.labresultoffset < 2880
      AND l.labresult BETWEEN 10 AND 1500
    GROUP BY l.patientunitstayid
),
c AS (
    SELECT m.patientunitstayid,
           COUNT(*) AS n_cult,
           MAX(CASE WHEN m.culturetakenoffset >= 1440
                     AND m.organism IS NOT NULL
                     AND LOWER(m.organism) NOT LIKE 'no growth%%'
                     AND LOWER(m.organism) NOT IN ('other','') THEN 1 ELSE 0 END) AS cult_pos
    FROM eicu_crd.microlab m JOIN p ON m.patientunitstayid = p.patientunitstayid
    WHERE m.culturesite ILIKE '%%blood%%' AND m.culturetakenoffset IS NOT NULL
    GROUP BY m.patientunitstayid
)
SELECT p.primary_grp, COUNT(*) AS n_landmark,
       SUM(CASE WHEN p.hospitaldischargestatus='Expired' THEN 1 ELSE 0 END) AS death_hosp,
       SUM(COALESCE(c.cult_pos,0)) AS cult_pos_after24,
       SUM(COALESCE(c.n_cult,0))   AS n_blood_cultures,
       SUM(COALESCE(d2.infect_icd,0)) AS infect_icd,
       SUM(CASE WHEN g.glu_max >= 180 THEN 1 ELSE 0 END) AS hyper,
       SUM(CASE WHEN g.glu_max IS NOT NULL THEN 1 ELSE 0 END) AS glu_measured
FROM p
LEFT JOIN d2 ON p.patientunitstayid = d2.patientunitstayid
LEFT JOIN g  ON p.patientunitstayid = g.patientunitstayid
LEFT JOIN c  ON p.patientunitstayid = c.patientunitstayid
GROUP BY p.primary_grp ORDER BY 2 DESC
"""

NWICU = r"""
WITH cls AS (
    SELECT hadm_id,
      MAX(CASE WHEN (icd_version=10 AND icd_code LIKE 'M32%%')
                OR (icd_version=9 AND icd_code LIKE '7100%%') THEN 1 ELSE 0 END) AS f_sle,
      MAX(CASE WHEN (icd_version=10 AND (icd_code LIKE 'M05%%' OR icd_code LIKE 'M06%%'))
                OR (icd_version=9 AND icd_code LIKE '714%%') THEN 1 ELSE 0 END) AS f_ra,
      MAX(CASE WHEN (icd_version=10 AND (icd_code LIKE 'M45%%' OR icd_code LIKE 'M46%%'))
                OR (icd_version=9 AND icd_code LIKE '720%%') THEN 1 ELSE 0 END) AS f_axspa,
      MAX(CASE WHEN (icd_version=10 AND icd_code LIKE 'M34%%')
                OR (icd_version=9 AND icd_code LIKE '7101%%') THEN 1 ELSE 0 END) AS f_ssc,
      MAX(CASE WHEN (icd_version=10 AND icd_code LIKE 'M31%%')
                OR (icd_version=9 AND icd_code LIKE '446%%') THEN 1 ELSE 0 END) AS f_vas
    FROM hosp.diagnoses_icd GROUP BY hadm_id
)
SELECT CASE WHEN f_sle=1 THEN 'SLE' WHEN f_ra=1 THEN 'RA'
            WHEN f_vas=1 THEN 'Vasculitis' WHEN f_ssc=1 THEN 'SSc'
            ELSE 'axSpA' END AS primary_grp,
       COUNT(DISTINCT c.hadm_id) AS n_hadm,
       COUNT(*) AS n_stay,
       SUM(CASE WHEN i.los >= 1 THEN 1 ELSE 0 END) AS n_landmark
FROM cls c JOIN icu.icustays i ON c.hadm_id = i.hadm_id
WHERE f_sle=1 OR f_ra=1 OR f_axspa=1 OR f_ssc=1 OR f_vas=1
GROUP BY 1 ORDER BY 3 DESC
"""

NWICU_EV = r"""
WITH cls AS (
    SELECT hadm_id,
      MAX(CASE WHEN (icd_version=10 AND icd_code LIKE 'M32%%')
                OR (icd_version=9 AND icd_code LIKE '7100%%') THEN 1 ELSE 0 END) AS f_sle,
      MAX(CASE WHEN (icd_version=10 AND (icd_code LIKE 'M05%%' OR icd_code LIKE 'M06%%'))
                OR (icd_version=9 AND icd_code LIKE '714%%') THEN 1 ELSE 0 END) AS f_ra,
      MAX(CASE WHEN (icd_version=10 AND (icd_code LIKE 'A40%%' OR icd_code LIKE 'A41%%'
                OR icd_code LIKE 'J1%%' OR icd_code LIKE 'N39%%' OR icd_code LIKE 'K65%%'
                OR icd_code LIKE 'L03%%' OR icd_code LIKE 'M86%%'))
                OR (icd_version=9 AND (LEFT(icd_code,3) IN ('038','481','482','483','484','485',
                '486','507','510','513','540','567','590','599','682','683','686')
                OR icd_code LIKE '9959%%')) THEN 1 ELSE 0 END) AS infect_icd
    FROM hosp.diagnoses_icd GROUP BY hadm_id
),
p0 AS (
    SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.los,
           a.hospital_expire_flag,
           CASE WHEN c.f_sle=1 THEN 'SLE' ELSE 'RA' END AS primary_grp,
           ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.intime) AS rn
    FROM cls c
    JOIN icu.icustays i ON c.hadm_id = i.hadm_id
    JOIN hosp.admissions a ON c.hadm_id = a.hadm_id
    WHERE c.f_sle=1 OR c.f_ra=1
),
p AS (SELECT * FROM p0 WHERE rn=1 AND los >= 1),
g AS (
    SELECT l.hadm_id, MAX(l.valuenum) AS glu_max
    FROM hosp.labevents l JOIN hosp.d_labitems d ON l.itemid = d.itemid
    JOIN p ON l.hadm_id = p.hadm_id
    WHERE d.label = 'Glucose' AND l.valuenum IS NOT NULL
      AND l.charttime >= p.intime AND l.charttime < p.intime + INTERVAL '48 hours'
      AND l.valuenum BETWEEN 10 AND 1500
    GROUP BY l.hadm_id
)
SELECT p.primary_grp, COUNT(*) AS n_landmark,
       SUM(p.hospital_expire_flag) AS death_hosp,
       SUM(COALESCE(c.infect_icd,0)) AS infect_icd,
       SUM(CASE WHEN g.glu_max >= 180 THEN 1 ELSE 0 END) AS hyper,
       SUM(CASE WHEN g.glu_max IS NOT NULL THEN 1 ELSE 0 END) AS glu_measured
FROM p
LEFT JOIN cls c ON p.hadm_id = c.hadm_id
LEFT JOIN g ON p.hadm_id = g.hadm_id
GROUP BY p.primary_grp ORDER BY 2 DESC
"""


def show(c, title, sql):
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)
    try:
        d = pd.read_sql_query(sql, c)
        print(d.to_string(index=False))
        return d
    except Exception as e:
        print("  FAILED: %s" % str(e)[:400])
        return None


def main():
    print("=" * 96)
    print("EXTERNAL FEASIBILITY OF THE SLE-vs-RA HEAD-TO-HEAD")
    print("=" * 96)

    ce = psycopg2.connect(dbname="eicu", **CONFIG)
    show(ce, "eICU-CRD: rheumatic ICU stays by disease", EICU)
    show(ce, "eICU-CRD: outcomes in the SLE / RA landmark cohort", EICU_EV)
    ce.close()

    cn = psycopg2.connect(dbname="nwicu", **CONFIG)
    show(cn, "NWICU: rheumatic ICU stays by disease", NWICU)
    show(cn, "NWICU: outcomes in the SLE / RA landmark cohort", NWICU_EV)
    cn.close()


if __name__ == "__main__":
    main()
