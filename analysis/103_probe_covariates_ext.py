# -*- coding: utf-8 -*-
"""Probe the harmonised covariate set and the REAL event counts in the two
external databases, before writing the extraction.

Two things decide whether external validation is possible at all:
  V1  can we build the same confounders (severity / ventilation / vasopressor)?
  V2  how many events does each timed outcome actually have in SLE stays?
"""
import os
import psycopg2

CONFIG = dict(
    host=os.environ.get("MIMIC_DB_HOST", "localhost"),
    port=int(os.environ.get("MIMIC_DB_PORT", "5432")),
    user=os.environ.get("MIMIC_DB_USER", "postgres"),
    password=os.environ.get("MIMIC_DB_PASSWORD", ""),
)


def run(title, sql, dbname, n=20):
    print("\n" + "=" * 72)
    print("[%s] %s" % (dbname, title))
    print("=" * 72)
    c = psycopg2.connect(dbname=dbname, **CONFIG)
    cur = c.cursor()
    try:
        cur.execute(sql)
        rows = cur.fetchall()
        print("  " + " | ".join(d[0] for d in cur.description))
        for r in rows[:n]:
            print("   ", " | ".join("NULL" if v is None else str(v) for v in r))
        print("  (%d rows)" % len(rows))
    except Exception as ex:
        print("  !! fail:", str(ex)[:400])
        c.rollback()
    c.close()


EICU_SLE = """
  WITH sle AS (
    SELECT DISTINCT patientunitstayid FROM eicu_crd.diagnosis
    WHERE diagnosisstring ILIKE '%lupus%' OR icd9code LIKE '7100%'
  ), p AS (
    SELECT pt.*, ROW_NUMBER() OVER (PARTITION BY pt.patienthealthsystemstayid
                                    ORDER BY pt.hospitaladmitoffset) AS rn
    FROM eicu_crd.patient pt JOIN sle ON pt.patientunitstayid = sle.patientunitstayid
  )
"""

NWICU_SLE = """
  WITH sle AS (
    SELECT DISTINCT hadm_id FROM hosp.diagnoses_icd
    WHERE (icd_version=10 AND icd_code LIKE 'M32%%') OR (icd_version=9 AND icd_code LIKE '7100%%')
  ), p AS (
    SELECT i.*, a.admittime, a.dischtime, a.deathtime, a.hospital_expire_flag,
           ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.intime) AS rn
    FROM icu.icustays i JOIN sle ON i.hadm_id = sle.hadm_id
    JOIN hosp.admissions a ON i.hadm_id = a.hadm_id
  )
"""


def main():
    # ---------------------------------------------------- eICU feasibility
    run("eICU SLE stays: total vs LOS>=24h (offset>=1440 min)", EICU_SLE + """
        SELECT COUNT(*) n_total,
               SUM(CASE WHEN unitdischargeoffset >= 1440 THEN 1 ELSE 0 END) n_ge24h,
               SUM(CASE WHEN unitdischargeoffset >= 2880 THEN 1 ELSE 0 END) n_ge48h
        FROM p WHERE rn = 1
    """, "eicu", n=5)

    run("eICU SLE: blood culture after 24h (excl. no growth / Other)", EICU_SLE + """
        SELECT
          COUNT(DISTINCT CASE WHEN m.culturetakenoffset >= 1440 THEN m.patientunitstayid END) n_cult_after24,
          COUNT(DISTINCT CASE WHEN m.culturetakenoffset >= 1440
                               AND m.organism IS NOT NULL
                               AND lower(m.organism) NOT LIKE 'no growth%'
                               AND lower(m.organism) <> 'other' THEN m.patientunitstayid END) n_pos_after24
        FROM p
        LEFT JOIN eicu_crd.microlab m
          ON m.patientunitstayid = p.patientunitstayid AND m.culturesite ILIKE '%blood%'
        WHERE p.rn = 1
    """, "eicu", n=5)

    run("eICU SLE: organism values in blood cultures", EICU_SLE + """
        SELECT m.organism, COUNT(*) n
        FROM p JOIN eicu_crd.microlab m ON m.patientunitstayid = p.patientunitstayid
        WHERE p.rn=1 AND m.culturesite ILIKE '%%blood%%'
        GROUP BY 1 ORDER BY n DESC LIMIT 12
    """, "eicu", n=14)

    run("eICU SLE: antibiotic timing", EICU_SLE + """
        SELECT
          COUNT(DISTINCT CASE WHEN m.drugstartoffset < 1440 THEN m.patientunitstayid END) n_abx_early,
          COUNT(DISTINCT CASE WHEN m.drugstartoffset >= 2880 THEN m.patientunitstayid END) n_abx_after48,
          COUNT(DISTINCT m.patientunitstayid) n_abx_any
        FROM p JOIN eicu_crd.medication m ON m.patientunitstayid = p.patientunitstayid
        WHERE p.rn=1 AND (m.drugname ILIKE '%%vancomycin%%' OR m.drugname ILIKE '%%meropenem%%'
           OR m.drugname ILIKE '%%piperacillin%%' OR m.drugname ILIKE '%%cefepime%%'
           OR m.drugname ILIKE '%%ceftriaxone%%' OR m.drugname ILIKE '%%linezolid%%'
           OR m.drugname ILIKE '%%levofloxacin%%' OR m.drugname ILIKE '%%ciprofloxacin%%')
    """, "eicu", n=5)

    run("eICU SLE: steroid exposure in first 24h (any)", EICU_SLE + """
        SELECT COUNT(DISTINCT m.patientunitstayid) n_with_gc
        FROM p JOIN eicu_crd.medication m ON m.patientunitstayid = p.patientunitstayid
        WHERE p.rn=1
          AND (m.drugname ILIKE '%%predni%%' OR m.drugname ILIKE '%%methylpred%%'
               OR m.drugname ILIKE '%%solumedrol%%' OR m.drugname ILIKE '%%dexameth%%'
               OR m.drugname ILIKE '%%hydrocort%%' OR m.drugname ILIKE '%%cortis%%'
               OR m.drugname ILIKE '%%betameth%%' OR m.drugname ILIKE '%%triamcin%%')
          AND m.drugstopoffset > 0 AND m.drugstartoffset < 1440
          AND COALESCE(m.drugordercancelled,'No') <> 'Yes'
    """, "eicu", n=5)

    run("eICU SLE: glucose coverage (first 48h)", EICU_SLE + """
        SELECT COUNT(DISTINCT p.patientunitstayid) n_with_glu,
               ROUND(MIN(l.labresult)::numeric,1) mn, ROUND(MAX(l.labresult)::numeric,1) mx
        FROM p JOIN eicu_crd.lab l ON l.patientunitstayid = p.patientunitstayid
        WHERE p.rn=1 AND l.labname IN ('glucose','bedside glucose')
          AND l.labresult IS NOT NULL AND l.labresultoffset BETWEEN 0 AND 2880
    """, "eicu", n=5)

    run("eICU: vasopressor via infusiondrug (name sample)", """
        SELECT drugname, COUNT(*) n FROM eicu_crd.infusiondrug
        WHERE drugname ILIKE '%norepinephrine%' OR drugname ILIKE '%levophed%'
           OR drugname ILIKE '%epinephrine%' OR drugname ILIKE '%dopamine%'
           OR drugname ILIKE '%phenylephrine%' OR drugname ILIKE '%vasopressin%'
           OR drugname ILIKE '%dobutamine%' OR drugname ILIKE '%neosynephrine%'
        GROUP BY 1 ORDER BY n DESC LIMIT 12
    """, "eicu", n=14)

    run("eICU: ventilation via treatment string", """
        SELECT treatmentstring, COUNT(*) n FROM eicu_crd.treatment
        WHERE treatmentstring ILIKE '%ventil%' OR treatmentstring ILIKE '%intubat%'
        GROUP BY 1 ORDER BY n DESC LIMIT 12
    """, "eicu", n=14)

    # ---------------------------------------------------- NWICU feasibility
    run("NWICU SLE stays: total vs LOS>=24h", NWICU_SLE + """
        SELECT COUNT(*) n_total,
               SUM(CASE WHEN los >= 1 THEN 1 ELSE 0 END) n_ge24h,
               SUM(CASE WHEN los >= 2 THEN 1 ELSE 0 END) n_ge48h
        FROM p WHERE rn = 1
    """, "nwicu", n=5)

    run("NWICU SLE: antibiotic timing", NWICU_SLE + """
        SELECT
          COUNT(DISTINCT CASE WHEN pr.starttime < p.intime + INTERVAL '24 hours' THEN pr.hadm_id END) n_abx_early,
          COUNT(DISTINCT CASE WHEN pr.starttime >= p.intime + INTERVAL '48 hours' THEN pr.hadm_id END) n_abx_after48,
          COUNT(DISTINCT pr.hadm_id) n_abx_any
        FROM p LEFT JOIN hosp.prescriptions pr ON pr.hadm_id = p.hadm_id
        WHERE p.rn=1 AND (pr.drug ILIKE '%%vancomycin%%' OR pr.drug ILIKE '%%meropenem%%'
           OR pr.drug ILIKE '%%piperacillin%%' OR pr.drug ILIKE '%%cefepime%%'
           OR pr.drug ILIKE '%%ceftriaxone%%' OR pr.drug ILIKE '%%linezolid%%'
           OR pr.drug ILIKE '%%levofloxacin%%' OR pr.drug ILIKE '%%ciprofloxacin%%')
    """, "nwicu", n=5)

    run("NWICU SLE: steroid in first 24h", NWICU_SLE + """
        SELECT COUNT(DISTINCT pr.hadm_id) n_with_gc
        FROM p JOIN hosp.prescriptions pr ON pr.hadm_id = p.hadm_id
        WHERE p.rn=1
          AND (pr.drug ILIKE '%%predni%%' OR pr.drug ILIKE '%%methylpred%%'
               OR pr.drug ILIKE '%%dexameth%%' OR pr.drug ILIKE '%%hydrocort%%'
               OR pr.drug ILIKE '%%betameth%%' OR pr.drug ILIKE '%%triamcin%%')
          AND pr.stoptime > p.intime AND pr.starttime < p.intime + INTERVAL '24 hours'
    """, "nwicu", n=5)

    run("NWICU: dose_val_rx format for steroids", """
        SELECT dose_val_rx, dose_unit_rx, doses_per_24_hrs, route, COUNT(*) n
        FROM hosp.prescriptions
        WHERE (drug ILIKE '%%predni%%' OR drug ILIKE '%%methylpred%%' OR drug ILIKE '%%dexameth%%')
        GROUP BY 1,2,3,4 ORDER BY n DESC LIMIT 15
    """, "nwicu", n=18)

    run("NWICU: glucose coverage", """
        SELECT COUNT(DISTINCT l.hadm_id) n_hadm, ROUND(MIN(l.valuenum)::numeric,1) mn,
               ROUND(MAX(l.valuenum)::numeric,1) mx
        FROM hosp.labevents l JOIN hosp.d_labitems d ON l.itemid=d.itemid
        WHERE d.label='Glucose' AND l.valuenum IS NOT NULL
    """, "nwicu", n=5)

    run("NWICU: vasopressor itemids in icu.inputevents", """
        SELECT i.itemid, di.label, COUNT(*) n
        FROM icu.inputevents i JOIN icu.d_items di ON i.itemid=di.itemid
        WHERE di.label ILIKE '%norepinephrine%' OR di.label ILIKE '%epinephrine%'
           OR di.label ILIKE '%dopamine%' OR di.label ILIKE '%phenylephrine%'
           OR di.label ILIKE '%vasopressin%' OR di.label ILIKE '%dobutamine%'
        GROUP BY 1,2 ORDER BY n DESC LIMIT 12
    """, "nwicu", n=14)

    run("NWICU: ventilation itemids in icu.procedureevents / chartevents", """
        SELECT 'procedureevents' src, di.label, COUNT(*) n
        FROM icu.procedureevents p JOIN icu.d_items di ON p.itemid=di.itemid
        WHERE di.label ILIKE '%ventil%' GROUP BY 1,2 ORDER BY n DESC LIMIT 6
    """, "nwicu", n=8)

    run("NWICU: procedures_icd in hosp?", """
        SELECT table_schema, table_name FROM information_schema.tables
        WHERE lower(table_name)='procedures_icd'
    """, "nwicu", n=6)


if __name__ == "__main__":
    main()
