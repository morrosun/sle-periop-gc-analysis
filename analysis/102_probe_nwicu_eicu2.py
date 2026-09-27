# -*- coding: utf-8 -*-
"""Resolve two blockers before any extraction.

B1  NWICU contains BOTH a `hosp`/`icu` schema pair AND a `mimiciv_hosp`/
    `mimiciv_icu` pair. If the latter is a leftover copy of MIMIC-IV, using it
    would silently produce a "validation" on the same patients we already
    analysed. This script proves which is which using row counts and patient
    overlap.
B2  NWICU seems to lack a microbiology table in `hosp`. Need to confirm whether
    a timed culture outcome exists at all, and what the fallback is.
B3  eICU microlab `organism` appears always non-null -> must check whether
    "no growth" is encoded as text, otherwise every culture looks positive.
"""
import os
import psycopg2

CONFIG = dict(
    host=os.environ.get("MIMIC_DB_HOST", "localhost"),
    port=int(os.environ.get("MIMIC_DB_PORT", "5432")),
    user=os.environ.get("MIMIC_DB_USER", "postgres"),
    password=os.environ.get("MIMIC_DB_PASSWORD", ""),
)


def run(title, sql, dbname="nwicu", n=20):
    print("\n" + "=" * 72)
    print("[%s] %s" % (dbname, title))
    print("=" * 72)
    c = psycopg2.connect(dbname=dbname, **CONFIG)
    cur = c.cursor()
    try:
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        print("  " + " | ".join(cols))
        for r in rows[:n]:
            print("   ", " | ".join("NULL" if v is None else str(v) for v in r))
        print("  (%d rows)" % len(rows))
    except Exception as ex:
        print("  !! fail:", str(ex)[:300])
        c.rollback()
    c.close()


def main():
    # ---------------------------------------------------------- B1
    run("B1 row counts per schema", """
        SELECT 'hosp.admissions' t, COUNT(*)::bigint n FROM hosp.admissions
        UNION ALL SELECT 'mimiciv_hosp.admissions', COUNT(*) FROM mimiciv_hosp.admissions
        UNION ALL SELECT 'hosp.diagnoses_icd', COUNT(*) FROM hosp.diagnoses_icd
        UNION ALL SELECT 'mimiciv_hosp.diagnoses_icd', COUNT(*) FROM mimiciv_hosp.diagnoses_icd
        UNION ALL SELECT 'hosp.labevents', COUNT(*) FROM hosp.labevents
        UNION ALL SELECT 'mimiciv_hosp.labevents', COUNT(*) FROM mimiciv_hosp.labevents
        UNION ALL SELECT 'mimiciv_hosp.microbiologyevents', COUNT(*) FROM mimiciv_hosp.microbiologyevents
        UNION ALL SELECT 'icu.icustays', COUNT(*) FROM icu.icustays
        UNION ALL SELECT 'mimiciv_icu.icustays', COUNT(*) FROM mimiciv_icu.icustays
        ORDER BY 1
    """, n=15)

    run("B1 subject overlap hosp vs mimiciv_hosp", """
        SELECT
          (SELECT COUNT(*) FROM (SELECT subject_id FROM hosp.patients
             INTERSECT SELECT subject_id FROM mimiciv_hosp.patients) z) AS shared_subjects,
          (SELECT COUNT(*) FROM hosp.patients) AS hosp_subjects,
          (SELECT COUNT(*) FROM mimiciv_hosp.patients) AS mimiciv_subjects
    """, n=5)

    # ---------------------------------------------------------- B2
    run("B2 does any NWICU microbiology table exist?", """
        SELECT table_schema, table_name FROM information_schema.tables
        WHERE lower(table_name) LIKE '%micro%' OR lower(table_name) LIKE '%culture%'
        ORDER BY 1,2
    """, n=15)

    run("B2 NWICU microbiology spec types (mimiciv_hosp)", """
        SELECT spec_type_desc, COUNT(*) n FROM mimiciv_hosp.microbiologyevents
        GROUP BY 1 ORDER BY n DESC LIMIT 20
    """, n=22)

    run("B2 NWICU antibiotic prescriptions available?", """
        SELECT COUNT(DISTINCT p.hadm_id) n_hadm, COUNT(*) n_rows
        FROM hosp.prescriptions p
        WHERE p.drug ILIKE '%vancomycin%' OR p.drug ILIKE '%meropenem%'
           OR p.drug ILIKE '%piperacillin%' OR p.drug ILIKE '%cefepime%'
           OR p.drug ILIKE '%ceftriaxone%' OR p.drug ILIKE '%linezolid%'
    """, n=5)

    run("B2 NWICU steroid prescriptions", """
        SELECT COUNT(DISTINCT p.hadm_id) n_hadm, COUNT(*) n_rows
        FROM hosp.prescriptions p
        WHERE p.drug ILIKE '%predni%' OR p.drug ILIKE '%methylpred%'
           OR p.drug ILIKE '%dexameth%' OR p.drug ILIKE '%hydrocort%'
           OR p.drug ILIKE '%betameth%' OR p.drug ILIKE '%triamcin%'
    """, n=5)

    run("B2 NWICU infection ICD availability (hosp.diagnoses_icd)", """
        SELECT icd_version, COUNT(DISTINCT hadm_id) n_hadm
        FROM hosp.diagnoses_icd
        WHERE (icd_version=10 AND (LEFT(icd_code,3) IN
                 ('A40','A41','J85','K65','L02','L03','M86','N10','R65','T81','T82','T84')))
           OR (icd_version=9 AND (LEFT(icd_code,3) IN
                 ('038','320','322','324','481','482','483','484','485','486','510','567','590','599','680','682','711','730','998')))
        GROUP BY 1 ORDER BY 1
    """, n=5)

    run("B2 NWICU hospital mortality", """
        SELECT hospital_expire_flag, COUNT(*) n FROM hosp.admissions GROUP BY 1 ORDER BY 1
    """, n=5)

    # ---------------------------------------------------------- B3
    run("B3 eICU microlab organism values", """
        SELECT organism, COUNT(*) n FROM eicu_crd.microlab
        GROUP BY 1 ORDER BY n DESC LIMIT 20
    """, dbname="eicu", n=22)

    run("B3 eICU blood culture rows in whole DB", """
        SELECT COUNT(*) n_rows, COUNT(DISTINCT patientunitstayid) n_stays,
               COUNT(DISTINCT CASE WHEN culturetakenoffset >= 1440 THEN patientunitstayid END) n_after24
        FROM eicu_crd.microlab WHERE culturesite ILIKE '%blood%'
    """, dbname="eicu", n=5)

    run("B3 eICU: does medication carry dose for SLE stays? sample", """
        SELECT m.drugname, m.dosage, m.frequency, m.routeadmin,
               m.drugstartoffset, m.drugstopoffset
        FROM eicu_crd.medication m
        JOIN eicu_crd.diagnosis d ON m.patientunitstayid = d.patientunitstayid
        WHERE (d.diagnosisstring ILIKE '%lupus%' OR d.icd9code LIKE '7100%')
          AND (m.drugname ILIKE '%predni%' OR m.drugname ILIKE '%methylpred%'
               OR m.drugname ILIKE '%dexameth%' OR m.drugname ILIKE '%hydrocort%'
               OR m.drugname ILIKE '%solumedrol%' OR m.drugname ILIKE '%cortis%')
        LIMIT 20
    """, dbname="eicu", n=22)


if __name__ == "__main__":
    main()
