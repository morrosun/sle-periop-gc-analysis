# -*- coding: utf-8 -*-
"""Second-pass probe: does each external DB actually contain what we need,
and in what shape?

eICU-CRD stores all timing as integer OFFSETS in minutes relative to unit
admission, and drug dose as a free-text string -> both need to be handled.
NWICU ships two parallel schemas (hosp/icu  vs  mimiciv_hosp/mimiciv_icu);
we must find out which one is populated.
"""
import os
import psycopg2

CONFIG = dict(
    host=os.environ.get("MIMIC_DB_HOST", "localhost"),
    port=int(os.environ.get("MIMIC_DB_PORT", "5432")),
    user=os.environ.get("MIMIC_DB_USER", "postgres"),
    password=os.environ.get("MIMIC_DB_PASSWORD", ""),
)


def run(dbname, title, sql, n=25):
    print("\n" + "=" * 72)
    print("[%s] %s" % (dbname, title))
    print("=" * 72)
    try:
        c = psycopg2.connect(dbname=dbname, **CONFIG)
    except Exception as ex:
        print("  !! connect fail:", ex)
        return
    try:
        cur = c.cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        print("  " + " | ".join(cols))
        for r in rows[:n]:
            print("   ", " | ".join("NULL" if v is None else str(v) for v in r))
        print("  (%d rows total)" % len(rows))
    except Exception as ex:
        print("  !! query fail:", ex)
        c.rollback()
    c.close()


def main():
    # ------------------------------------------------- NWICU schema ambiguity
    run("nwicu", "row counts: hosp vs mimiciv_hosp", """
        SELECT 'hosp.admissions' t, COUNT(*) FROM hosp.admissions
        UNION ALL SELECT 'mimiciv_hosp.admissions', COUNT(*) FROM mimiciv_hosp.admissions
        UNION ALL SELECT 'hosp.prescriptions', COUNT(*) FROM hosp.prescriptions
        UNION ALL SELECT 'mimiciv_hosp.prescriptions', COUNT(*) FROM mimiciv_hosp.prescriptions
        UNION ALL SELECT 'icu.icustays', COUNT(*) FROM icu.icustays
        UNION ALL SELECT 'mimiciv_icu.icustays', COUNT(*) FROM mimiciv_icu.icustays
        UNION ALL SELECT 'hosp.microbiologyevents', COUNT(*) FROM hosp.microbiologyevents
        UNION ALL SELECT 'mimiciv_hosp.microbiologyevents', COUNT(*) FROM mimiciv_hosp.microbiologyevents
        UNION ALL SELECT 'hosp.labevents', COUNT(*) FROM hosp.labevents
        ORDER BY 1
    """, n=20)

    run("nwicu", "SLE in NWICU (hosp.diagnoses_icd)", """
        SELECT icd_version, COUNT(DISTINCT hadm_id) n_hadm
        FROM hosp.diagnoses_icd
        WHERE (icd_version=10 AND icd_code LIKE 'M32%%') OR (icd_version=9 AND icd_code LIKE '7100%%')
        GROUP BY 1 ORDER BY 1
    """, n=10)

    run("nwicu", "NWICU microbiology: blood culture availability", """
        SELECT spec_type_desc, COUNT(*) n, COUNT(org_name) n_with_org
        FROM mimiciv_hosp.microbiologyevents
        WHERE spec_type_desc ILIKE '%%blood%%'
        GROUP BY 1 ORDER BY n DESC LIMIT 12
    """, n=15)

    run("nwicu", "NWICU glucose lab items", """
        SELECT d.label, COUNT(*) n
        FROM hosp.labevents l JOIN hosp.d_labitems d ON l.itemid=d.itemid
        WHERE d.label ILIKE '%%glucose%%'
        GROUP BY 1 ORDER BY n DESC LIMIT 10
    """, n=12)

    # ---------------------------------------------------------- eICU
    run("eicu", "eICU diagnosis strings mentioning lupus", """
        SELECT diagnosisstring, icd9code, COUNT(DISTINCT patientunitstayid) n
        FROM eicu_crd.diagnosis
        WHERE diagnosisstring ILIKE '%%lupus%%' OR icd9code LIKE '7100%%'
        GROUP BY 1,2 ORDER BY n DESC LIMIT 20
    """, n=22)

    run("eicu", "eICU steroid drug names in `medication`", """
        SELECT drugname, COUNT(*) n
        FROM eicu_crd.medication
        WHERE drugname ILIKE '%%predni%%' OR drugname ILIKE '%%methylpred%%'
           OR drugname ILIKE '%%dexameth%%' OR drugname ILIKE '%%hydrocort%%'
           OR drugname ILIKE '%%solumedrol%%' OR drugname ILIKE '%%cortis%%'
           OR drugname ILIKE '%%betameth%%' OR drugname ILIKE '%%triamcin%%'
        GROUP BY 1 ORDER BY n DESC LIMIT 25
    """, n=28)

    run("eicu", "eICU steroid DOSAGE string formats (prednisone)", """
        SELECT dosage, routeadmin, frequency, COUNT(*) n
        FROM eicu_crd.medication
        WHERE drugname ILIKE '%%predni%%'
        GROUP BY 1,2,3 ORDER BY n DESC LIMIT 25
    """, n=28)

    run("eicu", "eICU infusiondrug steroid rows", """
        SELECT drugname, drugrate, infusionrate, drugamount, COUNT(*) n
        FROM eicu_crd.infusiondrug
        WHERE drugname ILIKE '%%predni%%' OR drugname ILIKE '%%methylpred%%'
           OR drugname ILIKE '%%hydrocort%%' OR drugname ILIKE '%%dexameth%%'
           OR drugname ILIKE '%%solumedrol%%'
        GROUP BY 1,2,3,4 ORDER BY n DESC LIMIT 20
    """, n=22)

    run("eicu", "eICU microlab culture sites", """
        SELECT culturesite, COUNT(*) n, COUNT(organism) n_org
        FROM eicu_crd.microlab GROUP BY 1 ORDER BY n DESC LIMIT 15
    """, n=16)

    run("eicu", "eICU glucose lab names", """
        SELECT labname, COUNT(*) n FROM eicu_crd.lab
        WHERE labname ILIKE '%%gluc%%' GROUP BY 1 ORDER BY n DESC LIMIT 10
    """, n=12)

    run("eicu", "eICU timing columns sanity (offsets, minutes)", """
        SELECT COUNT(*) n,
               MIN(hospitaladmitoffset) min_hadm_off, MAX(hospitaladmitoffset) max_hadm_off,
               MIN(unitdischargeoffset) min_dc_off, MAX(unitdischargeoffset) max_dc_off,
               AVG(unitdischargeoffset) avg_dc_off
        FROM eicu_crd.patient
    """, n=5)

    run("eicu", "eICU age field format", """
        SELECT age, COUNT(*) n FROM eicu_crd.patient GROUP BY 1 ORDER BY n DESC LIMIT 12
    """, n=14)

    run("eicu", "eICU antibiotic availability (proxy for treated infection)", """
        SELECT COUNT(DISTINCT patientunitstayid) n_stays
        FROM eicu_crd.medication
        WHERE drugname ILIKE '%%vancomycin%%' OR drugname ILIKE '%%meropenem%%'
           OR drugname ILIKE '%%piperacillin%%' OR drugname ILIKE '%%cefepime%%'
           OR drugname ILIKE '%%ceftriaxone%%'
    """, n=5)

    run("eicu", "eICU apache severity availability", """
        SELECT COUNT(*) n_aps, COUNT(acutephysiologyscore) n_score
        FROM eicu_crd.apachepatientresult
    """, n=5)


if __name__ == "__main__":
    main()
