# -*- coding: utf-8 -*-
"""Probe: where are glucocorticoid records for SLE ICU patients, and what do they look like?"""
import os
import psycopg2

CONFIG = dict(
    host=os.environ.get("MIMIC_DB_HOST", "localhost"),
    port=int(os.environ.get("MIMIC_DB_PORT", "5432")),
    user=os.environ.get("MIMIC_DB_USER", "postgres"),
    password=os.environ.get("MIMIC_DB_PASSWORD", ""),
)

SLE = ("""
  SELECT DISTINCT d.hadm_id FROM mimiciv_hosp.diagnoses_icd d
  WHERE (d.icd_version=10 AND d.icd_code LIKE 'M32%%')
     OR (d.icd_version=9  AND d.icd_code LIKE '7100%%')
""")

GC_PAT = ("drug ILIKE '%%predni%%' OR drug ILIKE '%%methylpred%%' OR drug ILIKE '%%dexameth%%' "
          "OR drug ILIKE '%%hydrocort%%' OR drug ILIKE '%%cortisone%%' OR drug ILIKE '%%betameth%%' "
          "OR drug ILIKE '%%triamcin%%' OR drug ILIKE '%%fludrocort%%'")

def q(sql, label, limit=25):
    c = psycopg2.connect(dbname="mimiciv", **CONFIG); cur = c.cursor()
    try:
        cur.execute(sql); rows = cur.fetchall()
        print("=" * 100)
        print("[%s]" % label)
        for r in rows[:limit]:
            print("   ", r)
        if not rows:
            print("    (empty)")
    except Exception as e:
        c.rollback(); print("[%s] ERROR %s" % (label, str(e)[:300]))
    c.close()

# 1. top GC drug names in prescriptions among SLE ICU cohort
q("""
WITH sle AS (%s),
icu AS (SELECT DISTINCT i.hadm_id, i.stay_id, i.subject_id, i.intime, i.outtime
         FROM mimiciv_icu.icustays i JOIN sle ON i.hadm_id=sle.hadm_id)
SELECT p.drug, COUNT(*) n FROM mimiciv_hosp.prescriptions p
JOIN icu ON p.hadm_id=icu.hadm_id
WHERE %s
GROUP BY 1 ORDER BY 2 DESC LIMIT 30
""" % (SLE, GC_PAT), "SLE ICU: top GC drug names (prescriptions)")

# 2. does prescriptions have starttime/endtime + dose?
q("SELECT column_name, data_type FROM information_schema.columns "
  "WHERE table_schema='mimiciv_hosp' AND table_name='prescriptions' ORDER BY ordinal_position",
  "prescriptions columns")

# 3. inputevents (ICU) GC
q("SELECT table_name FROM information_schema.tables WHERE table_schema='mimiciv_icu' "
  "AND table_name LIKE 'input%%' ORDER BY 1", "mimiciv_icu input tables")

q("SELECT column_name, data_type FROM information_schema.columns "
  "WHERE table_schema='mimiciv_icu' AND table_name='inputevents' ORDER BY ordinal_position",
  "inputevents columns")

# 4. emar (ICU medication administration) - has dosing schedule
q("SELECT column_name, data_type FROM information_schema.columns "
  "WHERE table_schema='mimiciv_hosp' AND table_name='emar' ORDER BY ordinal_position",
  "emar columns")

q("""
SELECT column_name, data_type FROM information_schema.columns
WHERE table_schema='mimiciv_hosp' AND table_name='emar_detail' ORDER BY ordinal_position
""", "emar_detail columns")

# 5. which route/forms: check inputevents itemid for GC
q("""
SELECT itemid, label, COUNT(*) n FROM mimiciv_icu.inputevents
WHERE label ILIKE '%%predni%%' OR label ILIKE '%%methylpred%%' OR label ILIKE '%%dexameth%%'
   OR label ILIKE '%%hydrocort%%' OR label ILIKE '%%solumedrol%%' OR label ILIKE '%%decadron%%'
GROUP BY 1,2 ORDER BY 3 DESC LIMIT 30
""", "inputevents GC itemid")
