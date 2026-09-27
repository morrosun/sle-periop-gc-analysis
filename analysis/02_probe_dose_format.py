# -*- coding: utf-8 -*-
"""Probe: route distribution + dose string formats for GC prescriptions in SLE ICU."""
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
  WHERE (d.icd_version=10 AND d.icd_code LIKE 'M32%%') OR (d.icd_version=9 AND d.icd_code LIKE '7100%%')
""")
GC_PAT = ("p.drug ILIKE '%%predni%%' OR p.drug ILIKE '%%methylpred%%' OR p.drug ILIKE '%%dexameth%%' "
          "OR p.drug ILIKE '%%hydrocort%%' OR p.drug ILIKE '%%cortisone%%' OR p.drug ILIKE '%%betameth%%' "
          "OR p.drug ILIKE '%%triamcin%%' OR p.drug ILIKE '%%fludrocort%%'")

def q(sql, label, limit=40):
    c = psycopg2.connect(dbname="mimiciv", **CONFIG); cur = c.cursor()
    try:
        cur.execute(sql); rows = cur.fetchall()
        print("=" * 100); print("[%s]" % label)
        for r in rows[:limit]:
            print("   ", r)
        if not rows: print("    (empty)")
    except Exception as e:
        c.rollback(); print("[%s] ERROR %s" % (label, str(e)[:300]))
    c.close()

q("""
WITH sle AS (%s),
icu AS (SELECT DISTINCT i.hadm_id, i.stay_id FROM mimiciv_icu.icustays i JOIN sle ON i.hadm_id=sle.hadm_id)
SELECT p.route, COUNT(*) n FROM mimiciv_hosp.prescriptions p JOIN icu ON p.hadm_id=icu.hadm_id
WHERE %s GROUP BY 1 ORDER BY 2 DESC
""" % (SLE, GC_PAT), "SLE ICU GC prescriptions: route distribution")

q("""
WITH sle AS (%s),
icu AS (SELECT DISTINCT i.hadm_id, i.stay_id FROM mimiciv_icu.icustays i JOIN sle ON i.hadm_id=sle.hadm_id)
SELECT p.drug, p.prod_strength, p.dose_val_rx, p.dose_unit_rx, p.form_val_disp, p.form_unit_disp,
       p.doses_per_24_hrs, p.route, COUNT(*) n
FROM mimiciv_hosp.prescriptions p JOIN icu ON p.hadm_id=icu.hadm_id
WHERE %s
GROUP BY 1,2,3,4,5,6,7,8 ORDER BY 9 DESC LIMIT 40
""" % (SLE, GC_PAT), "SLE ICU GC: drug x strength x dose combos (top 40)")

q("""
WITH sle AS (%s),
icu AS (SELECT DISTINCT i.hadm_id, i.stay_id FROM mimiciv_icu.icustays i JOIN sle ON i.hadm_id=sle.hadm_id)
SELECT p.dose_unit_rx, COUNT(*) n FROM mimiciv_hosp.prescriptions p JOIN icu ON p.hadm_id=icu.hamd_id
WHERE %s GROUP BY 1 ORDER BY 2 DESC
""" % (SLE, GC_PAT), "dose units (typo guard)")

q("""
WITH sle AS (%s),
icu AS (SELECT DISTINCT i.hadm_id, i.stay_id FROM mimiciv_icu.icustays i JOIN sle ON i.hadm_id=sle.hadm_id)
SELECT p.dose_unit_rx, COUNT(*) n FROM mimiciv_hosp.prescriptions p JOIN icu ON p.hadm_id=icu.hadm_id
WHERE %s GROUP BY 1 ORDER BY 2 DESC
""" % (SLE, GC_PAT), "dose units")

# check whether starttime/stoptime overlap ICU window
q("""
WITH sle AS (%s),
icu AS (SELECT DISTINCT i.hadm_id, i.stay_id, i.intime, i.outtime FROM mimiciv_icu.icustays i JOIN sle ON i.hadm_id=sle.hadm_id)
SELECT
  SUM(CASE WHEN p.starttime IS NULL THEN 1 ELSE 0 END) AS n_null_start,
  SUM(CASE WHEN p.stoptime IS NULL THEN 1 ELSE 0 END) AS n_null_stop,
  COUNT(*) AS n_total
FROM mimiciv_hosp.prescriptions p JOIN icu ON p.hadm_id=icu.hadm_id
WHERE %s
""" % (SLE, GC_PAT), "prescriptions time completeness in SLE ICU")

# mimiciv_derived availability
q("SELECT table_schema, table_name FROM information_schema.tables "
  "WHERE table_schema='mimiciv_derived' AND (table_name ILIKE '%%sofa%%' OR table_name ILIKE '%%aps%%' "
  "OR table_name ILIKE '%%oasis%%' OR table_name ILIKE '%%charlson%%' OR table_name ILIKE '%%saps%%' "
  "OR table_name ILIKE '%%vent%%' OR table_name ILIKE '%%rrt%%' OR table_name ILIKE '%%vaso%%' "
  "OR table_name ILIKE '%%sepsis%%' OR table_name ILIKE '%%first%%' OR table_name ILIKE '%%weight%%') "
  "ORDER BY 1,2", "mimiciv_derived helper tables")
