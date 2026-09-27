# -*- coding: utf-8 -*-
"""Diagnose which join exploded the cohort."""
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
ICU = """
WITH sle AS (%s)
SELECT i.hadm_id, i.stay_id FROM mimiciv_icu.icustays i JOIN sle ON i.hadm_id=sle.hadm_id
""" % SLE

def q(sql, label):
    c = psycopg2.connect(dbname="mimiciv", **CONFIG); cur = c.cursor()
    try:
        cur.execute(sql); rows = cur.fetchall()
        print("%-34s : %s" % (label, rows[:8]))
    except Exception as e:
        c.rollback(); print("%-34s : ERROR %s" % (label, str(e)[:200]))
    c.close()

q(ICU.replace("SELECT i.hadm_id, i.stay_id", "SELECT COUNT(DISTINCT stay_id), COUNT(DISTINCT hadm_id)"), "SLE ICU stays / hadm")

q("""
WITH sle AS (%s),
icu AS (SELECT i.hadm_id, i.stay_id FROM mimiciv_icu.icustays i JOIN sle ON i.hadm_id=sle.hadm_id)
SELECT MAX(cnt), ROUND(AVG(cnt)::numeric,2) FROM (
  SELECT icu.stay_id, COUNT(*) AS cnt FROM icu
  JOIN mimiciv_derived.sofa s ON icu.stay_id=s.stay_id GROUP BY 1) t
""" % SLE, "rows per stay: sofa")

q("""
WITH sle AS (%s),
icu AS (SELECT i.hadm_id, i.stay_id FROM mimiciv_icu.icustays i JOIN sle ON i.hadm_id=sle.hadm_id)
SELECT MAX(cnt), ROUND(AVG(cnt)::numeric,2) FROM (
  SELECT icu.stay_id, COUNT(*) AS cnt FROM icu
  JOIN mimiciv_derived.oasis s ON icu.stay_id=s.stay_id GROUP BY 1) t
""" % SLE, "rows per stay: oasis")

q("""
WITH sle AS (%s),
icu AS (SELECT i.hadm_id, i.stay_id FROM mimiciv_icu.icustays i JOIN sle ON i.hadm_id=sle.hadm_id)
SELECT MAX(cnt), ROUND(AVG(cnt)::numeric,2) FROM (
  SELECT icu.stay_id, COUNT(*) AS cnt FROM icu
  JOIN mimiciv_derived.apsiii s ON icu.stay_id=s.stay_id GROUP BY 1) t
""" % SLE, "rows per stay: apsiii")

q("""
WITH sle AS (%s),
icu AS (SELECT i.hadm_id, i.stay_id FROM mimiciv_icu.icustays i JOIN sle ON i.hadm_id=sle.hadm_id)
SELECT MAX(cnt), ROUND(AVG(cnt)::numeric,2) FROM (
  SELECT icu.stay_id, COUNT(*) AS cnt FROM icu
  JOIN mimiciv_derived.sapsii s ON icu.stay_id=s.stay_id GROUP BY 1) t
""" % SLE, "rows per stay: sapsii")

q("""
WITH sle AS (%s),
icu AS (SELECT i.hadm_id, i.stay_id FROM mimiciv_icu.icustays i JOIN sle ON i.hadm_id=sle.hadm_id)
SELECT MAX(cnt), ROUND(AVG(cnt)::numeric,2) FROM (
  SELECT icu.stay_id, COUNT(*) AS cnt FROM icu
  JOIN mimiciv_derived.first_day_lab s ON icu.stay_id=s.stay_id GROUP BY 1) t
""" % SLE, "rows per stay: first_day_lab")

q("""
WITH sle AS (%s)
SELECT MAX(cnt), ROUND(AVG(cnt)::numeric,2) FROM (
  SELECT sle.hadm_id, COUNT(*) AS cnt FROM sle
  JOIN mimiciv_derived.charlson s ON sle.hadm_id=s.hadm_id GROUP BY 1) t
""" % SLE, "rows per hadm: charlson")

q("SELECT hr, COUNT(*) FROM mimiciv_derived.sofa GROUP BY 1 ORDER BY 1 LIMIT 10", "sofa hr values")
