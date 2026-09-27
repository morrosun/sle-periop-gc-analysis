# -*- coding: utf-8 -*-
"""
Probe: what SLE disease-activity proxies are actually available in MIMIC-IV?
  - lab: complement C3/C4, anti-dsDNA, ESR, CRP, urine protein, Hb, ANA
  - diagnoses: organ manifestations (nephritis / CNS / serositis / APS / cytopenia)
  - prior admissions
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

Q_LAB = r"""
SELECT d.label, d.fluid, COUNT(*) AS n,
       COUNT(DISTINCT l.subject_id) AS n_subj
FROM mimiciv_hosp.labevents l
JOIN mimiciv_hosp.d_labitems d ON l.itemid = d.itemid
WHERE d.label ILIKE ANY (ARRAY[
    '%%complement%%','%%c3%%','%%c4%%','%%dsdna%%','%%ds-dna%%','%%anti-dna%%',
    '%%sedimentation%%','%%esr%%','%%c-reactive%%','%%crp%%',
    '%%protein%%','%%albumin%%','%%hemoglobin%%','%%creatinine%%',
    '%%antinuclear%%','%%ana%%','%%urine%%'
])
GROUP BY d.label, d.fluid
ORDER BY n DESC
LIMIT 120
"""

Q_DX = r"""
WITH sle AS (
  SELECT DISTINCT d.hadm_id FROM mimiciv_hosp.diagnoses_icd d
  WHERE (d.icd_version=10 AND d.icd_code LIKE 'M32%%')
     OR (d.icd_version=9  AND d.icd_code LIKE '7100%%')
)
SELECT d.icd_version, d.icd_code, COUNT(*) AS n,
       MAX(CASE WHEN d.icd_version=10 THEN NULL ELSE NULL END) AS x
FROM mimiciv_hosp.diagnoses_icd d JOIN sle ON d.hadm_id = sle.hadm_id
WHERE (d.icd_version=10 AND (
        d.icd_code LIKE 'M32%%'      -- SLE & subentities
     OR d.icd_code LIKE 'N085%'      -- lupus nephritis (M32.14 also)
     OR d.icd_code LIKE 'D68%%'      -- APS-ish / coagulopathy
     OR d.icd_code LIKE 'D86%%'
     OR d.icd_code LIKE 'M541%%'
     OR d.icd_code LIKE 'G%%'        -- neuro
  ))
GROUP BY d.icd_version, d.icd_code
ORDER BY n DESC
LIMIT 60
"""

Q_SLE_SUB = r"""
WITH sle AS (
  SELECT DISTINCT d.hadm_id FROM mimiciv_hosp.diagnoses_icd d
  WHERE (d.icd_version=10 AND d.icd_code LIKE 'M32%%')
     OR (d.icd_version=9  AND d.icd_code LIKE '7100%%')
)
SELECT d.icd_version, d.icd_code,
       COUNT(*) AS n
FROM mimiciv_hosp.diagnoses_icd d JOIN sle ON d.hadm_id = sle.hadm_id
WHERE (d.icd_version=10 AND d.icd_code LIKE 'M32%')
   OR (d.icd_version=9  AND d.icd_code LIKE '710%')
GROUP BY d.icd_version, d.icd_code
ORDER BY n DESC
LIMIT 40
"""

Q_PREADM = r"""
WITH sle AS (
  SELECT DISTINCT d.hadm_id FROM mimiciv_hosp.diagnoses_icd d
  WHERE (d.icd_version=10 AND d.icd_code LIKE 'M32%')
     OR (d.icd_version=9  AND d.icd_code LIKE '7100%')
)
SELECT COUNT(*) AS n_admit, COUNT(DISTINCT a.subject_id) AS n_subj,
       AVG(cnt) AS mean_adm_per_subj
FROM (
  SELECT a.subject_id, a.hadm_id,
         COUNT(*) OVER (PARTITION BY a.subject_id) AS cnt
  FROM mimiciv_hosp.admissions a
  WHERE a.subject_id IN (SELECT DISTINCT i.subject_id FROM mimiciv_icu.icustays i JOIN sle ON i.hadm_id=sle.hadm_id)
) a
"""

def show(cur, title, sql, limit=40):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    print(" | ".join(cols))
    for r in rows[:limit]:
        print(" | ".join(str(x) for x in r))
    print("(total rows: %d)" % len(rows))

def main():
    c = psycopg2.connect(dbname="mimiciv", **CONFIG)
    cur = c.cursor()
    show(cur, "1. LAB ITEMS relevant to SLE activity", Q_LAB, 120)
    show(cur, "2. SLE SUB-ENTITY ICD CODES (organ manifestation)", Q_SLE_SUB, 40)
    cur.execute(Q_PREADM); print("\n3. admission counts\n", cur.fetchall())
    c.close()

if __name__ == "__main__":
    main()
