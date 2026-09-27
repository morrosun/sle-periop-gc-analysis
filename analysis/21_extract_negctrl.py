# -*- coding: utf-8 -*-
"""
Supplementary extraction for negative controls and a narrower (treated) infection definition.

  negative outcome : upper GI bleeding (should NOT be driven by GC in this ICU cohort)
  negative exposure: ondansetron (no plausible link with infection)
  narrow infection : positive blood culture after ICU admission, or new antibiotic
                     started > 48 h after ICU admission (incident treated infection)
  baseline sepsis  : antibiotics already on board before ICU admission
"""
import psycopg2
import pandas as pd
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root (data/ and out/ live here)

CONFIG = dict(
    host=os.environ.get("MIMIC_DB_HOST", "localhost"),
    port=int(os.environ.get("MIMIC_DB_PORT", "5432")),
    user=os.environ.get("MIMIC_DB_USER", "postgres"),
    password=os.environ.get("MIMIC_DB_PASSWORD", ""),
)
DATA = os.path.join(ROOT, "data")

SQL = r"""
WITH sle AS (
    SELECT DISTINCT d.hadm_id FROM mimiciv_hosp.diagnoses_icd d
    WHERE (d.icd_version=10 AND d.icd_code LIKE 'M32%%') OR (d.icd_version=9 AND d.icd_code LIKE '7100%%')
),
icu AS (
    SELECT i.hadm_id, i.stay_id, i.intime,
           ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.intime) AS rn
    FROM mimiciv_icu.icustays i JOIN sle ON i.hadm_id = sle.hadm_id
),
lm AS (SELECT * FROM icu WHERE rn = 1),

gib AS (
    SELECT DISTINCT d.hadm_id FROM mimiciv_hosp.diagnoses_icd d JOIN lm ON d.hadm_id = lm.hadm_id
    WHERE (d.icd_version=10 AND (
            d.icd_code LIKE 'K922%%' OR d.icd_code LIKE 'K250%%' OR d.icd_code LIKE 'K252%%'
         OR d.icd_code LIKE 'K254%%' OR d.icd_code LIKE 'K256%%' OR d.icd_code LIKE 'K260%%'
         OR d.icd_code LIKE 'K262%%' OR d.icd_code LIKE 'K264%%' OR d.icd_code LIKE 'K266%%'
         OR d.icd_code LIKE 'K270%%' OR d.icd_code LIKE 'K272%%' OR d.icd_code LIKE 'K274%%'
         OR d.icd_code LIKE 'K276%%' OR d.icd_code LIKE 'K280%%' OR d.icd_code LIKE 'K282%%'
         OR d.icd_code LIKE 'K284%%' OR d.icd_code LIKE 'K286%%' OR d.icd_code LIKE 'K290%%'
         OR d.icd_code LIKE 'I850%%' OR d.icd_code LIKE 'K921%%' OR d.icd_code LIKE 'K298%%'))
       OR (d.icd_version=9 AND (
            d.icd_code LIKE '5789%%' OR d.icd_code LIKE '5310%%' OR d.icd_code LIKE '5312%%'
         OR d.icd_code LIKE '5314%%' OR d.icd_code LIKE '5316%%' OR d.icd_code LIKE '5320%%'
         OR d.icd_code LIKE '5322%%' OR d.icd_code LIKE '5324%%' OR d.icd_code LIKE '5326%%'
         OR d.icd_code LIKE '5330%%' OR d.icd_code LIKE '5332%%' OR d.icd_code LIKE '5334%%'
         OR d.icd_code LIKE '5336%%' OR d.icd_code LIKE '5340%%' OR d.icd_code LIKE '5342%%'
         OR d.icd_code LIKE '5344%%' OR d.icd_code LIKE '5346%%' OR d.icd_code LIKE '5780%%'))
),
ondan AS (
    SELECT DISTINCT p.hadm_id FROM mimiciv_hosp.prescriptions p JOIN lm ON p.hadm_id = lm.hadm_id
    WHERE p.drug ILIKE '%%ondansetron%%' OR p.drug ILIKE '%%zofran%%'
),
abx AS (
    SELECT p.hadm_id, p.starttime,
           CASE WHEN p.starttime < lm.intime THEN 1 ELSE 0 END AS before_icu,
           CASE WHEN p.starttime >= lm.intime + INTERVAL '48 hours' THEN 1 ELSE 0 END AS new_after48
    FROM mimiciv_hosp.prescriptions p JOIN lm ON p.hadm_id = lm.hadm_id
    WHERE (p.drug ILIKE '%%vancomycin%%' OR p.drug ILIKE '%%meropenem%%' OR p.drug ILIKE '%%piperacillin%%'
        OR p.drug ILIKE '%%cefepime%%' OR p.drug ILIKE '%%ceftazidime%%' OR p.drug ILIKE '%%ceftriaxone%%'
        OR p.drug ILIKE '%%levofloxacin%%' OR p.drug ILIKE '%%ciprofloxacin%%' OR p.drug ILIKE '%%linezolid%%'
        OR p.drug ILIKE '%%daptomycin%%' OR p.drug ILIKE '%%amikacin%%' OR p.drug ILIKE '%%gentamicin%%'
        OR p.drug ILIKE '%%tobramycin%%' OR p.drug ILIKE '%%azithromycin%%' OR p.drug ILIKE '%%clindamycin%%'
        OR p.drug ILIKE '%%metronidazole%%' OR p.drug ILIKE '%%fluconazole%%' OR p.drug ILIKE '%%caspofungin%%'
        OR p.drug ILIKE '%%micafungin%%' OR p.drug ILIKE '%%voriconazole%%' OR p.drug ILIKE '%%ampicillin%%'
        OR p.drug ILIKE '%%nafcillin%%' OR p.drug ILIKE '%%oxacillin%%' OR p.drug ILIKE '%%ertapenem%%'
        OR p.drug ILIKE '%%imipenem%%' OR p.drug ILIKE '%%sulfamethoxazole%%' OR p.drug ILIKE '%%doxycycline%%'
        OR p.drug ILIKE '%%cefazolin%%' OR p.drug ILIKE '%%aztreonam%%' OR p.drug ILIKE '%%moxifloxacin%%')
      AND p.starttime IS NOT NULL
),
abx_agg AS (
    SELECT hadm_id,
           MAX(before_icu)   AS abx_before_icu,
           MAX(new_after48)  AS abx_new_after48
    FROM abx GROUP BY hadm_id
),
cult AS (
    SELECT m.hadm_id,
           MAX(CASE WHEN m.charttime >= lm.intime THEN 1 ELSE 0 END) AS culture_after_icu,
           MIN(EXTRACT(EPOCH FROM (m.charttime - lm.intime)) / 3600.0) AS culture_first_hour
    FROM mimiciv_hosp.microbiologyevents m JOIN lm ON m.hadm_id = lm.hadm_id
    WHERE m.spec_type_desc ILIKE '%%BLOOD%%'
      AND m.org_name IS NOT NULL
      AND m.charttime IS NOT NULL
    GROUP BY m.hadm_id
)
SELECT
    lm.hadm_id, lm.stay_id,
    CASE WHEN g.hadm_id  IS NOT NULL THEN 1 ELSE 0 END AS gi_bleed,
    CASE WHEN o.hadm_id  IS NOT NULL THEN 1 ELSE 0 END AS ondansetron,
    COALESCE(a.abx_before_icu, 0)   AS abx_before_icu,
    COALESCE(a.abx_new_after48, 0)  AS abx_new_after48,
    COALESCE(c.culture_after_icu, 0) AS culture_after_icu,
    c.culture_first_hour
FROM lm
LEFT JOIN gib  g ON lm.hadm_id = g.hadm_id
LEFT JOIN ondan o ON lm.hadm_id = o.hadm_id
LEFT JOIN abx_agg a ON lm.hadm_id = a.hadm_id
LEFT JOIN cult   c ON lm.hadm_id = c.hadm_id
ORDER BY lm.hadm_id
"""

def main():
    c = psycopg2.connect(dbname="mimiciv", **CONFIG)
    df = pd.read_sql_query(SQL, c)
    c.close()
    print("rows:", len(df))
    p = os.path.join(DATA, "negctrl_vars.csv")
    df.to_csv(p, index=False)
    print("saved", p)
    for v in ["gi_bleed", "ondansetron", "abx_before_icu", "abx_new_after48", "culture_after_icu"]:
        print("  %-20s %4d (%.1f%%)" % (v, int(df[v].sum()), 100 * df[v].mean()))

if __name__ == "__main__":
    main()
