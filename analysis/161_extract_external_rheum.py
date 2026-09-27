# -*- coding: utf-8 -*-
"""
161_extract_external_rheum.py

Build the SLE + RA cohorts in eICU-CRD and NWICU on the harmonised
variable set, so the head-to-head from 152/153 can be tested outside
MIMIC-IV.

Disease definition matches MIMIC-IV, with SLE taking priority over RA so
a patient coded with both lands in one group only:
    SLE   ICD-10 M32*  / ICD-9 710.0*  / eICU diagnosisstring 'lupus'
    RA    ICD-10 M05*,M06* / ICD-9 714* / eICU diagnosisstring 'rheumatoid'

Everything else is the transportable construct already validated in v3:
    landmark  ICU LOS >= 24 h, first ICU stay, age >= 18
    exposure  GC dose intensity in the first 24 h, prednisone-equivalent
              mg/day, strata 0 / >0-<10 / 10-<50 / >=50
    events    blood culture positive > 24 h (eICU only -- NWICU has no
              microbiology table), new antibiotic > 48 h, infection ICD,
              hospital death
    control   glucose >= 180 mg/dL within 48 h

eICU caveats handled here (all discovered by probing)
    * all timing is minutes offset from unit admission
    * `medication.dosage` is free text; `frequency` needs its own mapping
    * `drugstopoffset` is frequently < `drugstartoffset`
Outputs: data/cohort_eicu_rheum.csv, data/cohort_nwicu_rheum.csv
"""
import os
import re

import numpy as np
import pandas as pd
import psycopg2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root (data/ and out/ live here)

CONFIG = dict(
    host=os.environ.get("MIMIC_DB_HOST", "localhost"),
    port=int(os.environ.get("MIMIC_DB_PORT", "5432")),
    user=os.environ.get("MIMIC_DB_USER", "postgres"),
    password=os.environ.get("MIMIC_DB_PASSWORD", ""),
)
DATA = os.path.join(ROOT, "data")

# ============================================================== eICU cohort
EICU_SQL = r"""
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
rheum AS (
    SELECT patientunitstayid,
           CASE WHEN f_sle=1 THEN 'SLE' ELSE 'RA' END AS primary_grp
    FROM cls WHERE f_sle=1 OR f_ra=1
),
p0 AS (
    SELECT pt.patientunitstayid, pt.patienthealthsystemstayid, pt.gender, pt.age,
           pt.hospitaladmitoffset, pt.unitdischargeoffset,
           pt.hospitaldischargestatus, pt.unitdischargestatus,
           r.primary_grp,
           ROW_NUMBER() OVER (PARTITION BY pt.patienthealthsystemstayid
                              ORDER BY pt.hospitaladmitoffset) AS rn
    FROM eicu_crd.patient pt JOIN rheum r ON pt.patientunitstayid = r.patientunitstayid
),
p AS (SELECT * FROM p0 WHERE rn = 1),
dx2 AS (
    SELECT d.patientunitstayid,
      MAX(CASE WHEN TRIM(BOTH ' ' FROM u.code) LIKE '584%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '585%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '586%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'N17%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'N18%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'N19%%' THEN 1 ELSE 0 END) AS renal_fail,
      MAX(CASE WHEN TRIM(BOTH ' ' FROM u.code) LIKE '578%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'K922%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'K25%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'K26%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'K27%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'K28%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'I850%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'K921%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'K298%%' THEN 1 ELSE 0 END) AS gi_bleed,
      MAX(CASE WHEN TRIM(BOTH ' ' FROM u.code) LIKE 'J1%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'A40%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'A41%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'A49%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'K65%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'N10%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'N39%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'R65%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'L02%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'L03%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'L08%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'M86%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'T81%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'T82%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'T84%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '038%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '481%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '482%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '483%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '484%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '485%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '486%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '507%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '510%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '513%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '540%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '567%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '590%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '599%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '680%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '681%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '682%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '683%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '684%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '685%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '686%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '711%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '730%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '998%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '9959%%' THEN 1 ELSE 0 END) AS infect_icd,
      MAX(CASE WHEN TRIM(BOTH ' ' FROM u.code) LIKE 'J96%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'J80%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '5185%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '5188%%' THEN 1 ELSE 0 END) AS resp_fail,
      MAX(CASE WHEN TRIM(BOTH ' ' FROM u.code) LIKE 'R57%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '7855%%' THEN 1 ELSE 0 END) AS shock,
      MAX(CASE WHEN TRIM(BOTH ' ' FROM u.code) LIKE 'K72%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'K70%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '570%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '572%%' THEN 1 ELSE 0 END) AS liver_fail,
      MAX(CASE WHEN TRIM(BOTH ' ' FROM u.code) LIKE 'D65%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE 'D68%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '2866%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '2867%%'
               OR TRIM(BOTH ' ' FROM u.code) LIKE '2869%%' THEN 1 ELSE 0 END) AS coagulop
    FROM eicu_crd.diagnosis d,
         UNNEST(STRING_TO_ARRAY(COALESCE(d.icd9code,''), ',')) AS u(code)
    WHERE d.patientunitstayid IN (SELECT patientunitstayid FROM p)
    GROUP BY d.patientunitstayid
),
abx AS (
    SELECT m.patientunitstayid, MIN(m.drugstartoffset) AS abx_first_off
    FROM eicu_crd.medication m JOIN p ON m.patientunitstayid = p.patientunitstayid
    WHERE (m.drugname ILIKE '%%vancomycin%%' OR m.drugname ILIKE '%%meropenem%%'
        OR m.drugname ILIKE '%%piperacillin%%' OR m.drugname ILIKE '%%cefepime%%'
        OR m.drugname ILIKE '%%ceftazidime%%' OR m.drugname ILIKE '%%ceftriaxone%%'
        OR m.drugname ILIKE '%%levofloxacin%%' OR m.drugname ILIKE '%%ciprofloxacin%%'
        OR m.drugname ILIKE '%%linezolid%%' OR m.drugname ILIKE '%%daptomycin%%'
        OR m.drugname ILIKE '%%amikacin%%' OR m.drugname ILIKE '%%gentamicin%%'
        OR m.drugname ILIKE '%%tobramycin%%' OR m.drugname ILIKE '%%azithromycin%%'
        OR m.drugname ILIKE '%%clindamycin%%' OR m.drugname ILIKE '%%metronidazole%%'
        OR m.drugname ILIKE '%%fluconazole%%' OR m.drugname ILIKE '%%caspofungin%%'
        OR m.drugname ILIKE '%%micafungin%%' OR m.drugname ILIKE '%%voriconazole%%'
        OR m.drugname ILIKE '%%ampicillin%%' OR m.drugname ILIKE '%%nafcillin%%'
        OR m.drugname ILIKE '%%oxacillin%%' OR m.drugname ILIKE '%%ertapenem%%'
        OR m.drugname ILIKE '%%imipenem%%' OR m.drugname ILIKE '%%sulfamethoxazole%%'
        OR m.drugname ILIKE '%%doxycycline%%' OR m.drugname ILIKE '%%cefazolin%%'
        OR m.drugname ILIKE '%%aztreonam%%' OR m.drugname ILIKE '%%moxifloxacin%%')
      AND m.drugstartoffset IS NOT NULL
      AND COALESCE(m.drugordercancelled,'No') <> 'Yes'
    GROUP BY m.patientunitstayid
),
cult AS (
    SELECT m.patientunitstayid,
           COUNT(*) AS n_culture_any,
           MAX(CASE WHEN m.culturetakenoffset >= 1440 THEN 1 ELSE 0 END) AS culture_after24,
           MAX(CASE WHEN m.culturetakenoffset >= 1440
                     AND m.organism IS NOT NULL
                     AND LOWER(m.organism) NOT LIKE 'no growth%%'
                     AND LOWER(m.organism) NOT IN ('other','') THEN 1 ELSE 0 END) AS culture_pos_after24
    FROM eicu_crd.microlab m JOIN p ON m.patientunitstayid = p.patientunitstayid
    WHERE m.culturesite ILIKE '%%blood%%' AND m.culturetakenoffset IS NOT NULL
    GROUP BY m.patientunitstayid
),
glu AS (
    SELECT l.patientunitstayid, MAX(l.labresult) AS glu_max_48h
    FROM eicu_crd.lab l JOIN p ON l.patientunitstayid = p.patientunitstayid
    WHERE l.labname IN ('glucose','bedside glucose') AND l.labresult IS NOT NULL
      AND l.labresultoffset >= 0 AND l.labresultoffset < 2880
      AND l.labresult BETWEEN 10 AND 1500
    GROUP BY l.patientunitstayid
),
vaso AS (
    SELECT DISTINCT i.patientunitstayid FROM eicu_crd.infusiondrug i
    JOIN p ON i.patientunitstayid = p.patientunitstayid
    WHERE (i.drugname ILIKE '%%norepinephrine%%' OR i.drugname ILIKE '%%levophed%%'
        OR i.drugname ILIKE '%%epinephrine%%' OR i.drugname ILIKE '%%dopamine%%'
        OR i.drugname ILIKE '%%phenylephrine%%' OR i.drugname ILIKE '%%vasopressin%%'
        OR i.drugname ILIKE '%%dobutamine%%')
      AND i.infusionoffset >= 0 AND i.infusionoffset < 1440
),
vent AS (
    SELECT DISTINCT t.patientunitstayid FROM eicu_crd.treatment t
    JOIN p ON t.patientunitstayid = p.patientunitstayid
    WHERE t.treatmentstring ILIKE '%%mechanical ventilation%%' AND t.treatmentoffset < 1440
),
aps AS (
    SELECT a.patientunitstayid, MAX(a.acutephysiologyscore) AS acutephysiologyscore
    FROM eicu_crd.apachepatientresult a JOIN p ON a.patientunitstayid = p.patientunitstayid
    GROUP BY a.patientunitstayid
)
SELECT p.patientunitstayid AS stay_key, p.primary_grp, p.age AS age_raw, p.gender,
       p.unitdischargeoffset AS icu_los_min, p.hospitaladmitoffset AS preicu_min,
       p.hospitaldischargestatus,
       COALESCE(d.renal_fail,0) AS renal_fail, COALESCE(d.gi_bleed,0) AS gi_bleed,
       COALESCE(d.infect_icd,0) AS infect_icd, COALESCE(d.resp_fail,0) AS resp_fail,
       COALESCE(d.shock,0) AS shock, COALESCE(d.liver_fail,0) AS liver_fail,
       COALESCE(d.coagulop,0) AS coagulop,
       a.abx_first_off,
       COALESCE(c.n_culture_any,0) AS n_culture_any,
       COALESCE(c.culture_after24,0) AS culture_after24,
       COALESCE(c.culture_pos_after24,0) AS culture_pos_after24,
       g.glu_max_48h,
       CASE WHEN v.patientunitstayid IS NOT NULL THEN 1 ELSE 0 END AS vaso24,
       CASE WHEN ve.patientunitstayid IS NOT NULL THEN 1 ELSE 0 END AS vent24,
       s.acutephysiologyscore AS apache_aps
FROM p
LEFT JOIN dx2 d ON p.patientunitstayid = d.patientunitstayid
LEFT JOIN abx a ON p.patientunitstayid = a.patientunitstayid
LEFT JOIN cult c ON p.patientunitstayid = c.patientunitstayid
LEFT JOIN glu g ON p.patientunitstayid = g.patientunitstayid
LEFT JOIN vaso v ON p.patientunitstayid = v.patientunitstayid
LEFT JOIN vent ve ON p.patientunitstayid = ve.patientunitstayid
LEFT JOIN aps s ON p.patientunitstayid = s.patientunitstayid
ORDER BY p.patientunitstayid
"""

EICU_GC_SQL = r"""
SELECT m.patientunitstayid AS stay_key, m.drugname, m.dosage, m.frequency,
       m.routeadmin, m.drugstartoffset, m.drugstopoffset
FROM eicu_crd.medication m
WHERE m.patientunitstayid IN (
    SELECT DISTINCT d.patientunitstayid FROM eicu_crd.diagnosis d
    WHERE d.diagnosisstring ILIKE '%%lupus%%'
       OR d.diagnosisstring ILIKE '%%rheumatoid%%'
       OR d.icd9code LIKE '%%7100%%' OR d.icd9code LIKE '%%M32%%'
       OR d.icd9code LIKE '%%714%%' OR d.icd9code LIKE '%%M05%%'
       OR d.icd9code LIKE '%%M06%%')
  AND (m.drugname ILIKE '%%predni%%' OR m.drugname ILIKE '%%methylpred%%'
    OR m.drugname ILIKE '%%solumedrol%%' OR m.drugname ILIKE '%%dexameth%%'
    OR m.drugname ILIKE '%%hydrocort%%' OR m.drugname ILIKE '%%cortis%%'
    OR m.drugname ILIKE '%%betameth%%' OR m.drugname ILIKE '%%triamcin%%'
    OR m.drugname ILIKE '%%fludrocort%%')
"""

# ============================================================= NWICU cohort
NWICU_SQL = r"""
WITH cls AS (
    SELECT hadm_id,
      MAX(CASE WHEN (icd_version=10 AND icd_code LIKE 'M32%%')
                OR (icd_version=9 AND icd_code LIKE '7100%%') THEN 1 ELSE 0 END) AS f_sle,
      MAX(CASE WHEN (icd_version=10 AND (icd_code LIKE 'M05%%' OR icd_code LIKE 'M06%%'))
                OR (icd_version=9 AND icd_code LIKE '714%%') THEN 1 ELSE 0 END) AS f_ra,
      MAX(CASE WHEN (icd_version=10 AND LEFT(icd_code,3) IN ('N17','N18','N19'))
                OR (icd_version=9 AND LEFT(icd_code,3) IN ('584','585','586')) THEN 1 ELSE 0 END) AS renal_fail,
      MAX(CASE WHEN (icd_version=10 AND (icd_code LIKE 'K922%%' OR icd_code LIKE 'K25%%'
                OR icd_code LIKE 'K26%%' OR icd_code LIKE 'K27%%' OR icd_code LIKE 'K28%%'
                OR icd_code LIKE 'I850%%' OR icd_code LIKE 'K921%%' OR icd_code LIKE 'K298%%'))
                OR (icd_version=9 AND (icd_code LIKE '578%%' OR LEFT(icd_code,4) IN
                ('5310','5312','5314','5316','5320','5322','5324','5326','5330','5332',
                 '5334','5336','5340','5342','5344','5346'))) THEN 1 ELSE 0 END) AS gi_bleed,
      MAX(CASE WHEN (icd_version=10 AND (LEFT(icd_code,2)='J1'
                OR LEFT(icd_code,3) IN ('A40','A41','A49','J85','K61','K65','L02','L03','L08',
                                        'M86','N10','N39','R65','T81','T82','T84')))
                OR (icd_version=9 AND (LEFT(icd_code,3) IN ('038','320','322','324','420','421',
                '481','482','483','484','485','486','507','510','513','540','567','590','599',
                '680','681','682','683','684','685','686','711','730','998')
                OR icd_code LIKE '9959%%')) THEN 1 ELSE 0 END) AS infect_icd,
      MAX(CASE WHEN (icd_version=10 AND LEFT(icd_code,3) IN ('J96','J80'))
                OR (icd_version=9 AND LEFT(icd_code,4) IN ('5185','5188')) THEN 1 ELSE 0 END) AS resp_fail,
      MAX(CASE WHEN (icd_version=10 AND LEFT(icd_code,3)='R57')
                OR (icd_version=9 AND LEFT(icd_code,4)='7855') THEN 1 ELSE 0 END) AS shock,
      MAX(CASE WHEN (icd_version=10 AND LEFT(icd_code,3) IN ('K72','K70'))
                OR (icd_version=9 AND LEFT(icd_code,3) IN ('570','572')) THEN 1 ELSE 0 END) AS liver_fail,
      MAX(CASE WHEN (icd_version=10 AND LEFT(icd_code,3) IN ('D65','D68'))
                OR (icd_version=9 AND LEFT(icd_code,4) IN ('2866','2867','2869')) THEN 1 ELSE 0 END) AS coagulop
    FROM hosp.diagnoses_icd GROUP BY hadm_id
),
p0 AS (
    SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.los,
           a.admittime, a.hospital_expire_flag,
           CASE WHEN c.f_sle=1 THEN 'SLE' ELSE 'RA' END AS primary_grp,
           ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.intime) AS rn
    FROM cls c
    JOIN icu.icustays i ON c.hadm_id = i.hadm_id
    JOIN hosp.admissions a ON c.hadm_id = a.hadm_id
    WHERE c.f_sle=1 OR c.f_ra=1
),
p AS (SELECT * FROM p0 WHERE rn=1),
abx AS (
    SELECT pr.hadm_id, MIN(pr.starttime) AS abx_first_time
    FROM hosp.prescriptions pr JOIN p ON pr.hadm_id = p.hadm_id
    WHERE (pr.drug ILIKE '%%vancomycin%%' OR pr.drug ILIKE '%%meropenem%%'
        OR pr.drug ILIKE '%%piperacillin%%' OR pr.drug ILIKE '%%cefepime%%'
        OR pr.drug ILIKE '%%ceftazidime%%' OR pr.drug ILIKE '%%ceftriaxone%%'
        OR pr.drug ILIKE '%%levofloxacin%%' OR pr.drug ILIKE '%%ciprofloxacin%%'
        OR pr.drug ILIKE '%%linezolid%%' OR pr.drug ILIKE '%%daptomycin%%'
        OR pr.drug ILIKE '%%amikacin%%' OR pr.drug ILIKE '%%gentamicin%%'
        OR pr.drug ILIKE '%%tobramycin%%' OR pr.drug ILIKE '%%azithromycin%%'
        OR pr.drug ILIKE '%%clindamycin%%' OR pr.drug ILIKE '%%metronidazole%%'
        OR pr.drug ILIKE '%%fluconazole%%' OR pr.drug ILIKE '%%caspofungin%%'
        OR pr.drug ILIKE '%%micafungin%%' OR pr.drug ILIKE '%%voriconazole%%'
        OR pr.drug ILIKE '%%ampicillin%%' OR pr.drug ILIKE '%%nafcillin%%'
        OR pr.drug ILIKE '%%oxacillin%%' OR pr.drug ILIKE '%%ertapenem%%'
        OR pr.drug ILIKE '%%imipenem%%' OR pr.drug ILIKE '%%sulfamethoxazole%%'
        OR pr.drug ILIKE '%%doxycycline%%' OR pr.drug ILIKE '%%cefazolin%%'
        OR pr.drug ILIKE '%%aztreonam%%' OR pr.drug ILIKE '%%moxifloxacin%%')
      AND pr.starttime IS NOT NULL
    GROUP BY pr.hadm_id
),
glu AS (
    SELECT l.hadm_id, MAX(l.valuenum) AS glu_max_48h
    FROM hosp.labevents l JOIN hosp.d_labitems d ON l.itemid = d.itemid
    JOIN p ON l.hadm_id = p.hadm_id
    WHERE d.label = 'Glucose' AND l.valuenum IS NOT NULL
      AND l.charttime >= p.intime AND l.charttime < p.intime + INTERVAL '48 hours'
      AND l.valuenum BETWEEN 10 AND 1500
    GROUP BY l.hadm_id
),
vaso AS (
    SELECT DISTINCT pr.hadm_id FROM hosp.prescriptions pr JOIN p ON pr.hadm_id = p.hadm_id
    WHERE (pr.drug ILIKE '%%norepinephrine%%' OR pr.drug ILIKE '%%levophed%%'
        OR pr.drug ILIKE '%%epinephrine%%' OR pr.drug ILIKE '%%dopamine%%'
        OR pr.drug ILIKE '%%phenylephrine%%' OR pr.drug ILIKE '%%vasopressin%%'
        OR pr.drug ILIKE '%%dobutamine%%')
      AND pr.stoptime > p.intime AND pr.starttime < p.intime + INTERVAL '24 hours'
),
vent AS (
    SELECT DISTINCT pe.stay_id FROM icu.procedureevents pe
    JOIN icu.d_items di ON pe.itemid = di.itemid JOIN p ON pe.stay_id = p.stay_id
    WHERE di.label ILIKE '%%ventilation%%'
      AND pe.starttime >= p.intime AND pe.starttime < p.intime + INTERVAL '24 hours'
)
SELECT p.subject_id, p.hadm_id AS stay_key, p.stay_id, p.intime, p.primary_grp,
       p.los AS icu_los_days, p.hospital_expire_flag, pt.gender, pt.anchor_age AS age,
       EXTRACT(EPOCH FROM (p.intime - p.admittime))/3600.0 AS preicu_hours,
       COALESCE(c.renal_fail,0) AS renal_fail, COALESCE(c.gi_bleed,0) AS gi_bleed,
       COALESCE(c.infect_icd,0) AS infect_icd, COALESCE(c.resp_fail,0) AS resp_fail,
       COALESCE(c.shock,0) AS shock, COALESCE(c.liver_fail,0) AS liver_fail,
       COALESCE(c.coagulop,0) AS coagulop,
       a.abx_first_time, g.glu_max_48h,
       CASE WHEN v.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS vaso24,
       CASE WHEN ve.stay_id IS NOT NULL THEN 1 ELSE 0 END AS vent24
FROM p
JOIN hosp.patients pt ON p.subject_id = pt.subject_id
LEFT JOIN cls c ON p.hadm_id = c.hadm_id
LEFT JOIN abx a ON p.hadm_id = a.hadm_id
LEFT JOIN glu g ON p.hadm_id = g.hadm_id
LEFT JOIN vaso v ON p.hadm_id = v.hadm_id
LEFT JOIN vent ve ON p.stay_id = ve.stay_id
ORDER BY p.hadm_id
"""

NWICU_GC_SQL = r"""
SELECT pr.hadm_id AS stay_key, pr.drug, pr.dose_val_rx AS dose_txt,
       COALESCE(pr.doses_per_24_hrs, 1.0) AS freq24, pr.route,
       pr.starttime, pr.stoptime
FROM hosp.prescriptions pr
WHERE pr.hadm_id IN (
    SELECT DISTINCT hadm_id FROM hosp.diagnoses_icd
    WHERE (icd_version=10 AND (icd_code LIKE 'M32%%' OR icd_code LIKE 'M05%%'
                              OR icd_code LIKE 'M06%%'))
       OR (icd_version=9 AND (icd_code LIKE '7100%%' OR icd_code LIKE '714%%')))
  AND (pr.drug ILIKE '%%predni%%' OR pr.drug ILIKE '%%methylpred%%'
    OR pr.drug ILIKE '%%dexameth%%' OR pr.drug ILIKE '%%hydrocort%%'
    OR pr.drug ILIKE '%%betameth%%' OR pr.drug ILIKE '%%triamcin%%')
  AND pr.dose_unit_rx = 'mg'
  AND pr.dose_val_rx ~ '^[0-9]+(\.[0-9]+)?$'
  AND pr.starttime IS NOT NULL AND pr.stoptime IS NOT NULL
"""

# ============================================================ shared pieces
PE = [(r"methylpred|solumedrol|medrol", 1.25), (r"prednisolone", 1.0),
      (r"prednisone|predni(?!solone)", 1.0), (r"hydrocort|cortef", 0.25),
      (r"dexameth", 6.67), (r"betameth", 6.67), (r"triamcin", 1.25),
      (r"fludrocort", 0.25)]
ONCE = re.compile(r"(once|one|1\s*x\s*only|1xonly|x\s*1|single dose)", re.I)
FREQ_MAP = [(re.compile(r"\bqid\b", re.I), 4.0), (re.compile(r"\btid\b", re.I), 3.0),
            (re.compile(r"\bbid\b", re.I), 2.0),
            (re.compile(r"daily|every day|q\s*d\b|each day", re.I), 1.0)]


def pe_factor(name):
    n = (name or "").lower()
    for pat, f in PE:
        if re.search(pat, n):
            return f
    return None


def parse_freq(s):
    if s is None:
        return 1.0, False
    t = s.strip()
    if ONCE.search(t) and not re.search(r"daily|q\s*\d+\s*h|\bbid\b|\btid\b|\bqid\b", t, re.I):
        return 1.0, True
    m = re.search(r"q\s*(\d+)\s*h", t, re.I)
    if m:
        h = float(m.group(1))
        if h > 0:
            return 24.0 / h, False
    for pat, v in FREQ_MAP:
        if pat.search(t):
            return v, False
    return 1.0, False


def parse_dose(dosage, drugname):
    d = (dosage or "").strip()
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(mg|mcg|microgram|gram|g|ea|each|tab|tablet|ml|unit|unt)?",
                 d, re.I)
    if m:
        val = float(m.group(1))
        unit = (m.group(2) or "").lower()
        if unit in ("mg", ""):
            return val
        if unit in ("g", "gram"):
            return val * 1000.0
        if unit in ("mcg", "microgram"):
            return val / 1000.0
    m2 = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*MG\b", drugname or "", re.I)
    if m2:
        m3 = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)", d)
        mult = float(m3.group(1)) if m3 else 1.0
        if re.search(r"\bEA\b|\btab|tablet", d, re.I) or not m3:
            return float(m2.group(1)) * mult
        return float(m2.group(1))
    return None


def stratum(v):
    if v <= 0:
        return "G0_none"
    if v < 10:
        return "G1_low"
    if v < 50:
        return "G2_mod"
    return "G3_high"


def finalise(lm, abx_late_ok=True):
    lm["gc24_daily_pe_mg"] = lm.gc24_daily_pe_mg.fillna(0.0)
    lm["gc24_peak_daily_pe_mg"] = lm.gc24_peak_daily_pe_mg.fillna(0.0)
    lm["gc_any24"] = (lm.gc24_daily_pe_mg > 0).astype(int)
    lm["gc24_str"] = lm.gc24_daily_pe_mg.apply(stratum)
    lm["gc_str_num"] = lm.gc24_str.map({"G0_none": 0, "G1_low": 1, "G2_mod": 2, "G3_high": 3})
    lm["hyper_48h"] = np.where(lm.glu_max_48h >= 180, 1,
                               np.where(lm.glu_max_48h.isna(), np.nan, 0))
    lm["female"] = (lm.gender.astype(str).str.upper() == "F").astype(int)
    lm["organ_dysf"] = (lm.renal_fail + lm.resp_fail + lm.shock
                        + lm.liver_fail + lm.coagulop)
    return lm


def main():
    P = print

    # ------------------------------------------------------------- eICU
    c = psycopg2.connect(dbname="eicu", **CONFIG)
    de = pd.read_sql_query(EICU_SQL, c)
    ge = pd.read_sql_query(EICU_GC_SQL, c)
    c.close()
    P("=== eICU-CRD ===")
    P("  first-ICU stays (SLE+RA) : %d  %s" % (len(de), de.primary_grp.value_counts().to_dict()))
    P("  steroid order rows       : %d" % len(ge))
    de["age"] = pd.to_numeric(de.age_raw.replace("> 89", "90"), errors="coerce")
    de = de[de.age >= 18].copy()
    de["icu_los_days"] = de.icu_los_min / 1440.0
    lm = de[de.icu_los_min >= 1440].copy()
    P("  landmark (LOS >= 24 h)   : %d" % len(lm))
    rows = []
    for r in ge.itertuples(index=False):
        f = pe_factor(r.drugname)
        if f is None or r.drugstartoffset is None:
            continue
        start = float(r.drugstartoffset)
        stop = float(r.drugstopoffset) if r.drugstopoffset is not None else start
        if stop < start:
            stop = start
        if stop <= 0 or start >= 1440.0:
            continue
        dose = parse_dose(r.dosage, r.drugname)
        if dose is None or dose <= 0:
            continue
        freq24, is_once = parse_freq(r.frequency)
        if is_once:
            contrib = dose
        else:
            ov = max(0.0, min(stop, 1440.0) - max(start, 0.0))
            if ov <= 0:
                ov = 1440.0 - max(start, 0.0)
            contrib = dose * freq24 * (ov / 1440.0)
        rows.append((r.stay_key, dose * freq24 * f, contrib * f))
    g = pd.DataFrame(rows, columns=["stay_key", "daily_full", "contrib"])
    exp = (g.groupby("stay_key").agg(gc24_daily_pe_mg=("contrib", "sum"),
                                     gc24_peak_daily_pe_mg=("daily_full", "max")).reset_index()
           if len(g) else pd.DataFrame(columns=["stay_key", "gc24_daily_pe_mg",
                                                "gc24_peak_daily_pe_mg"]))
    lm = lm.merge(exp, on="stay_key", how="left")
    lm["death_hosp"] = (lm.hospitaldischargestatus == "Expired").astype(int)
    lm["intime"] = pd.NaT
    lm["abx_new_after48"] = ((lm.abx_first_off >= 2880) & lm.abx_first_off.notna()).astype(int)
    lm["abx_early"] = ((lm.abx_first_off < 1440) & lm.abx_first_off.notna()).astype(int)
    lm["db"] = "eicu"
    lm = finalise(lm)
    lm.to_csv(os.path.join(DATA, "cohort_eicu_rheum.csv"), index=False)
    P("  saved -> data/cohort_eicu_rheum.csv  (n=%d)" % len(lm))
    P("  any GC 24 h: %d (%.1f%%)   by disease:" % (lm.gc_any24.sum(), 100 * lm.gc_any24.mean()))
    for k, s in lm.groupby("primary_grp"):
        P("    %-4s n=%3d  anyGC %.1f%%  cult+ %d  infect %d  death %d  hyper %d/%d" % (
            k, len(s), 100 * s.gc_any24.mean(), int(s.culture_pos_after24.sum()),
            int(s.infect_icd.sum()), int(s.death_hosp.sum()),
            int(s.hyper_48h.sum()), s.glu_max_48h.notna().sum()))

    # ------------------------------------------------------------ NWICU
    c = psycopg2.connect(dbname="nwicu", **CONFIG)
    dn = pd.read_sql_query(NWICU_SQL, c)
    gn = pd.read_sql_query(NWICU_GC_SQL, c)
    c.close()
    P("")
    P("=== NWICU ===")
    P("  first-ICU stays (SLE+RA) : %d  %s" % (len(dn), dn.primary_grp.value_counts().to_dict()))
    P("  steroid order rows       : %d" % len(gn))
    dn["age"] = pd.to_numeric(dn.age, errors="coerce")
    dn = dn[dn.age >= 18].copy()
    lm2 = dn[dn.icu_los_days >= 1].copy()
    P("  landmark (LOS >= 24 h)   : %d" % len(lm2))
    for col in ["intime", "abx_first_time"]:
        lm2[col] = pd.to_datetime(lm2[col])
    for col in ["starttime", "stoptime"]:
        gn[col] = pd.to_datetime(gn[col])
    rows = []
    imap = dict(zip(lm2.stay_key, lm2.intime))
    for r in gn.itertuples(index=False):
        f = pe_factor(r.drug)
        if f is None:
            continue
        t0 = imap.get(r.stay_key)
        if t0 is None:
            continue
        win_end = t0 + pd.Timedelta(hours=24)
        start, stop = r.starttime, r.stoptime
        if stop < start:
            stop = start
        if stop <= t0 or start >= win_end:
            continue
        freq = float(r.freq24) if r.freq24 and r.freq24 > 0 else 1.0
        dose = float(r.dose_txt)
        total = (min(stop, win_end) - max(start, t0)).total_seconds() / 86400.0
        if total <= 0:
            total = (win_end - max(start, t0)).total_seconds() / 86400.0
        frac = min(1.0, max(0.0, total))
        rows.append((r.stay_key, dose * freq * f, dose * freq * f * frac))
    g2 = pd.DataFrame(rows, columns=["stay_key", "daily_full", "contrib"])
    exp2 = (g2.groupby("stay_key").agg(gc24_daily_pe_mg=("contrib", "sum"),
                                       gc24_peak_daily_pe_mg=("daily_full", "max")).reset_index()
            if len(g2) else pd.DataFrame(columns=["stay_key", "gc24_daily_pe_mg",
                                                  "gc24_peak_daily_pe_mg"]))
    lm2 = lm2.merge(exp2, on="stay_key", how="left")
    lm2["death_hosp"] = lm2.hospital_expire_flag.astype(int)
    lm2["abx_new_after48"] = ((lm2.abx_first_time >= lm2.intime + pd.Timedelta(hours=48))
                              & lm2.abx_first_time.notna()).astype(int)
    lm2["abx_early"] = ((lm2.abx_first_time < lm2.intime + pd.Timedelta(hours=24))
                        & lm2.abx_first_time.notna()).astype(int)
    lm2["culture_after24"] = np.nan
    lm2["culture_pos_after24"] = np.nan
    lm2["n_culture_any"] = np.nan
    lm2["db"] = "nwicu"
    lm2 = finalise(lm2)
    lm2.to_csv(os.path.join(DATA, "cohort_nwicu_rheum.csv"), index=False)
    P("  saved -> data/cohort_nwicu_rheum.csv  (n=%d)" % len(lm2))
    P("  any GC 24 h: %d (%.1f%%)   by disease:" % (lm2.gc_any24.sum(), 100 * lm2.gc_any24.mean()))
    for k, s in lm2.groupby("primary_grp"):
        P("    %-4s n=%3d  anyGC %.1f%%  infect %d  death %d  hyper %d/%d" % (
            k, len(s), 100 * s.gc_any24.mean(), int(s.infect_icd.sum()),
            int(s.death_hosp.sum()), int(s.hyper_48h.sum()), s.glu_max_48h.notna().sum()))


if __name__ == "__main__":
    main()
