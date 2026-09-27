# -*- coding: utf-8 -*-
"""
151_extract_rheum_cohort.py

Plan B, step 1: build the multi-disease "rheumatic / systemic autoimmune
disease ICU" cohort on the SAME harmonised variable set used for the
SLE-only analysis and the eICU/NWICU external validation.

Exactly the same transportable constructs as 112_extract_mimic_harmonized.py:
    landmark            ICU LOS >= 24 h, first ICU stay per admission, age >= 18
    exposure            glucocorticoid DOSE INTENSITY in the first 24 h
                        (prednisone-equivalent mg/day; window length
                        min(24 h, ICU LOS)) -- NOT cumulative dose
    strata              0 / >0-<10 / 10-<50 / >=50 mg/day
    events (timed)      blood culture positive > 24 h; new broad-spectrum
                        antibiotic > 48 h; 30-day death
    events (untimed)    infection ICD code; GI bleeding code (negative control)
    positive control    glucose >= 180 mg/dL within 48 h
    covariates          age, sex, vasopressor<=24h, ventilation<=24h,
                        renal/respiratory/hepatic failure, shock, coagulopathy,
                        organ-dysfunction count, SOFA

New in this script (required for the head-to-head):
    primary_grp         mutually exclusive disease label
    surg_service        admission under the SURG service -- RA patients
                        reach the ICU largely for surgical reasons, SLE
                        patients largely for medical reasons.  If this is
                        not measured it will masquerade as a disease effect.
    n_proc              number of procedures_icd rows (procedure intensity)
    immuno_any          non-glucocorticoid immunosuppressant / DMARD / biologic
    hcq                 hydroxychloroquine
    cytopenia           ICD D50-D77
    serositis           ICD J90/J91/I30/I31
    aps                 ICD D68.6 / 289.81
    n_hosp              total distinct admissions for that patient (chronicity)

Output: data/cohort_rheum_icu.csv
"""
import os
import re

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root (data/ and out/ live here)

CONFIG = dict(
    host=os.environ.get("MIMIC_DB_HOST", "localhost"),
    port=int(os.environ.get("MIMIC_DB_PORT", "5432")),
    user=os.environ.get("MIMIC_DB_USER", "postgres"),
    password=os.environ.get("MIMIC_DB_PASSWORD", ""),
)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")

BASE_SQL = r"""
WITH p0 AS (
    SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime, i.los,
           a.admittime, a.deathtime, a.hospital_expire_flag, a.admission_type,
           ag.age, pt.gender,
           ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.intime) AS rn,
           COUNT(*)     OVER (PARTITION BY i.hadm_id) AS n_icu
    FROM mimiciv_icu.icustays i
    JOIN rheum r      ON i.hadm_id = r.hadm_id
    JOIN mimiciv_hosp.admissions a ON i.hadm_id = a.hadm_id
    JOIN mimiciv_hosp.patients pt  ON i.subject_id = pt.subject_id
    LEFT JOIN mimiciv_derived.age ag ON i.hadm_id = ag.hadm_id
),
p AS (SELECT * FROM p0 WHERE rn = 1),
dx AS (
    SELECT d.hadm_id,
      MAX(CASE WHEN (d.icd_version=10 AND LEFT(d.icd_code,3) IN ('N17','N18','N19'))
               OR  (d.icd_version=9  AND LEFT(d.icd_code,3) IN ('584','585','586')) THEN 1 ELSE 0 END) AS renal_fail,
      MAX(CASE WHEN (d.icd_version=10 AND (d.icd_code LIKE 'K922%%' OR d.icd_code LIKE 'K250%%'
               OR d.icd_code LIKE 'K252%%' OR d.icd_code LIKE 'K254%%' OR d.icd_code LIKE 'K256%%'
               OR d.icd_code LIKE 'K260%%' OR d.icd_code LIKE 'K262%%' OR d.icd_code LIKE 'K264%%'
               OR d.icd_code LIKE 'K266%%' OR d.icd_code LIKE 'K270%%' OR d.icd_code LIKE 'K272%%'
               OR d.icd_code LIKE 'K274%%' OR d.icd_code LIKE 'K276%%' OR d.icd_code LIKE 'K280%%'
               OR d.icd_code LIKE 'K282%%' OR d.icd_code LIKE 'K284%%' OR d.icd_code LIKE 'K286%%'
               OR d.icd_code LIKE 'K290%%' OR d.icd_code LIKE 'I850%%' OR d.icd_code LIKE 'K921%%'
               OR d.icd_code LIKE 'K298%%'))
            OR (d.icd_version=9 AND (d.icd_code LIKE '5789%%' OR d.icd_code LIKE '5310%%'
               OR d.icd_code LIKE '5312%%' OR d.icd_code LIKE '5314%%' OR d.icd_code LIKE '5316%%'
               OR d.icd_code LIKE '5320%%' OR d.icd_code LIKE '5322%%' OR d.icd_code LIKE '5324%%'
               OR d.icd_code LIKE '5326%%' OR d.icd_code LIKE '5330%%' OR d.icd_code LIKE '5332%%'
               OR d.icd_code LIKE '5334%%' OR d.icd_code LIKE '5336%%' OR d.icd_code LIKE '5340%%'
               OR d.icd_code LIKE '5342%%' OR d.icd_code LIKE '5344%%' OR d.icd_code LIKE '5346%%'
               OR d.icd_code LIKE '5780%%')) THEN 1 ELSE 0 END) AS gi_bleed,
      MAX(CASE WHEN (d.icd_version = 10 AND (LEFT(d.icd_code,2) = 'J1'
               OR LEFT(d.icd_code,3) IN ('A40','A41','A49','J85','K61','K65','L02','L03','L08',
                                         'M86','N10','N39','R65','T81','T82','T84')))
            OR (d.icd_version = 9 AND (LEFT(d.icd_code,3) IN ('038','320','322','324','420','421',
               '481','482','483','484','485','486','507','510','513','540','567','590','599','680',
               '681','682','683','684','685','686','711','730','998') OR d.icd_code LIKE '9959%%'))
               THEN 1 ELSE 0 END) AS infect_icd,
      MAX(CASE WHEN (d.icd_version=10 AND LEFT(d.icd_code,3) IN ('J96','J80'))
               OR (d.icd_version=9  AND LEFT(d.icd_code,4) IN ('5185','5188')) THEN 1 ELSE 0 END) AS resp_fail,
      MAX(CASE WHEN (d.icd_version=10 AND LEFT(d.icd_code,3) IN ('R57'))
               OR (d.icd_version=9  AND LEFT(d.icd_code,4) IN ('7855')) THEN 1 ELSE 0 END) AS shock,
      MAX(CASE WHEN (d.icd_version=10 AND LEFT(d.icd_code,3) IN ('K72','K70'))
               OR (d.icd_version=9  AND LEFT(d.icd_code,3) IN ('570','572')) THEN 1 ELSE 0 END) AS liver_fail,
      MAX(CASE WHEN (d.icd_version=10 AND LEFT(d.icd_code,3) IN ('D65','D68'))
               OR (d.icd_version=9  AND LEFT(d.icd_code,4) IN ('2866','2867','2869')) THEN 1 ELSE 0 END) AS coagulop,
      MAX(CASE WHEN (d.icd_version=10 AND LEFT(d.icd_code,3) IN ('D50','D51','D52','D53','D55',
                          'D56','D57','D58','D59','D60','D61','D62','D63','D64','D69','D70','D71',
                          'D72','D73','D74','D75','D76','D77'))
               OR (d.icd_version=9  AND LEFT(d.icd_code,3) IN ('280','281','282','283','284','285',
                          '286','287','288')) THEN 1 ELSE 0 END) AS cytopenia,
      MAX(CASE WHEN (d.icd_version=10 AND LEFT(d.icd_code,3) IN ('J90','J91','I30','I31'))
               OR (d.icd_version=9  AND LEFT(d.icd_code,4) IN ('5111','5118','5119','4232','4233',
                          '4238','4239','5110','5119')) THEN 1 ELSE 0 END) AS serositis,
      MAX(CASE WHEN (d.icd_version=10 AND d.icd_code LIKE 'D68.6%%')
               OR (d.icd_version=9  AND d.icd_code LIKE '28981%%') THEN 1 ELSE 0 END) AS aps
    FROM mimiciv_hosp.diagnoses_icd d WHERE d.hadm_id IN (SELECT hadm_id FROM p) GROUP BY d.hadm_id
),
abx AS (
    SELECT pr.hadm_id, MIN(pr.starttime) AS abx_first_time
    FROM mimiciv_hosp.prescriptions pr JOIN p ON pr.hadm_id = p.hadm_id
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
cult AS (
    SELECT m.hadm_id,
           COUNT(*) AS n_culture_any,
           MAX(CASE WHEN m.charttime >= p.intime + INTERVAL '24 hours' THEN 1 ELSE 0 END) AS culture_after24,
           MAX(CASE WHEN m.charttime >= p.intime + INTERVAL '24 hours'
                     AND m.org_name IS NOT NULL THEN 1 ELSE 0 END) AS culture_pos_after24
    FROM mimiciv_hosp.microbiologyevents m JOIN p ON m.hadm_id = p.hadm_id
    WHERE m.spec_type_desc ILIKE '%%BLOOD%%' AND m.charttime IS NOT NULL
    GROUP BY m.hadm_id
),
glu AS (
    SELECT l.hadm_id, MAX(l.valuenum) AS glu_max_48h
    FROM mimiciv_hosp.labevents l
    JOIN mimiciv_hosp.d_labitems d ON l.itemid = d.itemid
    JOIN p ON l.hadm_id = p.hadm_id
    WHERE d.label = 'Glucose' AND l.valuenum IS NOT NULL
      AND l.charttime >= p.intime AND l.charttime < p.intime + INTERVAL '48 hours'
      AND l.valuenum BETWEEN 10 AND 1500
    GROUP BY l.hadm_id
),
vaso AS (
    SELECT DISTINCT ie.stay_id FROM mimiciv_icu.inputevents ie JOIN p ON ie.stay_id = p.stay_id
    WHERE ie.itemid IN (221906,221289,221662,221749,222315,221653,221986,228340)
      AND ie.starttime >= p.intime AND ie.starttime < p.intime + INTERVAL '24 hours'
),
vent AS (
    SELECT DISTINCT pe.stay_id FROM mimiciv_icu.procedureevents pe
    JOIN mimiciv_icu.d_items di ON pe.itemid = di.itemid
    JOIN p ON pe.stay_id = p.stay_id
    WHERE (di.label ILIKE '%%ventilation%%' OR pe.itemid IN (225792,225794,227194))
      AND pe.starttime >= p.intime AND pe.starttime < p.intime + INTERVAL '24 hours'
),
sofa AS (
    SELECT s.stay_id, MAX(s.sofa_24hours) AS sofa24
    FROM mimiciv_derived.sofa s JOIN p ON s.stay_id = p.stay_id GROUP BY s.stay_id
),
svc AS (
    SELECT s.hadm_id,
           MAX(CASE WHEN s.curr_service = 'SURG' THEN 1 ELSE 0 END) AS surg_service,
           MAX(CASE WHEN s.curr_service IN ('SURG','CSURG','GU','GYN','NSURG','ORTHO',
                                            'PSURG','TRAUM','TSURG','VSURG','ENT','DENT','OBS')
                    THEN 1 ELSE 0 END) AS surg_any
    FROM mimiciv_hosp.services s JOIN p ON s.hadm_id = p.hadm_id GROUP BY s.hadm_id
),
prc AS (
    SELECT hadm_id, COUNT(*) AS n_proc FROM mimiciv_hosp.procedures_icd
    WHERE hadm_id IN (SELECT hadm_id FROM p) GROUP BY hadm_id
),
imm AS (
    SELECT pr.hadm_id,
      MAX(CASE WHEN pr.drug ILIKE '%%mycophenol%%' OR pr.drug ILIKE '%%cellcept%%'
            OR pr.drug ILIKE '%%myfortic%%' OR pr.drug ILIKE '%%cyclophosph%%'
            OR pr.drug ILIKE '%%rituximab%%' OR pr.drug ILIKE '%%azathiopr%%'
            OR pr.drug ILIKE '%%tacrolimus%%' OR pr.drug ILIKE '%%methotrexate%%'
            OR pr.drug ILIKE '%%tocilizumab%%' OR pr.drug ILIKE '%%abatacept%%'
            OR pr.drug ILIKE '%%infliximab%%' OR pr.drug ILIKE '%%adalimumab%%'
            OR pr.drug ILIKE '%%etanercept%%' OR pr.drug ILIKE '%%sulfasalazine%%'
            OR pr.drug ILIKE '%%leflunomide%%' OR pr.drug ILIKE '%%tofacitinib%%'
            OR pr.drug ILIKE '%%upadacitinib%%' OR pr.drug ILIKE '%%baricitinib%%'
            OR pr.drug ILIKE '%%secukinumab%%' OR pr.drug ILIKE '%%ustekinumab%%'
            OR pr.drug ILIKE '%%belimumab%%' OR pr.drug ILIKE '%%anakinra%%'
            THEN 1 ELSE 0 END) AS immuno_any,
      MAX(CASE WHEN pr.drug ILIKE '%%hydroxychloroquine%%' OR pr.drug ILIKE '%%plaquenil%%'
               THEN 1 ELSE 0 END) AS hcq
    FROM mimiciv_hosp.prescriptions pr JOIN p ON pr.hadm_id = p.hadm_id
    GROUP BY pr.hadm_id
),
hosp AS (
    SELECT subject_id, COUNT(DISTINCT hadm_id) AS n_hosp
    FROM mimiciv_hosp.admissions
    WHERE subject_id IN (SELECT subject_id FROM p) GROUP BY subject_id
)
SELECT p.subject_id, p.hadm_id AS stay_key, p.stay_id, p.intime, p.outtime,
       p.los AS icu_los_days, p.n_icu, p.hospital_expire_flag, p.gender, p.age,
       r.primary_grp,
       EXTRACT(EPOCH FROM (p.intime - p.admittime))/3600.0 AS preicu_hours,
       COALESCE(d.renal_fail,0) AS renal_fail, COALESCE(d.gi_bleed,0) AS gi_bleed,
       COALESCE(d.infect_icd,0) AS infect_icd, COALESCE(d.resp_fail,0) AS resp_fail,
       COALESCE(d.shock,0) AS shock, COALESCE(d.liver_fail,0) AS liver_fail,
       COALESCE(d.coagulop,0) AS coagulop, COALESCE(d.cytopenia,0) AS cytopenia,
       COALESCE(d.serositis,0) AS serositis, COALESCE(d.aps,0) AS aps_dx,
       a.abx_first_time, g.glu_max_48h,
       COALESCE(c.n_culture_any,0) AS n_culture_any,
       COALESCE(c.culture_after24,0) AS culture_after24,
       COALESCE(c.culture_pos_after24,0) AS culture_pos_after24,
       CASE WHEN v.stay_id  IS NOT NULL THEN 1 ELSE 0 END AS vaso24,
       CASE WHEN ve.stay_id IS NOT NULL THEN 1 ELSE 0 END AS vent24,
       sf.sofa24,
       COALESCE(sv.surg_service,0) AS surg_service,
       COALESCE(sv.surg_any,0) AS surg_any,
       CASE WHEN p.admission_type IN ('ELECTIVE') THEN 1 ELSE 0 END AS elective,
       COALESCE(pc.n_proc,0) AS n_proc,
       COALESCE(im.immuno_any,0) AS immuno_any,
       COALESCE(im.hcq,0) AS hcq,
       COALESCE(hh.n_hosp,0) AS n_hosp,
       CASE WHEN p.deathtime IS NOT NULL AND p.deathtime <= p.intime + INTERVAL '30 days'
            THEN 1 ELSE 0 END AS death_30d
FROM p
JOIN rheum r     ON p.hadm_id = r.hadm_id
LEFT JOIN dx d   ON p.hadm_id = d.hadm_id
LEFT JOIN abx a  ON p.hadm_id = a.hadm_id
LEFT JOIN cult c ON p.hadm_id = c.hadm_id
LEFT JOIN glu g  ON p.hadm_id = g.hadm_id
LEFT JOIN vaso v ON p.stay_id = v.stay_id
LEFT JOIN vent ve ON p.stay_id = ve.stay_id
LEFT JOIN sofa sf ON p.stay_id = sf.stay_id
LEFT JOIN svc sv  ON p.hadm_id = sv.hadm_id
LEFT JOIN prc pc  ON p.hadm_id = pc.hadm_id
LEFT JOIN imm im  ON p.hadm_id = im.hadm_id
LEFT JOIN hosp hh ON p.subject_id = hh.subject_id
ORDER BY p.hadm_id
"""

GC_SQL = r"""
SELECT pr.hadm_id AS stay_key, pr.drug,
       CAST(pr.dose_val_rx AS DOUBLE PRECISION) AS dose_mg,
       COALESCE(pr.doses_per_24_hrs, 1.0) AS freq24,
       pr.route, pr.starttime, pr.stoptime
FROM mimiciv_hosp.prescriptions pr
WHERE pr.hadm_id IN (SELECT hadm_id FROM rheum)
  AND pr.dose_unit_rx = 'mg'
  AND pr.dose_val_rx ~ '^[0-9]+(\.[0-9]+)?$'
  AND pr.route IN ('IV','PO','PO/NG','IM')
  AND (pr.drug ILIKE '%%predni%%' OR pr.drug ILIKE '%%methylpred%%'
    OR pr.drug ILIKE '%%dexameth%%' OR pr.drug ILIKE '%%hydrocort%%'
    OR pr.drug ILIKE '%%betameth%%' OR pr.drug ILIKE '%%triamcin%%')
  AND pr.starttime IS NOT NULL AND pr.stoptime IS NOT NULL
"""

PE = [(r"methylpred|medrol|solumedrol", 1.25), (r"prednisolone", 1.0),
      (r"prednisone|predni", 1.0), (r"hydrocort|cortef", 0.25),
      (r"dexameth", 6.67), (r"betameth", 6.67), (r"triamcin", 1.25)]


def pe_factor(name):
    n = (name or "").lower()
    for pat, f in PE:
        if re.search(pat, n):
            return f
    return None


def build_exposure(gcdf, intime_map):
    rows = []
    for r in gcdf.itertuples(index=False):
        f = pe_factor(r.drug)
        if f is None:
            continue
        t0 = intime_map.get(r.stay_key)
        if t0 is None:
            continue
        win_end = t0 + pd.Timedelta(hours=24)
        start, stop = r.starttime, r.stoptime
        if stop < start:
            stop = start
        if stop <= t0 or start >= win_end:
            continue
        freq = float(r.freq24) if r.freq24 and r.freq24 > 0 else 1.0
        total = (min(stop, win_end) - max(start, t0)).total_seconds() / 86400.0
        if total <= 0:
            total = (win_end - max(start, t0)).total_seconds() / 86400.0
        frac = min(1.0, max(0.0, total))
        rows.append((r.stay_key, r.dose_mg * freq * f, r.dose_mg * freq * f * frac))
    g = pd.DataFrame(rows, columns=["stay_key", "daily_full", "contrib"])
    return g.groupby("stay_key").agg(gc24_daily_pe_mg=("contrib", "sum"),
                                     gc24_peak_daily_pe_mg=("daily_full", "max")).reset_index()


def main():
    P = print
    cls = pd.read_csv(os.path.join(DATA, "rheum_hadm_classified.csv"))
    cls = cls[cls.primary != "none"][["hadm_id", "primary"]]
    P("=== 151 multi-disease rheumatic ICU cohort ===")
    P("  classified admissions (non-'none') : %d" % len(cls))
    P("  by disease                         : %s" % cls.primary.value_counts().to_dict())

    c = psycopg2.connect(dbname="mimiciv", **CONFIG)
    cur = c.cursor()
    cur.execute("CREATE TEMP TABLE rheum(hadm_id BIGINT PRIMARY KEY, primary_grp TEXT)")
    execute_values(cur, "INSERT INTO rheum (hadm_id, primary_grp) VALUES %s",
                   list(cls.itertuples(index=False, name=None)), page_size=5000)
    c.commit()

    df = pd.read_sql_query(BASE_SQL, c)
    gc = pd.read_sql_query(GC_SQL, c)
    c.close()

    P("  first-ICU-stay rows pulled        : %d" % len(df))
    P("  steroid prescription rows         : %d" % len(gc))

    df["db"] = "mimiciv"
    df["age"] = pd.to_numeric(df.age, errors="coerce")
    df = df[df.age >= 18].copy()
    P("  adults >=18 y                     : %d" % len(df))
    lm = df[df.icu_los_days >= 1].copy()
    P("  landmark (ICU LOS >= 24 h)        : %d" % len(lm))
    P("  by disease (landmark)             : %s" % lm.primary_grp.value_counts().to_dict())

    for col in ["intime", "outtime", "abx_first_time"]:
        lm[col] = pd.to_datetime(lm[col])
    for col in ["starttime", "stoptime"]:
        gc[col] = pd.to_datetime(gc[col])
    imap = dict(zip(lm.stay_key, lm.intime))
    exp = build_exposure(gc, imap)
    lm = lm.merge(exp, on="stay_key", how="left")
    lm["gc24_daily_pe_mg"] = lm.gc24_daily_pe_mg.fillna(0.0)
    lm["gc24_peak_daily_pe_mg"] = lm.gc24_peak_daily_pe_mg.fillna(0.0)
    lm["gc_any24"] = (lm.gc24_daily_pe_mg > 0).astype(int)

    def stratum(v):
        if v <= 0:
            return "G0_none"
        if v < 10:
            return "G1_low"
        if v < 50:
            return "G2_mod"
        return "G3_high"
    lm["gc24_str"] = lm.gc24_daily_pe_mg.apply(stratum)
    lm["gc_str_num"] = lm.gc24_str.map({"G0_none": 0, "G1_low": 1, "G2_mod": 2, "G3_high": 3})

    lm["hyper_48h"] = np.where(lm.glu_max_48h >= 180, 1,
                               np.where(lm.glu_max_48h.isna(), np.nan, 0))
    lm["death_hosp"] = lm.hospital_expire_flag.astype(int)
    lm["female"] = (lm.gender.astype(str).str.upper() == "F").astype(int)
    lm["organ_dysf"] = (lm.renal_fail + lm.resp_fail + lm.shock
                        + lm.liver_fail + lm.coagulop)
    lm["abx_early"] = ((lm.abx_first_time < lm.intime + pd.Timedelta(hours=24))
                       & lm.abx_first_time.notna()).astype(int)
    lm["abx_new_after48"] = ((lm.abx_first_time >= lm.intime + pd.Timedelta(hours=48))
                             & lm.abx_first_time.notna()).astype(int)
    lm["is_sle"] = (lm.primary_grp == "SLE").astype(int)
    lm["is_ra"] = (lm.primary_grp == "RA").astype(int)

    out = os.path.join(DATA, "cohort_rheum_icu.csv")
    lm.to_csv(out, index=False)
    P("\n  saved -> %s" % out)

    P("\n=== exposure (landmark cohort n=%d) ===" % len(lm))
    P("  any GC in first 24 h : %d (%.1f%%)" % (lm.gc_any24.sum(), 100 * lm.gc_any24.mean()))
    P("  strata               : " + str(lm.gc24_str.value_counts().reindex(
        ["G0_none", "G1_low", "G2_mod", "G3_high"]).fillna(0).astype(int).to_dict()))
    pos = lm.loc[lm.gc24_daily_pe_mg > 0, "gc24_daily_pe_mg"]
    P("  dose among exposed   : n=%d median %.1f  IQR %.1f-%.1f  max %.1f" % (
        len(pos), pos.median(), pos.quantile(.25), pos.quantile(.75), pos.max()))

    P("\n=== exposure by disease ===")
    P("  %-12s %6s %9s %10s %10s %10s %10s" % (
        "disease", "n", "anyGC%", "G0", "G1_low", "G2_mod", "G3_high"))
    for g, s in lm.groupby("primary_grp"):
        vc = s.gc24_str.value_counts().reindex(["G0_none", "G1_low", "G2_mod", "G3_high"]).fillna(0).astype(int)
        P("  %-12s %6d %8.1f%% %10d %10d %10d %10d" % (
            g, len(s), 100 * s.gc_any24.mean(), vc["G0_none"], vc["G1_low"],
            vc["G2_mod"], vc["G3_high"]))

    P("\n=== outcomes ===")
    for v in ["infect_icd", "culture_after24", "culture_pos_after24", "abx_early",
              "abx_new_after48", "gi_bleed", "death_hosp", "death_30d"]:
        P("  %-20s %5d (%.1f%%)" % (v, int(lm[v].sum()), 100 * lm[v].mean()))
    P("  hyper >=180 (of measured) %d / %d (%.1f%%)" % (
        int(lm.hyper_48h.sum()), lm.glu_max_48h.notna().sum(),
        100 * lm.hyper_48h.mean(skipna=True)))

    P("\n=== covariates ===")
    for v in ["vaso24", "vent24", "renal_fail", "shock", "resp_fail",
              "surg_service", "surg_any", "elective", "immuno_any", "hcq",
              "cytopenia", "serositis"]:
        P("  %-14s %5d (%.1f%%)" % (v, int(lm[v].sum()), 100 * lm[v].mean()))
    P("  age median %.1f  female %.1f%%  sofa24 available %d  n_hosp median %.1f" % (
        lm.age.median(), 100 * lm.female.mean(), lm.sofa24.notna().sum(), lm.n_hosp.median()))

    P("\n=== SLE vs RA, side by side (the point of the exercise) ===")
    P("  %-16s %10s %10s" % ("", "SLE", "RA"))
    a, b = lm[lm.is_sle == 1], lm[lm.is_ra == 1]
    for v, lab in [("age", "age median"), ("female", "female %"), ("vaso24", "vasopressor %"),
                   ("vent24", "ventilation %"), ("renal_fail", "renal failure %"),
                   ("surg_service", "SURG service %"), ("surg_any", "any surg service %"),
                   ("elective", "elective admission %"),
                   ("immuno_any", "immunosuppr. %"), ("hcq", "hydroxychloroquine %"),
                   ("cytopenia", "cytopenia %"), ("serositis", "serositis %"),
                   ("n_proc", "procedures median"), ("n_hosp", "admissions median"),
                   ("sofa24", "SOFA median")]:
        if lab.endswith("%"):
            P("  %-16s %9.1f%% %9.1f%%" % (lab, 100 * a[v].mean(), 100 * b[v].mean()))
        else:
            P("  %-16s %10.1f %10.1f" % (lab, a[v].median(), b[v].median()))
    P("  %-16s %10d %10d" % ("n", len(a), len(b)))
    P("  %-16s %10d %10d" % ("culture+ >24h", int(a.culture_pos_after24.sum()),
                            int(b.culture_pos_after24.sum())))
    P("  %-16s %10d %10d" % ("hospital death", int(a.death_hosp.sum()),
                            int(b.death_hosp.sum())))


if __name__ == "__main__":
    main()
