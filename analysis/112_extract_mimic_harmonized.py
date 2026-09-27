# -*- coding: utf-8 -*-
"""
MIMIC-IV re-extraction on the HARMONISED variable set.

The v4 cohort (data/cohort_v4_fixes.csv) was built for the single-database
analysis and uses MIMIC-only constructs (SOFA, 30-day death, a wider
covariate set). To make a fair external comparison we rebuild the same
minimal, transportable variable set used for eICU and NWICU:

    age, sex, vasopressor<=24h, ventilation<=24h, renal failure,
    organ-dysfunction count, APACHE-equivalent severity (SOFA here)

Outcomes are likewise the transportable set:
    hyper_48h (positive control), abx_new_after48, infect_icd, gi_bleed,
    culture_pos_after24 (MIMIC + eICU only), hospital death.
"""
import psycopg2
import pandas as pd
import numpy as np
import os
import re
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
    WHERE (d.icd_version=10 AND d.icd_code LIKE 'M32%') OR (d.icd_version=9 AND d.icd_code LIKE '7100%')
),
p0 AS (
    SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime, i.los,
           a.admittime, a.deathtime, a.hospital_expire_flag, ag.age, pt.gender,
           ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.intime) AS rn
    FROM mimiciv_icu.icustays i
    JOIN sle ON i.hadm_id = sle.hadm_id
    JOIN mimiciv_hosp.admissions a ON i.hadm_id = a.hadm_id
    JOIN mimiciv_hosp.patients pt ON i.subject_id = pt.subject_id
    LEFT JOIN mimiciv_derived.age ag ON i.hadm_id = ag.hadm_id
),
p AS (SELECT * FROM p0 WHERE rn = 1),
dx AS (
    SELECT d.hadm_id,
      MAX(CASE WHEN (d.icd_version=10 AND LEFT(d.icd_code,3) IN ('N17','N18','N19'))
               OR  (d.icd_version=9  AND LEFT(d.icd_code,3) IN ('584','585','586')) THEN 1 ELSE 0 END) AS renal_fail,
      MAX(CASE WHEN (d.icd_version=10 AND (d.icd_code LIKE 'K922%' OR d.icd_code LIKE 'K250%'
               OR d.icd_code LIKE 'K252%' OR d.icd_code LIKE 'K254%' OR d.icd_code LIKE 'K256%'
               OR d.icd_code LIKE 'K260%' OR d.icd_code LIKE 'K262%' OR d.icd_code LIKE 'K264%'
               OR d.icd_code LIKE 'K266%' OR d.icd_code LIKE 'K270%' OR d.icd_code LIKE 'K272%'
               OR d.icd_code LIKE 'K274%' OR d.icd_code LIKE 'K276%' OR d.icd_code LIKE 'K280%'
               OR d.icd_code LIKE 'K282%' OR d.icd_code LIKE 'K284%' OR d.icd_code LIKE 'K286%'
               OR d.icd_code LIKE 'K290%' OR d.icd_code LIKE 'I850%' OR d.icd_code LIKE 'K921%'
               OR d.icd_code LIKE 'K298%'))
            OR (d.icd_version=9 AND (d.icd_code LIKE '5789%' OR d.icd_code LIKE '5310%'
               OR d.icd_code LIKE '5312%' OR d.icd_code LIKE '5314%' OR d.icd_code LIKE '5316%'
               OR d.icd_code LIKE '5320%' OR d.icd_code LIKE '5322%' OR d.icd_code LIKE '5324%'
               OR d.icd_code LIKE '5326%' OR d.icd_code LIKE '5330%' OR d.icd_code LIKE '5332%'
               OR d.icd_code LIKE '5334%' OR d.icd_code LIKE '5336%' OR d.icd_code LIKE '5340%'
               OR d.icd_code LIKE '5342%' OR d.icd_code LIKE '5344%' OR d.icd_code LIKE '5346%'
               OR d.icd_code LIKE '5780%')) THEN 1 ELSE 0 END) AS gi_bleed,
      MAX(CASE WHEN (d.icd_version = 10 AND (LEFT(d.icd_code,2) = 'J1'
               OR LEFT(d.icd_code,3) IN ('A40','A41','A49','J85','K61','K65','L02','L03','L08',
                                         'M86','N10','N39','R65','T81','T82','T84')))
            OR (d.icd_version = 9 AND (LEFT(d.icd_code,3) IN ('038','320','322','324','420','421',
               '481','482','483','484','485','486','507','510','513','540','567','590','599','680',
               '681','682','683','684','685','686','711','730','998') OR d.icd_code LIKE '9959%'))
               THEN 1 ELSE 0 END) AS infect_icd,
      MAX(CASE WHEN (d.icd_version=10 AND LEFT(d.icd_code,3) IN ('J96','J80'))
               OR (d.icd_version=9  AND LEFT(d.icd_code,4) IN ('5185','5188')) THEN 1 ELSE 0 END) AS resp_fail,
      MAX(CASE WHEN (d.icd_version=10 AND LEFT(d.icd_code,3) IN ('R57'))
               OR (d.icd_version=9  AND LEFT(d.icd_code,4) IN ('7855')) THEN 1 ELSE 0 END) AS shock,
      MAX(CASE WHEN (d.icd_version=10 AND LEFT(d.icd_code,3) IN ('K72','K70'))
               OR (d.icd_version=9  AND LEFT(d.icd_code,3) IN ('570','572')) THEN 1 ELSE 0 END) AS liver_fail,
      MAX(CASE WHEN (d.icd_version=10 AND LEFT(d.icd_code,3) IN ('D65','D68'))
               OR (d.icd_version=9  AND LEFT(d.icd_code,4) IN ('2866','2867','2869')) THEN 1 ELSE 0 END) AS coagulop
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
    SELECT DISTINCT ie.stay_id
    FROM mimiciv_icu.inputevents ie JOIN p ON ie.stay_id = p.stay_id
    WHERE ie.itemid IN (221906, 221289, 221662, 221749, 222315, 221653, 221986, 228340)
      AND ie.starttime >= p.intime AND ie.starttime < p.intime + INTERVAL '24 hours'
),
vent AS (
    SELECT DISTINCT pe.stay_id
    FROM mimiciv_icu.procedureevents pe
    JOIN mimiciv_icu.d_items di ON pe.itemid = di.itemid
    JOIN p ON pe.stay_id = p.stay_id
    WHERE (di.label ILIKE '%%ventilation%%' OR pe.itemid IN (225792, 225794, 227194))
      AND pe.starttime >= p.intime AND pe.starttime < p.intime + INTERVAL '24 hours'
),
sofa AS (
    SELECT s.stay_id, MAX(s.sofa_24hours) AS sofa24
    FROM mimiciv_derived.sofa s JOIN p ON s.stay_id = p.stay_id
    GROUP BY s.stay_id
)
SELECT p.subject_id, p.hadm_id AS stay_key, p.stay_id, p.intime, p.outtime,
       p.los AS icu_los_days, p.hospital_expire_flag, p.gender, p.age,
       EXTRACT(EPOCH FROM (p.intime - p.admittime))/3600.0 AS preicu_hours,
       COALESCE(d.renal_fail,0) AS renal_fail, COALESCE(d.gi_bleed,0) AS gi_bleed,
       COALESCE(d.infect_icd,0) AS infect_icd, COALESCE(d.resp_fail,0) AS resp_fail,
       COALESCE(d.shock,0) AS shock, COALESCE(d.liver_fail,0) AS liver_fail,
       COALESCE(d.coagulop,0) AS coagulop,
       a.abx_first_time, g.glu_max_48h,
       COALESCE(c.n_culture_any,0) AS n_culture_any,
       COALESCE(c.culture_after24,0) AS culture_after24,
       COALESCE(c.culture_pos_after24,0) AS culture_pos_after24,
       CASE WHEN v.stay_id IS NOT NULL THEN 1 ELSE 0 END AS vaso24,
       CASE WHEN ve.stay_id IS NOT NULL THEN 1 ELSE 0 END AS vent24,
       sf.sofa24,
       CASE WHEN p.deathtime IS NOT NULL AND p.deathtime <= p.intime + INTERVAL '30 days'
            THEN 1 ELSE 0 END AS death_30d
FROM p
LEFT JOIN dx d   ON p.hadm_id = d.hadm_id
LEFT JOIN abx a  ON p.hadm_id = a.hadm_id
LEFT JOIN cult c ON p.hadm_id = c.hadm_id
LEFT JOIN glu g  ON p.hadm_id = g.hadm_id
LEFT JOIN vaso v ON p.stay_id = v.stay_id
LEFT JOIN vent ve ON p.stay_id = ve.stay_id
LEFT JOIN sofa sf ON p.stay_id = sf.stay_id
ORDER BY p.hadm_id
"""

GC_SQL = """
SELECT pr.hadm_id AS stay_key, pr.drug,
       CAST(pr.dose_val_rx AS DOUBLE PRECISION) AS dose_mg,
       COALESCE(pr.doses_per_24_hrs, 1.0) AS freq24,
       pr.route, pr.starttime, pr.stoptime
FROM mimiciv_hosp.prescriptions pr
WHERE pr.hadm_id IN (SELECT DISTINCT hadm_id FROM mimiciv_hosp.diagnoses_icd
                     WHERE (icd_version=10 AND icd_code LIKE 'M32%%')
                        OR (icd_version=9 AND icd_code LIKE '7100%%'))
  AND pr.dose_unit_rx = 'mg'
  AND pr.dose_val_rx ~ '^[0-9]+(\\.[0-9]+)?$'
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
    for _, r in gcdf.iterrows():
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
    c = psycopg2.connect(dbname="mimiciv", **CONFIG)
    df = pd.read_sql_query(SQL, c)
    gc = pd.read_sql_query(GC_SQL, c)
    c.close()

    P = print
    P("=== MIMIC-IV harmonised ===")
    P("  SLE first-ICU stays        : %d" % len(df))
    P("  steroid prescription rows  : %d" % len(gc))

    df["db"] = "mimiciv"
    df["age"] = pd.to_numeric(df.age, errors="coerce")
    df = df[df.age >= 18].copy()
    P("  adults >=18 y              : %d" % len(df))
    lm = df[df.icu_los_days >= 1].copy()
    P("  landmark (ICU LOS >= 24 h) : %d" % len(lm))

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

    lm["hyper_48h"] = np.where(lm.glu_max_48h >= 180, 1,
                               np.where(lm.glu_max_48h.isna(), np.nan, 0))
    lm["death_hosp"] = lm.hospital_expire_flag.astype(int)
    lm["female"] = (lm.gender.astype(str).str.upper() == "F").astype(int)
    lm["organ_dysf"] = lm.renal_fail + lm.resp_fail + lm.shock + lm.liver_fail + lm.coagulop
    lm["abx_early"] = ((lm.abx_first_time < lm.intime + pd.Timedelta(hours=24))
                       & lm.abx_first_time.notna()).astype(int)
    lm["abx_new_after48"] = ((lm.abx_first_time >= lm.intime + pd.Timedelta(hours=48))
                             & lm.abx_first_time.notna()).astype(int)

    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, "cohort_mimic_harmonized.csv")
    lm.to_csv(out, index=False)
    P("\n  saved -> %s" % out)

    P("\n=== exposure (landmark cohort n=%d) ===" % len(lm))
    P("  any GC in first 24 h : %d (%.1f%%)" % (lm.gc_any24.sum(), 100 * lm.gc_any24.mean()))
    P("  strata               : " + lm.gc24_str.value_counts().reindex(
        ["G0_none", "G1_low", "G2_mod", "G3_high"]).fillna(0).astype(int).to_dict().__str__())
    pos = lm.loc[lm.gc24_daily_pe_mg > 0, "gc24_daily_pe_mg"]
    P("  dose among exposed   : n=%d median %.1f  IQR %.1f-%.1f  max %.1f" % (
        len(pos), pos.median(), pos.quantile(.25), pos.quantile(.75), pos.max()))

    P("\n=== outcomes (landmark cohort) ===")
    for v in ["infect_icd", "culture_after24", "culture_pos_after24",
              "abx_early", "abx_new_after48", "gi_bleed", "death_hosp", "death_30d"]:
        P("  %-20s %4d (%.1f%%)" % (v, int(lm[v].sum()), 100 * lm[v].mean()))
    P("  blood cultures total : %d" % int(lm.n_culture_any.sum()))
    P("  glucose measured     : %d (%.1f%%)" % (
        lm.glu_max_48h.notna().sum(), 100 * lm.glu_max_48h.notna().mean()))
    P("  hyper >=180 mg/dL    : %d (%.1f%%)" % (
        int(lm.hyper_48h.sum()), 100 * lm.hyper_48h.mean(skipna=True)))

    P("\n=== covariates ===")
    for v in ["vaso24", "vent24", "renal_fail", "shock", "resp_fail"]:
        P("  %-12s %4d (%.1f%%)" % (v, int(lm[v].sum()), 100 * lm[v].mean()))
    P("  age median %.1f  female %.1f%%  sofa24 available %d" % (
        lm.age.median(), 100 * lm.female.mean(), lm.sofa24.notna().sum()))


if __name__ == "__main__":
    main()
