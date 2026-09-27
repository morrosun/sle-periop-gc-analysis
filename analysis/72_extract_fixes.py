# -*- coding: utf-8 -*-
"""
Fix the positive-control failure: rebuild the cohort with (1) SLE-activity proxies
and (2) a clean temporal ordering.

Why this script exists
----------------------
The original analysis failed its positive control: GC->infection OR 1.26 vs
ondansetron->infection OR 1.27. Three structural defects, all addressed here:

  D1  outcome timing   : `infection` came from whole-hospitalisation ICD codes, so it
                        includes infections that were already present *before* ICU
                        admission. Reverse causation by construction.
                        -> add TIMED outcomes (blood culture / new antibiotic after
                           a 24 h landmark) and a prevalent-infection flag.
  D2  exposure timing  : GC was measured inside the same window as the outcome.
                        -> exposure = first 24 h of ICU; outcome = events after 24 h.
                        (a pre-ICU 7-day window does not exist: median pre-ICU time
                         is 1.8 h, only 3.8% have >= 7 days -- see out/71_probe_windows.txt)
  D3  no activity adj  : SLE disease activity drives both the GC dose and the
                        infection risk, and was not in the propensity model.
                        -> ICD-derived organ-manifestation proxies (100% coverage)
                        + a lab subset (C3/C4/CRP) for sensitivity.

A real positive control is also added: GC -> hyperglycaemia is a known
pharmacological effect. It separates "our exposure variable is broken" from
"the GC-infection association is genuinely absent".
"""
import psycopg2
import pandas as pd
import numpy as np
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
    WHERE (d.icd_version=10 AND d.icd_code LIKE 'M32%') OR (d.icd_version=9 AND d.icd_code LIKE '7100%')
),
icu AS (
    SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime, i.los,
           ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.intime) AS rn
    FROM mimiciv_icu.icustays i JOIN sle ON i.hadm_id = sle.hadm_id
),
lm AS (
    SELECT c.*, a.admittime, a.dischtime, a.deathtime, ag.age
    FROM icu c
    JOIN mimiciv_hosp.admissions a ON c.hadm_id = a.hadm_id
    LEFT JOIN mimiciv_derived.age ag ON c.hadm_id = ag.hadm_id
    WHERE c.rn = 1 AND ag.age >= 18
),
-- ---------------------------------------------------------------- exposures
gc_rx AS (
    SELECT p.hadm_id,
           CAST(p.dose_val_rx AS DOUBLE PRECISION) AS dose_mg,
           COALESCE(p.doses_per_24_hrs, 1.0) AS freq24,
           p.starttime, p.stoptime,
           CASE WHEN p.drug ILIKE '%methylpred%'   THEN 1.25
                WHEN p.drug ILIKE '%prednisolone%' THEN 1.0
                WHEN p.drug ILIKE '%predni%'       THEN 1.0
                WHEN p.drug ILIKE '%hydrocort%'    THEN 0.25
                WHEN p.drug ILIKE '%dexameth%'     THEN 6.67
                WHEN p.drug ILIKE '%betameth%'     THEN 6.67
                WHEN p.drug ILIKE '%triamcin%'     THEN 1.25
           END AS pe_factor
    FROM mimiciv_hosp.prescriptions p JOIN lm ON p.hadm_id = lm.hadm_id
    WHERE p.dose_unit_rx = 'mg'
      AND p.dose_val_rx ~ '^[0-9]+(\.[0-9]+)?$'
      AND p.route IN ('IV','PO','PO/NG','IM')
      AND p.starttime IS NOT NULL AND p.stoptime IS NOT NULL
      AND (p.drug ILIKE '%predni%' OR p.drug ILIKE '%methylpred%' OR p.drug ILIKE '%dexameth%'
        OR p.drug ILIKE '%hydrocort%' OR p.drug ILIKE '%betameth%' OR p.drug ILIKE '%triamcin%')
),
gc_lan AS (   -- exposure window = first 24 h of ICU (landmark), truncated at outtime
    SELECT x.hadm_id,
           SUM(CASE WHEN overlap_days > 0 THEN dose_mg*freq24*pe_factor*overlap_days ELSE 0 END) AS gc24_cum,
           MAX(CASE WHEN overlap_days > 0 THEN dose_mg*freq24*pe_factor END)                     AS gc24_peak
    FROM (
        SELECT g.hadm_id, g.dose_mg, g.freq24, g.pe_factor,
               GREATEST(0.0, LEAST(
                   EXTRACT(EPOCH FROM (LEAST(lm.intime + INTERVAL '24 hours', lm.outtime, g.stoptime)
                                       - GREATEST(lm.intime, g.starttime)))/86400.0)) AS overlap_days
        FROM gc_rx g JOIN lm ON g.hadm_id = lm.hadm_id
        WHERE g.pe_factor IS NOT NULL
    ) x GROUP BY hadm_id
),
-- ------------------------------------------------- activity proxies: ICD organ
organ AS (
    SELECT hadm_id,
      MAX(CASE WHEN icd_version=10 AND icd_code LIKE 'M321%' OR icd_version=9 AND icd_code LIKE '58381%' THEN 1 ELSE 0 END) AS act_nephritis,
      MAX(CASE WHEN icd_version=10 AND (icd_code LIKE 'M3212%' OR icd_code LIKE 'M3213%') THEN 1 ELSE 0 END) AS act_serositis,
      MAX(CASE WHEN icd_version=10 AND icd_code IN ('M3219','M3211','M3215','M3210') THEN 1 ELSE 0 END) AS act_other_organ,
      MAX(CASE WHEN icd_version=10 AND (icd_code LIKE 'D68%' OR icd_code LIKE 'D686%') THEN 1 ELSE 0 END) AS act_aps,
      MAX(CASE WHEN icd_version=10 AND (LEFT(icd_code,3) BETWEEN 'D50' AND 'D77')
               OR icd_version=9 AND (LEFT(icd_code,3) BETWEEN '280' AND '289') THEN 1 ELSE 0 END) AS act_cytopenia,
      MAX(CASE WHEN icd_version=10 AND (LEFT(icd_code,3) BETWEEN 'N17' AND 'N19')
               OR icd_version=9 AND (LEFT(icd_code,3) BETWEEN '584' AND '586') THEN 1 ELSE 0 END) AS act_renal_fail,
      MAX(CASE WHEN icd_version=10 AND (LEFT(icd_code,3) IN ('J90','J91','I30','I31'))
               OR icd_version=9 AND (LEFT(icd_code,3) IN ('511','420','423')) THEN 1 ELSE 0 END) AS act_effusion,
      COUNT(DISTINCT icd_code) AS n_dx_codes
    FROM mimiciv_hosp.diagnoses_icd WHERE hadm_id IN (SELECT hadm_id FROM lm) GROUP BY hadm_id
),
-- ------------------------------------------------- activity proxies: labs
labact AS (
    SELECT l.hadm_id,
           MIN(l.valuenum) FILTER (WHERE d.label='C3')  AS c3_min,
           MIN(l.valuenum) FILTER (WHERE d.label='C4')  AS c4_min,
           MAX(l.valuenum) FILTER (WHERE d.label='C-Reactive Protein') AS crp_max,
           MAX(l.valuenum) FILTER (WHERE d.label='Sedimentation Rate') AS esr_max,
           MAX(l.valuenum) FILTER (WHERE d.label='Protein/Creatinine Ratio') AS upcr_max
    FROM mimiciv_hosp.labevents l
    JOIN mimiciv_hosp.d_labitems d ON l.itemid = d.itemid
    JOIN lm ON l.hadm_id = lm.hadm_id
    WHERE d.label IN ('C3','C4','C-Reactive Protein','Sedimentation Rate','Protein/Creatinine Ratio')
      AND l.valuenum IS NOT NULL
      AND l.charttime >= lm.admittime - INTERVAL '1 day'
      AND l.charttime <  lm.intime + INTERVAL '2 days'
    GROUP BY l.hadm_id
),
-- ------------------------------------------------- positive control: glucose
glu AS (
    SELECT l.hadm_id,
           MAX(l.valuenum) FILTER (WHERE l.charttime >= lm.intime
                                     AND l.charttime <  lm.intime + INTERVAL '48 hours') AS glu_max_post48,
           MAX(l.valuenum) FILTER (WHERE l.charttime <  lm.intime
                                     AND l.charttime >= lm.intime - INTERVAL '48 hours') AS glu_max_pre48
    FROM mimiciv_hosp.labevents l
    JOIN mimiciv_hosp.d_labitems d ON l.itemid = d.itemid
    JOIN lm ON l.hadm_id = lm.hadm_id
    WHERE d.label='Glucose' AND l.valuenum IS NOT NULL
    GROUP BY l.hadm_id
),
ins AS (
    SELECT DISTINCT p.hadm_id FROM mimiciv_hosp.prescriptions p JOIN lm ON p.hadm_id=lm.hadm_id
    WHERE p.drug ILIKE '%insulin%' AND p.starttime >= lm.intime
      AND p.starttime < lm.intime + INTERVAL '48 hours'
),
-- ------------------------------------------------- timed outcomes
cult AS (
    SELECT m.hadm_id,
           MAX(CASE WHEN m.charttime <  lm.intime THEN 1 ELSE 0 END) AS culture_pre_icu,
           MAX(CASE WHEN m.charttime >= lm.intime + INTERVAL '24 hours' THEN 1 ELSE 0 END) AS culture_after24,
           MAX(CASE WHEN m.charttime >= lm.intime THEN 1 ELSE 0 END) AS culture_after_icu
    FROM mimiciv_hosp.microbiologyevents m JOIN lm ON m.hadm_id = lm.hadm_id
    WHERE m.spec_type_desc ILIKE '%BLOOD%' AND m.org_name IS NOT NULL AND m.charttime IS NOT NULL
    GROUP BY m.hadm_id
),
abx AS (
    SELECT p.hadm_id,
           MAX(CASE WHEN p.starttime < lm.intime THEN 1 ELSE 0 END) AS abx_before_icu,
           MAX(CASE WHEN p.starttime >= lm.intime + INTERVAL '48 hours' THEN 1 ELSE 0 END) AS abx_new_after48
    FROM mimiciv_hosp.prescriptions p JOIN lm ON p.hadm_id=lm.hadm_id
    WHERE (p.drug ILIKE '%vancomycin%' OR p.drug ILIKE '%meropenem%' OR p.drug ILIKE '%piperacillin%'
        OR p.drug ILIKE '%cefepime%' OR p.drug ILIKE '%ceftazidime%' OR p.drug ILIKE '%ceftriaxone%'
        OR p.drug ILIKE '%levofloxacin%' OR p.drug ILIKE '%ciprofloxacin%' OR p.drug ILIKE '%linezolid%'
        OR p.drug ILIKE '%daptomycin%' OR p.drug ILIKE '%amikacin%' OR p.drug ILIKE '%gentamicin%'
        OR p.drug ILIKE '%tobramycin%' OR p.drug ILIKE '%azithromycin%' OR p.drug ILIKE '%clindamycin%'
        OR p.drug ILIKE '%metronidazole%' OR p.drug ILIKE '%fluconazole%' OR p.drug ILIKE '%caspofungin%'
        OR p.drug ILIKE '%micafungin%' OR p.drug ILIKE '%voriconazole%' OR p.drug ILIKE '%ampicillin%'
        OR p.drug ILIKE '%nafcillin%' OR p.drug ILIKE '%oxacillin%' OR p.drug ILIKE '%ertapenem%'
        OR p.drug ILIKE '%imipenem%' OR p.drug ILIKE '%sulfamethoxazole%' OR p.drug ILIKE '%doxycycline%'
        OR p.drug ILIKE '%cefazolin%' OR p.drug ILIKE '%aztreonam%' OR p.drug ILIKE '%moxifloxacin%')
      AND p.starttime IS NOT NULL
    GROUP BY p.hadm_id
),
ondan AS (
    SELECT p.hadm_id,
           MAX(CASE WHEN p.stoptime > lm.intime AND p.starttime < lm.intime + INTERVAL '24 hours'
                    THEN 1 ELSE 0 END) AS ondan24
    FROM mimiciv_hosp.prescriptions p JOIN lm ON p.hadm_id=lm.hadm_id
    WHERE (p.drug ILIKE '%ondansetron%' OR p.drug ILIKE '%zofran%') AND p.starttime IS NOT NULL
    GROUP BY p.hadm_id
),
gib AS (
    SELECT DISTINCT d.hadm_id FROM mimiciv_hosp.diagnoses_icd d JOIN lm ON d.hadm_id=lm.hadm_id
    WHERE (d.icd_version=10 AND (d.icd_code LIKE 'K922%' OR d.icd_code LIKE 'K250%' OR d.icd_code LIKE 'K252%'
        OR d.icd_code LIKE 'K254%' OR d.icd_code LIKE 'K256%' OR d.icd_code LIKE 'K260%' OR d.icd_code LIKE 'K262%'
        OR d.icd_code LIKE 'K264%' OR d.icd_code LIKE 'K266%' OR d.icd_code LIKE 'K270%' OR d.icd_code LIKE 'K272%'
        OR d.icd_code LIKE 'K274%' OR d.icd_code LIKE 'K276%' OR d.icd_code LIKE 'K280%' OR d.icd_code LIKE 'K282%'
        OR d.icd_code LIKE 'K284%' OR d.icd_code LIKE 'K286%' OR d.icd_code LIKE 'K290%' OR d.icd_code LIKE 'I850%'
        OR d.icd_code LIKE 'K921%' OR d.icd_code LIKE 'K298%'))
       OR (d.icd_version=9 AND (d.icd_code LIKE '5789%' OR d.icd_code LIKE '5310%' OR d.icd_code LIKE '5312%'
        OR d.icd_code LIKE '5314%' OR d.icd_code LIKE '5316%' OR d.icd_code LIKE '5320%' OR d.icd_code LIKE '5322%'
        OR d.icd_code LIKE '5324%' OR d.icd_code LIKE '5326%' OR d.icd_code LIKE '5330%' OR d.icd_code LIKE '5332%'
        OR d.icd_code LIKE '5334%' OR d.icd_code LIKE '5336%' OR d.icd_code LIKE '5340%' OR d.icd_code LIKE '5342%'
        OR d.icd_code LIKE '5344%' OR d.icd_code LIKE '5346%' OR d.icd_code LIKE '5780%'))
),
inf AS (
    SELECT DISTINCT d.hadm_id FROM mimiciv_hosp.diagnoses_icd d JOIN lm ON d.hadm_id=lm.hadm_id
    WHERE (d.icd_version = 10 AND (
             LEFT(d.icd_code,2) = 'J1'
          OR LEFT(d.icd_code,3) IN ('A04','A40','A41','A49','J85','K61','K65','L02','L03','L08',
                                    'M86','N10','N39','R65','T79','T81','T82','T84')))
       OR (d.icd_version = 9  AND (
             LEFT(d.icd_code,3) IN ('038','320','322','324','420','421','481','482','483','484',
                                    '485','486','507','510','513','540','567','590','599','680',
                                    '681','682','683','684','685','686','711','730','998')
          OR d.icd_code LIKE '9959%'))
),
nadm AS (
    SELECT a.subject_id, COUNT(*) AS n_prior_adm
    FROM mimiciv_hosp.admissions a
    WHERE a.subject_id IN (SELECT subject_id FROM lm)
    GROUP BY a.subject_id
)
SELECT
    lm.subject_id, lm.hadm_id, lm.stay_id, lm.age, lm.intime, lm.outtime, lm.los AS icu_los,
    lm.admittime, lm.deathtime,
    LEAST(24.0, EXTRACT(EPOCH FROM (lm.outtime - lm.intime))/3600.0) AS win24_hours,
    CASE WHEN lm.los >= 1 THEN 1 ELSE 0 END AS los24,
    EXTRACT(EPOCH FROM (lm.intime - lm.admittime))/3600.0 AS preicu_hours,
    CASE WHEN lm.deathtime IS NOT NULL AND lm.deathtime <= lm.intime + INTERVAL '30 days' THEN 1 ELSE 0 END AS death_30d,
    -- outcomes
    CASE WHEN i.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS infection_any,
    COALESCE(a.abx_before_icu,0)   AS abx_before_icu,
    COALESCE(a.abx_new_after48,0)  AS abx_new_after48,
    COALESCE(c.culture_after24,0)  AS culture_after24,
    COALESCE(c.culture_after_icu,0) AS culture_after_icu,
    COALESCE(c.culture_pre_icu,0)  AS culture_pre_icu,
    -- exposure (24 h landmark)
    COALESCE(g24.gc24_cum,0)  AS gc24_cum_pe_mg,
    g24.gc24_peak             AS gc24_peak_daily_pe_mg,
    -- positive control outcome
    gl.glu_max_post48, gl.glu_max_pre48,
    CASE WHEN ii.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS insulin_after_icu,
    -- negative exposure / outcome
    COALESCE(o.ondan24,0) AS ondan24,
    CASE WHEN gb.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS gi_bleed,
    -- activity proxies
    COALESCE(og.act_nephritis,0) AS act_nephritis,
    COALESCE(og.act_serositis,0) AS act_serositis,
    COALESCE(og.act_other_organ,0) AS act_other_organ,
    COALESCE(og.act_aps,0) AS act_aps,
    COALESCE(og.act_cytopenia,0) AS act_cytopenia,
    COALESCE(og.act_renal_fail,0) AS act_renal_fail,
    COALESCE(og.act_effusion,0) AS act_effusion,
    COALESCE(og.n_dx_codes,0) AS n_dx_codes,
    la.c3_min, la.c4_min, la.crp_max, la.esr_max, la.upcr_max,
    COALESCE(na.n_prior_adm,1) AS n_prior_adm
FROM lm
LEFT JOIN gc_lan g24 ON lm.hadm_id = g24.hadm_id
LEFT JOIN inf i      ON lm.hadm_id = i.hadm_id
LEFT JOIN cult c     ON lm.hadm_id = c.hadm_id
LEFT JOIN abx a      ON lm.hadm_id = a.hadm_id
LEFT JOIN glu gl     ON lm.hadm_id = gl.hadm_id
LEFT JOIN ins ii     ON lm.hadm_id = ii.hadm_id
LEFT JOIN ondan o    ON lm.hadm_id = o.hadm_id
LEFT JOIN gib gb     ON lm.hadm_id = gb.hadm_id
LEFT JOIN organ og   ON lm.hadm_id = og.hadm_id
LEFT JOIN labact la  ON lm.hadm_id = la.hadm_id
LEFT JOIN nadm na    ON lm.subject_id = na.subject_id
ORDER BY lm.subject_id, lm.intime
"""


def main():
    c = psycopg2.connect(dbname="mimiciv", **CONFIG)
    df = pd.read_sql_query(SQL, c)
    c.close()

    df["win24_days"] = np.maximum(df["win24_hours"] / 24.0, 1.0 / 24.0)
    df["gc24_daily_pe_mg"] = df["gc24_cum_pe_mg"] / df["win24_days"]
    df["gc24_any"] = (df["gc24_cum_pe_mg"] > 0).astype(int)

    def stratum(v):
        if v <= 0:
            return "G0_none"
        if v < 10:
            return "G1_low"
        if v < 50:
            return "G2_mod"
        return "G3_high"

    df["gc24_str"] = df["gc24_daily_pe_mg"].apply(stratum)

    # incident hyperglycaemia: hyper after ICU, not before
    df["hyper_post"] = np.where(df.glu_max_post48 >= 180, 1, np.where(df.glu_max_post48.isna(), np.nan, 0))
    df["hyper_incident"] = np.where(
        (df.glu_max_post48 >= 180) & (df.glu_max_pre48 < 180), 1,
        np.where(df.glu_max_post48.isna(), np.nan, 0))

    # activity score (claims-based, 100% coverage)
    df["act_score"] = (df.act_nephritis * 2 + df.act_serositis + df.act_other_organ * 2
                       + df.act_aps + df.act_cytopenia + df.act_renal_fail + df.act_effusion)

    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, "cohort_v4_fixes.csv")
    df.to_csv(out, index=False)
    print("saved", out, " rows:", len(df))

    P = print
    P("\n=== landmark eligibility ===")
    P("  all stays                : %d" % len(df))
    P("  ICU LOS >= 24 h          : %d (%.1f%%)" % (df.los24.sum(), 100 * df.los24.mean()))
    sub = df[df.los24 == 1]
    P("\n=== exposure (GC in first 24 h of ICU) ===")
    P("  any GC   : %d (%.1f%%)" % (sub.gc24_any.sum(), 100 * sub.gc24_any.mean()))
    P("  strata   : " + sub.gc24_str.value_counts().reindex(
        ["G0_none", "G1_low", "G2_mod", "G3_high"]).to_dict().__str__())
    pos = sub.loc[sub.gc24_daily_pe_mg > 0, "gc24_daily_pe_mg"]
    P("  dose intensity among exposed (n=%d): median %.1f  IQR %.1f-%.1f  max %.1f" % (
        len(pos), pos.median(), pos.quantile(.25), pos.quantile(.75), pos.max()))

    P("\n=== timed outcomes (landmark cohort, n=%d) ===" % len(sub))
    for v in ["infection_any", "abx_before_icu", "culture_pre_icu", "abx_new_after48",
              "culture_after24", "gi_bleed", "death_30d"]:
        P("  %-20s %4d (%.1f%%)" % (v, int(sub[v].sum()), 100 * sub[v].mean()))
    P("  prevalent infection at ICU admission (abx_before_icu or culture_pre_icu): %d (%.1f%%)" % (
        ((sub.abx_before_icu == 1) | (sub.culture_pre_icu == 1)).sum(),
        100 * ((sub.abx_before_icu == 1) | (sub.culture_pre_icu == 1)).mean()))

    P("\n=== positive control: hyperglycaemia ===")
    P("  glu_max_post48 available : %d (%.1f%%)" % (sub.glu_max_post48.notna().sum(), 100 * sub.glu_max_post48.notna().mean()))
    P("  hyper >=180              : %d (%.1f%%)" % (int(sub.hyper_post.sum()), 100 * sub.hyper_post.mean(skipna=True)))
    P("  incident hyper           : %d (%.1f%%)" % (int(sub.hyper_incident.sum()), 100 * sub.hyper_incident.mean(skipna=True)))
    P("  insulin after ICU        : %d (%.1f%%)" % (int(sub.insulin_after_icu.sum()), 100 * sub.insulin_after_icu.mean()))
    P("  mean glu_max_post48 by GC: none %.1f | any %.1f" % (
        sub.loc[sub.gc24_any == 0, "glu_max_post48"].mean(),
        sub.loc[sub.gc24_any == 1, "glu_max_post48"].mean()))

    P("\n=== activity proxy coverage ===")
    for v in ["act_nephritis", "act_serositis", "act_other_organ", "act_aps",
              "act_cytopenia", "act_renal_fail", "act_effusion"]:
        P("  %-18s %4d (%.1f%%)" % (v, int(sub[v].sum()), 100 * sub[v].mean()))
    P("  act_score distribution: " + sub.act_score.value_counts().sort_index().to_dict().__str__())
    P("\n=== lab activity subset (landmark cohort) ===")
    for v in ["c3_min", "c4_min", "crp_max", "esr_max", "upcr_max"]:
        P("  %-10s n=%3d (%.1f%%)  median %.1f" % (
            v, int(sub[v].notna().sum()), 100 * sub[v].notna().mean(), sub[v].median()))


if __name__ == "__main__":
    main()
