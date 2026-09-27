# -*- coding: utf-8 -*-
"""
eICU-CRD extraction -- harmonised to the MIMIC-IV v4 definitions.

Harmonised design (identical in all three databases)
---------------------------------------------------
  cohort    SLE (ICD-10 M32* / ICD-9 7100* / eICU lupus diagnosis string),
            adults >= 18 y, FIRST ICU stay of the hospital stay
  landmark  ICU length of stay >= 24 h
  exposure  glucocorticoid (prednisone-equivalent) in the FIRST 24 h of ICU,
            expressed as prescribed DAILY dose (mg PE/day)
  outcomes  O1  blood culture positive  >= 24 h after ICU admission  (timed)
            O2  new systemic antibiotic started >= 48 h after ICU admission (timed)
            O3  any infection diagnosis code during the hospitalisation (UNTIMED)
            PC  max glucose in ICU [0, 48 h) >= 180 mg/dL   <- positive control
            NC  gastrointestinal bleeding diagnosis code    <- negative outcome
  covariates age, sex, mechanical ventilation <=24 h, vasopressor <=24 h,
             renal failure, organ-dysfunction count, APACHE acute physiology score

eICU quirks handled here
------------------------
  * all timing is integer MINUTES offset from unit admission
  * `age` is text ('> 89' for the de-identified top code)
  * `medication.dosage` is free text ('40 mg', '1 EA', '125 3') -> parsed in Python;
    when the dosage carries no unit the strength is taken from the drug name
  * `diagnosis.icd9code` is a comma-separated list, sometimes mixing ICD-9 and ICD-10
  * `infusiondrug` has only a single offset (no start/stop)
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
    SELECT DISTINCT patientunitstayid
    FROM eicu_crd.diagnosis
    WHERE diagnosisstring ILIKE '%lupus%' OR icd9code LIKE '7100%'
),
p0 AS (
    SELECT pt.patientunitstayid, pt.patienthealthsystemstayid, pt.gender, pt.age,
           pt.hospitaladmitoffset, pt.unitdischargeoffset,
           pt.hospitaldischargestatus, pt.unitdischargestatus,
           pt.unittype, pt.apacheadmissiondx,
           ROW_NUMBER() OVER (PARTITION BY pt.patienthealthsystemstayid
                              ORDER BY pt.hospitaladmitoffset) AS rn
    FROM eicu_crd.patient pt JOIN sle ON pt.patientunitstayid = sle.patientunitstayid
),
p AS (SELECT * FROM p0 WHERE rn = 1),
-- ------------------------------------------------------------ diagnosis codes
dxs AS (
    SELECT d.patientunitstayid,
           TRIM(BOTH ' ' FROM u.code) AS code
    FROM eicu_crd.diagnosis d,
         UNNEST(STRING_TO_ARRAY(COALESCE(d.icd9code,''), ',')) AS u(code)
    WHERE d.patientunitstayid IN (SELECT patientunitstayid FROM p)
),
dx AS (
    SELECT patientunitstayid,
      MAX(CASE WHEN code LIKE '7100%' OR code LIKE 'M32%' THEN 1 ELSE 0 END) AS dx_sle,
      MAX(CASE WHEN LEFT(code,3) IN ('584','585','586') OR LEFT(code,3) IN ('N17','N18','N19')
               THEN 1 ELSE 0 END) AS renal_fail,
      MAX(CASE WHEN LEFT(code,3) IN ('578') OR LEFT(code,4) IN ('5310','5312','5314','5316',
               '5320','5322','5324','5326','5330','5332','5334','5336','5340','5342','5344',
               '5346','5780','5781','5789') OR LEFT(code,4) IN ('K922','K250','K252','K254',
               'K256','K260','K262','K264','K266','K270','K272','K274','K276','K280','K282',
               'K284','K286','K290','K921','K298','I850')
               THEN 1 ELSE 0 END) AS gi_bleed,
      MAX(CASE WHEN LEFT(code,3) IN ('038','320','322','324','420','421','481','482','483',
               '484','485','486','507','510','513','540','567','590','599','680','681','682',
               '683','684','685','686','711','730','998')
               OR LEFT(code,3) IN ('A40','A41','A49','J85','K61','K65','L02','L03','L08',
               'M86','N10','N39','R65','T81','T82','T84')
               OR LEFT(code,2) = 'J1' OR LEFT(code,4) = '9959'
               THEN 1 ELSE 0 END) AS infect_icd,
      MAX(CASE WHEN LEFT(code,4) IN ('5185','5188') OR LEFT(code,3) IN ('J96','J80') THEN 1 ELSE 0 END) AS resp_fail,
      MAX(CASE WHEN LEFT(code,4) IN ('7855') OR LEFT(code,3) IN ('R57') THEN 1 ELSE 0 END) AS shock,
      MAX(CASE WHEN LEFT(code,3) IN ('570','572') OR LEFT(code,3) IN ('K72','K70') THEN 1 ELSE 0 END) AS liver_fail,
      MAX(CASE WHEN LEFT(code,4) IN ('2866','2869','2867') OR LEFT(code,3) IN ('D65','D68') THEN 1 ELSE 0 END) AS coagulop
    FROM dxs GROUP BY patientunitstayid
),
-- ------------------------------------------------------------ steroids (medication)
gc AS (
    SELECT m.patientunitstayid, m.drugname, m.dosage, m.frequency, m.routeadmin,
           m.drugstartoffset, m.drugstopoffset, m.prn, m.drugordercancelled
    FROM eicu_crd.medication m JOIN p ON m.patientunitstayid = p.patientunitstayid
    WHERE (m.drugname ILIKE '%predni%' OR m.drugname ILIKE '%methylpred%'
        OR m.drugname ILIKE '%solumedrol%' OR m.drugname ILIKE '%dexameth%'
        OR m.drugname ILIKE '%hydrocort%' OR m.drugname ILIKE '%cortis%'
        OR m.drugname ILIKE '%betameth%' OR m.drugname ILIKE '%triamcin%'
        OR m.drugname ILIKE '%fludrocort%')
),
-- ------------------------------------------------------------ antibiotics
abx AS (
    SELECT m.patientunitstayid, MIN(m.drugstartoffset) AS abx_first_off
    FROM eicu_crd.medication m JOIN p ON m.patientunitstayid = p.patientunitstayid
    WHERE (m.drugname ILIKE '%vancomycin%' OR m.drugname ILIKE '%meropenem%'
        OR m.drugname ILIKE '%piperacillin%' OR m.drugname ILIKE '%cefepime%'
        OR m.drugname ILIKE '%ceftazidime%' OR m.drugname ILIKE '%ceftriaxone%'
        OR m.drugname ILIKE '%levofloxacin%' OR m.drugname ILIKE '%ciprofloxacin%'
        OR m.drugname ILIKE '%linezolid%' OR m.drugname ILIKE '%daptomycin%'
        OR m.drugname ILIKE '%amikacin%' OR m.drugname ILIKE '%gentamicin%'
        OR m.drugname ILIKE '%tobramycin%' OR m.drugname ILIKE '%azithromycin%'
        OR m.drugname ILIKE '%clindamycin%' OR m.drugname ILIKE '%metronidazole%'
        OR m.drugname ILIKE '%fluconazole%' OR m.drugname ILIKE '%caspofungin%'
        OR m.drugname ILIKE '%micafungin%' OR m.drugname ILIKE '%voriconazole%'
        OR m.drugname ILIKE '%ampicillin%' OR m.drugname ILIKE '%nafcillin%'
        OR m.drugname ILIKE '%oxacillin%' OR m.drugname ILIKE '%ertapenem%'
        OR m.drugname ILIKE '%imipenem%' OR m.drugname ILIKE '%sulfamethoxazole%'
        OR m.drugname ILIKE '%doxycycline%' OR m.drugname ILIKE '%cefazolin%'
        OR m.drugname ILIKE '%aztreonam%' OR m.drugname ILIKE '%moxifloxacin%')
      AND m.drugstartoffset IS NOT NULL
      AND COALESCE(m.drugordercancelled,'No') <> 'Yes'
    GROUP BY m.patientunitstayid
),
-- ------------------------------------------------------------ microbiology
cult AS (
    SELECT m.patientunitstayid,
           COUNT(*) AS n_culture_any,
           MAX(CASE WHEN m.culturetakenoffset >= 1440 THEN 1 ELSE 0 END) AS culture_after24,
           MAX(CASE WHEN m.culturetakenoffset >= 1440
                     AND m.organism IS NOT NULL
                     AND LOWER(m.organism) NOT LIKE 'no growth%'
                     AND LOWER(m.organism) NOT IN ('other','') THEN 1 ELSE 0 END) AS culture_pos_after24,
           MAX(CASE WHEN m.culturetakenoffset >= 1440 THEN m.organism END) AS org_after24
    FROM eicu_crd.microlab m JOIN p ON m.patientunitstayid = p.patientunitstayid
    WHERE m.culturesite ILIKE '%blood%' AND m.culturetakenoffset IS NOT NULL
    GROUP BY m.patientunitstayid
),
-- ------------------------------------------------------------ positive control: glucose
glu AS (
    SELECT l.patientunitstayid,
           MAX(l.labresult) AS glu_max_48h
    FROM eicu_crd.lab l JOIN p ON l.patientunitstayid = p.patientunitstayid
    WHERE l.labname IN ('glucose','bedside glucose') AND l.labresult IS NOT NULL
      AND l.labresultoffset >= 0 AND l.labresultoffset < 2880
      AND l.labresult BETWEEN 10 AND 1500
    GROUP BY l.patientunitstayid
),
-- ------------------------------------------------------------ support
vaso AS (
    SELECT DISTINCT i.patientunitstayid FROM eicu_crd.infusiondrug i JOIN p ON i.patientunitstayid = p.patientunitstayid
    WHERE (i.drugname ILIKE '%norepinephrine%' OR i.drugname ILIKE '%levophed%'
        OR i.drugname ILIKE '%epinephrine%' OR i.drugname ILIKE '%dopamine%'
        OR i.drugname ILIKE '%phenylephrine%' OR i.drugname ILIKE '%neosynephrine%'
        OR i.drugname ILIKE '%vasopressin%' OR i.drugname ILIKE '%dobutamine%')
      AND i.infusionoffset >= 0 AND i.infusionoffset < 1440
),
vent AS (
    SELECT DISTINCT t.patientunitstayid FROM eicu_crd.treatment t JOIN p ON t.patientunitstayid = p.patientunitstayid
    WHERE t.treatmentstring ILIKE '%mechanical ventilation%' AND t.treatmentoffset < 1440
),
-- NB: apachepatientresult holds several rows per stay (APACHE IV and IVa
-- versions). Without aggregation it silently multiplies the cohort ~1.9x.
aps AS (
    SELECT a.patientunitstayid, MAX(a.acutephysiologyscore) AS acutephysiologyscore
    FROM eicu_crd.apachepatientresult a JOIN p ON a.patientunitstayid = p.patientunitstayid
    GROUP BY a.patientunitstayid
),
cult_any AS (
    SELECT m.patientunitstayid, COUNT(*) AS n_culture_any_site
    FROM eicu_crd.microlab m JOIN p ON m.patientunitstayid = p.patientunitstayid
    WHERE m.culturetakenoffset IS NOT NULL
    GROUP BY m.patientunitstayid
)
SELECT p.patientunitstayid AS stay_key, p.patienthealthsystemstayid AS hosp_key,
       p.age AS age_raw, p.gender, p.unitdischargeoffset AS icu_los_min,
       p.hospitaladmitoffset AS preicu_min,
       p.hospitaldischargestatus, p.unitdischargestatus,
       COALESCE(d.renal_fail,0) AS renal_fail, COALESCE(d.gi_bleed,0) AS gi_bleed,
       COALESCE(d.infect_icd,0) AS infect_icd, COALESCE(d.resp_fail,0) AS resp_fail,
       COALESCE(d.shock,0) AS shock, COALESCE(d.liver_fail,0) AS liver_fail,
       COALESCE(d.coagulop,0) AS coagulop,
       a.abx_first_off,
       COALESCE(c.n_culture_any,0) AS n_culture_any,
       COALESCE(c.culture_after24,0) AS culture_after24,
       COALESCE(c.culture_pos_after24,0) AS culture_pos_after24,
       COALESCE(ca.n_culture_any_site,0) AS n_culture_any_site,
       g.glu_max_48h,
       CASE WHEN v.patientunitstayid IS NOT NULL THEN 1 ELSE 0 END AS vaso24,
       CASE WHEN ve.patientunitstayid IS NOT NULL THEN 1 ELSE 0 END AS vent24,
       s.acutephysiologyscore AS apache_aps
FROM p
LEFT JOIN dx d  ON p.patientunitstayid = d.patientunitstayid
LEFT JOIN abx a ON p.patientunitstayid = a.patientunitstayid
LEFT JOIN cult c ON p.patientunitstayid = c.patientunitstayid
LEFT JOIN glu g ON p.patientunitstayid = g.patientunitstayid
LEFT JOIN vaso v ON p.patientunitstayid = v.patientunitstayid
LEFT JOIN vent ve ON p.patientunitstayid = ve.patientunitstayid
LEFT JOIN aps s  ON p.patientunitstayid = s.patientunitstayid
LEFT JOIN cult_any ca ON p.patientunitstayid = ca.patientunitstayid
ORDER BY p.patientunitstayid
"""

GC_SQL = """
SELECT m.patientunitstayid AS stay_key, m.drugname, m.dosage, m.frequency,
       m.routeadmin, m.drugstartoffset, m.drugstopoffset, m.prn
FROM eicu_crd.medication m
JOIN (%s) p ON m.patientunitstayid = p.patientunitstayid
WHERE (m.drugname ILIKE '%%predni%%' OR m.drugname ILIKE '%%methylpred%%'
    OR m.drugname ILIKE '%%solumedrol%%' OR m.drugname ILIKE '%%dexameth%%'
    OR m.drugname ILIKE '%%hydrocort%%' OR m.drugname ILIKE '%%cortis%%'
    OR m.drugname ILIKE '%%betameth%%' OR m.drugname ILIKE '%%triamcin%%'
    OR m.drugname ILIKE '%%fludrocort%%')
"""

# ---------------------------------------------------------------- dose parsing
PE = [
    (r"methylpred|solumedrol|medrol", 1.25),
    (r"prednisolone", 1.0),
    (r"prednisone|predni(?!solone)", 1.0),
    (r"hydrocort|cortef", 0.25),
    (r"dexameth", 6.67),
    (r"betameth", 6.67),
    (r"triamcin", 1.25),
    (r"fludrocort", 0.25),
]

ONCE = re.compile(r"(once|one|1\s*x\s*only|1xonly|x\s*1|single dose)", re.I)
FREQ_MAP = [
    (re.compile(r"q\s*1\s*h", re.I), 24.0),
    (re.compile(r"q\s*2\s*h", re.I), 12.0),
    (re.compile(r"q\s*3\s*h", re.I), 8.0),
    (re.compile(r"q\s*4\s*h", re.I), 6.0),
    (re.compile(r"q\s*6\s*h", re.I), 4.0),
    (re.compile(r"q\s*8\s*h", re.I), 3.0),
    (re.compile(r"q\s*12\s*h", re.I), 2.0),
    (re.compile(r"q\s*24\s*h|q\s*day|qd\b", re.I), 1.0),
    (re.compile(r"\bqid\b", re.I), 4.0),
    (re.compile(r"\btid\b", re.I), 3.0),
    (re.compile(r"\bbid\b", re.I), 2.0),
    (re.compile(r"daily|every day|q\s*d\b|each day", re.I), 1.0),
    (re.compile(r"q\s*(\d+)\s*h", re.I), None),  # handled generically below
]


def pe_factor(name):
    n = (name or "").lower()
    for pat, f in PE:
        if re.search(pat, n):
            return f
    return None


def parse_freq(s):
    """Doses per 24 h. 'Once'-type orders return None (handled as a single bolus)."""
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
        if v is not None and pat.search(t):
            return v, False
    return 1.0, False


def parse_dose(dosage, drugname):
    """mg per administration."""
    d = (dosage or "").strip()
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(mg|mcg|microgram|gram|g|ea|each|tab|tablet|ml|unit|unt)?", d, re.I)
    if m:
        val = float(m.group(1))
        unit = (m.group(2) or "").lower()
        if unit in ("mg", ""):
            return val
        if unit in ("g", "gram"):
            return val * 1000.0
        if unit in ("mcg", "microgram"):
            return val / 1000.0
        # EA / tablet / mL -> strength must come from the drug name
    # fall back to the strength embedded in the drug name
    n = drugname or ""
    m2 = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*MG\b", n, re.I)
    if m2:
        m3 = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)", d)
        mult = float(m3.group(1)) if m3 else 1.0
        if re.search(r"\bEA\b|\btab|tablet", d, re.I) or not m3:
            return float(m2.group(1)) * mult
        return float(m2.group(1))
    return None


def build_exposure(gcdf, cohort_keys):
    """Prescribed daily prednisone-equivalent dose (mg/day) inside the first 24 h."""
    rows = []
    for _, r in gcdf.iterrows():
        f = pe_factor(r.drugname)
        if f is None:
            continue
        start = r.drugstartoffset
        stop = r.drugstopoffset
        if start is None:
            continue
        start = float(start)
        stop = float(stop) if stop is not None else start
        if stop < start:                       # frequent eICU rounding artefact
            stop = start
        # window = first 24 h of ICU  -> [0, 1440] minutes
        win_end = 1440.0
        if stop <= 0 or start >= win_end:
            continue                            # order does not touch the window
        dose = parse_dose(r.dosage, r.drugname)
        if dose is None or dose <= 0:
            continue
        freq24, is_once = parse_freq(r.frequency)
        if is_once:
            contrib = dose                      # single bolus inside the window
        else:
            overlap = max(0.0, min(stop, win_end) - max(start, 0.0))
            if overlap <= 0:
                overlap = win_end - max(start, 0.0)   # assume it runs to window end
            frac = overlap / 1440.0
            contrib = dose * freq24 * frac
        rows.append((r.stay_key, dose * freq24 * f, contrib * f))
    if not rows:
        return pd.DataFrame({"stay_key": [], "gc24_daily_pe_mg": []})
    g = pd.DataFrame(rows, columns=["stay_key", "daily_full", "contrib"])
    out = g.groupby("stay_key").agg(gc24_daily_pe_mg=("contrib", "sum"),
                                    gc24_peak_daily_pe_mg=("daily_full", "max")).reset_index()
    return out


def main():
    c = psycopg2.connect(dbname="eicu", **CONFIG)
    df = pd.read_sql_query(SQL, c)
    gc = pd.read_sql_query(
        GC_SQL % "SELECT DISTINCT patientunitstayid FROM eicu_crd.diagnosis "
                 "WHERE diagnosisstring ILIKE '%%lupus%%' OR icd9code LIKE '7100%%'", c)
    c.close()

    P = print
    P("=== eICU raw ===")
    P("  SLE first-ICU stays          : %d" % len(df))
    P("  steroid order rows retrieved : %d" % len(gc))

    df["db"] = "eicu"
    df["age"] = pd.to_numeric(df.age_raw.replace("> 89", "90"), errors="coerce")
    df = df[df.age >= 18].copy()
    P("  adults >=18 y                : %d" % len(df))
    df["icu_los_days"] = df.icu_los_min / 1440.0
    lm = df[df.icu_los_min >= 1440].copy()
    P("  landmark (ICU LOS >= 24 h)   : %d" % len(lm))

    exp = build_exposure(gc, set(lm.stay_key))
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

    # outcomes
    lm["hyper_48h"] = np.where(lm.glu_max_48h >= 180, 1,
                               np.where(lm.glu_max_48h.isna(), np.nan, 0))
    lm["death_hosp"] = (lm.hospitaldischargestatus.astype(str).str.lower() == "expired").astype(int)
    lm["female"] = (lm.gender.astype(str).str.upper() == "FEMALE").astype(int)
    lm["organ_dysf"] = (lm.renal_fail + lm.resp_fail + lm.shock + lm.liver_fail + lm.coagulop)
    # timed antibiotic outcome
    lm["abx_early"] = (lm.abx_first_off < 1440).fillna(False).astype(int)
    lm["abx_new_after48"] = (lm.abx_first_off >= 2880).fillna(False).astype(int)

    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, "cohort_eicu.csv")
    lm.to_csv(out, index=False)
    P("\n  saved -> %s" % out)

    P("\n=== exposure (landmark cohort n=%d) ===" % len(lm))
    P("  any GC in first 24 h : %d (%.1f%%)" % (lm.gc_any24.sum(), 100 * lm.gc_any24.mean()))
    P("  strata               : " + lm.gc24_str.value_counts().reindex(
        ["G0_none", "G1_low", "G2_mod", "G3_high"]).fillna(0).astype(int).to_dict().__str__())
    pos = lm.loc[lm.gc24_daily_pe_mg > 0, "gc24_daily_pe_mg"]
    if len(pos):
        P("  dose among exposed   : n=%d median %.1f  IQR %.1f-%.1f  max %.1f" % (
            len(pos), pos.median(), pos.quantile(.25), pos.quantile(.75), pos.max()))

    P("\n=== outcomes (landmark cohort) ===")
    for v in ["infect_icd", "culture_after24", "culture_pos_after24", "n_culture_any",
              "n_culture_any_site", "abx_early", "abx_new_after48", "gi_bleed", "death_hosp"]:
        if v.startswith("n_culture"):
            P("  %-20s total=%d" % (v, int(lm[v].sum())))
        else:
            P("  %-20s %4d (%.1f%%)" % (v, int(lm[v].sum()), 100 * lm[v].mean()))
    P("  glucose measured     : %d (%.1f%%)" % (
        lm.glu_max_48h.notna().sum(), 100 * lm.glu_max_48h.notna().mean()))
    P("  hyper >=180 mg/dL    : %d (%.1f%%)" % (
        int(lm.hyper_48h.sum()), 100 * lm.hyper_48h.mean(skipna=True)))

    P("\n=== covariates ===")
    for v in ["vaso24", "vent24", "renal_fail", "shock", "resp_fail"]:
        P("  %-12s %4d (%.1f%%)" % (v, int(lm[v].sum()), 100 * lm[v].mean()))
    P("  age median %.1f  female %.1f%%  apache_aps available %d" % (
        lm.age.median(), 100 * lm.female.mean(), lm.apache_aps.notna().sum()))


if __name__ == "__main__":
    main()
