# -*- coding: utf-8 -*-
"""
150_probe_rheum_taxonomy.py

Step 0 of the "SLE vs RA head-to-head" redesign (Plan B).

Question this answers BEFORE any modelling:
  If the cohort is widened from SLE-only to "rheumatic / systemic
  autoimmune disease ICU patients", how many stays and -- more
  importantly -- how many EVENTS does each disease group actually
  carry?  A head-to-head comparison is only worth doing if both arms
  have enough events.

Design notes
------------
* Diagnosis codes are taken from MIMIC-IV `hosp.diagnoses_icd`, which
  stores ICD-9 and ICD-10 codes WITHOUT decimal points ('M3210',
  '7100').  Every prefix below is therefore decimal-free.
* RA is defined exactly as in the RA project (ICD-10 M05/M06,
  ICD-9 714) so the two studies remain comparable.
* SLE is defined exactly as in the SLE-only analysis (M32 / 710.0).
* MIMIC-IV codes M06.1 as "adult-onset Still disease" -- NOT RA.
  It is peeled off so the RA group stays clean, and reported
  separately (AOSD is itself a GC-treated systemic autoimmune disease).
* Counts are reported NON-EXCLUSIVELY first (any admission carrying the
  code) plus a full co-occurrence matrix, because secondary Sjogren is
  extremely common in RA and would otherwise silently re-label RA
  patients.  The mutually exclusive assignment used downstream is
  decided from that matrix.

Output
------
  out/150_probe_rheum.txt              human-readable report
  data/rheum_hadm_classified.csv       hadm_id -> disease flags + primary
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
OUT = os.path.join(ROOT, "out")
os.makedirs(DATA, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------------------------------
# disease taxonomy
# --------------------------------------------------------------------------
# name -> (icd10 prefixes, icd9 prefixes).  Decimal-free, as stored.
TAX = [
    ("SLE",         ("M32",),                                   ("7100",)),
    ("RA",          ("M05", "M06"),                              ("714",)),
    ("Vasculitis",  ("M30", "M31", "D69.0"),                     ("446", "2870")),
    ("IIM",         ("M33",),                                    ("7103", "7104")),
    ("SSc",         ("M34",),                                    ("7101",)),
    ("pSS",         ("M35.0",),                                  ("7102",)),
    ("MCTD",        ("M35.1",),                                  ("7108",)),
    ("PMR_GCA",     ("M35.3", "M31.5", "M31.6"),                 ("725", "4465")),
    ("Behcet",      ("M35.2",),                                  ("1361",)),
    ("APS",         ("D68.6",),                                  ("28981",)),
    ("axSpA",       ("M45", "M46"),                              ("720",)),
    ("PsA",         ("M07", "L40.5"),                            ("6960",)),
    ("Sarcoidosis", ("D86",),                                    ("135",)),
    ("RP",          ("M94.1",),                                  ("73399",)),
]
AOSD = ("M06.1",)          # peeled out of RA

DX_SQL = r"""
SELECT DISTINCT hadm_id, icd_version, icd_code
FROM mimiciv_hosp.diagnoses_icd
WHERE (icd_version = 10 AND (
         icd_code LIKE 'M05%%' OR icd_code LIKE 'M06%%' OR icd_code LIKE 'M07%%'
      OR icd_code LIKE 'M08%%' OR icd_code LIKE 'M30%%' OR icd_code LIKE 'M31%%'
      OR icd_code LIKE 'M32%%' OR icd_code LIKE 'M33%%' OR icd_code LIKE 'M34%%'
      OR icd_code LIKE 'M35%%' OR icd_code LIKE 'M45%%' OR icd_code LIKE 'M46%%'
      OR icd_code LIKE 'M94.1%%' OR icd_code LIKE 'D68.6%%' OR icd_code LIKE 'D69.0%%'
      OR icd_code LIKE 'D86%%' OR icd_code LIKE 'D89.1%%' OR icd_code LIKE 'L40.5%%'))
   OR (icd_version = 9 AND (
         icd_code LIKE '714%%' OR icd_code LIKE '710%%' OR icd_code LIKE '446%%'
      OR icd_code LIKE '725%%' OR icd_code LIKE '720%%' OR icd_code LIKE '1361%%'
      OR icd_code LIKE '2870%%' OR icd_code LIKE '28981%%' OR icd_code LIKE '6960%%'
      OR icd_code LIKE '135%%' OR icd_code LIKE '73399%%'))
"""

# first ICU stay per hadm, with age
ICU_SQL = r"""
WITH r AS (
    SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.los,
           ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.intime) AS rn
    FROM mimiciv_icu.icustays i
)
SELECT r.subject_id, r.hadm_id, r.stay_id, r.intime, r.los AS icu_los_days,
       a.hospital_expire_flag, ag.age, pt.gender
FROM r
JOIN mimiciv_hosp.admissions a ON r.hadm_id = a.hadm_id
JOIN mimiciv_hosp.patients pt  ON r.subject_id = pt.subject_id
LEFT JOIN mimiciv_derived.age ag ON r.hadm_id = ag.hadm_id
WHERE r.rn = 1
"""

# crude events, enough to decide which disease groups are analysable
EVENT_SQL = r"""
WITH p AS (
    SELECT i.hadm_id, i.stay_id, i.intime,
           ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.intime) AS rn
    FROM mimiciv_icu.icustays i
), q AS (SELECT * FROM p WHERE rn = 1)
SELECT q.hadm_id,
       COALESCE(m.culture_pos_after24, 0) AS culture_pos_after24,
       COALESCE(gl.hyper_48h, 0)          AS hyper_48h,
       COALESCE(dx.infect_icd, 0)         AS infect_icd,
       COALESCE(dx.count_icd, 0)          AS n_dx
FROM q
LEFT JOIN (
    SELECT m.hadm_id,
           MAX(CASE WHEN m.charttime >= q2.intime + INTERVAL '24 hours'
                     AND m.org_name IS NOT NULL THEN 1 ELSE 0 END) AS culture_pos_after24
    FROM mimiciv_hosp.microbiologyevents m
    JOIN q q2 ON m.hadm_id = q2.hadm_id
    WHERE m.spec_type_desc ILIKE '%%BLOOD%%' AND m.charttime IS NOT NULL
    GROUP BY m.hadm_id
) m ON q.hadm_id = m.hadm_id
LEFT JOIN (
    SELECT l.hadm_id,
           CASE WHEN MAX(l.valuenum) >= 180 THEN 1 ELSE 0 END AS hyper_48h
    FROM mimiciv_hosp.labevents l
    JOIN mimiciv_hosp.d_labitems d ON l.itemid = d.itemid
    JOIN q q2 ON l.hadm_id = q2.hadm_id
    WHERE d.label = 'Glucose' AND l.valuenum IS NOT NULL
      AND l.charttime >= q2.intime AND l.charttime < q2.intime + INTERVAL '48 hours'
      AND l.valuenum BETWEEN 10 AND 1500
    GROUP BY l.hadm_id
) gl ON q.hadm_id = gl.hadm_id
LEFT JOIN (
    SELECT d.hadm_id, COUNT(*) AS count_icd,
           MAX(CASE WHEN (d.icd_version = 10 AND (LEFT(d.icd_code,2) = 'J1'
                OR LEFT(d.icd_code,3) IN ('A40','A41','A49','J85','K61','K65','L02','L03','L08',
                                          'M86','N10','N39','R65','T81','T82','T84')))
                 OR (d.icd_version = 9 AND (LEFT(d.icd_code,3) IN ('038','320','322','324','420','421',
                    '481','482','483','484','485','486','507','510','513','540','567','590','599','680',
                    '681','682','683','684','685','686','711','730','998') OR d.icd_code LIKE '9959%%'))
                THEN 1 ELSE 0 END) AS infect_icd
    FROM mimiciv_hosp.diagnoses_icd d JOIN q q2 ON d.hadm_id = q2.hadm_id
    GROUP BY d.hadm_id
) dx ON q.hadm_id = dx.hadm_id
"""

GC_SQL = r"""
SELECT pr.hadm_id, COUNT(*) AS gc_rows,
       COUNT(DISTINCT pr.starttime::date) AS gc_days
FROM mimiciv_hosp.prescriptions pr
WHERE pr.dose_unit_rx = 'mg'
  AND pr.dose_val_rx ~ '^[0-9]+(\.[0-9]+)?$'
  AND pr.route IN ('IV','PO','PO/NG','IM')
  AND (pr.drug ILIKE '%%predni%%' OR pr.drug ILIKE '%%methylpred%%'
    OR pr.drug ILIKE '%%dexameth%%' OR pr.drug ILIKE '%%hydrocort%%'
    OR pr.drug ILIKE '%%betameth%%' OR pr.drug ILIKE '%%triamcin%%')
GROUP BY pr.hadm_id
"""


def classify(codes10, codes9):
    """Return the set of disease flags present for one admission."""
    flags = set()
    for name, p10, p9 in TAX:
        if any(c.startswith(p) for c in codes10 for p in p10):
            flags.add(name)
        if any(c.startswith(p) for c in codes9 for p in p9):
            flags.add(name)
    if any(c.startswith(AOSD) for c in codes10):
        flags.add("AOSD")
    return flags


def main():
    P = print
    c = psycopg2.connect(dbname="mimiciv", **CONFIG)

    P("=" * 78)
    P("0. PULL")
    P("=" * 78)
    dx = pd.read_sql_query(DX_SQL, c)
    icu = pd.read_sql_query(ICU_SQL, c)
    ev = pd.read_sql_query(EVENT_SQL, c)
    P("  rheum-coded (hadm, icd) rows : %d  (hadm %d)" % (len(dx), dx.hadm_id.nunique()))
    P("  first-ICU-stay admissions    : %d" % len(icu))

    # ---------------------------------------------------------------- flags
    g10 = dx[dx.icd_version == 10].groupby("hadm_id").icd_code.apply(list)
    g9 = dx[dx.icd_version == 9].groupby("hadm_id").icd_code.apply(list)
    flags = {}
    for h in dx.hadm_id.unique():
        flags[h] = classify(g10.get(h, []), g9.get(h, []))
    fl = pd.DataFrame([dict(hadm_id=h, **{n: int(n in s) for n, _, _ in TAX},
                            AOSD=int("AOSD" in s)) for h, s in flags.items()])
    fl["n_flags"] = fl[[n for n, _, _ in TAX]].sum(axis=1)

    NAMES = [n for n, _, _ in TAX]
    P("")
    P("=" * 78)
    P("1. NON-EXCLUSIVE COUNTS  (admissions carrying each code)")
    P("=" * 78)
    P("  %-14s %8s %10s %10s" % ("disease", "hadm", "w/ 1stICU", "LOS>=24h"))
    icu_s = icu.set_index("hadm_id")
    los_ok = np.asarray(icu_s.icu_los_days >= 1)
    for n in NAMES + ["AOSD"]:
        if n not in fl.columns:
            continue
        hs = set(fl.loc[fl[n] == 1, "hadm_id"])
        k = np.asarray(icu_s.index.isin(hs))
        P("  %-14s %8d %10d %10d" % (n, int(fl[n].sum()), int(k.sum()), int((k & los_ok).sum())))
    anyr = fl.loc[fl[NAMES].sum(axis=1) > 0, "hadm_id"]
    ka = np.asarray(icu_s.index.isin(set(anyr)))
    P("  %-14s %8d %10d %10d" % ("ANY rheum", len(anyr), int(ka.sum()),
                                 int((ka & los_ok).sum())))

    # ------------------------------------------------------- co-occurrence
    P("")
    P("=" * 78)
    P("2. CO-OCCURRENCE  (symmetric |I| and Jaccard) -- drives the mutually")
    P("   exclusive assignment.  Watch RA vs pSS: secondary sicca is common.")
    P("=" * 78)
    keep = [n for n in NAMES if fl[n].sum() >= 30]
    P("  " + " " * 12 + "".join("%8s" % n[:7] for n in keep))
    for a in keep:
        row = []
        for b in keep:
            ia, ib = fl[a] == 1, fl[b] == 1
            j = (ia & ib).sum() / max(1, (ia | ib).sum())
            row.append("%8.2f" % j if a != b else "%8s" % "-")
        P("  %-12s" % a + "".join(row))

    # --------------------------------------------------------------- events
    P("")
    P("=" * 78)
    P("3. EVENTS AMONG FIRST-ICU-STAY ADMISSIONS  (this is the binding")
    P("   constraint -- a head-to-head needs events in BOTH arms)")
    P("=" * 78)
    e = icu.merge(ev, on="hadm_id", how="left").merge(
        fl.rename(columns={"hadm_id": "hadm_id"}), on="hadm_id", how="left")
    e[NAMES + ["AOSD"]] = e[NAMES + ["AOSD"]].fillna(0).astype(int)
    e["age"] = pd.to_numeric(e.age, errors="coerce")
    e = e[e.age >= 18]
    e["death_hosp"] = e.hospital_expire_flag.astype(int)
    for c2 in ["culture_pos_after24", "hyper_48h", "infect_icd"]:
        e[c2] = e[c2].fillna(0).astype(int)

    lm = e[e.icu_los_days >= 1].copy()
    P("  %-14s %6s %6s %8s %8s %8s" % (
        "disease", "n24", "death", "cult+", "infect", "hyper"))
    for n in NAMES + ["AOSD"]:
        s = lm[lm[n] == 1]
        if len(s) == 0:
            continue
        P("  %-14s %6d %6d %8d %8d %8d" % (
            n, len(s), int(s.death_hosp.sum()), int(s.culture_pos_after24.sum()),
            int(s.infect_icd.sum()), int(s.hyper_48h.sum())))

    # ------------------------------------------------------------- exposure
    P("")
    P("=" * 78)
    P("4. GLUCOCORTICOID RECORDING DENSITY per disease (landmark cohort)")
    P("=" * 78)
    gc = pd.read_sql_query(GC_SQL, c)
    c.close()
    lm2 = lm.merge(gc, on="hadm_id", how="left")
    lm2["gc_rows"] = lm2.gc_rows.fillna(0)
    lm2["gc_any"] = (lm2.gc_rows > 0).astype(int)
    P("  %-14s %6s %9s %11s" % ("disease", "n24", "any GC%", "rows/stay"))
    for n in NAMES + ["AOSD"]:
        s = lm2[lm2[n] == 1]
        if len(s) == 0:
            continue
        P("  %-14s %6d %8.1f%% %11.2f" % (
            n, len(s), 100 * s.gc_any.mean(), s.gc_rows.sum() / len(s)))

    # mutually exclusive
    PRIORITY = ["SLE", "IIM", "SSc", "Vasculitis", "MCTD", "APS", "AOSD",
                "RA", "pSS", "PMR_GCA", "Behcet", "axSpA", "PsA", "RP",
                "Sarcoidosis"]
    fl["primary"] = "none"
    for n in reversed(PRIORITY):
        fl.loc[fl[n] == 1, "primary"] = n
    fl.to_csv(os.path.join(DATA, "rheum_hadm_classified.csv"), index=False)
    P("")
    P("  mutually exclusive assignment saved -> data/rheum_hadm_classified.csv")
    P("  " + str(fl.primary.value_counts().to_dict()))


if __name__ == "__main__":
    main()
