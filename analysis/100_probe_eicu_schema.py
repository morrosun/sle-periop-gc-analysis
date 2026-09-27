# -*- coding: utf-8 -*-
"""Probe eICU-CRD and NWICU schemas to see whether the MIMIC-IV v4 definitions
(exposure / outcome / covariates) can be reproduced.

What we need in each database
-----------------------------
E1  SLE cohort          : diagnosis table with ICD codes
E2  ICU stay + timing   : intime / outtime / los, admission anchor
E3  GC exposure         : drug name + dose + route + start/stop  (first 24 h of ICU)
E4  Timed infection     : microbiology culture with TIMESTAMP  and/or antibiotic starts
E5  Positive control    : glucose lab values with timestamps
E6  Covariates          : age, vasopressor, mechanical ventilation, renal failure ...
"""
import os
import psycopg2

CONFIG = dict(
    host=os.environ.get("MIMIC_DB_HOST", "localhost"),
    port=int(os.environ.get("MIMIC_DB_PORT", "5432")),
    user=os.environ.get("MIMIC_DB_USER", "postgres"),
    password=os.environ.get("MIMIC_DB_PASSWORD", ""),
)

Q = [
    ("eicu schemas", "eicu",
     "SELECT table_schema, table_name FROM information_schema.tables "
     "WHERE table_schema NOT IN ('pg_catalog','information_schema') ORDER BY 1,2"),
    ("eicu patient cols", "eicu",
     "SELECT column_name, data_type FROM information_schema.columns "
     "WHERE table_name='patient' ORDER BY ordinal_position"),
    ("nwicu schemas", "nwicu",
     "SELECT table_schema, table_name FROM information_schema.tables "
     "WHERE table_schema NOT IN ('pg_catalog','information_schema') ORDER BY 1,2"),
]


def show(cur, title):
    rows = cur.fetchall()
    print("\n--- %s (%d rows) ---" % (title, len(rows)))
    for r in rows:
        print("   ", " | ".join(str(x) for x in r))


def main():
    for dbname in ["eicu", "nwicu"]:
        try:
            c = psycopg2.connect(dbname=dbname, **CONFIG)
        except Exception as ex:
            print("!! cannot connect %s: %s" % (dbname, ex))
            continue
        cur = c.cursor()
        print("\n" + "=" * 70)
        print("DATABASE:", dbname)
        print("=" * 70)
        cur.execute(
            "SELECT table_schema, table_name FROM information_schema.tables "
            "WHERE table_schema NOT IN ('pg_catalog','information_schema') "
            "ORDER BY 1,2")
        show(cur, "%s tables" % dbname)
        c.close()

    # ---- detailed column probes for the tables we are most likely to need
    detail = {
        "eicu": ["patient", "diagnosis", "treatment", "medication", "infusiondrug",
                 "microlab", "lab", "apacheapsvar", "apachepatientresult",
                 "nursecharting", "intakeoutput", "vitalperiodic", "customLab"],
        "nwicu": ["patients", "admissions", "icustays", "diagnoses_icd", "procedures_icd",
                  "prescriptions", "microbiologyevents", "labevents", "inputevents",
                  "outputevents", "chartevents", "d_labitems", "d_icd_diagnoses"],
    }
    for dbname, tabs in detail.items():
        try:
            c = psycopg2.connect(dbname=dbname, **CONFIG)
        except Exception as ex:
            continue
        cur = c.cursor()
        for t in tabs:
            cur.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE lower(table_name)=lower(%s) ORDER BY ordinal_position", (t,))
            rows = cur.fetchall()
            if not rows:
                print("\n--- %s.%s : TABLE NOT FOUND ---" % (dbname, t))
                continue
            print("\n--- %s.%s ---" % (dbname, t))
            print("   ", ", ".join("%s:%s" % (a, b) for a, b in rows))
        c.close()


if __name__ == "__main__":
    main()
