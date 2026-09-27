# -*- coding: utf-8 -*-
import os
import psycopg2
CONFIG = dict(
    host=os.environ.get("MIMIC_DB_HOST", "localhost"),
    port=int(os.environ.get("MIMIC_DB_PORT", "5432")),
    user=os.environ.get("MIMIC_DB_USER", "postgres"),
    password=os.environ.get("MIMIC_DB_PASSWORD", ""),
)

def cols(schema, table, db="mimiciv"):
    c = psycopg2.connect(dbname=db, **CONFIG); cur = c.cursor()
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema=%s AND table_name=%s ORDER BY ordinal_position", (schema, table))
    r = [x[0] for x in cur.fetchall()]
    print("[%s.%s] %s\n" % (schema, table, r))
    c.close()

for t in ["first_day_lab", "charlson", "apsiii", "oasis", "sapsii", "sofa", "sepsis3", "ventilation", "vasoactive_agent", "first_day_rrt"]:
    cols("mimiciv_derived", t)
cols("mimiciv_icu", "icustays")
cols("mimiciv_hosp", "admissions")
for t in ["labevents", "d_labitems"]:
    cols("mimiciv_hosp", t)
