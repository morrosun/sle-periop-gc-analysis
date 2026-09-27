# -*- coding: utf-8 -*-
"""
154_extract_abx_outcomes.py

增加「新起始广谱抗生素」这一结局族，并把它做成**真正的 incident 事件**。

为什么需要这一步
----------------
v2 用的 abx_new_after48 是：
    MAX(CASE WHEN starttime >= intime + 48 h THEN 1 ELSE 0 END)
即"住院期间任意一剂广谱抗生素出现在 ICU 入室 48 h 之后"。
这个窗口 87.3% 的人其实在 24 h 内就已经用上抗生素了（中位 2.3 h），
所以该定义主要在测 **继续用药**（continuation），不是新发事件。
实测：SLE 组 56.1% 命中，正好等于 v2 记录的 243/433 —— 事件数"多"是这么来的。

本脚本重建三类定义，并统一窗口：
  A  legacy_any48  任意一剂 >=48 h（v2 口径，保留以对齐历史）
  B  new_agent     某 agent 首次启用落在 [24 h, 7 d)，且该 agent 在 24 h 前未用过
                    -> 真正的"新起始广谱抗生素"（agent 级 new-user）
  C  incident      在 B 的基础上，再要求 24 h landmark 时完全未用广谱抗生素
                    -> 最严格的 incident（新用户 + 新药）

同时补齐与 B 同窗口（24 h - 7 d）的菌血症与培养强度，使主/次结局口径一致，
并输出竞争风险所需的时间信息（到出院/死亡的小时数）。

输出: data/abx_outcomes_rheum.csv
"""
import os

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
CONFIG = dict(
    host=os.environ.get("MIMIC_DB_HOST", "localhost"),
    port=int(os.environ.get("MIMIC_DB_PORT", "5432")),
    user=os.environ.get("MIMIC_DB_USER", "postgres"),
    password=os.environ.get("MIMIC_DB_PASSWORD", ""),
)

# 广谱抗菌药 -> agent 类；每个药只归一类（注意：本串会按参数注入，不要写 %%，也不加 % 格式化）
AGENT_SQL = """
         CASE
           WHEN pr.drug ILIKE '%vancomycin%'       THEN 'vancomycin'
           WHEN pr.drug ILIKE '%piperacillin%'     THEN 'pip-tazo'
           WHEN pr.drug ILIKE '%meropenem%'        THEN 'meropenem'
           WHEN pr.drug ILIKE '%imipenem%'         THEN 'imipenem'
           WHEN pr.drug ILIKE '%ertapenem%'        THEN 'ertapenem'
           WHEN pr.drug ILIKE '%cefepime%'         THEN 'cefepime'
           WHEN pr.drug ILIKE '%ceftazidime%'      THEN 'ceftazidime'
           WHEN pr.drug ILIKE '%ceftriaxone%'      THEN 'ceftriaxone'
           WHEN pr.drug ILIKE '%cefazolin%'        THEN 'cefazolin'
           WHEN pr.drug ILIKE '%levofloxacin%'     THEN 'levofloxacin'
           WHEN pr.drug ILIKE '%ciprofloxacin%'    THEN 'ciprofloxacin'
           WHEN pr.drug ILIKE '%moxifloxacin%'     THEN 'moxifloxacin'
           WHEN pr.drug ILIKE '%linezolid%'        THEN 'linezolid'
           WHEN pr.drug ILIKE '%daptomycin%'       THEN 'daptomycin'
           WHEN pr.drug ILIKE '%amikacin%'         THEN 'amikacin'
           WHEN pr.drug ILIKE '%gentamicin%'       THEN 'gentamicin'
           WHEN pr.drug ILIKE '%tobramycin%'       THEN 'tobramycin'
           WHEN pr.drug ILIKE '%azithromycin%'     THEN 'azithromycin'
           WHEN pr.drug ILIKE '%clindamycin%'      THEN 'clindamycin'
           WHEN pr.drug ILIKE '%metronidazole%'    THEN 'metronidazole'
           WHEN pr.drug ILIKE '%fluconazole%'      THEN 'fluconazole'
           WHEN pr.drug ILIKE '%caspofungin%'      THEN 'caspofungin'
           WHEN pr.drug ILIKE '%micafungin%'       THEN 'micafungin'
           WHEN pr.drug ILIKE '%voriconazole%'     THEN 'voriconazole'
           WHEN pr.drug ILIKE '%ampicillin%'       THEN 'ampicillin'
           WHEN pr.drug ILIKE '%nafcillin%'        THEN 'nafcillin'
           WHEN pr.drug ILIKE '%oxacillin%'        THEN 'oxacillin'
           WHEN pr.drug ILIKE '%aztreonam%'        THEN 'aztreonam'
           WHEN pr.drug ILIKE '%sulfamethoxazole%' THEN 'tmp-smx'
           WHEN pr.drug ILIKE '%doxycycline%'      THEN 'doxycycline'
           ELSE NULL END AS agent
"""

# 需要剔除的"非治疗性/预防性"用药：外科预防与局部/非常规全身用药
EXCLUDE_AGENT = ["cefazolin", "doxycycline", "azithromycin", "gentamicin",
                 "tobramycin", "ampicillin", "nafcillin", "oxacillin"]

LANDMARK_H = 24.0
WINDOW_H = 168.0        # 7 d

lines = []


def P(s=""):
    print(s)
    lines.append(str(s))


d = pd.read_csv(os.path.join(DATA, "cohort_rheum_icu.csv"))
d["intime"] = pd.to_datetime(d.intime)
P("cohort rows = %d  (stay_key == hadm_id)" % len(d))

cc = psycopg2.connect(dbname="mimiciv", **CONFIG)
cur = cc.cursor()
cur.execute("CREATE TEMP TABLE prb(stay_key BIGINT PRIMARY KEY, intime TIMESTAMP)")
execute_values(cur, "INSERT INTO prb (stay_key, intime) VALUES %s",
               list(zip(d.stay_key.astype(int), d.intime)), page_size=5000)
cc.commit()

# ------------------------------------------------- 1) agent 级首启/末次
q_ag = ("WITH ag AS (SELECT pr.hadm_id, " + AGENT_SQL +
        ", pr.starttime FROM mimiciv_hosp.prescriptions pr "
        "WHERE pr.starttime IS NOT NULL AND pr.hadm_id IN (SELECT stay_key FROM prb)) "
        "SELECT hadm_id, agent, MIN(starttime) AS t_min, MAX(starttime) AS t_max "
        "FROM ag WHERE agent IS NOT NULL GROUP BY hadm_id, agent")
ag = pd.read_sql_query(q_ag, cc)
ag["t_min"] = pd.to_datetime(ag.t_min)
ag["t_max"] = pd.to_datetime(ag.t_max)
key = d.set_index("stay_key")["intime"]
ag["intime"] = ag.hadm_id.map(key)
ag["h_min"] = (ag.t_min - ag.intime).dt.total_seconds() / 3600.0
ag["h_max"] = (ag.t_max - ag.intime).dt.total_seconds() / 3600.0
P("  agent-level rows: %d ; hadm with >=1 agent: %d" % (len(ag), ag.hadm_id.nunique()))

# 治疗性 subset（剔除预防/非治疗药）
dt = ag[~ag.agent.isin(EXCLUDE_AGENT)].copy()
P("  therapeutic-subset agents retained: %s" % sorted(dt.agent.unique()))

# ------------------------------------- 2) 24 h - 7 d 窗口内的培养
q_cult = r"""
SELECT m.hadm_id AS stay_key,
       COUNT(*) FILTER (WHERE m.charttime >= p.intime + INTERVAL '24 hours'
                          AND m.charttime <  p.intime + INTERVAL '168 hours') AS n_culture_24_7d,
       MAX(CASE WHEN m.charttime >= p.intime + INTERVAL '24 hours'
                 AND m.charttime <  p.intime + INTERVAL '168 hours'
                 AND m.org_name IS NOT NULL THEN 1 ELSE 0 END) AS culture_pos_24_7d
FROM mimiciv_hosp.microbiologyevents m
JOIN prb p ON m.hadm_id = p.stay_key
WHERE m.spec_type_desc ILIKE '%BLOOD%' AND m.charttime IS NOT NULL
GROUP BY m.hadm_id
"""
cult = pd.read_sql_query(q_cult, cc)

# ------------------------------------- 3) 出院/死亡时间（竞争风险用）
q_adm = r"""
SELECT a.hadm_id AS stay_key, a.dischtime, a.deathtime, a.hospital_expire_flag
FROM mimiciv_hosp.admissions a
WHERE a.hadm_id IN (SELECT stay_key FROM prb)
"""
adm = pd.read_sql_query(q_adm, cc)

# ------------------- 4) 阴性暴露对照：前 24 h 内与感染无关的常规用药
# 这类药是"ICU 照护强度"的标记，但没有已知的感染药理效应。
# 若它们也给出与 GC 同向的"剂量-感染"关联，说明捕捉到的是体系性混杂。
q_neg = r"""
SELECT pr.hadm_id AS stay_key,
       MAX(CASE WHEN pr.starttime < p.intime + INTERVAL '24 hours'
                THEN 1 ELSE 0 END) AS ondan24
FROM mimiciv_hosp.prescriptions pr JOIN prb p ON pr.hadm_id = p.stay_key
WHERE pr.drug ILIKE '%ondansetron%'
GROUP BY pr.hadm_id
"""
q_neg2 = r"""
SELECT pr.hadm_id AS stay_key,
       MAX(CASE WHEN pr.drug ILIKE '%docusate%' AND pr.starttime < p.intime + INTERVAL '24 hours'
                THEN 1 ELSE 0 END) AS docusate24,
       MAX(CASE WHEN (pr.drug ILIKE '%acetaminophen%' OR pr.drug ILIKE '%paracetamol%')
                 AND pr.starttime < p.intime + INTERVAL '24 hours'
                THEN 1 ELSE 0 END) AS apap24,
       MAX(CASE WHEN (pr.drug ILIKE '%senna%' OR pr.drug ILIKE '%bisacodyl%')
                 AND pr.starttime < p.intime + INTERVAL '24 hours'
                THEN 1 ELSE 0 END) AS laxative24
FROM mimiciv_hosp.prescriptions pr JOIN prb p ON pr.hadm_id = p.stay_key
GROUP BY pr.hadm_id
"""
neg = pd.read_sql_query(q_neg, cc)[["stay_key", "ondan24"]]
neg2 = pd.read_sql_query(q_neg2, cc)
cc.close()

# ---------------------------------------------------------------- assemble
res = d[["stay_key", "primary_grp", "gc24_str", "intime"]].copy()

# A legacy：任意一剂 >=48 h（v2 口径）
mm = ag.groupby("hadm_id").h_max.max().reindex(res.stay_key)
res["legacy_any48"] = (mm >= 48).astype(int).values

# 24 h 前已用过的 agent 集合
pre = dt[dt.h_min < LANDMARK_H]
pre_pairs = set(zip(pre.hadm_id, pre.agent))
pre_agents = pre.groupby("hadm_id").agent.nunique().reindex(res.stay_key).fillna(0).astype(int)
res["n_agent_pre24"] = pre_agents.values
res["abx_naive_at24"] = (res.n_agent_pre24 == 0).astype(int)

# B new_agent：首启在 [24 h, 7 d) 且 24 h 前未用过该 agent
cand = dt[(dt.h_min >= LANDMARK_H) & (dt.h_min < WINDOW_H)]
cand = cand[[(r.hadm_id, r.agent) not in pre_pairs for r in cand.itertuples()]]
nb = cand.groupby("hadm_id").agg(n_agent_new=("agent", "nunique"),
                                 t_new_h=("h_min", "min"))
res["n_agent_new_24_7d"] = res.stay_key.map(nb.n_agent_new).fillna(0).astype(int)
res["t_new_agent_h"] = res.stay_key.map(nb.t_new_h)
res["new_agent_24_7d"] = (res.n_agent_new_24_7d > 0).astype(int)

# C incident：再要求 landmark 时完全 naive
res["incident_strict"] = ((res.abx_naive_at24 == 1) & (res.new_agent_24_7d == 1)).astype(int)

# 延伸到出院的新起始（不加 7 d 上界，用作敏感性）
cand2 = dt[dt.h_min >= LANDMARK_H]
cand2 = cand2[[(r.hadm_id, r.agent) not in pre_pairs for r in cand2.itertuples()]]
res["new_agent_anytime"] = res.stay_key.isin(cand2.hadm_id.unique()).astype(int)

# 培养
res = res.merge(cult, on="stay_key", how="left")
res["n_culture_24_7d"] = res.n_culture_24_7d.fillna(0).astype(int)
res["culture_pos_24_7d"] = res.culture_pos_24_7d.fillna(0).astype(int)

# 阴性暴露对照
res = res.merge(neg, on="stay_key", how="left").merge(neg2, on="stay_key", how="left")
for c in ["ondan24", "docusate24", "apap24", "laxative24"]:
    res[c] = res[c].fillna(0).astype(int)

# 出院 / 死亡（竞争风险）
res = res.merge(adm, on="stay_key", how="left")
for c in ["dischtime", "deathtime"]:
    res[c] = pd.to_datetime(res[c])
res["h_disch"] = (res.dischtime - res.intime).dt.total_seconds() / 3600.0
res["h_death"] = (res.deathtime - res.intime).dt.total_seconds() / 3600.0
# 到院外（= 可观察窗的终点）与 7 d 内是否死亡
res["h_censor"] = res.h_disch.fillna(res.h_death)
res["death_within7d"] = ((res.h_death >= 0) & (res.h_death < WINDOW_H)).astype(int)
# 窗口完整可观察：存活/在院 >= 7 d
res["window_complete"] = ((res.h_censor.isna()) | (res.h_censor >= WINDOW_H)).astype(int)
res["died_before_window_end"] = (1 - res.window_complete) & (res.death_within7d == 1)
res["died_before_window_end"] = res.died_before_window_end.astype(int)
res["discharged_before_window_end"] = ((1 - res.window_complete) & (res.death_within7d == 0)).astype(int)

res = res.drop(columns=["intime", "dischtime", "deathtime"])

# ---------------------------------------------------------------- report
P("")
P("=" * 96)
P("OUTCOME DEFINITIONS  (landmark = ICU intime + 24 h ; window = 24 h - 7 d)")
P("=" * 96)
P("  %-22s %26s %26s %26s" % ("definition", "ALL n=3361", "SLE n=433", "RA n=1203"))
sle = res[res.primary_grp == "SLE"]
ra = res[res.primary_grp == "RA"]
for c, lab in [("legacy_any48", "A legacy any dose >=48h"),
               ("new_agent_24_7d", "B new agent 24h-7d"),
               ("incident_strict", "C incident (naive+new)"),
               ("new_agent_anytime", "B2 new agent, no upper bd"),
               ("culture_pos_24_7d", "D culture + 24h-7d"),
               ("n_agent_pre24", "[n agents before 24h]")]:
    f = (lambda x: 100 * x.mean()) if c != "n_agent_pre24" else (lambda x: x.mean())
    P("  %-22s %14.1f %14.1f %14.1f" % (
        lab, f(res[c]), f(sle[c]), f(ra[c])))
both = res[res.primary_grp.isin(["SLE", "RA"])]
P("")
P("  head-to-head subset (SLE+RA, n=%d) events:" % len(both))
for c in ["legacy_any48", "new_agent_24_7d", "incident_strict", "culture_pos_24_7d"]:
    P("    %-22s %5d" % (c, int(both[c].sum())))
P("")
P("  window completeness (24h landmark -> 7 d):")
P("    complete window (still in hospital or discharged >= 7 d): %d (%.1f%%)" % (
    res.window_complete.sum(), 100 * res.window_complete.mean()))
P("    died before window end: %d (%.1f%%) ; discharged alive before window end: %d (%.1f%%)" % (
    res.died_before_window_end.sum(), 100 * res.died_before_window_end.mean(),
    res.discharged_before_window_end.sum(), 100 * res.discharged_before_window_end.mean()))
P("")
P("  dose-stratum event rate, primary outcomes:")
ORDER = ["G0_none", "G1_low", "G2_mod", "G3_high"]
P("  %-6s %-8s %6s %12s %12s %12s" % ("dis", "stratum", "n", "legacy48", "new_agent", "incident"))
for g in ["SLE", "RA"]:
    s = res[res.primary_grp == g]
    for gs in ORDER:
        t = s[s.gc24_str == gs]
        if len(t) == 0:
            continue
        P("  %-6s %-8s %6d %11.1f%% %11.1f%% %11.1f%%" % (
            g, gs, len(t), 100 * t.legacy_any48.mean(),
            100 * t.new_agent_24_7d.mean(), 100 * t.incident_strict.mean()))

res.to_csv(os.path.join(DATA, "abx_outcomes_rheum.csv"), index=False)
P("")
P("saved -> data/abx_outcomes_rheum.csv  (%d rows x %d cols)" % res.shape)
with open(os.path.join(OUT, "154_extract_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
