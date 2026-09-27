# -*- coding: utf-8 -*-
"""
168_activity_rule_engine.py
===========================
v8 —— 给关键词判定器补三条规则，并把它换成**连续变量**。

v7b 盲法核验实测（out/_probe_v8.txt 的证据驱动结果）：
  任务 A 「INACTIVE」PPV = 8.6%（3/35）；任务 B「ACTIVITY-POSITIVE」PPV = 15.6%（5/32）。
  误判机制**不是随机噪声**，而是三类可枚举的缺陷：

  (1) 裸词无边限定 —— `asymptomatic` 单独解释 44/54 的 INACTIVE 判定，
      但它命中的几乎全是**非风湿领域**：
        asymptomatic AFib with RVR / NSVT /  bacteriuria (E. coli) / hypertension /
        gallstones / hyponatremia / troponin elevation / hypoxemia / pneumothorax /
        ovarian cyst / aortic aneurysm / tamponade / soft BPs …
      同类：`remission` 命中 "lung cancer in remission"、"bladder cancer in remission"；
      `free of disease` 命中冠脉造影 "the RCA was a large caliber vessel and free of disease"；
      `has been well` 命中 "has been well controlled on outpatient lisinopril"（高血压）。

  (2) 否定/反转语境未处理 —— "has not achieved remission" 这类会被读成「无活动」。

  (3) 领域判断缺失 —— 任务 B 的 27 例假阳性几乎全由**非风湿领域的通用词**造成：
        `exacerbation`（CHF / HF / COPD / 胰腺炎 / 心衰）、`thrombocytopenia`
        （HIT / 万古霉素诱导 / 化疗 / lymphoma / MGUS 背景）、`pancytopenia`（MGUS-淋巴瘤）、
        `pneumonitis`（误吸）、`interstitial lung disease`（PMH 列表）、
        `rheumatic`（rheumatic fever）、`lupus nephritis`（ESRD 的既往病因，非当前活动）、
        "concern that this pain may represent a lupus flare"（**假说**而非断言）。

三条新规则
  R1 实体词限定 —— 弱线索（asymptomatic / doing well / free of / 裸 remission / well-controlled）
     必须与风湿病实体词在**同一句且距离最近**；否则不作数（＝编码手册硬规则 3）。
  R2 否定检测 —— ① 活动线索沿用 NegEx 前文窗口（"no evidence of active lupus" 不是活动）；
     ② 不活动线索若处于**反转语境**（not / never / no longer / failed to / unable to /
     refused）则作废；③ 假说语境（concern / possible / may / r/o / differential）不作断言。
  R3 领域判断 —— 线索必须落在**风湿病领域**：最近的风湿实体必须比最近的非风湿实体更近；
     通用器官词（thrombocytopenia / pancytopenia / pneumonitis / ILD / exacerbation …）
     还必须与风湿实体在 ±60 字符内共现；既往/sequela 标记（ESRD / s/p / history of /
     secondary to / transplant）后的线索作废；`rheumatic fever` 等黑名单直接剔除。

连续变量（替代三分类）
  aid_doc     活动度相关信息密度 = (n_act_pos + n_act_neg + n_inact + n_defer) / 千字符
              → 这一条才是三分类真正想抓的轴（"记录里到底有没有谈活动度"）
  aid_signed  带符号活动度信息密度 = (n_act_pos − n_inact) / 千字符  → 方向
  aid_act     纯活动信息密度 = n_act_pos / 千字符

输出
  data/gc_activity_density.csv   每 stay 一行（v7 标签 + v8 标签 + 连续变量 + 组件计数）
  out/168_engine_qc.txt          分布、逐词命中、与 v7 的对照
"""
import importlib.util
import os
import re
import sys

import numpy as np
import pandas as pd
import psycopg2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

L = []


def P(s=""):
    print(s)
    sys.stdout.flush()
    L.append(str(s))


def load_163():
    p = os.path.join(ROOT, "scripts", "163_extract_inactivity.py")
    spec = importlib.util.spec_from_file_location("s163", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = load_163()

# ==================================================================== 词表
# ---- 风湿病实体词（R1/R3 的「锚」）——注意**不含裸 disease**（癌症也会 disease）
RHEUM_ENT = re.compile(
    r"(lupus|\bsle\b|systemic lupus|rheumatoid arthritis|\brheumatoid\b|\bra\b|"
    r"synovitis|arthralgia|joint pain|morning stiffness|joint swelling|"
    r"(?<!osteo)arthritis|nephritis|serositis|vasculitis|myositis|"
    r"dermatomyositis|polymyositis|sjogren|scleroderma|systemic sclerosis|"
    r"connective tissue disease|\bctd\b|\bmctd\b|antiphospholipid|"
    r"disease activity|active disease|\bflare\b|flaring|quiescent|"
    r"hydroxychloroquine|\bhcq\b|azathioprine|mycophenolate|methotrexate|"
    r"rituximab|belimumab|tocilizumab|leflunomide|sulfasalazine|"
    r"rheumatolog\w*|immunosuppress\w*|"
    r"malar|discoid|photosensitivity|oral ulcer|anti-?dsdna|\bdsdna\b|"
    r"\bsledai\b|polymyalgia|antinuclear|"
    r"\bana\b|\bc3\b|\bc4\b|complement|rheumatic)", re.I)

# ---- 非风湿领域锚（R3 的「反锚」）
NONRHEUM = re.compile(
    r"(afib|atrial fibrillation|nsvt|ventricular tachycardia|\bvt\b|tachycardia|"
    r"bradycardia|hypotension|hypertension|blood pressure|\bbp\b|"
    r"congestive heart failure|\bchf\b|heart failure|\bhf\b|hfref|hfpef|"
    r"cardiomyopath|ejection fraction|mitral|aortic stenosis|tricuspid|"
    r"coronary|\brca\b|\blad\b|\blcx\b|\blcx\b|stenosis|stent|angioplasty|"
    r"catheteriz|\bcabg\b|anastomo|bypass graft|stemi|nstemi|\bacs\b|"
    r"troponin|precordial|myocardi|arrhythmi|"
    r"pneumonia|\bpna\b|aspiration|empyema|bronchit|"
    r"copd|asthma|emphysema|bronchospasm|wheez|tracheobronchomalacia|"
    r"pancreatitis|cholangitis|cholecystitis|gallstone|cholelithiasis|biliary|"
    r"\bercp\b|diverticul|colonoscopy|melena|colectomy|gastrectomy|"
    r"bacteriuria|\buti\b|urinary tract infection|e\. ?coli|"
    r"hyponatremia|hypernatremia|hypokalemia|hyperkalemia|hypoglycemia|"
    r"hyperglycemia|ketoacidosis|\bdka\b|anemia of chronic|iron deficiency|"
    r"cancer|carcinoma|adenocarcinoma|lymphoma|leukemi|myeloma|mgus|"
    r"metastatic|malignancy|neoplasm|tumou?r|chemotherap|"
    r"dementia|alzheimer|stroke|\bmca\b|thromboembol|"
    r"pneumothorax|pleural effusion|tamponade|pericardial effusion|"
    r"aneurysm|ovarian cyst|hernia|fracture|osteoporosis|osteoarthritis|"
    r"dialysis|\besrd\b|end[- ]stage|transplant|chronic kidney disease|\bckd\b|"
    r"sepsis|septic|bacteremi|bacteraemi|cellulitis|panniculitis|"
    r"cirrhosis|alcohol|withdrawal|bowel obstruction|\bsbo\b|"
    r"rheumatic fever|rheumatic heart|"
    r"bacteremia|abscess|osteomyelitis|cholecyst|"
    r"stem cell|bone marrow|neutropenic fever|febrile neutropenia)", re.I)

# ---- 既往 / sequela 标记（其后的线索不是「当前活动」）
PAST_MARK = re.compile(
    r"(\besrd\b|end[- ]stage|end stage|\bs/p\b|status post|"
    r"history of|\bh/o\b|prior to|previous|remote history|"
    r"secondary to|due to|attributed to|"
    r"dialysis|transplant|resolved|in remission|chronic\b)", re.I)

# ---- 假说 / 不确定语境（不是断言）
HYPOTH = re.compile(
    r"(\bconcern\w*|\bsuspicion|\bsuspect\w*|\bpossible\b|\bpossibly\b|"
    r"\bmay\b|\bmight\b|\bcould\b|question of|rule out|ruled out|\br/o\b|"
    r"differential|\bversus\b|\bvs\.?\b|evaluate for|workup|"
    r"\bif\b|\bwhether\b|considered less likely|not excluded|"
    r"cannot exclude|\bunclear\b)", re.I)

# ---- 反转语境（把「不活动」反转为「不活动不成立」）
# 注意：**不能**用裸 \bno\b / \bwithout\b —— 实测会误否决
#   "A-fib not on Coumadin, cardiomyopathy, lupus in remission"（窗口里有个 not）
#   "no constitutional symptoms; no evidence of active disease"
# 只保留真正构成「反转」的构造。
INVERT = re.compile(
    r"(no longer|failed to|failure to|unable to|refus\w*|declin\w*|"
    r"not\s+(?:yet\s+)?(?:in\b|achieved|sustained|respond|resolv)|"
    r"never\s+(?:in\b|achieved|sustained))", re.I)
INVERT_WIN = 25

# ---- 黑名单短语（词表命中即剔除）
BLACKLIST = re.compile(r"(rheumatic fever|rheumatic heart|inflammatory bowel|"
                       r"osteoarthritis|degenerative joint)", re.I)

BOUND = M.BOUND

# ==================================================================== 规则开关
# v10 用于**规则消融**：逐条关掉 R1/R2/R3，看 PPV 如何变化。
# 默认全开 —— 关掉任何一条才会改变判定结果，故常规流程与已落盘结果不受影响。
RULES = {"R1": True, "R2": True, "R3": True}


def set_rules(r1=True, r2=True, r3=True):
    """设置启用的规则（消融用）。返回旧设置，便于还原。"""
    old = dict(RULES)
    RULES["R1"], RULES["R2"], RULES["R3"] = bool(r1), bool(r2), bool(r3)
    return old


WS = re.compile(r"\s+")


def norm(t):
    """
    **关键前处理**：MIMIC 出院记录是硬换行的（约 70 字符断行），
    若不先把空白压平，句子会被切碎，药物与理由分到两句，
    句子级规则会全部失效（实测 4/5 例「因感染暂缓」因此漏诊）。
    """
    return WS.sub(" ", (t or "")).strip()


def sent_bounds(text):
    """返回句子区间列表 [(a, b), ...]（文本应已 norm 过）"""
    bnds = []
    a = 0
    for m in re.finditer(r"(?<=[.;!?])(?=\s)|\n", text):
        bnds.append((a, m.start()))
        a = m.end()
    if a < len(text):
        bnds.append((a, len(text)))
    return bnds


def sent_at(bnds, pos):
    for a, b in bnds:
        if a <= pos <= b:
            return (a, b)
    return (0, len(bnds) and 1 or 0)


def _last_dist(rx, text, s, e, window):
    """窗口内最后一次命中的「距离」——(最近端到 s/e 的距离)。"""
    lo = max(0, s - window)
    hi = min(len(text), e + window)
    best_b = None
    best_a = None
    for m in rx.finditer(text[lo:hi]):
        a0, b0 = lo + m.start(), lo + m.end()
        db = s - b0 if b0 <= s else (a0 - e if a0 >= e else 0)
        if best_b is None or db < best_b:
            best_b, best_a = db, (a0, b0)
    return best_b, best_a


def domain_ok(text, s, e, window=300, tight=False):
    """
    R3：最近的风湿实体必须存在，且不比最近的非风湿锚更远。
    tight=True 时窗口收紧到 150 字符（用于弱线索）。
    """
    w = 150 if tight else window
    d_r, _ = _last_dist(RHEUM_ENT, text, s, e, w)
    d_n, _ = _last_dist(NONRHEUM, text, s, e, max(60, w // 3))
    if d_r is None:
        return False
    if d_n is not None and d_n < d_r:
        return False
    return True


def invert_ok(text, s, window=None):
    """R2②：线索之前（不跨句）没有反转语境。窗口故意很窄——反转必须紧贴线索。"""
    window = INVERT_WIN if window is None else window
    pre = text[max(0, s - window):s]
    return not bool(INVERT.search(pre))


def sent_at2(bnds, pos):
    for i, (a, b) in enumerate(bnds):
        if a <= pos <= b:
            return i, (a, b)
    return 0, (0, 0)


def sent_has_rheum(text, s, e, bnds=None, back=1):
    """
    R1 的核心：弱线索必须与风湿病实体在**同一句**（或紧邻上一句，容忍 ";" 被切句）。
    """
    bnds = bnds if bnds is not None else sent_bounds(text)
    i, _ = sent_at2(bnds, s)
    lo = bnds[max(0, i - back)][0] if bnds else 0
    _, hi = bnds[i] if bnds else (0, len(text))
    return bool(RHEUM_ENT.search(text[lo:hi]))


def hypoth_hit(text, s, window=65):
    """R2③：线索之前（不跨句）有假说/不确定语境。"""
    pre = text[max(0, s - window):s]
    last = None
    for m in BOUND.finditer(pre):
        last = m.end()
    if last is not None:
        pre = pre[last:]
    return bool(HYPOTH.search(pre))


def past_hit(text, s, window=48):
    """R3：线索之前（不跨句）有既往/sequela 标记。"""
    pre = text[max(0, s - window):s]
    last = None
    for m in BOUND.finditer(pre):
        last = m.end()
    if last is not None:
        pre = pre[last:]
    return bool(PAST_MARK.search(pre))


# ==================================================================== 不活动线索（分 tier）
# tier: weak  —— 极端通用，必须靠实体词限定（R1）
#       med   —— 词表自带实体槽，但槽位可能被非风湿实体填上
WEAK = "weak"
MED = "med"
INACT_V8 = [
    # ---- 明确无活动：no evidence / no signs 句式
    # 实体槽必须包含 rheumatolog* / autoimmune / connective tissue
    #（实测漏掉 "no evidence of active rheumatologic disease" → 整例丢失）
    ("no_evidence", MED, r"\bno\s+(evidence|signs?|symptoms?|clinical signs?)\s+(of|for)\s+"
     r"(active\s+|any\s+)?(lupus|sle|ra\b|rheumatoid|disease|flares?|"
     r"disease activity|active disease|synovitis|arthritis|arthralgia|"
     r"nephritis|serositis|vasculitis|rheumatolog\w*|autoimmune|"
     r"connective tissue)\b"),
    ("no_evidence", MED, r"\bno\s+(active|flaring|new)\s+(lupus|sle|ra\b|rheumatoid|disease|"
     r"arthritis|synovitis|joint)\b"),
    ("no_evidence", MED, r"\b(no|without)\s+(further\s+|any\s+|recurrent\s+)?(flare|flares|flaring)\b"),
    ("no_evidence", MED, r"\bwithout\s+(evidence of\s+)?(active\s+)?(disease|flare|activity|synovitis)\b"),
    # ---- 缓解
    ("remission", WEAK, r"\b(in|achieved|sustained|remains? in)\s+(complete |clinical |full )?remission\b"),
    ("remission", WEAK, r"\bremission\b"),
    ("remission", MED, r"\b(complete|clinical|full|sustained|drug[- ]free)\s+remission\b"),
    # ---- 静止
    ("quiescent", MED, r"\b(lupus|sle|ra\b|rheumatoid arthritis|disease|arthritis)\s+"
     r"(is|was|remains?|appears?|had been|has been)\s+(in\s+)?"
     r"(remission|inactive|quiescent|dormant|well[- ]controlled|"
     r"stable|controlled|minimal)\b"),
    ("quiescent", MED, r"\b(quiescent|dormant|inactive|clinically inactive)\s+"
     r"(lupus|sle|ra\b|rheumatoid|disease|arthritis)\b"),
    ("quiescent", MED, r"\bdisease activity\s*(:|was|is|remained|appears? to be)?\s*"
     r"(none|nil|negative|zero|0|inactive|quiescent|minimal|low)\b"),
    ("quiescent", MED, r"\bstable\s+(lupus|sle|ra\b|rheumatoid|disease activity)\b"),
    # 问题列表式："Rheumatoid arthritis. Chronic, stable." / "SLE: stable"
    # 注意分隔符要允许**纯空白**——硬换行归一化后 "Rheumatoid arthritis\nChronic, stable"
    # 会变成 "Rheumatoid arthritis Chronic, stable"（实测 A-018 因此丢失）
    ("quiescent", MED, r"\b(lupus|sle|rheumatoid arthritis|\bra\b|rheumatoid|"
     r"sjogren|scleroderma|vasculitis|\bctd\b|polymyalgia|myositis)\b"
     r"[\s.:;,]+(chronic[,\s]+)?"
     r"(stable|controlled|quiescent|inactive|no activity|doing well|"
     r"well[- ]controlled)\b"),
    ("quiescent", MED, r"\bno\s+disease[- ]related\b"),
    ("quiescent", WEAK, r"\b(well[- ]controlled|mildly active|minimally active|"
     r"controlled|stable)\s+(lupus|sle|ra\b|rheumatoid|disease|arthritis)\b"),
    # ---- 无症状
    ("asymptomatic", MED, r"\bdenies\s+(any\s+)?(joint pain|arthralgia|morning stiffness|joint "
     r"swelling|rash|oral ulcer|photosensitivity)\b"),
    ("asymptomatic", MED, r"\bno\s+(joint pain|arthralgia|morning stiffness|joint swelling|"
     r"synovitis on exam|rash|oral ulcer)\b"),
    ("asymptomatic", WEAK, r"\basymptomatic\b"),
    ("asymptomatic", WEAK, r"\bwithout\s+(complaint|symptom)"),
    # ---- free of / doing well
    ("free_of", WEAK, r"\bfree of\s+(active\s+)?(disease|symptoms|joint|synovitis|flares?)\b"),
    ("free_of", WEAK, r"\bhas been\s+(doing\s+)?well\b"),
    ("free_of", WEAK, r"\bdoing well\s+(from|from a|in terms of)\b"),
    ("free_of", MED, r"\bdoing well\s+(from|from a|in terms of)[^.;]{0,40}"
     r"(rheumat|arthritis|lupus|sle|disease)\b"),
]
INACT_V8_RX = [(k, t, re.compile(p, re.I)) for k, t, p in INACT_V8]

# ==================================================================== 活动线索（分 tier）
# specific —— 词本身即风湿特异，自带领域
# generic  —— 通用器官/事件词，必须与风湿实体在 ±60 字符内共现（R3）
ACT_V8 = [
    ("flar", "med", r"\b(lupus|sle|systemic lupus|rheumatoid arthritis|ra|disease)\s+"
     r"(flare|flares|flaring|exacerbation|activity|active)\b"),
    ("flar", "generic", r"\b(flare|flares|flaring|exacerbation)\b"),
    ("flar", "med", r"\b(active|severe|worsening|uncontrolled|refractory)\s+"
     r"(lupus|sle|ra|rheumatoid|arthritis|disease|sarcoidosis|vasculitis)\b"),
    ("flar", "specific", r"\bsynovitis\b"),
    ("flar", "specific", r"\bpolymyalgia\b"),
    ("flar", "specific", r"\brheumatic\b"),
    ("flar", "specific", r"\binflammatory (arthritis|arthralgia)\b"),
    ("renal", "specific", r"\blupus nephritis\b"),
    ("renal", "generic", r"\bglomerulonephritis\b"),
    ("renal", "generic", r"\bnephrotic\b"),
    ("renal", "generic", r"\bproteinuria\b"),
    ("renal", "med", r"\brenal flare\b"),
    ("renal", "generic", r"\bcrescentic\b"),
    ("renal", "generic", r"\bmembranous nephropathy\b"),
    ("pulm", "generic", r"\binterstitial lung disease\b"),
    ("pulm", "generic", r"\bpneumonitis\b"),
    ("pulm", "generic", r"\balveolitis\b"),
    ("pulm", "generic", r"\borganizing pneumonia\b"),
    ("pulm", "specific", r"\b(diffuse )?alveolar hemorrhage\b"),
    ("pulm", "specific", r"\bpulmonary hemorrhage\b"),
    ("pulm", "specific", r"\bpleuritis\b"),
    ("pulm", "specific", r"\bpleurisy\b"),
    ("pulm", "specific", r"\bser[oa]sitis\b"),
    ("heme", "specific", r"\bhemolytic anemia\b"),
    ("heme", "specific", r"\baiha\b"),
    ("heme", "specific", r"\bevans syndrome\b"),
    ("heme", "specific", r"\bimmune thrombocytopenia\b"),
    ("heme", "generic", r"\bthrombocytopenia\b"),
    ("heme", "generic", r"\bitp\b"),
    ("heme", "generic", r"\bttp\b"),
    ("heme", "generic", r"\bmacrophage activation\b"),
    ("heme", "generic", r"\bhemophagocytic\b"),
    ("heme", "generic", r"\bpancytopenia\b"),
    ("neuro", "specific", r"\bnpsle\b"),
    ("neuro", "specific", r"\bneuropsychiatric lupus\b"),
    ("neuro", "specific", r"\bcerebritis\b"),
    ("neuro", "generic", r"\bseizure\b"),
    ("neuro", "specific", r"\bstatus epilepticus\b"),
    ("neuro", "generic", r"\bmyelitis\b"),
    ("neuro", "generic", r"\boptic neuritis\b"),
    ("neuro", "specific", r"\bcns vasculitis\b"),
    ("neuro", "specific", r"\bmyositis\b"),
    ("neuro", "specific", r"\bdermatomyositis\b"),
    ("neuro", "specific", r"\bpolymyositis\b"),
]
ACT_V8_RX = [(k, t, p, re.compile(p, re.I)) for k, t, p in ACT_V8]

# 通用器官词所需的共现半径
GEN_RADIUS = 60
# bare flare / exacerbation 必须**紧贴**风湿实体：否则 "COPD exacerbation" /
# "acute HF exacerbation" / "acute exacerbation of chronic pancreatitis" 都会被
# 算作疾病活动（实测为任务 B 最大的单一误判来源）
TIGHT_RADIUS = {r"\b(flare|flares|flaring|exacerbation)\b": 30}

# 「慢性病 + 维持用药」列表（PMH 列表里 "interstitial lung disease on prednisone"
# 不是活动断言，实测 B-008 因此误判）
ON_DRUG = None   # 在下方药物表定义后初始化（运行时查找，故此处可先置空）

ACTIV_RX = M.ACTIV_RX


# ==================================================================== 评估函数
def eval_activity(text):
    """
    返回 dict：
      act_pos, act_pos_ok, act_neg, act_neg_ok        （去重前的命中数）
      act_pos_kept   —— 通过全部过滤的阳性命中（含原文片段）
      act_neg_kept   —— 通过全部过滤的否定命中
      dropped_*      —— 各规则剔除计数
    """
    bnds = sent_bounds(text)
    n_pos = 0
    n_neg = 0
    all_pos = 0      # 宽口径：只过「黑名单/假说/既往」三道，不过领域与共现半径
    all_neg = 0
    kept_pos = []
    kept_neg = []
    drop = {"neg": 0, "hypoth": 0, "blacklist": 0, "past": 0, "domain": 0,
            "radius": 0, "chronic": 0}
    for k, tier, p, rx in ACT_V8_RX:
        rad = TIGHT_RADIUS.get(p, GEN_RADIUS)
        for m in rx.finditer(text):
            s, e = m.start(), m.end()
            frag = text[max(0, s - 30):min(len(text), e + 30)].replace("\n", " ")
            if RULES["R3"] and BLACKLIST.search(text[max(0, s - 40):e + 40]):
                drop["blacklist"] += 1
                continue
            if RULES["R2"] and hypoth_hit(text, s):
                drop["hypoth"] += 1
                continue
            if RULES["R3"] and past_hit(text, s):
                drop["past"] += 1
                continue
            neg = M.negated(text, s) if RULES["R2"] else False
            if neg:
                all_neg += 1
            else:
                all_pos += 1
            if RULES["R3"] and tier == "generic":
                # R3：① 「慢性病 + 维持用药」列表不算活动；② 必须与风湿实体 ±rad 共现
                if chronic_maintenance(text, e):
                    drop["chronic"] += 1
                    continue
                d_r, _ = _last_dist(RHEUM_ENT, text, s, e, rad)
                d_n, _ = _last_dist(NONRHEUM, text, s, e, rad)
                if d_r is None or d_r > rad or (d_n is not None and d_n < d_r):
                    drop["radius"] += 1
                    continue
            elif RULES["R3"]:
                if not domain_ok(text, s, e, window=200):
                    drop["domain"] += 1
                    continue
            if neg:
                n_neg += 1
                kept_neg.append((k, frag))
            else:
                n_pos += 1
                kept_pos.append((k, frag))
    return dict(act_pos=n_pos, act_neg=n_neg,
                act_all_pos=all_pos, act_all_neg=all_neg,
                act_pos_kept=kept_pos, act_neg_kept=kept_neg, drop=drop)


def eval_inactivity(text):
    bnds = sent_bounds(text)
    kept = []
    drop = {"invert": 0, "domain": 0, "hypoth": 0, "sent": 0}
    for k, tier, rx in INACT_V8_RX:
        for m in rx.finditer(text):
            s, e = m.start(), m.end()
            frag = text[max(0, s - 40):min(len(text), e + 60)].replace("\n", " ")
            if RULES["R3"] and BLACKLIST.search(text[max(0, s - 40):e + 40]):
                drop["domain"] += 1
                continue
            if RULES["R2"] and not invert_ok(text, s):
                drop["invert"] += 1
                continue
            if RULES["R2"] and hypoth_hit(text, s):
                drop["hypoth"] += 1
                continue
            if RULES["R3"] and not domain_ok(text, s, e, tight=(tier == WEAK)):
                drop["domain"] += 1
                continue
            # R1 核心：弱线索必须与风湿实体同句（或紧邻上一句）
            if RULES["R1"] and tier == WEAK and not sent_has_rheum(text, s, e, bnds):
                drop["sent"] += 1
                continue
            kept.append((k, tier, frag))
    kinds = sorted(set(k for k, _, _ in kept))
    return dict(inact_n=len(kept), inact_kinds=",".join(kinds),
                inact_kept=kept, drop=drop)


# ---- 暂缓的判断要素
# v7 的药物表**只有糖皮质激素**，实测把 5/5 例人类认定的「因感染暂缓免疫抑制」全部漏掉
# （原文分别是 methotrexate / CellCept / leflunomide / Actemra 被 held/off of）。
# 编码手册明说该标签覆盖「激素 / 免疫抑制剂」，所以药物表必须加上 DMARD。
GC_DRUG = (r"(prednisone|prednisolone|methylprednisolone|methylpred|medrol|"
           r"solu-?\s?medrol|dexamethasone|hydrocortisone|steroids?|"
           r"glucocorticoids?|corticosteroids?|immunosuppress\w*|"
           r"steroid[- ]sparing)")
DMARD = (r"(methotrexate|\bmtx\b|azathioprine|mycophenolate|mycophenolic|"
         r"cellcept|leflunomide|lefluonamide|hydroxychloroquine|\bhcq\b|"
         r"rituximab|tocilizumab|actemra|belimumab|sulfasalazine|"
         r"cyclophosphamide|tacrolimus|abatacept|ustekinumab|tofacitinib|"
         r"baricitinib|infliximab|adalimumab|etanercept|golimumab|"
         r"certolizumab|secukinumab|ixekizumab|anakinra)")
DRUG = r"(" + GC_DRUG + r"|" + DMARD + r")"
HOLD = (r"(held|hold|holding|withheld|withhold|deferred|defer|delay\w*|"
        r"discontinu\w*|stop\w*|avoid\w*|off of|off (?:her|his|the|their)\b|"
        r"not\s+(?:given|started|initiated|restarted|resumed|administered)|"
        r"on hold|contraindicat\w*|precluded)")
INFE = (r"(infection|infectious|sepsis|septic|bacteremia|bacteraemia|bacteremic|"
        r"pneumonia|bloodstream infection|positive blood culture|"
        r"urinary tract infection|\buti\b|covid|aspiration pneumonia|"
        r"empiric antibiotics|cellulitis|abscess|osteomyelitis)")
# 句级三要素共现：药物 ∧ 减停 ∧ 感染
# （v7 用的是**整篇**共现 → 19/20 假阳性；这里收紧到同一句）
# 但「同一句」还不够：MIMIC 的句子可长到 300+ 字符，实测出现
#   "IV vancomycin and piperacillin/tazobactam were stopped, she was given oral
#    prednisone"（held 说的是抗生素）、
#   "MTX held during admission, will restart"（没有任何感染归因）
# → 必须再要求三个要素**彼此邻近**，故改用「有序 + 有界间隔」的 6 种语序。
DEFER_ORD = [
    r"\b" + DRUG + r"\b[^;]{0,40}?\b" + HOLD + r"\b[^;]{0,60}?\b" + INFE + r"\b",
    r"\b" + HOLD + r"\b[^;]{0,40}?\b" + DRUG + r"\b[^;]{0,60}?\b" + INFE + r"\b",
    r"\b" + DRUG + r"\b[^;]{0,60}?\b" + INFE + r"\b[^;]{0,60}?\b" + HOLD + r"\b",
    r"\b" + INFE + r"\b[^;]{0,60}?\b" + DRUG + r"\b[^;]{0,40}?\b" + HOLD + r"\b",
    r"\b" + HOLD + r"\b[^;]{0,60}?\b" + INFE + r"\b[^;]{0,60}?\b" + DRUG + r"\b",
    r"\b" + INFE + r"\b[^;]{0,60}?\b" + HOLD + r"\b[^;]{0,40}?\b" + DRUG + r"\b",
]
DEFER_SENT = re.compile("|".join("(?:%s)" % p for p in DEFER_ORD), re.I)

# 「慢性病 + 维持用药」：紧跟 "<药> on" 的通用器官词不是活动断言
ON_DRUG = re.compile(
    r"^\s*(?:on|while on|taking|maintained on)\s+"
    r"(?:her|his|their|the|home|chronic)?\s*\b" + DRUG + r"\b", re.I)


def chronic_maintenance(text, e, span=36):
    return bool(ON_DRUG.match(text[e:e + span]))


def eval_defer(text):
    """
    INF_DEFER = 明确的「因感染而减停」陈述。
    v8 口径：**同一句**内同时出现〔药物（GC 或 DMARD）〕〔减停动词〕〔感染〕。
    删掉了 v7 的 pattern E（"药 + held" 无理由）与整篇感染/减停共现弱通道。
    """
    bnds = sent_bounds(text)
    kept = []
    for a, b in bnds:
        sent = text[a:b]
        if len(sent.strip()) < 15:
            continue
        if DEFER_SENT.search(sent):
            kept.append((a, sent.strip()[:240].replace("\n", " ")))
    return dict(defer_n=len(kept), defer_kept=kept)


# ==================================================================== 主流程
def main():
    coh = pd.read_csv(os.path.join(DATA, "cohort_rheum_icu.csv"))
    coh["stay_key"] = coh.stay_key.astype(int)
    both = coh[coh.primary_grp.isin(["SLE", "RA"])].copy()
    keys = sorted(both.stay_key.unique().tolist())
    P("SLE+RA stays = %d (SLE %d / RA %d)" % (
        len(both), int((both.primary_grp == "SLE").sum()),
        int((both.primary_grp == "RA").sum())))

    cc = psycopg2.connect(host=os.environ.get("MIMIC_DB_HOST", "localhost"),
                          port=int(os.environ.get("MIMIC_DB_PORT", "5432")),
                          user=os.environ.get("MIMIC_DB_USER", "postgres"),
                          password=os.environ.get("MIMIC_DB_PASSWORD", ""),
                          dbname="mimiciv")
    parts = []
    B = 400
    for i in range(0, len(keys), B):
        ch = keys[i:i + B]
        q = ("SELECT hadm_id, note_seq, text FROM mimiciv_note.discharge "
             "WHERE hadm_id IN (%s)" % ",".join(map(str, ch)))
        parts.append(pd.read_sql(q, cc))
    cc.close()
    notes = pd.concat(parts, ignore_index=True)
    notes = (notes.sort_values(["hadm_id", "note_seq"])
                  .drop_duplicates("hadm_id", keep="last"))
    P("notes = %d" % len(notes))

    v7 = pd.read_csv(os.path.join(DATA, "gc_inactivity.csv"))
    v7m = {int(r.stay_key): r for r in v7.itertuples()}

    recs = []
    snippet_rows = []
    for r in notes.itertuples():
        t = r.text or ""
        secs = M.split_sections(t)
        hc = norm(M.sec(secs, "brief hospital course", "hospital course"))
        dd = norm(M.sec(secs, "discharge diagnosis"))
        moa = norm(M.sec(secs, "medications on admission"))
        if len(hc) + len(dd) < 200:
            narr = norm(t)          # 段落切分失败则退回全文
        else:
            narr = hc + " " + dd
        nchar = max(1.0, len(narr) / 1000.0)

        ia = eval_inactivity(narr)
        ac = eval_activity(narr)
        df = eval_defer(narr + " " + moa)
        n_eval = sum(len(rx.findall(narr)) for rx in ACTIV_RX)
        n_eval_any = int(n_eval > 0)

        # ---- 连续变量
        info_n = ac["act_pos"] + ac["act_neg"] + ia["inact_n"] + df["defer_n"]
        aid_doc = info_n / nchar
        aid_signed = (ac["act_pos"] - ia["inact_n"]) / nchar
        aid_act = ac["act_pos"] / nchar
        # 宽口径：活动线索只过黑名单/假说/既往，不过领域与共现半径
        info_broad = (ac["act_all_pos"] + ac["act_all_neg"] + ia["inact_n"]
                      + df["defer_n"] + n_eval)
        aid_broad = info_broad / nchar
        aid_eval = n_eval / nchar
        aid_bin = int(info_n > 0)

        # ---- 分半信度（句级奇偶分半；两半各自用自身长度归一）
        sents = [narr[a:b] for a, b in sent_bounds(narr)]
        s1 = " ".join(sents[0::2])
        s2 = " ".join(sents[1::2])

        def _half_info(txt):
            if len(txt.strip()) < 25:
                return np.nan
            i = (eval_inactivity(txt)["inact_n"] + eval_defer(txt)["defer_n"]
                 + eval_activity(txt)["act_all_pos"]
                 + eval_activity(txt)["act_all_neg"])
            return i / max(1.0, len(txt) / 1000.0)

        aid_h1 = _half_info(s1)
        aid_h2 = _half_info(s2)

        # ---- v8 三分类（与 v7 同样的优先级）
        if ia["inact_n"] > 0:
            lab = "INACTIVE"
        elif df["defer_n"] > 0:
            lab = "INF_DEFER"
        else:
            lab = "UNDERDOC"

        v7r = v7m.get(int(r.hadm_id))
        recs.append(dict(
            stay_key=int(r.hadm_id),
            v7_label=(v7r.inact3 if v7r is not None else None),
            v8_label=lab,
            v8_inact_n=ia["inact_n"], v8_inact_kinds=ia["inact_kinds"],
            v8_defer_n=df["defer_n"],
            v8_act_pos=ac["act_pos"], v8_act_neg=ac["act_neg"],
            v8_act_all_pos=ac["act_all_pos"], v8_act_all_neg=ac["act_all_neg"],
            n_eval=n_eval, n_eval_any=n_eval_any,
            info_n=info_n, info_broad=info_broad, narr_char=len(narr),
            aid_doc=aid_doc, aid_signed=aid_signed, aid_act=aid_act,
            aid_broad=aid_broad, aid_eval=aid_eval,
            aid_h1=aid_h1, aid_h2=aid_h2,
            aid_bin=aid_bin,
            # 被三条规则剔除的量（用于证明规则真的在起作用）
            dropped_inact_total=sum(ia["drop"].values()),
            dropped_act_total=sum(ac["drop"].values()),
            d_act_hypoth=ac["drop"]["hypoth"], d_act_past=ac["drop"]["past"],
            d_act_radius=ac["drop"]["radius"], d_act_blacklist=ac["drop"]["blacklist"],
            d_act_domain=ac["drop"]["domain"], d_act_chronic=ac["drop"]["chronic"],
            d_inact_invert=ia["drop"]["invert"], d_inact_domain=ia["drop"]["domain"],
            d_inact_hypoth=ia["drop"]["hypoth"], d_inact_sent=ia["drop"]["sent"],
        ))
        for k, tier, frag in ia["inact_kept"][:4]:
            snippet_rows.append(dict(stay_key=int(r.hadm_id), src="v8_INACT",
                                     kind=k, tier=tier, frag=frag[:180]))
        for k, frag in ac["act_pos_kept"][:4]:
            snippet_rows.append(dict(stay_key=int(r.hadm_id), src="v8_ACT+",
                                     kind=k, tier="", frag=frag[:180]))
        for i, frag in df["defer_kept"][:3]:
            snippet_rows.append(dict(stay_key=int(r.hadm_id), src="v8_DEFER",
                                     kind="defer#%d" % i, tier="", frag=frag[:180]))

    res = pd.DataFrame(recs)
    out_csv = os.path.join(DATA, "gc_activity_density.csv")
    res.to_csv(out_csv, index=False)
    pd.DataFrame(snippet_rows).to_csv(os.path.join(OUT, "168_v8_snippets.csv"),
                                      index=False)
    P("saved %s %s" % (out_csv, res.shape))

    # ================================================================ QC
    P("")
    P("=" * 96)
    P("168 QC —— v8 三规则引擎与连续变量")
    P("=" * 96)
    P("")
    P("--- 1. v7 vs v8 三分类分布 ---")
    ct = pd.crosstab(res.v7_label.fillna("(no note)"), res.v8_label)
    P(ct.to_string())
    P("")
    P("--- 2. 三条规则各剔除了多少命中（合计） ---")
    P("  不活动线索：反转语境 %d / 非风湿领域 %d / 假说语境 %d / 非同句实体 %d" % (
        int(res.d_inact_invert.sum()), int(res.d_inact_domain.sum()),
        int(res.d_inact_hypoth.sum()), int(res.d_inact_sent.sum())))
    P("  活动线索  ：假说 %d / 既往标记 %d / 维持用药列表 %d / "
      "共现半径 %d / 黑名单 %d / 领域 %d" % (
          int(res.d_act_hypoth.sum()), int(res.d_act_past.sum()),
          int(res.d_act_chronic.sum()), int(res.d_act_radius.sum()),
          int(res.d_act_blacklist.sum()), int(res.d_act_domain.sum())))
    P("")
    P("--- 3. 连续变量分布 ---")
    P(res[["aid_doc", "aid_broad", "aid_signed", "aid_act", "aid_eval",
           "info_n", "info_broad", "narr_char"]]
      .describe(percentiles=[.1, .25, .5, .75, .9]).to_string())
    P("  零信息量（info_n == 0）n = %d (%.1f%%)"
      % (int((res.info_n == 0).sum()), 100 * (res.info_n == 0).mean()))
    P("  零信息量（info_broad == 0）n = %d (%.1f%%)"
      % (int((res.info_broad == 0).sum()), 100 * (res.info_broad == 0).mean()))
    P("  分半信度：可用 %d 例；Spearman(aid_h1, aid_h2) = %.3f"
      % (int(res.aid_h1.notna().sum()),
         res[["aid_h1", "aid_h2"]].dropna().corr(method="spearman").iloc[0, 1]))
    P("")
    P("--- 4. 三分类 → 连续变量的单调性（v8 通道内 aid_doc 中位数） ---")
    for k in ["UNDERDOC", "INF_DEFER", "INACTIVE"]:
        s = res[res.v8_label == k]
        if len(s):
            P("  %-10s n=%4d  aid_doc med=%.3f  aid_signed med=%+.3f  "
              "info_n med=%.1f" % (
                  k, len(s), s.aid_doc.median(), s.aid_signed.median(),
                  s.info_n.median()))

    txt = "\n".join(L)
    with open(os.path.join(OUT, "168_engine_qc.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    P("")
    P("saved: out/168_engine_qc.txt")


if __name__ == "__main__":
    main()
