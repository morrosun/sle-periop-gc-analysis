# -*- coding: utf-8 -*-
"""
158_extract_gc_indication.py
============================
从 MIMIC-IV 出院记录文本抽取 GC 的「用药指征」与「院前/出院激素状态」，
用来直接回答方案 B 的效度死结：**没拿到 GC 的那一组到底是谁？**

抽取四类信息
  (1) 指征（indication）：以 GC 词的 ±260 字符上下文作为语料，在其中做
      指征词共现计数 -> 11 个互斥指征标签 + 多标签标记
  (2) 院前 GC：`Medications on Admission` 段（住院前家庭用药清单）中是否含全身 GC
      以及可解析的日剂量 —— 这是判定「慢性使用者」最可靠的结构化信号
  (3) 出院 GC：`Discharge Medications` 段中的 GC 与日剂量
  (4) 未用 GC 的语境：感染 / 舒适医疗 / 明确停减激素

输出
  data/gc_indication.csv
  out/158_indication_qc.txt   （含人工可核对的原文片段）
"""
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


def log(*a):
    print(*a)
    sys.stdout.flush()


# ------------------------------------------------------------------ 区段切分
SEC_RE = re.compile(r"^\s*([A-Z][A-Za-z0-9 /&,'\-\(\)]{2,52}):\s*$", re.M)


def split_sections(text):
    """按 'Xxx Yyy:' 独占一行的形式切段。返回 {小写段名: 正文}。"""
    marks = [(m.start(), m.end(), m.group(1).strip().lower())
             for m in SEC_RE.finditer(text)]
    out = {}
    for i, (_, s, name) in enumerate(marks):
        e = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        out.setdefault(name, []).append(text[s:e])
    return {k: "\n".join(v) for k, v in out.items()}


def sec(secs, *frags):
    """按段名包含关系取段（兼容不同院区的标题变体）。"""
    return "\n".join(v for k, v in secs.items()
                     if any(f in k for f in frags))


# ------------------------------------------------------------------ 词表
# 全身 GC 药名（不含裸词 steroid）
GC_DRUG = re.compile(
    r"\b(prednisone|prednisolone|methylprednisolone|methylpred|medrol|"
    r"solu-?\s?medrol|dexamethasone|decadron|hydrocortisone|betamethasone|"
    r"triamcinolone|kenalog|aristospan|cortisone|prednisone-equivalent)\b", re.I)
# GC 词（含裸词 steroid，用于指征上下文语料）
GC_WORD = re.compile(
    r"\b(prednisone|prednisolone|methylprednisolone|methylpred|medrol|"
    r"solu-?\s?medrol|dexamethasone|decadron|hydrocortisone|betamethasone|"
    r"triamcinolone|kenalog|cortisone|steroids?|glucocorticoids?|"
    r"corticosteroids?)\b", re.I)

# ---- 指征类别（在 GC 上下文中做共现）
# 分两大类：SPEC_* = 风湿病特异的疾病活动/器官受累（效度上才是「激素的指征」）
#           GEN_*  = 通用器官衰竭（AKI/哮喘/ARDS），本身是死亡的强混杂，必须单列
IND = {
    "flar": [   # 风湿病本身活动（疾病特异）
        r"\b(lupus|sle|systemic lupus|rheumatoid arthritis|ra|disease)\s+"
        r"(flare|flares|flaring|exacerbation|activity|active)\b",
        r"\b(flare|flares|flaring|exacerbation)\b",
        r"\b(active|severe|worsening|uncontrolled|refractory)\s+"
        r"(lupus|sle|ra|rheumatoid|arthritis|disease|sarcoidosis|vasculitis)\b",
        r"\bsynovitis\b", r"\bpolymyalgia\b", r"\brheumatic\b",
        r"\binflammatory (arthritis|arthralgia|bowel)\b",
    ],
    "renal": [  # 疾病特异肾受累（不含 AKI）
        r"\blupus nephritis\b", r"\bnephritis\b", r"\bglomerulonephritis\b",
        r"\bnephrotic\b", r"\bproteinuria\b", r"\brenal flare\b",
        r"\bcrescentic\b", r"\bmembranous nephropathy\b",
    ],
    "pulm": [   # 疾病特异肺受累（不含哮喘/COPD/ARDS）
        r"\binterstitial lung disease\b", r"\bpneumonitis\b", r"\balveolitis\b",
        r"\borganizing pneumonia\b", r"\bdiffuse alveolar hemorrhage\b",
        r"\balveolar hemorrhage\b", r"\bpulmonary hemorrhage\b",
        r"\bpleuritis\b", r"\bpleurisy\b", r"\bser[oa]sitis\b",
    ],
    "heme": [   # 疾病特异血液受累
        r"\bhemolytic anemia\b", r"\baiha\b", r"\bevans syndrome\b",
        r"\bthrombocytopenia\b", r"\bitp\b", r"\bttp\b",
        r"\bmacrophage activation\b", r"\bhemophagocytic\b",
        r"\bpancytopenia\b", r"\bimmune thrombocytopenia\b",
    ],
    "neuro": [  # 疾病特异神经/肌肉受累
        r"\bnpsle\b", r"\bneuropsychiatric lupus\b", r"\bcerebritis\b",
        r"\bseizure\b", r"\bstatus epilepticus\b", r"\bmyelitis\b",
        r"\boptic neuritis\b", r"\bcns vasculitis\b",
        r"\bmyositis\b", r"\bdermatomyositis\b", r"\bpolymyositis\b",
    ],
    "shock": [  # 应激剂量 / 休克（非疾病特异，但直接指向「为什么用激素」）
        r"\bstress[- ]?dose\b", r"\bstress dose\b",
        r"\bseptic shock\b", r"\bshock\b", r"\bvasopressor\w*\b",
        r"\brefractory hypotension\b", r"\badrenal insufficiency\b",
        r"\bhypotension\b",
    ],
    "other": [
        r"\bdrug reaction\b", r"\bhypersensitivity\b", r"\banaphylaxis\b",
        r"\btransfusion reaction\b", r"\btransplant\b", r"\bgout\b",
        r"\bcontrast\b", r"\bchemotherapy\b", r"\bcerebral edema\b",
    ],
    # ---- 通用器官衰竭（单列，绝不并入 spec）
    "genrenal": [r"\bacute kidney injury\b", r"\baki\b", r"\brenal failure\b"],
    "genpulm": [r"\basthma\b", r"\bcopd\b", r"\bbronchospasm\b", r"\bwheez\w*\b",
                r"\bards\b", r"\bhypoxemic respiratory failure\b",
                r"\brespiratory failure\b"],
    "genheme": [r"\bleukopenia\b", r"\bneutropenia\b", r"\banemia\b"],
    "genneuro": [r"\bencephalopathy\b", r"\bmeningitis\b"],
}
SPEC = ["flar", "renal", "pulm", "heme", "neuro"]
GENERIC = ["genrenal", "genpulm", "genheme", "genneuro"]
IND_ORDER = SPEC + ["shock", "other"] + GENERIC
IND_RX = {k: [re.compile(p, re.I) for p in v] for k, v in IND.items()}

# ---- 慢性激素描述
CHRONIC_RX = re.compile(
    r"(chronic(ally)?\s+(on\s+|taking\s+|receiv\w+\s+)?(prednisone|steroids?|"
    r"glucocorticoids?|corticosteroids?))|"
    r"(long[- ]term\s+(steroid|glucocorticoid|corticosteroid|prednisone))|"
    r"(home\s+(dose\s+)?(prednisone|steroids?))|"
    r"(maintenance\s+(prednisone|steroids?))|"
    r"(on\s+prednisone\s+chronically)|"
    r"(years?\s+of\s+(prednisone|steroids?))", re.I)

# ---- 未用 GC / 减停的语境
CTX = {
    "ctx_infection": [
        r"\bsepsis\b", r"\bseptic\b", r"\bbacteremia\b", r"\bbacteraemia\b",
        r"\bpneumonia\b", r"\bbloodstream infection\b", r"\binfection\b",
        r"\bpositive (blood )?culture\b", r"\bempiric antibiotics\b",
    ],
    "ctx_care_limited": [
        r"\bcomfort care\b", r"\bcomfort measures\b", r"\bwithdraw\w*\s+care\b",
        r"\bgoals of care\b", r"\bDNR\b", r"\bDNI\b", r"\bhospice\b",
        r"\bpalliative\b",
    ],
    "ctx_hold_gc": [
        r"\b(held|hold|discontinu\w*|taper\w*|stop\w*|reduc\w*|decreas\w*|"
        r"wean\w*)\b[^.]{0,60}\b(prednisone|steroids?|glucocorticoids?|"
        r"corticosteroids?)\b",
        r"\b(prednisone|steroids?|glucocorticoids?|corticosteroids?)\b"
        r"[^.]{0,60}\b(held|hold|discontinued|tapered|stopped|reduced|"
        r"decreased|weaned)\b",
    ],
}
CTX_RX = {k: [re.compile(p, re.I) for p in v] for k, v in CTX.items()}

# ---- 剂量解析
FREQ = {"daily": 1.0, "qd": 1.0, "qday": 1.0, "every day": 1.0,
        "bid": 2.0, "b.i.d": 2.0, "twice": 2.0, "q12": 2.0,
        "tid": 3.0, "t.i.d": 3.0, "three times": 3.0, "q8": 3.0,
        "qid": 4.0, "q.i.d": 4.0, "four times": 4.0, "q6": 4.0,
        "qod": 0.5, "every other day": 0.5, "q48": 0.5}

DOSE_RX = re.compile(
    r"\b(prednisone|prednisolone|methylprednisolone|methylpred)\b"
    r"[^0-9\n]{0,40}?([0-9]+(?:\.[0-9]+)?)\s*(mg|milligrams)",
    re.I)


def dose_of(text):
    """从一段药单里解析 GC 日剂量（泼尼松等效 mg/日），取最大者。"""
    best = np.nan
    for m in DOSE_RX.finditer(text):
        drug = m.group(1).lower()
        mg = float(m.group(2))
        tail = text[m.end(): m.end() + 60].lower()
        f = 1.0
        for k, v in FREQ.items():
            if k in tail:
                f = v
                break
        pe = {"prednisone": 1.0, "prednisolone": 1.0,
              "methylprednisolone": 1.25, "methylpred": 1.25}[drug]
        d = mg * f * pe
        if not np.isfinite(best) or d > best:
            best = d
    return best


# ------------------------------------------------------------------ 主流程
def main():
    cc = psycopg2.connect(host=os.environ.get("MIMIC_DB_HOST", "localhost"),
                          port=int(os.environ.get("MIMIC_DB_PORT", "5432")),
                          user=os.environ.get("MIMIC_DB_USER", "postgres"),
                          password=os.environ.get("MIMIC_DB_PASSWORD", ""),
                          dbname="mimiciv")
    coh = pd.read_csv(os.path.join(DATA, "cohort_rheum_icu.csv"))
    keys = sorted(coh.stay_key.astype(int).unique().tolist())
    log("cohort hadm:", len(keys))

    parts = []
    B = 400
    for i in range(0, len(keys), B):
        ch = keys[i:i + B]
        q = ("SELECT hadm_id, note_seq, text FROM mimiciv_note.discharge "
             "WHERE hadm_id IN (%s)" % ",".join(map(str, ch)))
        parts.append(pd.read_sql(q, cc))
        log("  pulled %d/%d" % (min(i + B, len(keys)), len(keys)))
    cc.close()

    notes = pd.concat(parts, ignore_index=True)
    notes = (notes.sort_values(["hadm_id", "note_seq"])
                  .drop_duplicates("hadm_id", keep="last"))
    log("notes:", len(notes), " unique hadm:", notes.hadm_id.nunique())

    recs = []
    for r in notes.itertuples():
        t = r.text or ""
        secs = split_sections(t)
        hc = sec(secs, "brief hospital course", "hospital course")
        dd = sec(secs, "discharge diagnosis")
        moa = sec(secs, "medications on admission")
        dcm = sec(secs, "discharge medications")
        pmh = sec(secs, "past medical history")

        # 若 Hospital Course 段没切出来，退化为全篇
        narr = hc if len(hc) > 400 else t

        # --- GC 上下文语料
        ctx_buf = []
        for m in GC_WORD.finditer(narr):
            ctx_buf.append(narr[max(0, m.start() - 260): m.end() + 260])
        gcctx = " \n ".join(ctx_buf)
        blob = gcctx + "\n" + dd

        # --- 指征命中（三套口径）
        #   ind_*  = GC 上下文共现 -> 「为什么给 GC」（G0 组按构造必为 0，不可跨剂量层比）
        #   indN_* = 全叙述共现     -> 「这次住院有没有风湿活动/器官受累」，跨层可比
        #   indD_* = 仅出院诊断段   -> 由医师书写的编码式总结，最保守、与 GC 完全无关
        hits = {k: int(any(p.search(blob) for p in IND_RX[k]))
                for k in IND_ORDER}
        n_hit = sum(hits.values())
        prio = next((k for k in IND_ORDER if hits[k]), "none")

        blobN = hc + "\n" + dd
        blobD = dd
        hitsN = {k: int(any(p.search(blobN) for p in IND_RX[k]))
                 for k in IND_ORDER}
        hitsD = {k: int(any(p.search(blobD) for p in IND_RX[k]))
                 for k in IND_ORDER}

        def _spec(h, suf=""):
            return int(any(h[k] for k in SPEC))
        def _gen(h):
            return int(any(h[k] for k in GENERIC))

        # --- 上下文标志
        cxh = {k: int(any(p.search(blob) for p in CTX_RX[k]))
               for k in CTX}

        # --- 院前 / 出院 GC
        home_gc = int(bool(GC_DRUG.search(moa)))
        dc_gc = int(bool(GC_DRUG.search(dcm)))
        n_home_drugs = len(re.findall(r"(?m)^\s*\d+\.\s", moa))

        recs.append(dict(
            stay_key=int(r.hadm_id),
            note_found=1, note_len=len(t),
            n_gc_mentions=len(GC_WORD.findall(narr)),
            n_gc_drug_mentions=len(GC_DRUG.findall(narr)),
            n_strict_mentions=len(re.findall(r"\bsteroids?\b", narr, re.I)),
            moa_len=len(moa), dcm_len=len(dcm),
            n_home_drugs=n_home_drugs,
            home_gc=home_gc, home_gc_dose_mg=dose_of(moa),
            home_gc_chronic_phrase=int(bool(CHRONIC_RX.search(t))),
            pmh_gc_phrase=int(bool(GC_DRUG.search(pmh))),
            dc_gc=dc_gc, dc_gc_dose_mg=dose_of(dcm),
            **{("ind_" + k): v for k, v in hits.items()},
            **{("indN_" + k): v for k, v in hitsN.items()},
            **{("indD_" + k): v for k, v in hitsD.items()},
            ind_n_hit=n_hit, ind_priority=prio,
            indN_n_hit=sum(hitsN.values()),
            indD_n_hit=sum(hitsD.values()),
            # 疾病特异活动/器官受累（三套口径）
            ind_spec=int(any(hits[k] for k in SPEC)),
            indN_spec=_spec(hitsN), indD_spec=_spec(hitsD),
            ind_gen=int(any(hits[k] for k in GENERIC)),
            indN_gen=_gen(hitsN), indD_gen=_gen(hitsD),
            ind_any_flar_or_organ=int(any(hits[k] for k in SPEC)),
            indN_any_organ=_spec(hitsN),
            indD_any_organ=_spec(hitsD),
            indN_any_flar=int(hitsN["flar"]),
            ind_shock_only=int(hits["shock"] and prio == "shock"),
            **cxh,
            gcctx_len=len(gcctx),
        ))

    res = pd.DataFrame(recs)
    coh2 = coh.copy()
    coh2["stay_key"] = coh2.stay_key.astype(int)
    # 没抽到记录 = 没有出院记录
    miss = coh2[~coh2.stay_key.isin(res.stay_key)][["stay_key"]].copy()
    for c in res.columns:
        if c != "stay_key":
            miss[c] = 0
    miss["ind_priority"] = "no_note"
    miss["note_found"] = 0
    res = pd.concat([res, miss], ignore_index=True)
    res = res.fillna({"ind_priority": "no_note", "home_gc_dose_mg": np.nan,
                      "dc_gc_dose_mg": np.nan})
    res["ind_priority"] = res.ind_priority.replace({"none": "none_mentioned"})

    out_csv = os.path.join(DATA, "gc_indication.csv")
    res.to_csv(out_csv, index=False)
    log("saved", out_csv, res.shape)

    # ---------------------------------------------------------------- QC
    m = coh2.merge(res, on="stay_key", how="left")
    L = []
    P = lambda *a: L.append(" ".join(str(x) for x in a))
    P("=" * 78)
    P("158 GC 指征抽取 QC")
    P("=" * 78)
    P("cohort n = %d ; 有出院记录 n = %d (%.1f%%)" % (
        len(m), int(m.note_found.sum()), 100 * m.note_found.mean()))
    P("注: note_seq 为全局计数，每 hadm 仅 1 份出院记录")
    P("")
    P("--- 指征标签分布（仅 note_found=1） ---")
    nn = m[m.note_found == 1]
    vc = nn.ind_priority.value_counts()
    for k, v in vc.items():
        P("  %-16s %5d  (%.1f%%)" % (k, v, 100 * v / len(nn)))
    P("")
    P("--- GC 在叙述中被提及的比例（note_found=1） ---")
    P("  完全未提及 GC 词: %d (%.1f%%)" % (
        (nn.n_gc_mentions == 0).sum(), 100 * (nn.n_gc_mentions == 0).mean()))
    P("  中位 GC 提及次数: %.1f" % nn.n_gc_mentions.median())
    P("")
    P("--- 院前 GC（Medications on Admission）---")
    P("  moa 段切出率: %.1f%% ; 中位药条数 %.0f" % (
        100 * (nn.moa_len > 0).mean(), nn.n_home_drugs.median()))
    P("  home_gc 阳性: %d (%.1f%%)" % (
        nn.home_gc.sum(), 100 * nn.home_gc.mean()))
    P("  慢性激素短语: %d (%.1f%%)" % (
        nn.home_gc_chronic_phrase.sum(), 100 * nn.home_gc_chronic_phrase.mean()))
    P("  可解析出院日剂量: %d" % nn.dc_gc_dose_mg.notna().sum())
    P("")
    P("=== 核心表 1: 院前 GC 状态 × 首 24h GC 剂量层（SLE / RA 分列）===")
    order = ["0", ">0-10", "10-50", ">=50"]
    for g in ["SLE", "RA"]:
        s = nn[nn.primary_grp == g]
        P("  %s  n = %d" % (g, len(s)))
        P("    %-14s %6s %8s %8s %8s" % ("24h 剂量层", "n", "院前GC%",
                                         "慢性短语%", "未提GC%"))
        for st in ["G0_none", "G1_low", "G2_mod", "G3_high"]:
            if "gc24_str" not in s.columns:
                break
            sub = s[s.gc24_str == st] if s.gc24_str.dtype == object else s
            if not len(sub):
                continue
            P("    %-14s %6d %7.1f%% %7.1f%% %7.1f%%" % (
                st, len(sub), 100 * sub.home_gc.mean(),
                100 * sub.home_gc_chronic_phrase.mean(),
                100 * (sub.n_gc_mentions == 0).mean()))
    P("")
    P("=== 核心表 2: 指征构成 × 病种 ===")
    P("  %-16s %10s %10s" % ("指征", "SLE%", "RA%"))
    for k in IND_ORDER + ["none_mentioned"]:
        col = "ind_" + k if k != "none_mentioned" else None
        if col and col in nn.columns:
            a = 100 * nn[nn.primary_grp == "SLE"][col].mean()
            b = 100 * nn[nn.primary_grp == "RA"][col].mean()
        else:
            a = 100 * (nn[nn.primary_grp == "SLE"].ind_priority
                       == "none_mentioned").mean()
            b = 100 * (nn[nn.primary_grp == "RA"].ind_priority
                       == "none_mentioned").mean()
        P("  %-16s %9.1f%% %9.1f%%" % (k, a, b))
    P("")
    P("=== 核心表 3: 指征构成 × 剂量层（合并两病种），三套口径并列 ===")
    P("  三套口径含义: GCctx = GC 上下文共现（为什么给 GC，G0 按构造为 0）")
    P("                narr  = 全叙述共现（这次住院有没有活动/器官受累）")
    P("                dx    = 仅出院诊断段（与 GC 无关，跨层最可比）")
    P("  spec = 风湿病特异的疾病活动/器官受累 ; gen = 通用器官衰竭（AKI/哮喘/ARDS）")
    P("  %-10s %6s %7s %7s %7s %7s %7s %7s %8s" % (
        "剂量层", "n", "GCcx spec", "na spec", "dx spec", "na gen",
        "na shock", "dx shock", "na 未提GC"))
    for st in ["G0_none", "G1_low", "G2_mod", "G3_high"]:
        if "gc24_str" not in nn.columns or nn.gc24_str.dtype != object:
            break
        sub = nn[nn.gc24_str == st]
        if not len(sub):
            continue
        P("  %-10s %6d %7.1f%% %7.1f%% %7.1f%% %6.1f%% %7.1f%% %7.1f%% %8.1f%%" % (
            st, len(sub), 100 * sub.ind_spec.mean(),
            100 * sub.indN_spec.mean(), 100 * sub.indD_spec.mean(),
            100 * sub.indN_gen.mean(), 100 * sub.indN_shock.mean(),
            100 * sub.indD_shock.mean(),
            100 * (sub.n_gc_mentions == 0).mean()))
    P("")
    P("=== 核心表 3b: 三套口径 × 病种 ===")
    P("  %-8s %7s %7s %7s %7s %7s %7s %7s %8s" % (
        "病种", "GCcx", "na", "dx", "na", "dx", "na", "dx", "n"))
    P("  %-8s %7s %7s %7s %7s %7s %7s %7s %8s" % (
        "", "spec", "spec", "spec", "gen", "gen", "shock", "shock", ""))
    for g in ["SLE", "RA"]:
        sub = nn[nn.primary_grp == g]
        P("  %-8s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %6.1f%% %6.1f%% %6.1f%% %8d" % (
            g, 100 * sub.ind_spec.mean(),
            100 * sub.indN_spec.mean(), 100 * sub.indD_spec.mean(),
            100 * sub.indN_gen.mean(), 100 * sub.indD_gen.mean(),
            100 * sub.indN_shock.mean(), 100 * sub.indD_shock.mean(),
            len(sub)))
    P("")
    P("=== 未用 GC 语境 ===")
    for k in CTX:
        col = k
        if col in nn.columns:
            P("  %-18s %6.1f%%  (SLE %.1f%% / RA %.1f%%)" % (
                k, 100 * nn[col].mean(),
                100 * nn[nn.primary_grp == "SLE"][col].mean(),
                100 * nn[nn.primary_grp == "RA"][col].mean()))

    txt = "\n".join(L)
    with open(os.path.join(OUT, "158_indication_qc.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
