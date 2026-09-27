# -*- coding: utf-8 -*-
"""
163_extract_inactivity.py
=========================
v7 —— 把 v6 的「无活动记录」层（indN_spec == 0）细分成三个可解释通道：

  INACTIVE    真无活动 —— 出院记录**明确**记载疾病无活动 / 缓解 / 稳定
  INF_DEFER   因感染暂缓 —— 明确记载因感染（风险）而暂缓 / 停用 / 未给 GC
  UNDERDOC    记录不足 —— 既无「有活动」也无「无活动」记载，无法判定

同时修一个 v6 留下的**系统性测量错误**：
  158 号脚本的指征词表对**否定句不敏感**。病历写
      "no evidence of active lupus"、"denies joint pain / no flare"
  会被 `flar` 词表（`(active|severe|worsening|...)\s+(lupus|sle|ra|...)`）
  命中，于是被算成「有疾病活动」。也就是说 v6 的 spec+ 层里混进了
  一批「被明确记载为无活动」的人 —— 这会**同时污染** spec+ 和 spec- 的对照。
  本脚本用 NegEx 式前文窗口否定检测重算，输出校正后的 spec_true。

另外附一个与文本完全无关的结构化证据：
  本次入院**主诊断码**（diagnoses_icd.seq_num = 1）是不是风湿病码 ——
  用来判断「风湿病是不是这次住院的原因」，是「真无活动」的硬校验。

输出
  data/gc_inactivity.csv        每 stay 一行
  out/163_inactivity_qc.txt     分布 + 交叉 + 复现校验
  out/163_inactivity_audit.csv  分层抽样原文片段（供人工双标）
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

L = []


def P(s=""):
    print(s)
    sys.stdout.flush()
    L.append(str(s))


# ------------------------------------------------------------------ 复用 158 的切段
SEC_RE = re.compile(r"^\s*([A-Z][A-Za-z0-9 /&,'\-\(\)]{2,52}):\s*$", re.M)


def split_sections(text):
    marks = [(m.start(), m.end(), m.group(1).strip().lower())
             for m in SEC_RE.finditer(text)]
    out = {}
    for i, (_, s, name) in enumerate(marks):
        e = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        out.setdefault(name, []).append(text[s:e])
    return {k: "\n".join(v) for k, v in out.items()}


def sec(secs, *frags):
    return "\n".join(v for k, v in secs.items() if any(f in k for f in frags))


# 与 158 完全一致的 SPEC / GENERIC 词表（用于复现校验）
IND = {
    "flar": [
        r"\b(lupus|sle|systemic lupus|rheumatoid arthritis|ra|disease)\s+"
        r"(flare|flares|flaring|exacerbation|activity|active)\b",
        r"\b(flare|flares|flaring|exacerbation)\b",
        r"\b(active|severe|worsening|uncontrolled|refractory)\s+"
        r"(lupus|sle|ra|rheumatoid|arthritis|disease|sarcoidosis|vasculitis)\b",
        r"\bsynovitis\b", r"\bpolymyalgia\b", r"\brheumatic\b",
        r"\binflammatory (arthritis|arthralgia|bowel)\b",
    ],
    "renal": [
        r"\blupus nephritis\b", r"\bnephritis\b", r"\bglomerulonephritis\b",
        r"\bnephrotic\b", r"\bproteinuria\b", r"\brenal flare\b",
        r"\bcrescentic\b", r"\bmembranous nephropathy\b",
    ],
    "pulm": [
        r"\binterstitial lung disease\b", r"\bpneumonitis\b", r"\balveolitis\b",
        r"\borganizing pneumonia\b", r"\bdiffuse alveolar hemorrhage\b",
        r"\balveolar hemorrhage\b", r"\bpulmonary hemorrhage\b",
        r"\bpleuritis\b", r"\bpleurisy\b", r"\bser[oa]sitis\b",
    ],
    "heme": [
        r"\bhemolytic anemia\b", r"\baiha\b", r"\bevans syndrome\b",
        r"\bthrombocytopenia\b", r"\bitp\b", r"\bttp\b",
        r"\bmacrophage activation\b", r"\bhemophagocytic\b",
        r"\bpancytopenia\b", r"\bimmune thrombocytopenia\b",
    ],
    "neuro": [
        r"\bnpsle\b", r"\bneuropsychiatric lupus\b", r"\bcerebritis\b",
        r"\bseizure\b", r"\bstatus epilepticus\b", r"\bmyelitis\b",
        r"\boptic neuritis\b", r"\bcns vasculitis\b",
        r"\bmyositis\b", r"\bdermatomyositis\b", r"\bpolymyositis\b",
    ],
    "genrenal": [r"\bacute kidney injury\b", r"\baki\b", r"\brenal failure\b"],
    "genpulm": [r"\basthma\b", r"\bcopd\b", r"\bbronchospasm\b", r"\bwheez\w*\b",
                r"\bards\b", r"\bhypoxemic respiratory failure\b",
                r"\brespiratory failure\b"],
    "genheme": [r"\bleukopenia\b", r"\bneutropenia\b", r"\banemia\b"],
    "genneuro": [r"\bencephalopathy\b", r"\bmeningitis\b"],
}
SPEC = ["flar", "renal", "pulm", "heme", "neuro"]
GENERIC = ["genrenal", "genpulm", "genheme", "genneuro"]
IND_RX = {k: [re.compile(p, re.I) for p in v] for k, v in IND.items()}

# ------------------------------------------------------------------ 否定检测
# NegEx 式：命中点**前文 55 字符**窗口内出现否定线索即判为否定句
NEG_CUE = re.compile(
    r"\b(no|not|non|never|without|denies|denied|deny|negative|absent|"
    r"absence of|free of|ruled out|no evidence of|no sign|no signs|"
    r"unremarkable|resolved|improved|no longer|neither|nor)\b", re.I)
BOUND = re.compile(r"[.;:!?]|\bbut\b|\bhowever\b|\balthough\b|\bthough\b|\bother than\b",
                   re.I)


def negated(text, start, window=55):
    """命中点 start 之前 window 字符内是否有否定线索（不跨句边界）。"""
    pre = text[max(0, start - window):start]
    last = None
    for m in BOUND.finditer(pre):
        last = m.end()
    if last is not None:
        pre = pre[last:]
    return bool(NEG_CUE.search(pre))


# ------------------------------------------------------------------ 真无活动
INACT = {
    "remission": [
        r"\b(in|achieved|sustained|remains? in)\s+(complete |clinical |full )?remission\b",
        r"\bremission\b",
    ],
    "no_evidence": [
        r"\bno\s+(evidence|signs?|symptoms?|clinical signs?)\s+(of|for)\s+"
        r"(active\s+|any\s+)?(lupus|sle|ra\b|rheumatoid|disease|flares?|"
        r"disease activity|active disease|synovitis|arthritis|arthralgia|"
        r"nephritis|serositis|vasculitis)\b",
        r"\bno\s+(active|flaring|new)\s+(lupus|sle|ra\b|rheumatoid|disease|"
        r"arthritis|synovitis|joint)\b",
        r"\b(no|without)\s+(further\s+|any\s+|recurrent\s+)?(flare|flares|flaring)\b",
        r"\bwithout\s+(evidence of\s+)?(active\s+)?(disease|flare|activity|synovitis)\b",
    ],
    "quiescent": [
        r"\b(lupus|sle|ra\b|rheumatoid arthritis|disease|arthritis)\s+"
        r"(is|was|remains?|appears?|had been|has been)\s+(in\s+)?"
        r"(remission|inactive|quiescent|dormant|well[- ]controlled|"
        r"stable|controlled|minimal)\b",
        r"\b(quiescent|dormant|inactive|clinically inactive|well[- ]controlled|"
        r"mildly active|minimally active)\s+"
        r"(lupus|sle|ra\b|rheumatoid|disease|arthritis)\b",
        r"\bdisease activity\s*(:|was|is|remained|appears? to be)?\s*"
        r"(none|nil|negative|zero|0|inactive|quiescent|minimal|low)\b",
        r"\bstable\s+(lupus|sle|ra\b|rheumatoid|disease activity)\b",
        r"\bno\s+disease[- ]related\b",
    ],
    "asymptomatic": [
        r"\bdenies\s+(any\s+)?(joint pain|arthralgia|morning stiffness|joint "
        r"swelling|rash|oral ulcer|photosensitivity)\b",
        r"\bno\s+(joint pain|arthralgia|morning stiffness|joint swelling|"
        r"synovitis on exam|rash|oral ulcer)\b",
        r"\basymptomatic\b",
        r"\bwithout\s+(complaint|symptom)",
    ],
    "free_of": [
        r"\bfree of\s+(active\s+)?(disease|symptoms|joint|synovitis|flares?)\b",
        r"\bhas been\s+(doing\s+)?well\b",
        r"\bdoing well\s+(from|from a|in terms of)\b",
    ],
}
INACT_RX = [(k, re.compile(p, re.I))
            for k, v in INACT.items() for p in v]

# ------------------------------------------------------------------ 因感染暂缓
GCMED = (r"(prednisone|prednisolone|methylprednisolone|methylpred|medrol|"
         r"solu-?\s?medrol|dexamethasone|hydrocortisone|steroids?|"
         r"glucocorticoids?|corticosteroids?|immunosuppress\w*)")
INFS = (r"(infection|infectious|sepsis|septic|bacteremia|bacteraemia|"
        r"bacteremic|pneumonia|bloodstream infection|positive blood culture|"
        r"urinary tract infection|uti\b|covid|aspiration pneumonia|"
        r"empiric antibiotics)")
ACT = (r"(held|hold|withheld|withhold|deferred|defer|delay\w*|"
       r"discontinu\w*|stop\w*|avoid\w*|"
       r"not\s+(?:given|started|initiated|restarted|resumed|administered)|"
       r"on hold|contraindicat\w*|precluded)")
LINK = (r"(due to|because of|secondary to|in the setting of|in light of|"
        r"given|for|concerning for|concerning|worri\w* (?:for|about|of)|"
        r"related to|amid|as a result of|owing to|amid|following)")

DEFER_RX = [
    # 句型 A：药(动作) -> 因为 -> 感染
    re.compile(r"\b" + GCMED + r"\b[^.;]{0,40}?\b" + ACT +
               r"\b[^.;]{0,90}?\b" + LINK + r"\b[^.;]{0,50}?\b" + INFS + r"\b", re.I),
    # 句型 B：动作 -> 因为 -> 感染 -> 药
    re.compile(r"\b" + ACT + r"\b[^.;]{0,90}?\b" + LINK + r"\b[^.;]{0,50}?\b"
               + INFS + r"\b[^.;]{0,50}?\b" + GCMED + r"\b", re.I),
    # 句型 C：感染 -> 阻止 -> 药
    re.compile(r"\b" + INFS + r"\b[^.;]{0,90}?\b" + ACT + r"\b[^.;]{0,60}?\b"
               + GCMED + r"\b", re.I),
    # 句型 D：担忧感染 -> 药
    re.compile(r"\b(concern|worri\w*|fear\w*|suspicion|suspected|"
               r"in the setting of|given)\b[^.;]{0,40}?\b" + INFS +
               r"\b[^.;]{0,90}?\b" + GCMED + r"\b", re.I),
    # 句型 E：药 -> 暂缓（无理由，但同段有感染）
    re.compile(r"\b" + GCMED + r"\b[^.;]{0,50}?\b(on hold|was held|withheld|"
               r"was not given|not administered)\b", re.I),
]

# 活动度评估证据（有没有查/评价过疾病活动）
ACTIV_RX = [re.compile(p, re.I) for p in [
    r"\bC-?reactive protein\b", r"\bCRP\b", r"\bESR\b",
    r"\bsedimentation rate\b", r"\bcomplement\b", r"\bC3\b", r"\bC4\b",
    r"\banti-?dsDNA\b", r"\bds-?DNA\b", r"\bSLEDAI\b", r"\bDAS-?28\b",
    r"\bdisease activity\b", r"\bANA\b", r"\brheumatolog\w*\b",
    r"\bimmunosuppress\w*\b", r"\bflare\b",
]]

# 感染语境（与 158 一致，用于构造「感染 + 减停」的弱证据通道）
CTX_INF_RX = re.compile(
    r"\b(sepsis|septic|bacteremi\w*|bacteraemi\w*|pneumonia|"
    r"bloodstream infection|infection|positive (blood )?culture|"
    r"empiric antibiotics)\b", re.I)
CTX_HOLD_RX = [
    re.compile(r"\b(held|hold|discontinu\w*|taper\w*|stop\w*|reduc\w*|"
               r"decreas\w*|wean\w*)\b[^.]{0,60}\b(prednisone|steroids?|"
               r"glucocorticoids?|corticosteroids?)\b", re.I),
    re.compile(r"\b(prednisone|steroids?|glucocorticoids?|corticosteroids?)\b"
               r"[^.]{0,60}\b(held|hold|discontinued|tapered|stopped|reduced|"
               r"decreased|weaned)\b", re.I),
]


def spec_eval(text):
    """返回 (raw 命中数, 被否定数, 未被否定数)。raw>0 等价于 158 的 indN_spec。"""
    raw = neg = pos = 0
    for k in SPEC:
        for rx in IND_RX[k]:
            for m in rx.finditer(text):
                raw += 1
                if negated(text, m.start()):
                    neg += 1
                else:
                    pos += 1
    return raw, neg, pos


# ------------------------------------------------------------------ 主诊断码
def code_class(code, ver):
    c = str(code).upper().strip()
    if int(ver) == 9:
        if c.startswith("7100"):
            return "SLE"
        if c.startswith("714"):
            return "RA"
        if c.startswith("710") or c.startswith("69") or c.startswith("725"):
            return "RHEUM_OTHER"
        return "OTHER"
    if c.startswith("M32"):
        return "SLE"
    if c[:3] in ("M05", "M06", "M08"):
        return "RA"
    if c.startswith("M3") or c[:3] in ("M45", "M46", "M30", "M31", "M33",
                                       "M34", "M35", "M0"):
        return "RHEUM_OTHER"
    return "OTHER"


# ------------------------------------------------------------------ 主流程
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
    dxp = []
    for i in range(0, len(keys), B):
        ch = keys[i:i + B]
        q = ("SELECT hadm_id, seq_num, icd_code, icd_version "
             "FROM mimiciv_hosp.diagnoses_icd WHERE hadm_id IN (%s)"
             % ",".join(map(str, ch)))
        dxp.append(pd.read_sql(q, cc))
    cc.close()

    notes = pd.concat(parts, ignore_index=True)
    notes = (notes.sort_values(["hadm_id", "note_seq"])
                  .drop_duplicates("hadm_id", keep="last"))
    dx = pd.concat(dxp, ignore_index=True)
    dx["hadm_id"] = dx.hadm_id.astype(int)
    d1 = dx[dx.seq_num == 1].drop_duplicates("hadm_id")
    d1["pdx_class"] = [code_class(c, v) for c, v in
                       zip(d1.icd_code, d1.icd_version)]
    P("notes %d ; 主诊断 d1 %d" % (len(notes), len(d1)))

    recs = []
    for r in notes.itertuples():
        t = r.text or ""
        secs = split_sections(t)
        hc = sec(secs, "brief hospital course", "hospital course")
        dd = sec(secs, "discharge diagnosis")
        moa = sec(secs, "medications on admission")
        blobN = hc + "\n" + dd
        blob = blobN + "\n" + moa

        # ---- 真无活动证据
        inact_hits = []
        for k, rx in INACT_RX:
            m = rx.search(blobN)
            if m:
                inact_hits.append((k, m.group(0)[:60]))
        inact_any = int(bool(inact_hits))
        inact_kinds = ",".join(sorted(set(k for k, _ in inact_hits)))

        # ---- 因感染暂缓证据
        defer_hits = []
        for i, rx in enumerate(DEFER_RX):
            m = rx.search(blob)
            if m:
                defer_hits.append(m.group(0)[:150])
        defer_any = int(bool(defer_hits))

        # ---- 活动度评估证据
        activ_n = sum(len(rx.findall(blobN)) for rx in ACTIV_RX)
        activ_any = int(activ_n > 0)

        # ---- spec 复现 + 否定校正
        raw, negcnt, poscnt = spec_eval(blobN)
        ctx_inf = int(bool(CTX_INF_RX.search(blobN)))
        ctx_hold = int(any(rx.search(blobN) for rx in CTX_HOLD_RX))

        recs.append(dict(
            stay_key=int(r.hadm_id),
            inact_any=inact_any, inact_kinds=inact_kinds,
            inact_snippet=(inact_hits[0][1] if inact_hits else ""),
            defer_any=defer_any,
            defer_snippet=(defer_hits[0] if defer_hits else ""),
            activ_any=activ_any, activ_n=activ_n,
            spec_raw_hits=raw, spec_neg_hits=negcnt, spec_pos_hits=poscnt,
            ctx_inf2=ctx_inf, ctx_hold2=ctx_hold,
            hc_len=len(hc), dd_len=len(dd), moa_len2=len(moa),
            note_len2=len(t),
        ))
    res = pd.DataFrame(recs)
    res = res.merge(d1[["hadm_id", "pdx_class", "icd_code"]]
                    .rename(columns={"hadm_id": "stay_key", "icd_code": "pdx_code"}),
                    on="stay_key", how="left")
    res["pdx_class"] = res.pdx_class.fillna("NODX")

    # ---- 三分（互斥，证据优先：明确无活动 > 明确因感染暂缓 > 记录不足）
    grp = []
    for r in res.itertuples():
        if r.inact_any == 1:
            grp.append("INACTIVE")
        elif r.defer_any == 1 or (r.ctx_inf2 == 1 and r.ctx_hold2 == 1):
            grp.append("INF_DEFER")
        else:
            grp.append("UNDERDOC")
    res["inact3"] = grp
    # 重叠标记（不做优先级裁剪，供敏感分析）
    res["inact_and_defer"] = ((res.inact_any == 1) & (res.defer_any == 1)).astype(int)
    # 记录不足的三个维度
    res["doc_thin"] = ((res.hc_len < 3000) | (res.note_len2 < 8000)).astype(int)
    res["no_activ_eval"] = (res.activ_any == 0).astype(int)
    res["nonrheum_pdx"] = (~res.pdx_class.isin(["SLE", "RA"])).astype(int)
    # 校正后的 spec：存在**未被否定**的疾病特异活动命中
    res["spec_true"] = (res.spec_pos_hits > 0).astype(int)
    res["spec_falsepos"] = ((res.spec_raw_hits > 0) &
                            (res.spec_pos_hits == 0)).astype(int)

    out_csv = os.path.join(DATA, "gc_inactivity.csv")
    res.to_csv(out_csv, index=False)
    P("saved %s %s" % (out_csv, res.shape))

    # ================================================================ QC
    m = both.merge(res, on="stay_key", how="left")
    nn = m[m.stay_key.isin(res.stay_key)].copy()
    P("")
    P("=" * 92)
    P("163 QC ——「无活动记录」细分")
    P("=" * 92)
    P("可抽文本 %d / %d (%.1f%%)" % (len(nn), len(both), 100 * len(nn) / len(both)))
    P("")
    P("--- 0. 复现校验：重算 spec_raw 是否等于 158 的 indN_spec ---")
    gi = pd.read_csv(os.path.join(DATA, "gc_indication.csv"))
    chk = nn.merge(gi[["stay_key", "indN_spec", "n_gc_mentions"]], on="stay_key",
                   how="left")
    if "indN_spec" in chk.columns:
        ok = (chk.spec_raw_hits.clip(0, 1) == chk.indN_spec.fillna(0)).mean()
        P("  一致率 = %.3f  (不一致 %d / %d)" % (
            ok, int((chk.spec_raw_hits.clip(0, 1) !=
                     chk.indN_spec.fillna(0)).sum()), len(chk)))
        P("  说明：若 < 1.0，多半是 158 用 hc 退化文本、此处用纯 hc+dd 所致。")
    P("")
    P("--- 1. 否定校正：spec+ 里有多少其实是「被否定」的 ---")
    P("  spec_raw 阳性 n = %d" % int((nn.spec_raw_hits > 0).sum()))
    P("  其中所有命中均被否定（= 假阳性）n = %d (%.1f%% of spec_raw+)"
      % (int(nn.spec_falsepos.sum()),
         100 * nn.spec_falsepos.sum() / max(1, int((nn.spec_raw_hits > 0).sum()))))
    P("  校正后 spec_true 阳性 n = %d" % int(nn.spec_true.sum()))
    P("  按病种：")
    for g in ["SLE", "RA"]:
        s = nn[nn.primary_grp == g]
        P("    %s  raw+ %d (%.1f%%)  ->  true+ %d (%.1f%%)  被否定 %d" % (
            g, int((s.spec_raw_hits > 0).sum()),
            100 * (s.spec_raw_hits > 0).mean(), int(s.spec_true.sum()),
            100 * s.spec_true.mean(), int(s.spec_falsepos.sum())))
    P("")
    P("--- 2. 三分分布（全队列 / 原 spec- 层 / 原 spec+ 层） ---")
    for lab, sub in [("ALL with note", nn), ("v6 spec-", nn[nn.spec_true == 0]),
                     ("v6 spec+", nn[nn.spec_true == 1])]:
        vc = sub.inact3.value_counts()
        P("  %-14s n=%4d   " % (lab, len(sub)) +
          "  ".join("%s=%d(%.1f%%)" % (k, v, 100 * v / len(sub))
                    for k, v in vc.items()))
    P("")
    P("--- 3. 三分 × 病种（在 v6 spec- 层内） ---")
    sp0 = nn[nn.spec_true == 0]
    ct = pd.crosstab(sp0.primary_grp, sp0.inact3)
    P(ct.to_string())
    P("")
    P("--- 4. 三分的临床画像（v6 spec- 层内） ---")
    c2 = both.merge(res, on="stay_key", how="left")
    c2 = c2.merge(gi[["stay_key", "home_gc", "dc_gc", "n_gc_mentions",
                      "ctx_infection", "ctx_hold_gc", "indN_spec"]],
                  on="stay_key", how="left")
    c2 = c2[c2.spec_true == 0]
    P("  %-12s %6s %8s %8s %8s %8s %8s %8s" % (
        "通道", "n", "homeGC%", "usedGC%", "SOFA", "death30%", "spec_neg%",
        "未提GC%"))
    for k in ["INACTIVE", "INF_DEFER", "UNDERDOC"]:
        s = c2[c2.inact3 == k]
        if not len(s):
            continue
        P("  %-12s %6d %7.1f%% %7.1f%% %8.1f %8.1f%% %8.1f%% %8.1f%%" % (
            k, len(s), 100 * s.home_gc.mean(), 100 * (s.gc_str_num >= 1).mean(),
            s.sofa24.median(), 100 * s.death_30d.mean(),
            100 * s.spec_neg_hits.gt(0).mean(),
            100 * (s.n_gc_mentions.fillna(0) == 0).mean()))
    P("")
    P("--- 5. 三分 × 剂量层（v6 spec- 层内） ---")
    P(pd.crosstab(c2.inact3, c2.gc24_str).to_string())
    P("")
    P("--- 6. 主诊断是否为风湿（结构化硬校验） ---")
    P(pd.crosstab(c2.inact3, c2.pdx_class).to_string())
    P("")
    P("--- 7. 三个「记录不足」维度在三分内 ---")
    for k in ["INACTIVE", "INF_DEFER", "UNDERDOC"]:
        s = c2[c2.inact3 == k]
        if not len(s):
            continue
        P("  %-12s doc_thin=%.1f%%  no_activ_eval=%.1f%%  nonrheum_pdx=%.1f%%" % (
            k, 100 * s.doc_thin.mean(), 100 * s.no_activ_eval.mean(),
            100 * s.nonrheum_pdx.mean()))
    P("")
    P("--- 8. 重叠：同时「明确无活动」且「明确因感染暂缓」 n = %d ---"
      % int(c2.inact_and_defer.sum()))

    txt = "\n".join(L)
    with open(os.path.join(OUT, "163_inactivity_qc.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)

    # ---- 人工双标抽样
    rs = np.random.RandomState(20260916)
    aud = []
    for k in ["INACTIVE", "INF_DEFER", "UNDERDOC"]:
        s = c2[c2.inact3 == k]
        take = min(40, len(s))
        for r in s.sample(take, random_state=rs).itertuples():
            aud.append(dict(stay_key=r.stay_key, disease=r.primary_grp,
                            inact3=k, gc24_str=r.gc24_str,
                            death_30d=int(r.death_30d),
                            home_gc=int(r.home_gc),
                            inact_kinds=r.inact_kinds,
                            inact_snippet=r.inact_snippet,
                            defer_snippet=r.defer_snippet,
                            spec_neg_hits=int(r.spec_neg_hits),
                            pdx_class=r.pdx_class,
                            hc_len=int(r.hc_len)))
    pd.DataFrame(aud).to_csv(os.path.join(OUT, "163_inactivity_audit.csv"),
                             index=False)
    P("")
    P("saved: out/163_inactivity_qc.txt + out/163_inactivity_audit.csv")
    print(txt)


if __name__ == "__main__":
    main()
