# -*- coding: utf-8 -*-
"""
165_make_blind_sheets.py
========================
v7b —— 为「人工双标核验」生成**盲法**抽样表与封印键。

背景
  v7 的三分类（INACTIVE / INF_DEFER / UNDERDOC）与否定校正（spec_true）
  全部由正则词表自动判定。163 号脚本虽导出了 163_inactivity_audit.csv，
  但那份表**带有算法标签与算法命中片段**，评分者看到后无法独立判定 ——
  不能用于一致性/κ 估计。本脚本生成真正盲的版本。

设计
  任务 A（三分类）：抽样框 = spec_true == 0 且 spec_raw_hits == 0（排除 18 例
      全否定假阳性，使其只出现在任务 B，同一病例不被评两次）
        INACTIVE  抽 35 / 57
        INF_DEFER 全抽 25 / 25
        UNDERDOC  抽 40 / 792
      总 100 例。分层原因：两层极稀疏，等比例抽会把 κ 变成「UNDERDOC 内部一致性」。
      事后用抽样权重校正回真实层构成。

  任务 B（否定规则）：抽样框 = spec_raw_hits > 0
        全抽 spec_falsepos 18 例（算法判「全被否定」）
        抽 spec_falsepos == 0 的 32 例（算法判「存在未否定命中」）
      总 50 例。用于估否定规则的敏感度/特异度 + κ。

盲法
  评分者只看到 {不透明 case_id, 住院经过段 + 出院诊断 + 入院前用药}。
  case_id 随机重排且**不含可推断信息**。病种、剂量、结局、算法标签全部
  只存在于封印键 out/adjud_sealed_key.csv，解封前不看。

摘录规则（关键：与算法标签无关）
  blob = 住院经过段 + 出院诊断段 + 入院前用药段（与 163 的判定输入一致）
  超过 CAP=4000 字符 → 保留前 2500 + 后 1400，中间以占位符替代。
  该规则只依赖文本长度，与任何标签无关 —— 因此截断只会**稀释**一致性，
  不会系统性地偏向某个通道。同时记录算法命中片段是否幸存于截断后文本，
  以便做「证据完整可见」子集的敏感性分析。

输出
  out/adjud_blind_A.md       100 例盲表（任务 A）
  out/adjud_blind_B.md        50 例盲表（任务 B）
  out/adjud_blind_A.csv / _B.csv   同内容 CSV（留档）
  out/adjud_sealed_key.csv         封印键（含算法标签，评分者不得见）
  out/adjud_codebook.md            编码手册（两评分者完全一致）
  out/165_blind_qc.txt             抽样与截断 QC
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


def load_163():
    """按路径加载 163 号脚本，复用其**完全一致**的词表与判定函数。"""
    p = os.path.join(ROOT, "scripts", "163_extract_inactivity.py")
    spec = importlib.util.spec_from_file_location("mod163", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M163 = load_163()

SEED = 20260916
CAP = 5000
HEAD = 3200
TAIL = 1700
CHUNK = 25
TRUNC_MARK = "\n\n[...middle of hospital course omitted for length...]\n\n"

L = []


def P(s=""):
    print(s)
    sys.stdout.flush()
    L.append(str(s))


# ---------------------------------------------------------------- 复用 163 切段
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


def build_blob(text):
    secs = split_sections(text or "")
    hc = sec(secs, "brief hospital course", "hospital course")
    dd = sec(secs, "discharge diagnosis")
    moa = sec(secs, "medications on admission")
    return hc.strip(), dd.strip(), moa.strip()


def join_blob(hc, dd, moa):
    b = hc + "\n\n[Discharge diagnosis]\n" + dd
    if moa:
        b += "\n\n[Medications on admission]\n" + moa
    return b.strip()


def split_blob(b):
    """把拼好的串还原成 (hc, dd, moa)；截断导致缺 marker 时退化处理。"""
    MK_D = "\n\n[Discharge diagnosis]\n"
    MK_M = "\n\n[Medications on admission]\n"
    moa = ""
    if MK_M in b:
        b, moa = b.split(MK_M, 1)
    dd = ""
    if MK_D in b:
        hc, dd = b.split(MK_D, 1)
    else:
        hc = b
    return hc, dd, moa


def verdict(hc, dd, moa):
    """用 163 的原始词表，重算该文本下 163 会给出的判定。"""
    blobN = hc + "\n" + dd
    blob = blobN + "\n" + moa
    inact_any = any(rx.search(blobN) for _, rx in M163.INACT_RX)
    defer_any = any(rx.search(blob) for rx in M163.DEFER_RX)
    ctx_inf = bool(M163.CTX_INF_RX.search(blobN))
    ctx_hold = any(rx.search(blobN) for rx in M163.CTX_HOLD_RX)
    if inact_any:
        lab = "INACTIVE"
    elif defer_any or (ctx_inf and ctx_hold):
        lab = "INF_DEFER"
    else:
        lab = "UNDERDOC"
    raw, neg, pos = M163.spec_eval(blobN)
    fp = int(raw > 0 and pos == 0)
    if raw == 0:
        verdict = "no match"
    elif pos > 0:
        verdict = "affirmative"
    else:
        verdict = "purely-negated"
    return lab, fp, raw, pos, verdict


def truncate(blob):
    """纯按长度截断，与任何标签无关。"""
    if len(blob) <= CAP:
        return blob, 0, len(blob)
    return blob[:HEAD] + TRUNC_MARK + blob[-TAIL:], 1, len(blob)


def si(v, default=0):
    """NaN 安全的 int。"""
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return int(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- 主流程
def main():
    gi = pd.read_csv(os.path.join(DATA, "gc_inactivity.csv"))
    gi["stay_key"] = gi.stay_key.astype(int)
    coh = pd.read_csv(os.path.join(DATA, "cohort_rheum_icu.csv"))
    coh["stay_key"] = coh.stay_key.astype(int)
    m = coh[coh.primary_grp.isin(["SLE", "RA"])].merge(gi, on="stay_key",
                                                       how="inner")
    gind = pd.read_csv(os.path.join(DATA, "gc_indication.csv"))
    gind["stay_key"] = gind.stay_key.astype(int)
    m = m.merge(gind[["stay_key", "home_gc", "n_gc_mentions"]], on="stay_key",
                how="left")

    # ---------------- 任务 A 抽样框
    fa = m[(m.spec_true == 0) & (m.spec_raw_hits == 0)]
    P("任务 A 抽样框（spec- 且无被否定命中）= %d" % len(fa))
    P("  " + "  ".join("%s=%d" % (k, v)
                       for k, v in fa.inact3.value_counts().items()))
    fa = fa.copy()
    rs = np.random.RandomState(SEED)
    picks = []
    for k, want in [("INACTIVE", 35), ("INF_DEFER", 25), ("UNDERDOC", 40)]:
        pool = fa[fa.inact3 == k]
        take = min(want, len(pool))
        pick = pool.sample(take, random_state=rs).copy()
        pick["task"] = "A"
        pick["stratum"] = k
        picks.append(pick)
        P("  A | %-10s 抽 %d / %d" % (k, take, len(pool)))
    A = pd.concat(picks, ignore_index=True)

    # ---------------- 任务 B 抽样框
    fb = m[m.spec_raw_hits > 0].copy()
    P("")
    P("任务 B 抽样框（spec_raw > 0）= %d" % len(fb))
    picksB = []
    for k, want in [("purely negated (spec_falsepos=1)", 18),
                    ("affirmative (spec_falsepos=0)", 32)]:
        fl = fb.spec_falsepos == (1 if "=1" in k else 0)
        pool = fb[fl]
        take = min(want, len(pool))
        pick = pool.sample(take, random_state=rs).copy()
        pick["task"] = "B"
        pick["stratum"] = k
        picksB.append(pick)
        P("  B | %-34s 抽 %d / %d" % (k, take, len(pool)))
    B = pd.concat(picksB, ignore_index=True)

    sel = pd.concat([A, B], ignore_index=True)
    keys = sorted(sel.stay_key.unique().tolist())
    P("")
    P("合计抽取 %d 例（去重后 %d）" % (len(sel), len(keys)))

    # ---------------- 取原文
    cc = psycopg2.connect(host=os.environ.get("MIMIC_DB_HOST", "localhost"),
                          port=int(os.environ.get("MIMIC_DB_PORT", "5432")),
                          user=os.environ.get("MIMIC_DB_USER", "postgres"),
                          password=os.environ.get("MIMIC_DB_PASSWORD", ""),
                          dbname="mimiciv")
    parts = []
    Bs = 400
    for i in range(0, len(keys), Bs):
        ch = keys[i:i + Bs]
        q = ("SELECT hadm_id, note_seq, text FROM mimiciv_note.discharge "
             "WHERE hadm_id IN (%s)" % ",".join(map(str, ch)))
        parts.append(pd.read_sql(q, cc))
    cc.close()
    notes = pd.concat(parts, ignore_index=True)
    notes = (notes.sort_values(["hadm_id", "note_seq"])
                  .drop_duplicates("hadm_id", keep="last"))
    notes["hadm_id"] = notes.hadm_id.astype(int)
    txt = dict(zip(notes.hadm_id, notes.text))

    recs = []
    for r in sel.itertuples():
        t = txt.get(int(r.stay_key), "")
        hc, dd, moa = build_blob(t)
        blob = join_blob(hc, dd, moa)
        ex, trunc, rawlen = truncate(blob)

        # 算法判定：全文 vs 评分者实际看到的（截断后）文本
        lab_full, fp_full, raw_full, pos_full, v_full = verdict(hc, dd, moa)
        hcv, ddv, moav = split_blob(ex)
        lab_vis, fp_vis, raw_vis, pos_vis, v_vis = verdict(hcv, ddv, moav)

        recs.append(dict(task=r.task, stratum=r.stratum, stay_key=int(r.stay_key),
                         disease=r.primary_grp, algo_label=r.inact3,
                         algo_label_recomputed=lab_full,
                         algo_label_visible=lab_vis,
                         spec_true=si(r.spec_true),
                         spec_raw=si(r.spec_raw_hits),
                         spec_pos=si(r.spec_pos_hits),
                         spec_neg=si(r.spec_neg_hits),
                         spec_falsepos=si(r.spec_falsepos),
                         spec_falsepos_full=fp_full,
                         spec_falsepos_visible=fp_vis,
                         spec_verdict_full=v_full,
                         spec_verdict_visible=v_vis,
                         death_30d=si(r.death_30d),
                         gc24_str=r.gc24_str, home_gc=si(r.home_gc),
                         pdx_class=r.pdx_class, note_len=si(r.note_len2),
                         excerpt_len=len(ex), excerpt_rawlen=rawlen,
                         truncated=trunc,
                         algo_stable=int(lab_vis == lab_full),
                         algo_stable_B=int(v_vis == v_full),
                         excerpt=ex))
    df = pd.DataFrame(recs)

    # ---------------- 自检：重算的全文判定必须复现 163 的落盘标签
    P("")
    P("--- 复现自检：用 163 词表重算 != gc_inactivity.csv 的标签 ---")
    bad = df[df.algo_label_recomputed != df.algo_label]
    P("  不一致 %d / %d" % (len(bad), len(df)))
    if len(bad):
        P(bad[["stay_key", "algo_label", "algo_label_recomputed"]].head(10)
          .to_string())
    badB = df[df.spec_falsepos_full != df.spec_falsepos]
    P("  B：重算 spec_falsepos 与落盘不一致 %d / %d" % (len(badB), len(df)))
    if len(badB):
        P(badB[["stay_key", "spec_raw", "spec_pos", "spec_falsepos",
                "spec_falsepos_full"]].head(10).to_string())

    # ---------------- 不透明 case_id（随机顺序）
    df = df.sample(frac=1.0, random_state=rs).reset_index(drop=True)
    nA = int((df.task == "A").sum())
    nB = int((df.task == "B").sum())
    cid = []
    ia = ib = 0
    for t in df.task:
        if t == "A":
            ia += 1
            cid.append("A-%03d" % ia)
        else:
            ib += 1
            cid.append("B-%03d" % ib)
    df["case_id"] = cid

    # ---------------- 盲表（分卷，避免单次读取被截断）
    files = {}

    def write_blind(sub, task, title):
        sub = sub.reset_index(drop=True)
        nparts = int(np.ceil(len(sub) / CHUNK))
        for pi in range(nparts):
            part = sub.iloc[pi * CHUNK:(pi + 1) * CHUNK]
            fn = "adjud_blind_%s_p%d.md" % (task, pi + 1)
            lines = ["# %s (part %d/%d)" % (title, pi + 1, nparts), "",
                     "每例只给出**不透明编号**与病历文本。表中**不提供**病种分组、激素",
                     "剂量、临床结局、以及任何自动词表判定结果 —— 请勿从外部推断它们，",
                     "只依据文本**实际写明**的内容判定。病历正文中的疾病名、药品名",
                     "属于原文，按原文理解即可。",
                     "",
                     "判定结果写入 CSV（每卷一个），三列：case_id,label,evidence_quote",
                     "evidence_quote 必须是从该例文本中**逐字复制**的片段（≤120 字符）。",
                     ""]
            for r in part.itertuples():
                lines += ["", "=== %s ===" % r.case_id, "", r.excerpt, ""]
            with open(os.path.join(OUT, fn), "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            files[fn] = len(part)
        sub[["case_id", "excerpt"]].to_csv(
            os.path.join(OUT, "adjud_blind_%s.csv" % task), index=False,
            encoding="utf-8")
        return nparts

    a = df[df.task == "A"]
    b = df[df.task == "B"]
    na = write_blind(a, "A",
                     "TASK A —— 三分类盲法核验（INACTIVE / INF_DEFER / UNDERDOC）")
    nb = write_blind(b, "B",
                     "TASK B —— 否定规则盲法核验（ACTIVITY-POSITIVE / PURELY-NEGATED）")

    # ---------------- 编码手册（两评分者逐字相同）
    CODEBOOK = """# 编码手册（两评分者使用完全相同的版本）

你只能看到：不透明病例编号 + 病历节选（住院经过 / 出院诊断 / 入院前用药）。
表中**没有**病种分组、激素剂量、临床结局，也没有任何自动词表判定结果。
请只依据文本**实际写明**的内容判定；不要用「严重程度」或「是否给了激素」反推。

--------------------------------------------------------------------------
## 任务 A：把每一例判为三个互斥标签之一

判定优先级固定为 **INACTIVE > INF_DEFER > UNDERDOC**（先看第一条，命中即止）。

### INACTIVE —— 明确记载疾病无活动
文本中存在**明确陈述**「风湿病处于无活动 / 缓解 / 静止 / 稳定 / 控制良好」，
**或**明确陈述「没有活动性疾病证据 / 无复发 / 否认风湿相关症状」。例：
- "lupus is in remission"、"disease activity: none"、"quiescent disease"
- "no evidence of active lupus"、"no flare"、"denies joint pain / morning stiffness"
- "stable SLE"、"free of active disease"、"doing well from rheumatological standpoint"
- "disease activity was minimal / low"、"inactive disease"

### INF_DEFER —— 明确记载因感染而暂缓/停用激素或免疫抑制
文本中存在**明确的因果陈述**：激素 / 免疫抑制剂被 **held / withheld / deferred /
delayed / stopped / discontinued / not given**，并把这归因于**感染或感染风险**；
也包括同一段落中同时出现「感染」与「激素减停」的情形。例：
- "prednisone held due to concern for sepsis"
- "steroids deferred in the setting of pneumonia"
- "immunosuppression withheld because of positive blood culture"
- "on hold in light of aspiration pneumonia"

**注意**：本标签描述的是**病历写明的用药理由**，不是「实际给没给」。
即使后续或此前给了激素，只要文本明确写了「因感染而暂缓」，仍判 INF_DEFER。

### UNDERDOC —— 记录不足以判定
以上都不成立：文本既没有明确说「无活动」，也没有明确说「因感染而暂缓」。
**最常见的正确判定就是 UNDERDOC。**

### 三条硬规则（防止把「沉默」读成「无活动」）
1. **「没提到活动」不等于「无活动」** —— 未提及一律 UNDERDOC。
2. 「未复查抗体 / 未做活动度评分」**不等于**无活动 —— 仍为 UNDERDOC。
3. 反面证据（如 "no acute complaints"、"afebrile"、"no new symptoms"，
   或只针对非风湿问题）**不等于**风湿病无活动，除非明确指向风湿病本身。

--------------------------------------------------------------------------
## 任务 B：判断一段被词表命中「活动」的文本，是否真的写了活动

词表把某些词句标为「疾病特异活动」（如 flare、active lupus、lupus nephritis、
alveolitis、thrombocytopenia、NPSLE 等），但它对**否定句不敏感**。请判定：

### ACTIVITY-POSITIVE
文本中存在**至少一处未否定的、肯定性的**疾病特异活动陈述。例：
- "presented with a lupus flare"、"active lupus nephritis on biopsy"
- "worsening synovitis"、"new onset seizure attributed to NPSLE"
- "thrombocytopenia consistent with ITP, treated with steroids"

### PURELY-NEGATED
文本中**所有**与活动相关的词句都是被否定的、已解决的、或只出现在排除语境中。例：
- "no evidence of active lupus"、"denies joint pain"
- "no flare"、"resolved synovitis"、"negative for arthritis on exam"
- "ruled out lupus nephritis"、"no signs of active disease"

否定线索：no, not, without, denies, negative, absent, free of, ruled out,
no evidence of, no longer, resolved, improved, unremarkable, normal。

### 一条硬规则
只要存在**一处**肯定性活动陈述，即为 ACTIVITY-POSITIVE ——
不要因为文本同时也说「不活动」就改判（那属于同时记载，仍算阳性）。

--------------------------------------------------------------------------
## 输出格式

每卷盲表对应一个 CSV，写入 `out/` 目录：
`adjud_<RaterName>_A_p<卷号>.csv` 与 `adjud_<RaterName>_B_p<卷号>.csv`
列：`case_id,label,evidence_quote`

- `label` 只能取上表中的四个字符串之一（大小写敏感）：
  INACTIVE / INF_DEFER / UNDERDOC / ACTIVITY-POSITIVE / PURELY-NEGATED
- `evidence_quote` 必须是该例文本中**逐字复制**的片段（≤120 字符）；
  若判为「无证据」类（UNDERDOC 或 PURELY-NEGATED）且找不到代表性原句，写 NONE。
- **每一例都必须有且只有一行**，不得跳过、不得合并。
"""
    with open(os.path.join(OUT, "adjud_codebook.md"), "w",
              encoding="utf-8") as f:
        f.write(CODEBOOK)

    # ---------------- 封印键
    key = df[["case_id", "task", "stratum", "stay_key", "disease", "algo_label",
              "algo_label_recomputed", "algo_label_visible",
              "spec_true", "spec_raw", "spec_pos", "spec_neg", "spec_falsepos",
              "spec_falsepos_visible", "spec_verdict_full", "spec_verdict_visible",
              "death_30d", "gc24_str", "home_gc",
              "pdx_class", "note_len", "excerpt_len", "excerpt_rawlen",
              "truncated", "algo_stable", "algo_stable_B"]]
    key.to_csv(os.path.join(OUT, "adjud_sealed_key.csv"), index=False)

    P("")
    P("=" * 92)
    P("165 QC —— 盲法抽样")
    P("=" * 92)
    P("任务 A 盲表 n=%d ; 任务 B 盲表 n=%d" % (nA, nB))
    P("")
    P("--- 截断情况 ---")
    for t in ["A", "B"]:
        s = df[df.task == t]
        P("  %s: 截断 %d / %d (%.1f%%) ; 摘录长度中位 %d, 最大 %d, 总字符 %d"
          % (t, int(s.truncated.sum()), len(s),
             100 * s.truncated.mean(), int(s.excerpt_len.median()),
             int(s.excerpt_len.max()), int(s.excerpt_len.sum())))
    P("--- 截断是否改变了算法自己的判定（输入保真度）---")
    for t, lab in [("A", "inact3"), ("B", "spec_falsepos")]:
        s = df[df.task == t]
        col = "algo_stable" if t == "A" else "algo_stable_B"
        P("  任务 %s：判定被截断改变的 %d / %d (%.1f%%)"
          % (t, int((s[col] == 0).sum()), len(s),
             100 * (s[col] == 0).mean()))
        chg = s[s[col] == 0]
        if len(chg):
            P("      例：" + ", ".join(chg.case_id.head(8).tolist()))
    P("--- 盲表分卷 ---")
    for fn, n in files.items():
        P("  %-28s %3d 例" % (fn, n))
    P("")
    P("--- 抽样代表性核对（抽取 vs 全层）---")
    P("  %-28s %8s %10s %10s" % ("层", "抽取 n", "抽取内病种", "全层病种"))
    for k in ["INACTIVE", "INF_DEFER", "UNDERDOC"]:
        s = fa[fa.inact3 == k]
        d = a[a.stratum == k]
        allSLE = 100 * (m[(m.spec_true == 0) & (m.spec_raw_hits == 0) &
                          (m.inact3 == k)].primary_grp == "SLE").mean()
        P("  %-28s %8d %9.1f%% %9.1f%%" % (
            k, len(d), 100 * (d.disease == "SLE").mean(), allSLE))
    P("")
    P("--- 任务 B 抽样代表性 ---")
    for k in ["purely negated (spec_falsepos=1)", "affirmative (spec_falsepos=0)"]:
        P("  %-34s 抽 %d" % (k, int((b.stratum == k).sum())))
    P("")
    P("注意：任务 A 抽样框已排除 %d 例全否定假阳性（它们只出现在任务 B），"
      % int(df.spec_falsepos.sum()))
    P("      故任务 A 的层构成基于 n=%d，非 874。" % len(fa))

    with open(os.path.join(OUT, "165_blind_qc.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    P("")
    P("saved: adjud_blind_{A_p1..,B_p1..}.md, adjud_blind_{A,B}.csv,")
    P("       adjud_sealed_key.csv, adjud_codebook.md, 165_blind_qc.txt")
    print("\n".join(L))


if __name__ == "__main__":
    main()
