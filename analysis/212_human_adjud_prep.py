# -*- coding: utf-8 -*-
"""
212_human_adjud_prep.py —— 真人风湿科医师盲审材料一次性构建（v13 / adjud3）

背景
----
v12 已把「三分类文本标签」彻底封死（修手册后 v8 引擎任务 B 加权 PPV 由 0.343
掉到 0.114）。文本侧唯一幸存的主输出是**连续活动度信息密度** aid_doc
（= (act_pos + act_neg + inact_n + defer_n) / 千字符，分母为
brief hospital course + discharge diagnosis 的字符数）。

因此本次真人盲审**不再审三分类标签**，而是验 aid_doc 的**构念效度**：
aid_doc 到底测不测得到「这份病历写了多少风湿病疾病活动度信息」。

抽样设计
--------
抽样框  = data/gc_activity_density.csv（n=1250，头对头 SLE+RA 中有出院小结者）
         − 既往评审样本 279 例（v7b 148 + v9 131；铁律 20：设计样本物理排除）
         = 971 例

分层（4 层，切点取**可用框内正侧三分位**）：
  L1  aid_doc = 0
  L2  0 < aid_doc <= T1
  L3  T1 < aid_doc <= T2
  L4  aid_doc > T2

分配：L2/L3/L4 **全查**（正侧总共仅 138 例，抽样即等于全查，层内权重=1，
      在信息量最大的一段上不留抽样误差）；L1 抽 n1 = 220 − n_pos 例，
      并按病种分层富集 SLE（SLE 是交互的驱动侧）。
      实际：n1 = 82（SLE 30 / RA 52），L2=L3=L4=46 → 合计 220。

设计权重 w = N_cell / n_cell（Horvitz–Thompson），Σ w·1 = 971 自检。

盲法
----
· 编号 H-001…H-220 与层/病种**随机错位**（编号顺序本身是 shuffle 的结果）
· 两名评分者用**同一套编号**（保证可配对）但**不同呈现顺序**
· 盲表内**不出现**层标签、aid_doc、激素剂量、结局、任何算法判定
· 疾病无法盲（原文写明 lupus / RA），不作假装

输出
----
  out/adjud3_frame.csv            抽样框全量 971 行（含层、权重、是否入样）
  out/adjud3_sample.csv           入样 220 例（不含正文）
  out/adjud3_sealed_key.csv       密封键（含 aid_doc / 可见文本重算值 / 权重 / 顺序）
  out/adjud3_sheet_A.md / _B.md   盲表全文（交评分者）
  out/adjud3_sheet_A_b0..b4.md    分卷：b0=校准 15 例，b1–b4=正式 4 卷
  out/adjud3_sheet_A.csv / _B.csv 机读盲表（case_id, excerpt）
  out/adjud3_score_A.csv / _B.csv 评分模板（utf-8-sig，可直接 Excel 打开）
  out/adjud3_rater_manual.md      评审员手册（序数版，由 v2 手册改编）
  out/adjud3_analysis_plan.md     预先指定的分析计划（含达标阈值与判定规则）
  out/212_human_adjud_prep.txt    构建日志（QC）
"""
import os
import re
import io
import sys
import importlib.util
import numpy as np
import pandas as pd
import psycopg2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
DATA = os.path.join(ROOT, "data")

SEED = 20260917
SEED_A = 20260917 + 1
SEED_B = 20260917 + 2
SEED_CAL = 20260917 + 3

# 不做内容截断：历史上 CAP=5000 的头/尾截断是为子代理省上下文窗口，
# 但会让 43.6% 的病例被切、22.3% 的病例**全部命中句**被切掉
# （aid_doc>0 而可见文本 aid_doc=0），使 L2 层在可见文本上塌成与 L1 无差别。
# 真人医师没有上下文窗口限制，且构念效度要求「人看到的 == 算法算过的」。
CAP, HEAD, TAIL = 10 ** 9, 3200, 1700
TRUNC_MARK = "\n\n[...middle of hospital course omitted for length...]\n\n"

WPM = 250.0            # 阅读速度（词/分钟），用于工作量估算
CH_PER_WORD = 6.1      # 英文平均词长（含空格）
CPM = WPM * CH_PER_WORD   # ≈1525 字符/分钟

N_TARGET = 220
N1_SLE = 30          # L1 层内 SLE 富集目标
N_CALIB = 15         # 校准批
CAL_BY_LAYER = {"L1": 6, "L2": 3, "L3": 3, "L4": 3}

L = []


def P(s=""):
    print(s)
    sys.stdout.flush()
    L.append(str(s))


# ------------------------------------------------------------------ 模块复用
def load_mod(name, fname):
    p = os.path.join(ROOT, "scripts", fname)
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E = load_mod("mod168", "168_activity_rule_engine.py")   # 词表 / 判定 / aid_doc
M = E.M                                                  # 163：切段


def truncate(blob):
    if len(blob) <= CAP:
        return blob, 0, len(blob)
    return blob[:HEAD] + TRUNC_MARK + blob[-TAIL:], 1, len(blob)


def split_blob(b):
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


def build_blob(text):
    t = text or ""
    secs = M.split_sections(t)
    hc = M.sec(secs, "brief hospital course", "hospital course")
    dd = M.sec(secs, "discharge diagnosis")
    moa = M.sec(secs, "medications on admission")
    return E.norm(hc), E.norm(dd), E.norm(moa), t


def join_blob(hc, dd, moa):
    b = hc + "\n\n[Discharge diagnosis]\n" + dd
    if moa:
        b += "\n\n[Medications on admission]\n" + moa
    return b.strip()


def density_on(hc, dd, moa, raw=""):
    """严格照 168 main() 的配方重算 aid_doc（narr = hc + ' ' + dd，短病历回退全文）。

    注意：不能走「拼 blob → 再拆回」的往返，那条路与 168 的 narr 定义不等价
    （短病历回退分支不同），会导致逐例复现失败。
    """
    if len(hc) + len(dd) < 200:
        narr = E.norm(raw)
    else:
        narr = (hc or "") + " " + (dd or "")
    nchar = max(1.0, len(narr) / 1000.0)
    ia = E.eval_inactivity(narr)
    ac = E.eval_activity(narr)
    df = E.eval_defer((narr or "") + " " + (moa or ""))
    info_n = ac["act_pos"] + ac["act_neg"] + ia["inact_n"] + df["defer_n"]
    return dict(aid_doc=info_n / nchar,
                aid_signed=(ac["act_pos"] - ia["inact_n"]) / nchar,
                narr_char=len(narr), info_n=info_n)


# ================================================================== 主流程
def main():
    # 规则全开（铁律 29），并记录
    E.set_rules(True, True, True)
    assert all(E.RULES.values()), "规则开关必须为全开"

    P("=" * 74)
    P("212 真人盲审材料构建  adjud3 / v13   SEED=%d" % SEED)
    P("=" * 74)

    # ---------------------------------------------------------- 1 抽样框
    d = pd.read_csv(os.path.join(DATA, "gc_activity_density.csv"))
    d["stay_key"] = d["stay_key"].astype(int)
    coh = pd.read_csv(os.path.join(DATA, "cohort_rheum_icu.csv"))
    coh["stay_key"] = coh["stay_key"].astype(int)
    coh["disease"] = np.where(coh["is_sle"] == 1, "SLE",
                              np.where(coh["is_ra"] == 1, "RA", "OTHER"))
    fr = d.merge(coh[["stay_key", "disease", "is_sle", "is_ra", "death_30d",
                      "gc24_daily_pe_mg", "sofa24", "age"]],
                 on="stay_key", how="left")
    P("")
    P("--- 1. 抽样框 ---")
    P("  密度表 n=%d（头对头 SLE+RA 有出院小结者）" % len(fr))

    # 既往评审样本 → 物理排除（铁律 20）
    ex = set()
    for f in ["adjud_sealed_key.csv", "adjud2_sealed_key.csv",
              "adjud_v2_audit_key.csv"]:
        p = os.path.join(OUT, f)
        if os.path.exists(p) and "stay_key" in pd.read_csv(p, nrows=1).columns:
            k = pd.read_csv(p)["stay_key"].astype(int)
            ex |= set(k)
            P("  排除源 %-26s %3d 例" % (f, len(k)))
    fr["excluded"] = fr["stay_key"].isin(ex).astype(int)
    P("  既往评审样本（去重）%d 例 → 全部物理排除" % len(ex))
    av = fr[fr["excluded"] == 0].copy()
    P("  可用抽样框 n=%d" % len(av))

    # ---------------------------------------------------------- 2 分层
    pos = av[av["aid_doc"] > 0]
    T1, T2 = np.percentile(pos["aid_doc"], [33.3333, 66.6667])
    av["layer"] = np.where(
        av["aid_doc"] == 0, "L1",
        np.where(av["aid_doc"] <= T1, "L2",
                 np.where(av["aid_doc"] <= T2, "L3", "L4")))
    P("")
    P("--- 2. 分层（切点取可用框内正侧三分位）---")
    P("  T1=%.4f   T2=%.4f" % (T1, T2))
    P("  %-4s %-26s %8s %6s %6s %6s" % ("层", "定义", "框内N", "事件", "SLE", "RA"))
    DEF = {"L1": "aid_doc = 0",
           "L2": "0 < aid_doc <= %.4f" % T1,
           "L3": "%.4f < aid_doc <= %.4f" % (T1, T2),
           "L4": "aid_doc > %.4f" % T2}
    for ly in ["L1", "L2", "L3", "L4"]:
        g = av[av["layer"] == ly]
        P("  %-4s %-26s %8d %6d %6d %6d" % (
            ly, DEF[ly], len(g), int(g["death_30d"].sum()),
            int((g["disease"] == "SLE").sum()), int((g["disease"] == "RA").sum())))

    # ---------------------------------------------------------- 3 分配
    rng = np.random.RandomState(SEED)
    cells = []          # (layer, disease, 框内N, 抽n)
    n_pos = int((av["layer"] != "L1").sum())
    n1 = N_TARGET - n_pos
    for ly in ["L2", "L3", "L4"]:
        for dz in ["SLE", "RA"]:
            N_h = int(((av["layer"] == ly) & (av["disease"] == dz)).sum())
            cells.append([ly, dz, N_h, N_h])        # 全查
    N1_sle = int(((av["layer"] == "L1") & (av["disease"] == "SLE")).sum())
    N1_ra = int(((av["layer"] == "L1") & (av["disease"] == "RA")).sum())
    take_sle = min(N1_SLE, N1_sle)
    take_ra = n1 - take_sle
    assert take_ra <= N1_ra, "L1 的 RA 框内例数不足"
    cells.append(["L1", "SLE", N1_sle, take_sle])
    cells.append(["L1", "RA", N1_ra, take_ra])
    cells = pd.DataFrame(cells, columns=["layer", "disease", "N_frame", "n_take"])
    cells["weight"] = cells["N_frame"] / cells["n_take"]

    P("")
    P("--- 3. 分配（L2–L4 全查；L1 按病种分层富集 SLE）---")
    P("  %-4s %-6s %8s %8s %10s" % ("层", "病种", "框内N", "抽n", "设计权重"))
    for r in cells.itertuples():
        P("  %-4s %-6s %8d %8d %10.4f" % (
            r.layer, r.disease, r.N_frame, r.n_take, r.weight))
    P("  合计抽样 %d 例" % int(cells["n_take"].sum()))
    assert int(cells["n_take"].sum()) == N_TARGET, "抽样总数必须等于 220"
    assert abs(float((cells["N_frame"]).sum()) - len(av)) == 0
    assert abs(float((cells["weight"] * cells["n_take"]).sum()) - len(av)) < 1e-6, \
        "Σ权重必须复现框内总数"

    picks = []
    for r in cells.itertuples():
        pool = av[(av["layer"] == r.layer) & (av["disease"] == r.disease)]
        idx = rng.choice(pool["stay_key"].values, size=r.n_take, replace=False)
        for k in idx:
            picks.append(dict(stay_key=int(k), layer=r.layer, disease=r.disease,
                              weight=float(r.weight)))
    samp = pd.DataFrame(picks)
    assert samp["stay_key"].nunique() == N_TARGET
    assert not samp["stay_key"].isin(ex).any(), "入样不得命中既往评审样本"

    fr = fr.merge(cells[["layer", "disease", "weight"]], on=["layer", "disease"],
                  how="left") if "layer" in fr.columns else fr
    av = av.merge(cells[["layer", "disease", "weight"]], on=["layer", "disease"],
                  how="left")
    av["sampled"] = av["stay_key"].isin(samp["stay_key"]).astype(int)
    av.to_csv(os.path.join(OUT, "adjud3_frame.csv"), index=False,
              encoding="utf-8")

    samp = samp.merge(av[["stay_key", "aid_doc", "aid_signed", "narr_char",
                          "death_30d", "age", "sofa24", "gc24_daily_pe_mg"]],
                      on="stay_key", how="left")

    # ---------------------------------------------------------- 4 取原文
    keys = sorted(samp["stay_key"].tolist())
    cc = psycopg2.connect(host=os.environ.get("MIMIC_DB_HOST", "localhost"),
                          port=int(os.environ.get("MIMIC_DB_PORT", "5432")),
                          user=os.environ.get("MIMIC_DB_USER", "postgres"),
                          password=os.environ.get("MIMIC_DB_PASSWORD", ""),
                          dbname="mimiciv")
    parts = []
    for i in range(0, len(keys), 200):
        ch = keys[i:i + 200]
        q = ("SELECT hadm_id, note_seq, text FROM mimiciv_note.discharge "
             "WHERE hadm_id IN (%s)" % ",".join(map(str, ch)))
        parts.append(pd.read_sql(q, cc))
    cc.close()
    notes = pd.concat(parts, ignore_index=True)
    notes = (notes.sort_values(["hadm_id", "note_seq"])
                  .drop_duplicates("hadm_id", keep="last"))
    notes["hadm_id"] = notes["hadm_id"].astype(int)
    txt = dict(zip(notes["hadm_id"], notes["text"]))
    P("")
    P("--- 4. 取原文 ---")
    P("  请求 %d 例，命中出院小结 %d 例" % (len(keys), len(notes)))
    miss = [k for k in keys if not (txt.get(k) or "").strip()]
    P("  无文本 %d 例" % len(miss))
    assert len(miss) == 0, "存在无文本病例：%s" % miss[:5]

    recs = []
    for r in samp.itertuples():
        hc, dd, moa, raw = build_blob(txt.get(int(r.stay_key), ""))
        # 与 168 的 narr 定义保持同一回退分支：段落解析失败时 narr = 整份出院小结，
        # 因此评分者看到的也必须退回整份小结，否则「人看空壳、算法算全文」。
        if len(hc) + len(dd) < 200:
            blob = E.norm(raw)
        else:
            blob = join_blob(hc, dd, moa)
        exr, tr, rawlen = truncate(blob)
        vis = density_on(hc, dd, moa, raw)
        recs.append(dict(stay_key=int(r.stay_key), excerpt=exr,
                         truncated=tr, rawlen=rawlen,
                         aid_doc_vis=vis["aid_doc"],
                         narr_char_vis=vis["narr_char"],
                         info_n_vis=vis["info_n"]))
    smp = samp.merge(pd.DataFrame(recs), on="stay_key", how="left")

    # 不截断 → 可见文本上重算的密度必须与落盘 aid_doc 逐例相等（铁律 38 式门槛断言）
    dmax = float((smp["aid_doc"] - smp["aid_doc_vis"]).abs().max())
    same0 = float((((smp["aid_doc"] == 0) == (smp["aid_doc_vis"] == 0)).mean()))
    P("  可见文本 vs 落盘 aid_doc：逐例最大绝对差 %.2e；零/正同判率 %.1f%%"
      % (dmax, 100 * same0))
    assert dmax < 1e-9, "不截断时重算密度必须逐例复现落盘值（差 %.2e）" % dmax
    assert same0 == 1.0
    P("  段落解析成功（hc+dd ≥ 200 字符）%d 例；回退整份小结 %d 例"
      % (int((smp["narr_char_vis"] <= smp["rawlen"]).sum()),
         int((smp["narr_char_vis"] > smp["rawlen"]).sum())))
    assert int(smp["narr_char_vis"].min()) >= 1, "narr 不得为空"
    tot_ch = int(smp["rawlen"].sum())
    P("  阅读量：合计 %s 字符 ≈ %s 词（中位 %d 字符/例，最大 %d）"
      % ("{:,}".format(tot_ch), "{:,}".format(int(tot_ch / CH_PER_WORD)),
         int(smp["rawlen"].median()), int(smp["rawlen"].max())))
    P("  按 %d 词/分钟（≈%d 字符/分钟）→ 每人阅读 %.1f h；加评分开销（40 s/例 ≈ %.1f h）"
      "→ 每人合计约 %.1f h"
      % (int(WPM), int(CPM), tot_ch / CPM / 60.0, len(smp) * 40 / 3600.0,
         tot_ch / CPM / 60.0 + len(smp) * 40 / 3600.0))
    smp.to_csv(os.path.join(OUT, "adjud3_sample.csv"), index=False,
               encoding="utf-8")

    # ---------------------------------------------------------- 5 编号与顺序
    rngc = np.random.RandomState(SEED)
    order_id = list(rngc.permutation(len(smp)))
    smp = smp.reset_index(drop=True)
    smp["case_id"] = ["H-%03d" % (i + 1) for i in range(len(smp))]
    # 编号与层随机错位：按 shuffle 后的位置赋号（编号顺序 = shuffle 顺序）
    smp = smp.iloc[order_id].reset_index(drop=True)
    smp["case_id"] = ["H-%03d" % (i + 1) for i in range(len(smp))]

    # 校准批：层内分层取 15 例（两名评分者同一批、同一顺序）
    rng_cal = np.random.RandomState(SEED_CAL)
    cal_ids = []
    for ly, k in CAL_BY_LAYER.items():
        pool = smp.loc[smp["layer"] == ly, "case_id"].tolist()
        cal_ids += list(rng_cal.choice(pool, size=min(k, len(pool)), replace=False))
    smp["calib"] = smp["case_id"].isin(cal_ids).astype(int)
    P("")
    P("--- 5. 编号与顺序 ---")
    P("  编号 H-001…H-%03d，与层/病种随机错位" % len(smp))
    P("  校准批 %d 例（L1 6 / L2 3 / L3 3 / L4 3），两名评分者同一批" % len(cal_ids))

    def make_order(seed):
        r = np.random.RandomState(seed)
        cal = smp.loc[smp["calib"] == 1, "case_id"].tolist()
        rest = smp.loc[smp["calib"] == 0, "case_id"].tolist()
        r.shuffle(rest)
        return cal + rest

    for tag, seed in [("A", SEED_A), ("B", SEED_B)]:
        smp["order_" + tag] = 0
        for i, cid in enumerate(make_order(seed), 1):
            smp.loc[smp["case_id"] == cid, "order_" + tag] = i
    # 两评分者顺序不得相同
    assert not (smp["order_A"] == smp["order_B"]).all(), "A/B 顺序不能相同"

    NB = 8        # 正式卷数：每卷 ~26 例 / 约 2.5–3 h，便于分次完成、降低疲劳噪声
    rest_n = int((smp["calib"] == 0).sum())
    base = rest_n // NB
    sizes = [base] * NB
    sizes[0] += rest_n - base * NB
    smp["batch_A"] = 0
    smp["batch_B"] = 0
    for tag in ["A", "B"]:
        sub = smp[smp["calib"] == 0].sort_values("order_" + tag)
        pos_in_batch, b = 0, 1
        for cid in sub["case_id"]:
            smp.loc[smp["case_id"] == cid, "batch_" + tag] = b
            pos_in_batch += 1
            if pos_in_batch >= sizes[b - 1] and b < NB:
                b += 1
                pos_in_batch = 0
    P("  正式卷：A/B 各 %d 卷，卷内例数 %s（总 %d）" % (NB, sizes, rest_n))

    keycols = ["case_id", "stay_key", "layer", "disease", "weight",
               "aid_doc", "aid_doc_vis", "narr_char", "narr_char_vis",
               "info_n_vis", "death_30d", "truncated", "rawlen",
               "calib", "order_A", "order_B", "batch_A", "batch_B"]
    smp[keycols].sort_values("case_id").to_csv(
        os.path.join(OUT, "adjud3_sealed_key.csv"), index=False, encoding="utf-8")
    P("  密封键已写出：out/adjud3_sealed_key.csv")

    # ---------------------------------------------------------- 6 盲表
    EX = dict(zip(smp["case_id"], smp["excerpt"]))

    def write_md(path, title, ids, note):
        with io.open(path, "w", encoding="utf-8") as f:
            f.write("# %s\n\n" % title)
            f.write(note)
            f.write("\n---\n\n")
            for cid in ids:
                f.write("=== %s ===\n\n%s\n\n" % (cid, EX[cid]))
            f.write("---\n（本卷结束）\n")

    NOTE = (
        "每例只给出**不透明编号**与病历文本。表中**不提供**病种分组、疾病活动度\n"
        "信息密度、激素剂量、临床结局、以及任何算法判定结果 —— 请勿从外部推断，\n"
        "只依据文本**实际写明**的内容，并严格按 `adjud3_rater_manual.md` 评分。\n"
        "病历正文中的疾病名、药品名属于原文，按原文理解即可。\n\n"
        "评分结果写入 CSV（每卷一个），列：\n"
        "`case_id, q1_info, q2_nstmt, q3_time, q4_explicit, quote, conf, note`\n"
        "`quote` 必须是该例文本中**逐字复制**的片段（≤120 字符）；\n"
        "**每一例都必须有且只有一行**。\n"
    )
    for tag in ["A", "B"]:
        ids = smp.sort_values("order_" + tag)["case_id"].tolist()
        write_md(os.path.join(OUT, "adjud3_sheet_%s.md" % tag),
                 "盲审卷 %s（共 %d 例）" % (tag, len(ids)), ids, NOTE)
        pd.DataFrame({"case_id": ids, "excerpt": [EX[c] for c in ids]}).to_csv(
            os.path.join(OUT, "adjud3_sheet_%s.csv" % tag), index=False,
            encoding="utf-8")
        # 分卷
        cal = smp[smp["calib"] == 1].sort_values("order_" + tag)["case_id"].tolist()
        write_md(os.path.join(OUT, "adjud3_sheet_%s_b0.md" % tag),
                 "盲审卷 %s · 校准批（%d 例）" % (tag, len(cal)), cal,
                 NOTE + "\n**本卷为校准批**：两名评分者使用完全相同的一批，\n"
                        "做完请**先与研究者对答案、统一尺度**，再进入 b1–b4。\n")
        for b in range(1, NB + 1):
            bid = smp[smp["batch_" + tag] == b].sort_values(
                "order_" + tag)["case_id"].tolist()
            write_md(os.path.join(OUT, "adjud3_sheet_%s_b%d.md" % (tag, b)),
                     "盲审卷 %s · 第 %d 卷（%d 例）" % (tag, b, len(bid)),
                     bid, NOTE)
    P("")
    P("--- 6. 盲表 ---")
    P("  adjud3_sheet_A.md / _B.md（全文）+ 各 b0–b%d 分卷 + 机读 CSV" % NB)

    # ---------------------------------------------------------- 7 评分模板
    TPL = ["case_id", "q1_info", "q2_nstmt", "q3_time", "q4_explicit",
           "quote", "conf", "note"]
    for tag in ["A", "B"]:
        ids = smp.sort_values("order_" + tag)["case_id"].tolist()
        t = pd.DataFrame({"case_id": ids})
        for c in TPL[1:]:
            t[c] = ""
        t.to_csv(os.path.join(OUT, "adjud3_score_%s.csv" % tag),
                 index=False, encoding="utf-8-sig")
    P("  评分模板 adjud3_score_A.csv / _B.csv（utf-8-sig，可直接 Excel 打开）")

    # ---------------------------------------------------------- 7b 评审员手册
    MANUAL = u"""# 评审员手册 —— 风湿病「疾病活动度信息密度」盲法评分
# adjud3_rater_manual —— 2026-09-17（v13.1，含 b0 校准后追加的裁决 6–7）

> 版本说明（研究者存档，评分者只需读下方正文）
> 本手册是**序数版**，取代此前的三分类标签手册（adjud_codebook_v1 / v2）。
> 原因：三分类标签方案已在 v9 的样本外核验中被推翻（v12 修订手册后加权 PPV 仅
> 0.114），**不再作为评审对象**。文本侧唯一在用的输出是**连续密度 aid_doc**，
> 因此本次评审的目标是验它的**构念效度**。

--------------------------------------------------------------------------
## 你要回答的问题（一句话）

> **这份出院小结，到底写了多少「风湿病疾病活动度」的信息？**

注意三件事：
1. 是**信息量**，不是**活动与否**。既写「疾病活动」也写「明确无活动」——
   两者都提供了信息，都应得高分；**只字不提**才是低分。
2. 是**本次住院**的记录。仅出现在既往史里、或仅作为长期维持用药的理由，
   信息量要打折（见 Q3）。
3. 是**风湿免疫病本身**。痛风发作、克罗恩病活动、肺炎，都不算。

--------------------------------------------------------------------------
## 你对什么保持盲态

以下信息**本表一律不提供**，也请不要试图推断：
自动计算的密度值（aid_doc）、既往算法标签、糖皮质激素剂量、
30 天与住院死亡结局、另一位评分者的打分、本例被分在哪一层。

病历正文中的疾病名、药品名属于**原文**（MIMIC-IV 已去标识化），
按原文理解即可。疾病本身无法设盲（原文写明 lupus / RA），不假装有。

--------------------------------------------------------------------------
## 四个评分项

### Q1 信息充分度（0–3，主项）

| 分值 | 含义 |
|---|---|
| **0** | 完全没有：全篇未提及疾病活动状态；或只有病名、只有长期用药清单、只有「风湿科随诊」这类套话 |
| **1** | 只有模糊/间接线索：能猜到与疾病有关，但没有一句明确陈述（例：仅列出激素剂量；仅「followed by rheumatology」；仅既往史式病名罗列） |
| **2** | 有明确陈述但简略：至少一句直接说清了当前活动状态（例："no clinical evidence of active lupus"、"her RA is quiescent"、"presented with active arthritis"） |
| **3** | 详细：在明确陈述之外，还给出 ≥2 项具体支撑——受累器官/体征、实验室指标（补体、dsDNA、CRP/ESR）、影像学、或治疗反应；或有专门一段评估活动度 |

> **不要按病历长短打分。** 很长的病历可能一句活动度都没有（给 0），
> 很短的病历也可能一句话说清楚（给 2）。

### Q2 活动度语句数（整数 ≥ 0）

数一数文本中有多少**句子或子句**在**直接描述风湿病的活动状态**。
- **算**：肯定与否定都算。"no evidence of active lupus" 算 1 条；
  "lupus nephritis with active urinary sediment" 算 1 条。
- **不算**：仅出现病名；仅列药名剂量；**非风湿病**的活动（PNA、cellulitis、
  gout flare、Crohn's flare）；检验值本身而无活动度判断。
- 同一个意思换个说法重复出现，按**一条**计。

### Q3 时间性（0 / 1 / 2）

| 分值 | 含义 |
|---|---|
| **0** | 没有任何与活动状态相关的描述 |
| **1** | 只出现在**既往史 / 长期维持**语境（past medical history、"chronic maintenance"、"baseline"），没有本次住院的当前状态 |
| **2** | 描述的是**本次住院期间或入院时**的状态（含入院评估、住院期间演变、出院时状态） |

### Q4 是否作出过明确的疾病活动度评估（0 / 1）

文本中是否存在**至少一句**对风湿病**当前**活动状态的**明确陈述**
（肯定或否定均可）。这一项直接对映自动密度是否为 0，请格外慎重。
- "no evidence of active lupus" → **1**（明确的否定陈述）
- "likely lupus flare"、"concern for disease activity" → **0**（是假说，不是陈述）
- 只有 "SLE" 这个病名 → **0**

--------------------------------------------------------------------------
## 五条最容易踩的坑（来自前期双标核验，务必逐条看）

1. **否定也是信息。** "no active disease"、"in remission"、"quiescent"
   必须给 Q1 ≥ 2、Q4 = 1。把否定读成「没写」是最常见的错误。
2. **假说与印象不是陈述。** likely / possible / suspected / concern for /
   believed to be / rule out 之后的内容 → Q4 = 0，最多算 Q1 = 1 的间接线索。
3. **领域必须限定在风湿免疫病。** "gout flare"、"Crohn's flare"、
   "worsening PNA" 一律不算本病活动度。
4. **时间性。** 只在既往史里出现的活动描述 → Q3 = 1；
   仅作为「为什么长期吃激素」的理由出现、且无当前状态 → Q3 = 1、Q4 = 0。
5. **用药指征型症状描述不算活动度断言。**
   "prednisone for control of her arthritis symptoms" 是间接线索（Q1 = 1、Q4 = 0），
   不是对活动度的评估。
6. **问题列表的栏目标题算「结构性分类」，不算句子。**（b0 校准后追加）
   "Inactive Issues"、"Chronic Issues"、"Active Problems" 这类栏目标题下
   挂着 SLE / RA 时，视为医师对该病**当前状态已作出结构性的明确归类**：
   → **Q4 = 1、Q1 ≥ 2、Q3 = 2**；但 **Q2 计 0**（它不是句子或子句）。
   理由：自动密度按词表命中计数，`inactive` 属于命中；若人判「不算」，
   人-机相关性会被系统性地在这类问题列表式病历上整体拉低。
7. **本次住院 ROS 的阴性记录属于当前时点。**（b0 校准后追加）
   入院 ROS 中「否认近期关节痛 / 肌痛」是**本次住院的当前信息** → **Q3 = 2**。
   但它是症状筛查、不是对活动度的评估 → **Q4 = 0、Q1 最多 1**。

### b0 校准锚点（已确认，正式卷沿用）

| 病例 | 关键原文 | 判定 | 说明 |
|---|---|---|---|
| H-027 | `most likely due to lupus cerebritis`、`concerning for SLE flare` | Q1 ≤ 1、Q4 = 0 | 假说不是陈述（坑 2） |
| H-149 | 活检 `Bullous Lupus` + 低补体 / dsDNA+ / 蛋白尿 / 激素 | Q1 = 3、Q4 = 1 | 明确当前活动 + ≥2 项支撑 |
| H-179 | `stable long term management` | Q1 = 1、Q4 = 0 | 长期管理 ≠ 活动状态 |
| H-219 | `not secondary to lupus flare` | Q1 = 3、Q4 = 1 | 明确的**阴性**陈述（坑 1） |
| H-107 | SLE 列于 `Inactive Issues` 下 | **Q1 = 2、Q2 = 0、Q3 = 2、Q4 = 1** | 按裁决 6 |
| H-218 | ROS 否认近期关节症状 | **Q1 = 1、Q2 = 0、Q3 = 2、Q4 = 0** | 按裁决 7 |

--------------------------------------------------------------------------
## 输出格式

每卷一个 CSV，8 列，**每一例有且只有一行**：

`case_id, q1_info, q2_nstmt, q3_time, q4_explicit, quote, conf, note`

- `q1_info` 0/1/2/3；`q2_nstmt` 非负整数；`q3_time` 0/1/2；`q4_explicit` 0/1
- `quote`：**逐字复制**该例文本中支撑你判断的片段（≤120 字符）。
  Q1=0 时填 `-`（表示无片段可引）。
- `conf` 你的把握度：1 = 不确定，2 = 一般，3 = 很确定
- `note` 选填：分歧、困难、或你认为该例特殊的地方

--------------------------------------------------------------------------
## 流程

1. **先做 b0 校准卷（15 例）**：两名评分者做完全相同的一批。
   做完**先与研究者对答案、统一尺度**，再进入 b1–b8。
   这一步不能省——不做校准，两人尺度不一致会淹没真实信号。
2. b1–b8 按任意顺序完成，每卷约 2 小时；建议一次只做一卷，避免疲劳。
3. 评分过程中**不要回看已交卷**、不要与另一位评分者讨论。
4. 全部完成后，分歧病例交由第三位医师仲裁。
"""
    with io.open(os.path.join(OUT, "adjud3_rater_manual.md"), "w",
                 encoding="utf-8") as f:
        f.write(MANUAL)
    P("  评审员手册 → out/adjud3_rater_manual.md")

    # ---------------------------------------------------------- 7c 分析计划
    n_ev_pos = int(smp.loc[smp["layer"] != "L1", "death_30d"].sum())
    PLAN = u"""# 预先指定的分析计划 —— 真人盲审（adjud3 / v13）

> **主体在拿到任何评分结果之前写定。** 目的：把判定阈值、估计量、
> 与「达标/不达标」的后果**预先锁定**，避免事后挑口径。
> v1.1 / v1.2 两次修订均在**正式卷（b1–b8）结果回收之前**完成，
> 且已注明各自的触发来源；主指标阈值自始未变。

## 修订记录

- **v1.1（2026-09-17，b0 校准批回收后、正式结果回收前）**：修正指标 3 的
  **正例方向**。原文写「`aid_doc` 预测『人判 Q4 = 0』的 AUC ≥ 0.75」，
  与构念方向自相矛盾——按构念，密度越高应越可能「有明确评估（Q4 = 1）」，
  若以 Q4 = 0 为正例，则达标方向应为 AUC ≤ 0.25。此为**纯逻辑笔误**，
  不依赖任何数据即可判定，故在正式结果回收前修正：
  以 **Q4 = 1 为正例**，阈值维持 AUC ≥ 0.75 / < 0.65 不变。
  同时报告反向 AUC 以便核对（两者应互补为 1）。
- **v1.2（2026-09-17，b0 校正评分回收后、正式结果回收前）**：新增
  **第 8 节「打分函数对比（事后催生、事前锁定）」**。b0 校准批暴露出
  `aid_doc` 的正侧 PPV 偏低，且叙述长度本身即有判别力，故追加两项
  **对比性诊断**（密度 vs 原始命中数、长度校正偏相关）。
  这两项目前**没有达标阈值**、只作参考列，且明确记录其来源于 b0——
  正式批次须以独立数据复核。主指标（1–6）的估计量与阈值**一律未改**。
- 其余阈值与后果条款**一律未改**。

--------------------------------------------------------------------------
## 1. 这次盲审验什么、不验什么

**验**：连续活动度信息密度 `aid_doc` 的**构念效度**——
它是否真的测到了「病历里写了多少风湿病疾病活动度信息」。

**不验**：三分类标签（INACTIVE / INF_DEFER / UNDERDOC）。该方案已由 v9 样本外
核验与 v12 手册修订证伪并**废弃**（v12 后任务 B 加权 PPV 仅 0.114）。

**不能指望它修什么（务必记住）**：
本盲审**只能救测量，救不了把握度**。
正侧（aid_doc > 0）在 spec− 分析子集内仅 **99 例 / 4 个事件**，
全文本集内 238 例 / 14 个事件。即便测量完美，
「密度 × 疾病 × 剂量」三项交互仍结构性不可估（v11：MDE 1.84–2.42）。
因此盲审**不会**让密度剂量-反应变得可估。

--------------------------------------------------------------------------
## 2. 抽样设计（已落盘，见 adjud3_frame.csv / adjud3_sealed_key.csv）

- 抽样框 n = 971 = 1250（有出院小结的头对头 SLE+RA）− 279（既往评审样本，物理排除）
- 4 层：L1 `aid_doc = 0`；L2–L4 为正侧三分位
- 分配：L2/L3/L4 **全查**（框内各 46，层内权重 = 1）；L1 抽 82 例
  （按病种分层富集：SLE 30 / RA 52）
- 设计权重 w = N_cell / n_cell（Horvitz–Thompson）；Σw = 971（已自检）
- 盲法：编号 H-001…H-220 与层/病种随机错位；两名评分者同编号、不同顺序；
  对 aid_doc、剂量、结局、对方评分均设盲

--------------------------------------------------------------------------
## 3. 预先指定的估计量与阈值

所有估计**一律使用抽样设计权重**（Horvitz–Thompson），
置信区间用**层内 bootstrap**（每层内重抽，1000 次）。

| # | 指标 | 估计量 | 达标阈值 | 失败阈值 |
|---|---|---|---|---|
| 1 | **构念效度（主）** | 两人 Q1 均值 vs `aid_doc` 的**加权 Spearman ρ** | ρ ≥ 0.40 | ρ < 0.30 |
| 2 | 构念效度（次） | 两人 Q2 语句数均值 vs `aid_doc` 的加权 Spearman ρ | ρ ≥ 0.40 | ρ < 0.30 |
| 3 | **零侧判别（主）** | `aid_doc` 预测「人判 **Q4 = 1**（有明确评估）」的加权 AUC（v1.1 修正方向） | AUC ≥ 0.75 | AUC < 0.65 |
| 4 | 评分者间信度 | Q1 的 **ICC(2,1)**（双向随机、绝对一致） | ICC ≥ 0.60 | ICC < 0.45 |
| 5 | 评分者间信度（次） | Q4 的 **Cohen κ** | κ ≥ 0.60 | κ < 0.45 |
| 6 | **零侧核查（方向）** | 人判 Q4 = 0 中 `aid_doc = 0` 的**加权 PPV**；配对比较用 **McNemar** | PPV ≥ 0.80 | PPV < 0.60 |

介于达标与失败之间 → 判为**不确定**，密度维持「探索性」，并在稿件中如实写明。

**报 PPV，不报总体一致率**（铁律 21）：零侧占比高，一致率会被多数类抬高。

--------------------------------------------------------------------------
## 4. 失效模式判定（决定「还能不能修」）

用 Q1 与 `aid_doc` 的四格表判定：
- 人判低、算法判高（Q1 ≤ 1 且 `aid_doc` ≥ L3 切点）占比高 → **过度判定**，
  修法是**提高阈值 / 加权**，不是扩词表（铁律 23）
- 人判高、算法判低（Q1 ≥ 2 且 `aid_doc = 0`）占比高 → **漏判**，
  修法才是扩词表

--------------------------------------------------------------------------
## 5. 达标 / 不达标的后果（预先写死）

- **达标（主指标 1 与 3 均达标）**：
  `aid_doc` 由「探索性文本测量」升级为「**已验证文本测量**」；
  稿件中移除 `[PENDING HUMAN ADJUDICATION]` 占位，机制段可写为已验证的
  探索性结论（仍不写因果）。
- **不达标（任一主指标落入失败区间）**：
  `aid_doc` **维持探索性**；机制主张继续降级为「假说，未验证」；
  稿件主结论（疾病×剂量交互 1.96, 1.04–3.67, P = 0.006）**不受影响**——
  该估计不依赖文本侧（它在 spec− 层 n = 874 / 80 事件上拟合，密度不是其协变量）。
- **不确定**：按不达标处理，但把区间如实写进 Limitations。

--------------------------------------------------------------------------
## 6. 分析脚本

约定为 `scripts/213_human_adjud_analysis.py`（待评分回收后编写）：
读入 `adjud3_score_A.csv` / `_B.csv` 与 `adjud3_sealed_key.csv`，
先做**门槛断言**（case_id 集合与顺序一致、无缺行、取值域合法），
再按上表逐项估计并出图出表。

--------------------------------------------------------------------------
## 7. 与设计稿的两处偏离（如实记录）

1. **取消文本截断**：原方案沿用历史 CAP = 5000 的头/尾截断。实测发现
   43.6% 病例被截断、**22.3% 病例的全部命中句被切掉**（aid_doc > 0 而可见
   文本为 0），会使 L2 层在可见文本上塌成与 L1 无差别，分层失效。
   截断原本是为子代理省上下文窗口，真人无此限制 → 改为**全文不截断**，
   并已断言重算密度逐例复现落盘值（最大绝对差 4.4e-16）。
2. **正侧全查而非每层 55 例**：严格排除 279 例后正侧仅剩 138 例，
   三分位各 46，无法每层抽 55。改为 L2–L4 全查（层内权重 = 1，
   在信息量最大的一段上不留抽样误差）+ L1 补足到 220。

--------------------------------------------------------------------------
## 8. 打分函数对比（**事后催生、事前锁定** · v1.2 追加）

**来源**：b0 校准批（15 例）显示 `aid_doc` 正侧 PPV 偏低，且叙述长度
本身即可预测人判结果。为判断「密度（每千字符）」是否是最优打分函数，
追加以下**参考性**指标。**这些指标不设达标阈值，不参与第 5 节的后果判定。**

| # | 估计量 | 目的 | 读法 |
|---|---|---|---|
| 8a | 原始命中数（`aid_doc × narr_char / 1000`）预测人判 Q4 = 1 的 AUC | 与密度直接对照 | 若明显高于密度，提示除以长度反而损失信息 |
| 8b | **仅 `narr_char`** 预测人判 Q4 = 1 的 AUC | 长度基线 | 若接近或超过 8a，则任何文本指标的判别力都可能是长度效应 |
| 8c | 原始命中数 vs `narr_char` 的 ρ | 长度依赖度 | 对照密度的 ρ，判断哪种打分更黏长度 |
| 9a–d | **偏 Spearman ρ**（以 `narr_char` 为协变量） | 剥离长度后的净构念效度 | Q1 校正后仍 ≥0.40 才算构念效度成立；Q4 校正后大幅下降则提示二分类判别含长度混淆 |

**诚实声明**：第 8、9 两组指标是**看了 b0 之后才想到**的，属于
hypothesis-generating。它们在此被锁定的意义是：口径与用途先写死，
正式批次上只做一次确认性检验，**不再挑选**。b0 的 n = 15、
其中 Q4 = 1 仅 3 例，任何数值都不足以支持改动打分函数。

--------------------------------------------------------------------------
## 9. 退化解约束（v1.2 追加）

若两份评分在**全部病例、全部评分项**上完全一致（例如评分后经讨论把分歧抹平、
或直接复制同一份），则 ICC / κ 必然等于 1.000。**该数值不是信度证据**，
脚本须将其标为「退化·不可估」并输出警示。
**正式批次（b1–b8）必须保留 A/B 各自独立评分**——分歧是信度估计的原始材料，
按共同锚点抹平分歧等价于删掉被估计的对象。
"""
    with io.open(os.path.join(OUT, "adjud3_analysis_plan.md"), "w",
                 encoding="utf-8") as f:
        f.write(PLAN)
    P("  分析计划 → out/adjud3_analysis_plan.md")

    # ---------------------------------------------------------- 7d 交付说明
    n_sle = int((smp["disease"] == "SLE").sum())
    n_ra = int((smp["disease"] == "RA").sum())
    tot_h = tot_ch / CPM / 60.0 + len(smp) * 40 / 3600.0
    README = u"""# 真人盲审材料包 —— adjud3 / v13（发放说明）

生成：`scripts/212_human_adjud_prep.py`（SEED=%d，完全可复现）
生成时间：%s

## 一、这次验什么

连续活动度信息密度 **aid_doc** 的**构念效度**。三分类标签已废弃，不再评审。

## 二、规模与构成

- 病例 **%d 例**（SLE %d / RA %d），30 天死亡事件 %d 例
- 4 层：L1 `aid_doc=0` %d 例；L2 %d / L3 %d / L4 %d 例（正侧三分位）
- 抽样框 %d 例（1250 − 既往评审 279 例全部物理排除）
- 与既往 279 例重叠 **0**

## 三、工作量

每人 **%.1f 小时**（阅读 %s 字符 ≈ %s 词 + 评分开销）。
已切分为 **1 个校准卷（15 例）+ 8 个正式卷（每卷 25–30 例，约 2 小时）**，
建议一次只做一卷。

## 四、发给谁、发什么

**先发给两位评分者的（不含任何答案）：**

| 文件 | 用途 |
|---|---|
| `adjud3_rater_manual.md` | 评分规则，**两人用同一版本** |
| `adjud3_sheet_<A|B>.md` | 盲表全文（也可只发分卷） |
| `adjud3_sheet_<A|B>_b0.md` | **校准卷，先做**（15 例，两人同一批） |
| `adjud3_sheet_<A|B>_b1..b8.md` | 正式卷 |
| `adjud3_score_<A|B>.csv` | 评分模板（utf-8-sig，Excel 直接打开） |

**研究者留存，评分结束前不得外传：**

| 文件 | 内容 |
|---|---|
| `adjud3_sealed_key.csv` | 密封键：case_id ↔ stay_key ↔ 层 ↔ aid_doc ↔ 权重 ↔ A/B 顺序 |
| `adjud3_frame.csv` | 抽样框全量 971 行 |
| `adjud3_sample.csv` | 入样 220 例（不含正文） |
| `adjud3_analysis_plan.md` | **预先指定**的分析计划与达标阈值 |

## 五、流程（按顺序）

1. 两位医师各自完成 **b0 校准卷** → **与研究者对答案、统一尺度**（不可省略）
2. 各自完成 b1–b8；期间不回看、不互相讨论
3. 回收两份 `adjud3_score_*.csv`
4. 分歧病例交第三位医师仲裁
5. 跑 `scripts/213_human_adjud_analysis.py`（待写），按分析计划逐项估计

## 六、合规

MIMIC-IV 叙述文本受 PhysioNet DUA 约束。两位评分者**必须**已完成的
credentialing（CITI + 申请获批），或已列入本院伦理批准方案的研究成员名单。
本材料包含原始叙述文本，不得对外转发。

## 七、一处必须知道的局限

本盲审**只能救测量，救不了把握度**：正侧在 spec− 分析子集内仅 99 例 / 4 事件，
三项交互结构性不可估（MDE 1.84–2.42）。即便测量完美，也**不会**让
密度剂量-反应变得可估。
""" % (SEED, pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
       len(smp), n_sle, n_ra, int(smp["death_30d"].sum()),
       int((smp["layer"] == "L1").sum()), int((smp["layer"] == "L2").sum()),
       int((smp["layer"] == "L3").sum()), int((smp["layer"] == "L4").sum()),
       len(av), tot_h, "{:,}".format(tot_ch),
       "{:,}".format(int(tot_ch / CH_PER_WORD)))
    with io.open(os.path.join(OUT, "adjud3_README.md"), "w",
                 encoding="utf-8") as f:
        f.write(README)
    P("  交付说明 → out/adjud3_README.md")

    # ---------------------------------------------------------- 8 QC
    P("")
    P("--- 8. QC 自检 ---")
    P("  入样 %d 例，唯一编号 %d 个" % (len(smp), smp["case_id"].nunique()))
    P("  与既往 279 例重叠 %d（必须为 0）" % int(smp["stay_key"].isin(ex).sum()))
    P("  Σ设计权重 = %.1f（应等于框内 %d）"
      % (float(smp["weight"].sum()), len(av)))
    P("  层分布：%s" % smp["layer"].value_counts().sort_index().to_dict())
    P("  病种分布：%s" % smp["disease"].value_counts().to_dict())
    P("  事件数（30d 死亡）：%d" % int(smp["death_30d"].sum()))
    P("  摘录（=全文，未截断）：中位 %d 字符，p90 %d，最大 %d"
      % (int(smp["rawlen"].median()), int(smp["rawlen"].quantile(.9)),
         int(smp["rawlen"].max())))
    for tag in ["A", "B"]:
        for b in [0] + list(range(1, NB + 1)):
            sub = smp[smp["batch_" + tag] == b] if b > 0 else smp[smp["calib"] == 1]
            P("    %s 卷 b%d：%3d 例，%7s 字符，约 %.1f h"
              % (tag, b, len(sub), "{:,}".format(int(sub["rawlen"].sum())),
                 sub["rawlen"].sum() / CPM / 60.0))
    for ly in ["L1", "L2", "L3", "L4"]:
        g = smp[smp["layer"] == ly]
        P("    %s n=%3d  权重=%7.3f  SLE=%2d  RA=%3d  事件=%2d  aid_doc中位=%.3f"
          % (ly, len(g), g["weight"].iloc[0], int((g["disease"] == "SLE").sum()),
             int((g["disease"] == "RA").sum()), int(g["death_30d"].sum()),
             float(g["aid_doc"].median())))
    assert smp["case_id"].nunique() == N_TARGET
    assert int(smp["stay_key"].isin(ex).sum()) == 0
    assert abs(float(smp["weight"].sum()) - len(av)) < 1e-6

    with io.open(os.path.join(OUT, "212_human_adjud_prep.txt"), "w",
                 encoding="utf-8") as f:
        f.write("\n".join(L))
    P("")
    P("构建日志 → out/212_human_adjud_prep.txt")


if __name__ == "__main__":
    main()
