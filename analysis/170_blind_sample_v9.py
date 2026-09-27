# -*- coding: utf-8 -*-
"""
170_blind_sample_v9.py
======================
v9 —— 为「v8 判定器的**样本外**复现」生成全新的盲法抽样。

为什么要新样本
  v8 的三条规则（实体词限定 / 否定检测 / 领域判断）是**看着** v7b 那 98 例的
  失败机制设计的 —— 因此 v8 在 v7b 金标准上的 PPV（1.000 / 0.833 / 1.000）是
  **样本内**估计，偏乐观。本脚本抽一批**与 v7b 完全不重叠**的病例（148 例全部排除），
  在同一份编码手册、同一套隔离双标流程下重测。

与 165 的差异
  - 抽样分层改为按 **v8 引擎标签**（要检验的是 v8，样本必须含足够 v8 阳性）；
  - 排除集 = adjud_sealed_key.csv 的全部 148 例（任务 A + 任务 B 都不复用）；
  - 种子改为 20260917；输出文件名统一为 adjud2_*，**不覆盖 v7b 的任何文件**；
  - 新增「v8 引擎标签重算自检」：直接调用 168 的规则引擎，确认落盘标签可复现。

输出
  out/adjud2_blind_A_p*.md / _B_p*.md   盲表（分卷）
  out/adjud2_blind_A.csv / _B.csv       同内容 CSV
  out/adjud2_sealed_key.csv             封印键（评分者不得见）
  out/adjud2_codebook.md                编码手册（与 v7b 逐字相同）
  out/170_blind_qc.txt                  抽样与截断 QC
"""
import importlib.util
import os
import shutil
import sys

import numpy as np
import pandas as pd
import psycopg2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
SCR = os.path.join(ROOT, "scripts")
os.makedirs(OUT, exist_ok=True)


def load_mod(tag, path):
    spec = importlib.util.spec_from_file_location(tag, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[tag] = mod
    spec.loader.exec_module(mod)
    return mod


M165 = load_mod("m165", os.path.join(SCR, "165_make_blind_sheets.py"))
M168 = load_mod("m168", os.path.join(SCR, "168_activity_rule_engine.py"))

SEED = 20260917
CAP, HEAD, TAIL, CHUNK = M165.CAP, M165.HEAD, M165.TAIL, M165.CHUNK

N_A_UD = 40          # 任务 A：未记录层抽样数
N_B_POS = 35         # 任务 B：v8 判为活动阳性
N_B_NEG = 25         # 任务 B：v8 判为非阳性

L = []


def P(s=""):
    L.append(str(s))
    print(s)


def label_v8(hc, dd, moa, full_text=""):
    """复用 168 的规则函数，按 168 完全一致的优先级给出 v8 三分类。

    与 168 主流程保持一致：narr = 归一化(住院经过 + 出院诊断)；段落切分失败
    （两者合计 <200 字符）时退回归一化全文。INF_DEFER 的判据用的是 narr + 入院前用药。
    """
    if len(hc) + len(dd) < 200:
        narr = M168.norm(full_text)
    else:
        narr = M168.norm(hc + " " + dd)
    ia = M168.eval_inactivity(narr)
    df = M168.eval_defer(narr + " " + M168.norm(moa))
    if ia["inact_n"] > 0:
        return "INACTIVE"
    if df["defer_n"] > 0:
        return "INF_DEFER"
    return "UNDERDOC"


def main():
    gi = pd.read_csv(os.path.join(DATA, "gc_inactivity.csv"))
    gi["stay_key"] = gi.stay_key.astype(int)
    coh = pd.read_csv(os.path.join(DATA, "cohort_rheum_icu.csv"))
    coh["stay_key"] = coh.stay_key.astype(int)
    den = pd.read_csv(os.path.join(DATA, "gc_activity_density.csv"))
    den["stay_key"] = den.stay_key.astype(int)
    gind = pd.read_csv(os.path.join(DATA, "gc_indication.csv"))
    gind["stay_key"] = gind.stay_key.astype(int)

    m = coh[coh.primary_grp.isin(["SLE", "RA"])].merge(gi, on="stay_key",
                                                       how="inner")
    m = m.merge(den[["stay_key", "v8_label", "v8_act_pos", "aid_doc",
                     "aid_broad"]], on="stay_key", how="left")
    m = m.merge(gind[["stay_key", "home_gc", "n_gc_mentions"]], on="stay_key",
                how="left")

    # ---------------- 排除集：v7b 的全部 148 例
    key = pd.read_csv(os.path.join(OUT, "adjud_sealed_key.csv"))
    key["stay_key"] = key.stay_key.astype(int)
    used = set(key.stay_key.tolist())
    P("v7b 已评病例（全部排除）= %d（去重后 %d）" % (len(key), len(used)))

    rs = np.random.RandomState(SEED)

    # ---------------- 任务 A 抽样框（与 165 同口径：spec- 且无被否定命中）
    fa_all = m[(m.spec_true == 0) & (m.spec_raw_hits == 0)].copy()
    fa = fa_all[~fa_all.stay_key.isin(used)].copy()
    P("")
    P("任务 A 抽样框 = %d（fresh %d，已排除 v7b 用过的 %d）"
      % (len(fa_all), len(fa), len(fa_all) - len(fa)))
    P("  全框 v8 构成：" + "  ".join(
        "%s=%d" % (k, v) for k, v in fa_all.v8_label.value_counts().items()))
    P("  fresh v8 构成：" + "  ".join(
        "%s=%d" % (k, v) for k, v in fa.v8_label.value_counts().items()))

    picks = []
    for k, want in [("INACTIVE", 999), ("INF_DEFER", 999),
                    ("UNDERDOC", N_A_UD)]:
        pool = fa[fa.v8_label == k]
        take = min(want, len(pool))
        if take == 0:
            P("  A | %-10s 池为空，跳过" % k)
            continue
        pick = pool.sample(take, random_state=rs).copy()
        pick["task"] = "A"
        pick["stratum"] = k
        picks.append(pick)
        P("  A | v8 %-10s 抽 %d / %d" % (k, take, len(pool)))
    A = pd.concat(picks, ignore_index=True)

    # ---------------- 任务 B 抽样框
    fb_all = m[m.spec_raw_hits > 0].copy()
    fb = fb_all[~fb_all.stay_key.isin(used)].copy()
    P("")
    P("任务 B 抽样框 = %d（fresh %d）" % (len(fb_all), len(fb)))
    P("  全框：v8 活动阳性 %d / 非阳性 %d"
      % (int((fb_all.v8_act_pos > 0).sum()),
         int((fb_all.v8_act_pos == 0).sum())))
    picksB = []
    for pos, want, nm in [(1, N_B_POS, "v8 activity-positive"),
                          (0, N_B_NEG, "v8 not positive")]:
        pool = fb[(fb.v8_act_pos > 0) == (pos == 1)]
        take = min(want, len(pool))
        pick = pool.sample(take, random_state=rs).copy()
        pick["task"] = "B"
        pick["stratum"] = nm
        picksB.append(pick)
        P("  B | %-24s 抽 %d / %d" % (nm, take, len(pool)))
    B = pd.concat(picksB, ignore_index=True)

    sel = pd.concat([A, B], ignore_index=True)
    keys = sorted(sel.stay_key.unique().tolist())
    P("")
    P("合计抽取 %d 例（去重 %d）" % (len(sel), len(keys)))
    ov = used & set(keys)
    P("与 v7b 抽样重叠 = %d 例 %s" % (len(ov), "" if not ov else sorted(ov)))
    assert not ov, "抽样必须与 v7b 完全不重叠"

    # ---------------- 取原文
    cc = psycopg2.connect(host=os.environ.get("MIMIC_DB_HOST", "localhost"),
                          port=int(os.environ.get("MIMIC_DB_PORT", "5432")),
                          user=os.environ.get("MIMIC_DB_USER", "postgres"),
                          password=os.environ.get("MIMIC_DB_PASSWORD", ""),
                          dbname="mimiciv")
    parts = []
    for i in range(0, len(keys), 400):
        ch = keys[i:i + 400]
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
        hc, dd, moa = M165.build_blob(t)
        blob = M165.join_blob(hc, dd, moa)
        ex, trunc, rawlen = M165.truncate(blob)
        hcv, ddv, moav = M165.split_blob(ex)
        # v8 引擎标签重算（对**截断后**文本，确认可见证据下的判定）
        lab_v8_vis = label_v8(hcv, ddv, moav, t)
        recs.append(dict(
            task=r.task, stratum=r.stratum, stay_key=int(r.stay_key),
            disease=r.primary_grp,
            v7_label=r.inact3, v8_label=r.v8_label,
            v8_label_sealed=r.v8_label,
            v8_label_visible=lab_v8_vis,
            v8_act_pos=int(r.v8_act_pos),
            aid_doc=float(r.aid_doc) if np.isfinite(r.aid_doc) else np.nan,
            spec_true=int(r.spec_true), spec_raw=int(r.spec_raw_hits),
            spec_pos=int(r.spec_pos_hits), spec_neg=int(r.spec_neg_hits),
            spec_falsepos=int(r.spec_falsepos),
            death_30d=int(r.death_30d), home_gc=M165.si(r.home_gc),
            pdx_class=r.pdx_class,
            excerpt_len=len(ex), excerpt_rawlen=rawlen, truncated=trunc,
            excerpt=ex))
    df = pd.DataFrame(recs)

    # ---------------- 自检
    P("")
    P("--- 自检 ---")
    P("  抽取例数 %d，其中任务 A %d / 任务 B %d"
      % (len(df), int((df.task == "A").sum()), int((df.task == "B").sum())))
    P("  截断例数 %d（占 %.1f%%）"
      % (int(df.truncated.sum()), 100 * df.truncated.mean()))
    if df.v8_label_visible.notna().any():
        bad = df[df.v8_label_visible.notna() &
                 (df.v8_label_visible != df.v8_label)]
        P("  v8 标签重算（截断文本）不一致 %d / %d" % (len(bad), len(df)))
    P("  病种：SLE %d / RA %d" % (int((df.disease == "SLE").sum()),
                                  int((df.disease == "RA").sum())))

    # ---------------- 不透明 case_id
    df = df.sample(frac=1.0, random_state=rs).reset_index(drop=True)
    cid = []
    ia = ib = 0
    for t in df.task:
        if t == "A":
            ia += 1
            cid.append("N-A-%03d" % ia)
        else:
            ib += 1
            cid.append("N-B-%03d" % ib)
    df["case_id"] = cid

    # ---------------- 盲表
    def write_blind(sub, task, title):
        sub = sub.reset_index(drop=True)
        nparts = int(np.ceil(len(sub) / CHUNK))
        for pi in range(nparts):
            part = sub.iloc[pi * CHUNK:(pi + 1) * CHUNK]
            fn = "adjud2_blind_%s_p%d.md" % (task, pi + 1)
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
            P("  写出 %-28s %3d 例" % (fn, len(part)))
        sub[["case_id", "excerpt"]].to_csv(
            os.path.join(OUT, "adjud2_blind_%s.csv" % task), index=False,
            encoding="utf-8")
        return nparts

    P("")
    P("--- 盲表 ---")
    nA = write_blind(df[df.task == "A"], "A", "任务 A：疾病活动度记录三分类（盲评）")
    nB = write_blind(df[df.task == "B"], "B", "任务 B：活动阳性 vs 纯否定（盲评）")

    # ---------------- 编码手册（沿用 v7b，逐字相同）
    src_cb = os.path.join(OUT, "adjud_codebook.md")
    dst_cb = os.path.join(OUT, "adjud2_codebook.md")
    if os.path.exists(src_cb):
        shutil.copyfile(src_cb, dst_cb)
        P("  编码手册：复制 v7b 版本 -> adjud2_codebook.md（逐字相同）")
    else:
        raise AssertionError("缺少 v7b 编码手册 %s" % src_cb)

    # ---------------- 封印键
    keycols = ["case_id", "task", "stratum", "stay_key", "disease",
               "v7_label", "v8_label", "v8_act_pos", "aid_doc", "spec_true",
               "spec_raw", "spec_pos", "spec_neg", "spec_falsepos",
               "death_30d", "home_gc", "pdx_class", "excerpt_len",
               "excerpt_rawlen", "truncated"]
    df[keycols].to_csv(os.path.join(OUT, "adjud2_sealed_key.csv"),
                       index=False, encoding="utf-8")

    # ---------------- 抽样代表性
    P("")
    P("--- 抽样代表性（v9 抽取 vs 全框）---")
    P("  %-12s %8s %10s %10s" % ("层", "抽取 n", "抽取 SLE%", "全框 SLE%"))
    for k in ["INACTIVE", "INF_DEFER", "UNDERDOC"]:
        d = df[(df.task == "A") & (df.stratum == k)]
        allk = fa_all[fa_all.v8_label == k]
        if not len(d):
            continue
        P("  %-12s %8d %9.1f%% %9.1f%%" % (
            k, len(d), 100 * (d.disease == "SLE").mean(),
            100 * (allk.primary_grp == "SLE").mean()))

    P("")
    P("saved: adjud2_blind_{A_p%d..,B_p%d..}.md, adjud2_blind_{A,B}.csv," % (nA, nB))
    P("       adjud2_sealed_key.csv, adjud2_codebook.md, 170_blind_qc.txt")

    with open(os.path.join(OUT, "170_blind_qc.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
