# -*- coding: utf-8 -*-
"""
207_manual_v2_sheet.py —— 为「v2 手册重审」构建盲法重审表。

用途
----
v2 手册对 v1 的 6 处修订**全部只收窄阳性标签**（见 out/adjud_codebook_v2.md）：
  · 任务 A：INF_DEFER 删除「同段共现」子句            → 只可能 INF_DEFER → UNDERDOC
  · 任务 B：新增领域限定 / 断言要求 / 时间性 / 用药指征 → 只可能 POSITIVE → NEGATED
因此唯一可能发生标签迁移的，是**已判为阳性的病例**。本脚本把它们全部挑出，
再掺入一组**对照**（原判非阳性）以避免「整表皆阳性」的锚定，
拼成两张与 v7b/v9 完全同构的盲表（只给不透明编号 + 病历文本）。

输出
----
  out/adjud_v2_audit_A.md / _B.md      盲表（交评分者）
  out/adjud_v2_audit_A.csv / _B.csv    case_id,excerpt（机读）
  out/adjud_v2_audit_key.csv           密封键：case_id,task,batch,prior_ref_std,grp
  out/207_manual_v2_sheet.txt          构建日志（QC）
"""
import os
import glob
import io
import re
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
SEED = 20260916           # 固定，保证可复现
N_CTRL_A = 4              # 任务 A 对照（原判 UNDERDOC）
N_CTRL_B = 6              # 任务 B 对照（原判 PURELY-NEGATED）

L = []


def P(s=""):
    L.append(s)
    print(s)


# ------------------------------------------------------------------ 读金标准
def load_gold():
    """返回 (case_id, task, batch, prior_ref_std) 的表。"""
    recs = []
    for batch, f in [("v7b", "adjud_merged.csv"), ("v9", "adjud2_merged.csv")]:
        M = pd.read_csv(os.path.join(OUT, f))
        M.columns = [c.replace("\ufeff", "") for c in M.columns]
        M["case_id"] = M.case_id.astype(str).str.strip()
        for r in M.itertuples():
            recs.append(dict(case_id=r.case_id, task=r.task, batch=batch,
                             prior_ref_std=r.ref_std))
    return pd.DataFrame(recs)


def load_excerpts():
    """把四张盲表的 excerpt 拼成 {case_id: excerpt}。"""
    d = {}
    for f in ["adjud_blind_A.csv", "adjud_blind_B.csv",
              "adjud2_blind_A.csv", "adjud2_blind_B.csv"]:
        x = pd.read_csv(os.path.join(OUT, f))
        x.columns = [c.replace("\ufeff", "") for c in x.columns]
        x["case_id"] = x.case_id.astype(str).str.strip()
        for r in x.itertuples():
            d[r.case_id] = r.excerpt
    return d


def write_sheet(path, title, cases, ex, task):
    """写盲表 md。cases = [case_id...]。"""
    with io.open(path, "w", encoding="utf-8") as f:
        f.write("# %s\n\n" % title)
        f.write("每例只给出**不透明编号**与病历文本。表中**不提供**病种分组、激素\n"
                "剂量、临床结局、以及任何自动词表判定结果 —— 请勿从外部推断它们，\n"
                "只依据文本**实际写明**的内容，并严格按 `out/adjud_codebook_v2.md` 判定。\n"
                "病历正文中的疾病名、药品名属于原文，按原文理解即可。\n\n")
        f.write("判定结果写入 CSV（每卷一个），三列：case_id,label,evidence_quote\n"
                "evidence_quote 必须是该例文本中**逐字复制**的片段（≤120 字符）。\n"
                "**每一例都必须有且只有一行**。\n")
        for i, cid in enumerate(cases, 1):
            f.write("\n\n=== %s ===\n\n" % cid)
            f.write(ex[cid])
        f.write("\n")
    P("  wrote %s  (%d cases, %d chars)"
      % (os.path.basename(path), len(cases), os.path.getsize(path)))


def main():
    G = load_gold()
    E = load_excerpts()

    P("=" * 96)
    P("207 —— v2 手册重审：盲表构建")
    P("=" * 96)
    P("金标准合计 n=%d" % len(G))

    POS = {("A", "INF_DEFER"), ("B", "ACTIVITY-POSITIVE")}
    G["grp"] = np.where(
        [((r.task, r.prior_ref_std) in POS) for r in G.itertuples()],
        "positive", "control")

    pos = G[G.grp == "positive"].copy()
    P("原判阳性（= 唯一可能迁移者）n=%d" % len(pos))
    P(pos.groupby(["task", "batch", "prior_ref_std"]).size().to_string())

    # 对照：按 task 分层随机抽
    rng = np.random.default_rng(SEED)
    ctrl_rows = []
    for task, n in [("A", N_CTRL_A), ("B", N_CTRL_B)]:
        cand = G[(G.grp == "control") & (G.task == task)]
        # 按 batch 等比例抽，保证两批都出现
        take = []
        for b, sub in cand.groupby("batch"):
            k = max(1, int(round(n * len(sub) / len(cand))))
            take.append(sub.sample(min(k, len(sub)), random_state=int(rng.integers(1e9))))
        t = pd.concat(take, ignore_index=True).drop_duplicates("case_id")
        if len(t) > n:
            t = t.sample(n, random_state=int(rng.integers(1e9)))
        ctrl_rows.append(t)
    ctrl = pd.concat(ctrl_rows, ignore_index=True) if ctrl_rows else pd.DataFrame()
    P("对照 n=%d" % len(ctrl))
    if len(ctrl):
        P(ctrl.groupby(["task", "batch", "prior_ref_std"]).size().to_string())

    AUD = pd.concat([pos, ctrl], ignore_index=True)
    # 检查每个 case_id 都有 excerpt
    missing = [c for c in AUD.case_id if c not in E]
    assert not missing, "缺 excerpt: %s" % missing
    AUD.to_csv(os.path.join(OUT, "adjud_v2_audit_key.csv"), index=False)
    P("密封键已落盘 out/adjud_v2_audit_key.csv")

    # 分任务出两张表；顺序随机化（固定种子）
    for task, title, outb in [
            ("A", "任务 A 重审：三分类（INACTIVE / INF_DEFER / UNDERDOC）· 按 v2 手册",
             "adjud_v2_audit_A"),
            ("B", "任务 B 重审：活动阳性 vs 纯否定 · 按 v2 手册",
             "adjud_v2_audit_B")]:
        sub = AUD[AUD.task == task].sample(frac=1.0, random_state=SEED)
        ids = sub.case_id.tolist()
        P("任务 %s：%d 例（阳性 %d + 对照 %d）"
          % (task, len(ids), int((sub.grp == "positive").sum()),
             int((sub.grp == "control").sum())))
        write_sheet(os.path.join(OUT, outb + ".md"), title, ids, E, task)
        pd.DataFrame([dict(case_id=c, excerpt=E[c]) for c in ids]).to_csv(
            os.path.join(OUT, outb + ".csv"), index=False)
        pd.DataFrame([dict(case_id=c, label="", evidence_quote="") for c in ids]).to_csv(
            os.path.join(OUT, outb + "_TEMPLATE.csv"), index=False)

    P("")
    P("重审总量：阳性 %d + 对照 %d = %d 例"
      % (int((AUD.grp == "positive").sum()), int((AUD.grp == "control").sum()),
         len(AUD)))
    with io.open(os.path.join(OUT, "207_manual_v2_sheet.txt"), "w",
                 encoding="utf-8") as f:
        f.write("\n".join(L))
    P("saved: out/207_manual_v2_sheet.txt")


if __name__ == "__main__":
    main()
