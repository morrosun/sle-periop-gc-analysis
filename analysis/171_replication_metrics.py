# -*- coding: utf-8 -*-
"""
171_replication_metrics.py
==========================
v9 —— v8 判定器（与 v7 对照）在**全新盲法样本**上的样本外复现。

输入
  out/adjud2_sealed_key.csv          封印键（v7/v8 算法标签 + 分层 + 分层规模）
  out/adjud2_R{1,2}_{A,B}_p*.csv     两名隔离子代理的盲法判定
  out/adjud2_third.csv               （可选）分歧病例的第三仲裁
  out/_v8.json / _v7b_results.json   样本内（v7b）对照值

输出
  out/171_replication.txt            全文报告
  out/table_v9_*.csv                 各表
  out/171_disputes.md                分歧清单（供第三仲裁）
  out/adjud2_merged.csv              逐例合并表（含参考标准）

与 v7b 的唯一差别是**样本不同**：v9 的 131 例与 v7b 的 148 例零重叠。
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
DATA = os.path.join(ROOT, "data")
SEED = 20260918

CH = ["INACTIVE", "INF_DEFER", "UNDERDOC"]
LBL_A = CH
LBL_B = ["ACTIVITY-POSITIVE", "PURELY-NEGATED"]
FRAME_A = {"INACTIVE": 32, "INF_DEFER": 9, "UNDERDOC": 815}   # = 856
FRAME_B = {"ACTIVITY-POSITIVE": 126, "PURELY-NEGATED": 268}   # = 394

L = []


def P(s=""):
    print(s)
    sys.stdout.flush()
    L.append(str(s))


# ------------------------------------------------------------------ 统计工具
def cohen_kappa(a, b, labels):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0:
        return np.nan, np.nan, np.nan
    po = float((a == b).mean())
    pe = sum((a == c).mean() * (b == c).mean() for c in labels)
    return po, pe, ((po - pe) / (1 - pe) if pe < 1 else np.nan)


def w_kappa(a, b, labels, w):
    a, b = np.asarray(a), np.asarray(b)
    w = np.asarray(w, dtype=float)
    sw = w.sum()
    po = float((w * (a == b)).sum() / sw)
    pe = sum((w * (a == c)).sum() / sw * (w * (b == c)).sum() / sw
             for c in labels)
    return po, pe, ((po - pe) / (1 - pe) if pe < 1 else np.nan)


def ck2(df, test_col, truth_col, pos, wcol=None):
    a = df[test_col].astype(str).values
    b = df[truth_col].astype(str).values
    w = df[wcol].astype(float).values if wcol else np.ones(len(df))
    TP = w[(a == pos) & (b == pos)].sum()
    FP = w[(a == pos) & (b != pos)].sum()
    FN = w[(a != pos) & (b == pos)].sum()
    TN = w[(a != pos) & (b != pos)].sum()
    return dict(n_test_pos=int((a == pos).sum()), TP=TP, FP=FP, FN=FN, TN=TN,
                sens=TP / (TP + FN) if TP + FN else np.nan,
                spec=TN / (TN + FP) if TN + FP else np.nan,
                ppv=TP / (TP + FP) if TP + FP else np.nan,
                npv=TN / (TN + FN) if TN + FN else np.nan,
                acc=(TP + TN) / w.sum())


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def boot_strat(df, fn, wcol, strat, nrep=2000):
    rng = np.random.RandomState(SEED)
    groups = [g for _, g in df.groupby(strat)]
    vals = []
    for _ in range(nrep):
        s = pd.concat([g.sample(len(g), replace=True, random_state=rng)
                       for g in groups], ignore_index=True)
        vals.append(fn(s))
    v = np.array([x for x in vals if np.isfinite(x)])
    return (np.nanpercentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan))


def load_ratings(tag):
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, "adjud2_R?_*.csv"))):
        m = re.match(r"adjud2_R(\d)_([AB])_p(\d)\.csv", os.path.basename(f))
        if not m:
            continue
        raw = open(f, encoding="utf-8-sig").read().splitlines()
        has_header = raw[0].strip().lower().startswith("case_id")
        d = pd.read_csv(f, encoding="utf-8-sig",
                        header=0 if has_header else None,
                        names=None if has_header else
                        ["case_id", "label", "evidence_quote"])
        for c in ["case_id", "label", "evidence_quote"]:
            if c not in d.columns:
                d[c] = ""
        d = d[["case_id", "label", "evidence_quote"]].copy()
        d["rater"] = "R" + m.group(1)
        d["task"] = m.group(2)
        d["part"] = int(m.group(3))
        d["case_id"] = d.case_id.astype(str).str.strip()
        d["label"] = d.label.astype(str).str.strip()
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------ 主流程
def main():
    key = pd.read_csv(os.path.join(OUT, "adjud2_sealed_key.csv"))
    key["case_id"] = key.case_id.astype(str).str.strip()
    key["stay_key"] = key.stay_key.astype(int)
    rt = load_ratings(key)

    P("=" * 100)
    P("v9 —— v8 判定器的样本外复现（全新盲法样本，与 v7b 零重叠）")
    P("=" * 100)
    P("")
    P("--- 0. 数据完整性自检 ---")
    P("  封印键 %d 例（任务 A %d / 任务 B %d）"
      % (len(key), int((key.task == "A").sum()), int((key.task == "B").sum())))
    ok = True
    for r in ["R1", "R2"]:
        for t in ["A", "B"]:
            nk = int(((key.task == t)).sum())
            d = rt[(rt.rater == r) & (rt.task == t)]
            dup = int(d.case_id.duplicated().sum())
            miss = set(key[key.task == t].case_id) - set(d.case_id)
            extra = set(d.case_id) - set(key[key.task == t].case_id)
            P("  %s 任务 %s：%d 行（应 %d） 重复 %d 缺失 %d 多余 %d"
              % (r, t, len(d), nk, dup, len(miss), len(extra)))
            ok &= (len(d) == nk and dup == 0 and not miss and not extra)
    for r in ["R1", "R2"]:
        for t, legal in [("A", LBL_A), ("B", LBL_B)]:
            d = rt[(rt.rater == r) & (rt.task == t)]
            bad = sorted(set(d.label) - set(legal))
            if bad:
                P("  !! %s 任务 %s 出现非法标签：%s" % (r, t, bad))
                ok = False
    P("  行数/标签合法性：%s" % ("全部通过" if ok else "**有问题**"))
    assert ok, "评分数据未通过自检"

    # ---------------- 评分者间一致性
    P("")
    P("--- 1. 评分者间一致性（隔离子代理 R1 vs R2）---")
    wide = rt.pivot_table(index=["task", "case_id"], columns="rater",
                          values="label", aggfunc="first").reset_index()
    wide = wide.merge(key[["case_id", "task", "stratum"]], on=["case_id", "task"],
                      how="left")
    WA = wide[wide.task == "A"]
    WB = wide[wide.task == "B"]
    poA, peA, kA = cohen_kappa(WA.R1, WA.R2, LBL_A)
    poB, peB, kB = cohen_kappa(WB.R1, WB.R2, LBL_B)
    P("  任务 A：n=%d  一致率 %.3f  期望 %.3f  κ=%.3f  PABAK=%+.3f"
      % (len(WA), poA, peA, kA, 2 * poA - 1))
    P("  任务 B：n=%d  一致率 %.3f  期望 %.3f  κ=%.3f  PABAK=%+.3f"
      % (len(WB), poB, peB, kB, 2 * poB - 1))
    dis = pd.concat([WA[WA.R1 != WA.R2], WB[WB.R1 != WB.R2]], ignore_index=True)
    P("  分歧病例 = %d（A %d / B %d）"
      % (len(dis), int((dis.task == "A").sum()), int((dis.task == "B").sum())))

    tk = pd.DataFrame([
        dict(task="A", n=len(WA), po=poA, pe=peA, kappa=kA, pabak=2 * poA - 1),
        dict(task="B", n=len(WB), po=poB, pe=peB, kappa=kB, pabak=2 * poB - 1)])
    tk.to_csv(os.path.join(OUT, "table_v9_interrater.csv"), index=False)

    if len(dis):
        P("")
        P("  分歧清单（需第三仲裁）：")
        for r in dis.itertuples():
            P("    %-8s R1=%-18s R2=%-18s" % (r.case_id, r.R1, r.R2))
        bl = pd.concat([pd.read_csv(os.path.join(OUT, "adjud2_blind_%s.csv" % t))
                        for t in ["A", "B"]], ignore_index=True)
        bl["case_id"] = bl.case_id.astype(str).str.strip()
        d = dis.merge(bl, on="case_id", how="left")
        with open(os.path.join(OUT, "171_disputes.md"), "w",
                  encoding="utf-8") as f:
            f.write("# v9 盲评分歧清单（第三仲裁用）\n\n")
            f.write("规则：读编码手册后独立判定；不得参考算法标签。\n\n")
            for r in d.itertuples():
                f.write("## %s（任务 %s）\n\nR1=%s  R2=%s\n\n```\n%s\n```\n\n"
                        % (r.case_id, r.task, r.R1, r.R2,
                           str(r.excerpt)[:6000]))
        tp = os.path.join(OUT, "adjud2_third.csv")
        if not os.path.exists(tp):
            pd.DataFrame(columns=["case_id", "third_label", "reason"]).to_csv(
                tp, index=False, encoding="utf-8")
            P("")
            P("  已写出仲裁模板 %s（表头 case_id,third_label,reason）" % tp)

    # ---------------- 参考标准
    tp = os.path.join(OUT, "adjud2_third.csv")
    ref = wide.copy()
    ref["ref_std"] = np.where(ref.R1 == ref.R2, ref.R1, None)
    n_th = 0
    if os.path.exists(tp):
        th = pd.read_csv(tp, encoding="utf-8-sig")
        th["case_id"] = th.case_id.astype(str).str.strip()
        tm = dict(zip(th.case_id, th.third_label.astype(str).str.strip()))
        m = ref.ref_std.isna()
        ref.loc[m, "ref_std"] = ref.loc[m, "case_id"].map(tm)
        n_th = int(m.sum())
    P("")
    P("--- 2. 参考标准 ---")
    P("  一致直接采用 %d 例；第三仲裁填充 %d / %d 例（未填则留空）"
      % (int(ref.ref_std.notna().sum()) - n_th, n_th, len(dis)))
    ref["r12_agree"] = (ref.R1 == ref.R2).astype(int)

    # ---------------- 合并封印键
    df = ref.merge(key, on=["case_id", "task"], how="left")
    df["v8_engine"] = np.where(df.task == "A", df.v8_label,
                               np.where(df.v8_act_pos > 0, "ACTIVITY-POSITIVE",
                                        "PURELY-NEGATED"))
    df["v7_engine"] = np.where(df.task == "A", df.v7_label,
                               np.where(df.spec_falsepos == 0,
                                        "ACTIVITY-POSITIVE",
                                        "PURELY-NEGATED"))
    # 设计权重（按 v8 引擎标签分层反推总体）
    wa = {c: FRAME_A[c] / max(1, int((df[(df.task == "A") &
                                        (df.v8_label == c)]).shape[0]))
          for c in CH}
    wb = {}
    for c, pos in [("ACTIVITY-POSITIVE", 1), ("PURELY-NEGATED", 0)]:
        n = int(((df.task == "B") & ((df.v8_act_pos > 0) == (pos == 1))).shape[0])
        wb[c] = FRAME_B[c] / max(1, n)
    df["w"] = np.where(df.task == "A", df.v8_label.map(wa),
                       df.v8_engine.map(wb))
    P("  设计权重：A 层 %s ｜ B 层 %s"
      % (" ".join("%s=%.2f" % (k, v) for k, v in wa.items()),
         " ".join("%s=%.2f" % (k, v) for k, v in wb.items())))

    DA = df[(df.task == "A") & df.ref_std.notna()].copy()
    DB = df[(df.task == "B") & df.ref_std.notna()].copy()

    # ---------------- 引擎 vs 参考标准
    P("")
    P("=" * 100)
    P("--- 3. 样本外 PPV：v8 引擎 vs 人类参考（这才是 v8 的三条规则是否真的有效）")
    P("=" * 100)
    rows = []
    for eng_col, eng, tasks in [("v8_engine", "v8", [("A", DA), ("B", DB)]),
                                ("v7_engine", "v7", [("A", DA), ("B", DB)])]:
        d_all = pd.concat([t[1].assign(task=t[0]) for t in tasks],
                          ignore_index=True)
        for t, d in tasks:
            legal = LBL_A if t == "A" else LBL_B
            for c in legal:
                un = ck2(d, eng_col, "ref_std", c)
                wt = ck2(d, eng_col, "ref_std", c, wcol="w")
                lo, hi = wilson(wt["TP"], wt["TP"] + wt["FP"]) if t == "A" else \
                    wilson(un["TP"], un["TP"] + un["FP"])
                rows.append(dict(engine=eng, task=t, cls=c,
                                 n_algo_pos=un["n_test_pos"], TP=un["TP"],
                                 FP=un["FP"], FN=un["FN"], TN=un["TN"],
                                 ppv=un["ppv"], ppv_w=wt["ppv"],
                                 w_lo=lo, w_hi=hi,
                                 sens=un["sens"], spec=un["spec"],
                                 ppv_lo=lo, ppv_hi=hi,
                                 npv=un["npv"], acc=un["acc"]))
    PV = pd.DataFrame(rows)
    P("  %-5s %-5s %-18s %5s %4s %4s %4s %4s %8s %8s %8s"
      % ("引擎", "任务", "类别", "n阳", "TP", "FP", "FN", "TN", "PPV", "PPV加", "sens"))
    for r in PV.itertuples():
        P("  %-5s %-5s %-18s %5d %4d %4d %4d %4d %8s %8s %8s"
          % (r.engine, r.task, r.cls, r.n_algo_pos, r.TP, r.FP, r.FN, r.TN,
             "%.3f" % r.ppv if np.isfinite(r.ppv) else "—",
             "%.3f" % r.ppv_w if np.isfinite(r.ppv_w) else "—",
             "%.3f" % r.sens if np.isfinite(r.sens) else "—"))
    PV.to_csv(os.path.join(OUT, "table_v9_ppv.csv"), index=False)

    # ---------------- 算法 vs 参考的一致性
    P("")
    P("--- 4. 算法 vs 参考标准的一致性（κ）---")
    for eng_col, eng in [("v8_engine", "v8"), ("v7_engine", "v7")]:
        for t, d, labels in [("A", DA, LBL_A), ("B", DB, LBL_B)]:
            po, pe, k = cohen_kappa(d[eng_col], d.ref_std, labels)
            P("  %-3s 任务 %s：n=%d 一致率 %.3f 期望 %.3f κ=%.3f PABAK=%+.3f"
              % (eng, t, len(d), po, pe, k, 2 * po - 1))
    kk = []
    for eng_col, eng in [("v8_engine", "v8"), ("v7_engine", "v7")]:
        for t, d, labels in [("A", DA, LBL_A), ("B", DB, LBL_B)]:
            po, pe, k = cohen_kappa(d[eng_col], d.ref_std, labels)
            kk.append(dict(engine=eng, task=t, n=len(d), po=po, pe=pe,
                           kappa=k, pabak=2 * po - 1))
    pd.DataFrame(kk).to_csv(os.path.join(OUT, "table_v9_kappa.csv"), index=False)

    # ---------------- 通道规模反推
    P("")
    P("--- 5. 通道规模（分层反推总体，只有在第 3 节站得住时才有意义）---")
    ch = []
    DALL = df[df.task == "A"]          # 全样本（含仲裁未填者），反推总体规模
    for eng_col, eng in [("v8_engine", "v8"), ("v7_engine", "v7")]:
        for c in LBL_A:
            m_ = (DALL[eng_col] == c)
            # Horvitz–Thompson：Σ 设计权重 = 该标签在全框的例数
            ch.append(dict(engine=eng, cls=c, n_sampled=int(m_.sum()),
                           n_projected=float(DALL.loc[m_, "w"].sum())))
    CHD = pd.DataFrame(ch)
    P("  抽样框（任务 A）= %d；抽取 %d（其中 %d 例参考标准待仲裁，不参与 PPV）"
      % (sum(FRAME_A.values()), len(DALL), int(DALL.ref_std.isna().sum())))
    for r in CHD.itertuples():
        P("    %-3s %-10s 抽到 %3d 例 → 全框估计 %6.1f 例（全框真值见抽样框）"
          % (r.engine, r.cls, r.n_sampled, r.n_projected))
    # v8 的自洽性：v8 就是分层变量，Σw 必须等于抽样框层规模
    for c in LBL_A:
        got = float(CHD[(CHD.engine == "v8") & (CHD.cls == c)].iloc[0].n_projected)
        assert abs(got - FRAME_A[c]) < 1e-6, "v8 通道反推与抽样框不一致：%s" % c
    CHD.to_csv(os.path.join(OUT, "table_v9_channels.csv"), index=False)

    # ---------------- 连续变量的样本外判别力
    P("")
    P("--- 6. 连续变量 aid_doc 的样本外判别力（对本次新参考标准）---")
    ad = pd.read_csv(os.path.join(DATA, "gc_activity_density.csv"))
    ad["stay_key"] = ad.stay_key.astype(int)
    base = df.drop(columns=[c for c in ["aid_doc", "aid_broad", "info_n",
                                        "narr_char"] if c in df.columns])
    df2 = base.merge(ad[["stay_key", "aid_doc", "aid_broad", "info_n",
                         "narr_char"]], on="stay_key", how="left")
    DA2 = df2[(df2.task == "A") & df2.ref_std.notna()]
    DB2 = df2[(df2.task == "B") & df2.ref_std.notna()]
    yA = DA2.ref_std.isin(["INACTIVE", "INF_DEFER"]).astype(int)
    yB = (DB2.ref_std == "ACTIVITY-POSITIVE").astype(int)
    arows = []
    for nm in ["aid_doc", "aid_broad", "info_n", "narr_char"]:
        for tag, y, d in [("A（有 vs 无活动度陈述）", yA, DA2),
                          ("B（活动 vs 纯否定）", yB, DB2)]:
            v = d[nm].values
            if len(set(y)) < 2:
                continue
            auc = roc_auc_score(y, v)
            arows.append(dict(task=tag, var=nm, auc=auc, n=len(d),
                              n_pos=int(y.sum())))
            P("    %-24s %-10s AUC=%.3f (n=%d, 阳性 %d)"
              % (tag, nm, auc, len(d), int(y.sum())))
    pd.DataFrame(arows).to_csv(os.path.join(OUT, "table_v9_auc.csv"),
                               index=False)

    # ---------------- 与样本内对照
    P("")
    P("=" * 100)
    P("--- 7. 样本内（v7b，n=148）vs 样本外（v9，n=%d）" % len(df))
    P("=" * 100)
    cmp_rows = []
    import json
    with open(os.path.join(OUT, "_v8.json"), encoding="utf-8") as fh:
        J8 = json.load(fh)
    inA = {r.cls: r for r in PV[(PV.engine == "v8") & (PV.task == "A")].itertuples()}
    P("  %-18s %14s %14s" % ("类别", "样本内 PPV", "样本外 PPV"))
    ins = pd.DataFrame(J8["ppv"])
    ins8 = ins[ins.engine == "v8"]
    for c in CH:
        i = ins8[ins8.cls == c]
        o = inA.get(c)
        a = float(i.iloc[0].ppv) if len(i) else np.nan
        b = o.ppv if o is not None else np.nan
        P("  %-18s %14s %14s" % (
            c, "%.3f (%d/%d)" % (a, int(i.iloc[0].TP),
                                 int(i.iloc[0].n_algo_pos)) if len(i) else "—",
            "%.3f (%d/%d)" % (b, o.TP, o.n_algo_pos)
            if o is not None and np.isfinite(b) else "—"))
        cmp_rows.append(dict(cls=c, ppv_in=a, ppv_out=b,
                             n_in=int(i.iloc[0].n_algo_pos) if len(i) else 0,
                             n_out=o.n_algo_pos if o is not None else 0))
    k_in = pd.read_csv(os.path.join(OUT, "table_v8_kappa.csv"))
    k_out = pd.DataFrame(kk)
    # 任务 B 的样本内对照（v7b 的 v8(narrow)）
    insB = pd.DataFrame(J8["ppv_B"])
    ib = insB[insB.engine == "v8(narrow)"]
    ob = PV[(PV.engine == "v8") & (PV.task == "B") &
            (PV.cls == "ACTIVITY-POSITIVE")]
    if len(ib) and len(ob):
        cmp_rows.append(dict(
            cls="ACTIVITY-POSITIVE", ppv_in=float(ib.iloc[0].ppv),
            ppv_out=float(ob.iloc[0].ppv), n_in=int(ib.iloc[0].n_algo_pos),
            n_out=int(ob.iloc[0].n_algo_pos)))
    P("")
    P("  %-22s %10s %10s" % ("κ（算法 vs 人类）", "样本内", "样本外"))
    for t, eng_in in [("A", "v8"), ("B", "v8(narrow)")]:
        a = k_in[(k_in.task == t) & (k_in.engine == eng_in)]
        b = k_out[(k_out.task == t) & (k_out.engine == "v8")]
        va = float(a.iloc[0].kappa) if len(a) else np.nan
        vb = float(b.iloc[0].kappa) if len(b) else np.nan
        P("  %-22s %10s %10s" % ("任务 " + t, "%.3f" % va, "%.3f" % vb))
        for r in cmp_rows:
            r["kappa_in_%s" % t] = va
            r["kappa_out_%s" % t] = vb
    pd.DataFrame(cmp_rows).to_csv(os.path.join(OUT, "table_v9_compare.csv"),
                                  index=False)

    df.to_csv(os.path.join(OUT, "adjud2_merged.csv"), index=False)
    P("")
    P("saved: 171_replication.txt, table_v9_{ppv,kappa,channels,auc,compare,"
      "interrater}.csv, adjud2_merged.csv")

    with open(os.path.join(OUT, "171_replication.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
