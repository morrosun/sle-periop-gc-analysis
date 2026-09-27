# -*- coding: utf-8 -*-
"""
166_double_coding_metrics.py
============================
v7b —— 盲法双标核验的度量与仲裁。

输入
  out/adjud_sealed_key.csv   封印键（算法标签 + 分层 + 权重所需的分层规模）
  out/adjud_ratings_long.csv 两评分者 R1/R2 的逐例判定
  out/adjud_third.csv        （可选）对 R1≠R2 病例的第三仲裁

输出
  out/166_disagreements.md        R1≠R2 病例清单（供人工仲裁）
  out/166_double_coding.txt       完整度量报告
  out/table_v7b_*.csv             各表
  out/_v7b_results.json           令牌驱动报告所需数值

方法要点
  · 抽样是**分层**的（稀有通道被过抽）→ 报告 (a) 未加权 与 (b) 按分层权重反推的估计
    权重 w_h = N_h / n_h，h = 算法分层；敏感度/特异度本身不受抽样影响，
    但 PPV/NPV 与总一致率受构成影响，必须加权。
  · κ = (po - pe)/(1 - pe)；同时报 PABAK = 2po - 1（稀疏层 κ 悖论的对冲）。
  · 截断改变了算法自身判定的病例（A 3 例 / B 4 例）另做排除性敏感性分析。
  · κ 的 CI 用分层 bootstrap（2000 次，按算法分层内重抽）。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
DATA = os.path.join(ROOT, "data")
SEED = 20260916

# 抽样框规模（由 165 输出，硬编码于此以免再连数据库；见 165_blind_qc.txt）
FRAME_A = {"INACTIVE": 54, "INF_DEFER": 23, "UNDERDOC": 779}
FRAME_B = {"affirmative": 376, "purely-negated": 18}
CLS_A = ["INACTIVE", "INF_DEFER", "UNDERDOC"]

L = []


def P(s=""):
    print(s)
    sys.stdout.flush()
    L.append(str(s))


def cohen_kappa(a, b, labels):
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a)
    if n == 0:
        return np.nan, np.nan, np.nan
    po = float((a == b).mean())
    pe = 0.0
    for c in labels:
        pe += (a == c).mean() * (b == c).mean()
    k = (po - pe) / (1 - pe) if pe < 1 else np.nan
    return po, pe, k


def w_kappa(a, b, labels, w):
    a = np.asarray(a)
    b = np.asarray(b)
    w = np.asarray(w, dtype=float)
    sw = w.sum()
    po = float((w * (a == b)).sum() / sw)
    pe = 0.0
    for c in labels:
        pe += (w * (a == c)).sum() / sw * (w * (b == c)).sum() / sw
    k = (po - pe) / (1 - pe) if pe < 1 else np.nan
    return po, pe, k


def boot_kappa(df, a_col, b_col, labels, wcol=None, nrep=2000, strat="stratum"):
    rng = np.random.RandomState(SEED)
    ks, pbs = [], []
    groups = [g for _, g in df.groupby(strat)]
    for _ in range(nrep):
        parts = [g.sample(len(g), replace=True, random_state=rng)
                 for g in groups]
        s = pd.concat(parts, ignore_index=True)
        if wcol:
            po, pe, k = w_kappa(s[a_col], s[b_col], labels, s[wcol])
        else:
            po, pe, k = cohen_kappa(s[a_col], s[b_col], labels)
        ks.append(k)
        pbs.append(2 * po - 1)
    ks = np.array([x for x in ks if np.isfinite(x)])
    pbs = np.array(pbs)
    return (np.nanpercentile(ks, [2.5, 97.5]),
            np.nanpercentile(pbs, [2.5, 97.5]))


def ck2(df, test_col, truth_col, pos, wcol=None):
    """测试=算法，真值=参考标准。返回标准的 sens/spec/PPV/NPV/ACC。

      TP = 算法阳性 且 真值阳性      FP = 算法阳性 且 真值阴性
      FN = 算法阴性 且 真值阳性      TN = 算法阴性 且 真值阴性
    """
    a = df[test_col].astype(str).values
    b = df[truth_col].astype(str).values
    w = (df[wcol].astype(float).values if wcol else np.ones(len(df)))
    TP = w[(a == pos) & (b == pos)].sum()
    FP = w[(a == pos) & (b != pos)].sum()
    FN = w[(a != pos) & (b == pos)].sum()
    TN = w[(a != pos) & (b != pos)].sum()
    return dict(
        n_test_pos=int((a == pos).sum()), TP=TP, FP=FP, FN=FN, TN=TN,
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


def load_ratings():
    """直接解析 12 个评分 CSV（R2 的 A 系列无表头，需自适应）。"""
    import glob
    import re
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, "adjud_R?_*.csv"))):
        m = re.match(r"adjud_R(\d)_([AB])_p(\d)\.csv", os.path.basename(f))
        raw = open(f, encoding="utf-8-sig").read()
        has_header = raw.splitlines()[0].strip().startswith("case_id")
        d = pd.read_csv(f, encoding="utf-8-sig",
                        header=0 if has_header else None,
                        names=None if has_header else
                        ["case_id", "label", "evidence_quote"])
        for c in ["case_id", "label", "evidence_quote"]:
            if c not in d.columns:
                d[c] = ""
        d = d[["case_id", "label", "evidence_quote"]].copy()
        d["case_id"] = d.case_id.astype(str).str.strip()
        d["label"] = d.label.astype(str).str.strip()
        d["rater"] = "R" + m.group(1)
        d["task"] = m.group(2)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def main():
    key = pd.read_csv(os.path.join(OUT, "adjud_sealed_key.csv"))
    rt = load_ratings()
    rt.to_csv(os.path.join(OUT, "adjud_ratings_long.csv"), index=False)
    w = rt.pivot_table(index="case_id", columns="rater", values="label",
                       aggfunc="first")
    q = rt[rt.rater == "R1"].set_index("case_id").evidence_quote
    q2 = rt[rt.rater == "R2"].set_index("case_id").evidence_quote
    df = key.merge(w, on="case_id", how="left")

    # ---------------- 第三仲裁（若已存在）
    tp = os.path.join(OUT, "adjud_third.csv")
    if os.path.exists(tp):
        th = pd.read_csv(tp, encoding="utf-8-sig")
        th.columns = [c.strip() for c in th.columns]
        df = df.merge(th[["case_id", "third_label"]], on="case_id", how="left")
        P("已载入第三仲裁 %d 例" % int(df.third_label.notna().sum()))
    else:
        df["third_label"] = np.nan

    # ---------------- 参考标准：R1==R2 用二者共同判定；否则用第三仲裁
    def ref(r):
        if r.R1 == r.R2:
            return r.R1
        if isinstance(r.third_label, str) and r.third_label.strip():
            return r.third_label.strip()
        return np.nan

    df["ref_std"] = df.apply(ref, axis=1)
    df["r12_agree"] = (df.R1 == df.R2).astype(int)
    df["algo_eq_R1"] = (df.algo_label == df.R1).astype(int)
    df["algo_eq_R2"] = (df.algo_label == df.R2).astype(int)

    # 设计权重
    wa = FRAME_A
    df["w"] = np.where(df.task == "A",
                       df.stratum.map(wa).astype(float) /
                       df.groupby(["task", "stratum"]).stratum.transform("size"),
                       1.0)
    # 任务 B 的分层规模
    nb = {"affirmative (spec_falsepos=0)": 376,
          "purely negated (spec_falsepos=1)": 18}
    df.loc[df.task == "B", "w"] = (
        df.loc[df.task == "B", "stratum"].map(nb).astype(float) /
        df.loc[df.task == "B"].groupby("stratum").stratum.transform("size"))

    # =============================================================== 分歧清单
    dis = df[(df.R1 != df.R2) | (df.third_label.isna() & (df.R1 != df.R2))]
    bl = pd.concat([pd.read_csv(os.path.join(OUT, "adjud_blind_%s.csv" % t))
                    for t in ["A", "B"]], ignore_index=True).set_index("case_id")
    with open(os.path.join(OUT, "166_disagreements.md"), "w",
              encoding="utf-8") as f:
        f.write("# R1 与 R2 判定不一致的病例（待第三仲裁）\n\n")
        f.write("填写方式：在 `out/adjud_third.csv` 中补 `case_id,third_label,"
                "reason` 三列。\n\n")
        f.write("| case_id | task | 算法 | R1 | R2 |\n|---|---|---|---|---|\n")
        for r in dis.itertuples():
            f.write("| %s | %s | %s | %s | %s |\n"
                    % (r.case_id, r.task, r.algo_label, r.R1, r.R2))
        f.write("\n---\n\n")
        for r in dis.itertuples():
            f.write("## %s （算法 %s / R1 %s / R2 %s）\n\n" % (
                r.case_id, r.algo_label, r.R1, r.R2))
            f.write("- R1 引用：`%s`\n" % str(q.get(r.case_id, ""))[:200])
            f.write("- R2 引用：`%s`\n\n" % str(q2.get(r.case_id, ""))[:200])
            f.write("```\n%s\n```\n\n" % str(bl.loc[r.case_id, "excerpt"])[:2000])
    P("不一致病例 %d / %d（%.1f%%）→ out/166_disagreements.md"
      % (len(dis), len(df), 100 * len(dis) / len(df)))

    # 任务 B 的算法判定：用 spec_verdict_full，并映射到评分者标签体系
    MAP_B = {"affirmative": "ACTIVITY-POSITIVE",
             "purely-negated": "PURELY-NEGATED"}
    df["algo_B"] = df.spec_verdict_full.map(MAP_B)
    df.loc[df.task == "A", "algo_B"] = np.nan

    # =============================================================== 度量
    res = {"n": int(len(df)), "n_disagree": int(len(dis))}
    tab_k, tab_m, tab_cm, tab_cf = [], [], [], []

    B_READER_LABELS = ["ACTIVITY-POSITIVE", "PURELY-NEGATED"]
    for task, labels, algo_col in [("A", CLS_A, "algo_label"),
                                   ("B", B_READER_LABELS, "algo_B")]:
        d = df[df.task == task].copy()
        P("")
        P("=" * 96)
        P("任务 %s：%s（n=%d）"
          % (task, "三分类 INACTIVE/INF_DEFER/UNDERDOC" if task == "A"
             else "否定规则 ACTIVITY-POSITIVE/PURELY-NEGATED", len(d)))
        P("=" * 96)

        # ---- 评分者间一致性
        po, pe, k = cohen_kappa(d.R1, d.R2, labels)
        pw_, pew, kw = w_kappa(d.R1, d.R2, labels, d.w)
        b0 = boot_kappa(d, "R1", "R2", labels)[0]
        P("R1 vs R2  ：一致 %.3f  期望一致 %.3f  κ=%.3f (95%%CI %.3f–%.3f)  "
          "PABAK=%.3f" % (po, pe, k, b0[0], b0[1], 2 * po - 1))
        P("             分层加权：一致 %.3f  κ=%.3f" % (pw_, kw))
        tab_k.append(dict(task=task, pair="R1 vs R2", agreement=po,
                          expected=pe, kappa=k, k_lo=b0[0], k_hi=b0[1],
                          pabak=2 * po - 1, agreement_w=pw_, kappa_w=kw,
                          n=len(d)))

        # ---- 算法 vs 评分者
        for pair, bc in [("Alg vs R1", "R1"), ("Alg vs R2", "R2")]:
            po, pe, k = cohen_kappa(d[algo_col], d[bc], labels)
            pw_, _, kw = w_kappa(d[algo_col], d[bc], labels, d.w)
            bb = boot_kappa(d, algo_col, bc, labels)[0]
            P("%-10s：一致 %.3f  期望一致 %.3f  κ=%.3f (95%%CI %.3f–%.3f)  "
              "PABAK=%.3f" % (pair, po, pe, k, bb[0], bb[1], 2 * po - 1))
            P("             分层加权：一致 %.3f  κ=%.3f" % (pw_, kw))
            tab_k.append(dict(task=task, pair=pair, agreement=po, expected=pe,
                              kappa=k, k_lo=bb[0], k_hi=bb[1],
                              pabak=2 * po - 1, agreement_w=pw_, kappa_w=kw,
                              n=len(d)))

        da = d[d.ref_std.notna()].copy()      # 含第三仲裁；仅在两者皆无时剔除
        po, pe, k = cohen_kappa(da[algo_col], da.ref_std, labels)
        P("Alg vs 参考标准*：n=%d  一致 %.3f  κ=%.3f  PABAK=%.3f"
          % (len(da), po, k, 2 * po - 1))

        # ---- 主指标：算法各标签的「确证率」= P(参考标准 == 算法 | 算法 = c)
        P("")
        P("□ 主指标 —— 算法各标签的确证率（分母为算法标签，样本量充裕）")
        P("  %-16s %6s %8s %10s %-22s" % ("算法标签", "n", "确证", "确证率",
                                           "95% CI (Wilson)"))
        for c in ([x for x in labels if x in set(da[algo_col].dropna())]):
            s = da[da[algo_col] == c]
            kk = int((s.ref_std == c).sum())
            lo, hi = wilson(kk, len(s))
            P("  %-16s %6d %8d %9.1f%% %-22s"
              % (c, len(s), kk, 100 * kk / max(1, len(s)),
                 "%.1f–%.1f%%" % (100 * lo, 100 * hi)))
            tab_cf.append(dict(task=task, algo_label=c, n=len(s), confirmed=kk,
                               rate=kk / max(1, len(s)), lo=lo, hi=hi,
                               dest=str(dict(s.ref_std.value_counts()))))

        # ---- 反向：参考标准各标签 → 算法标签分布
        P("")
        P("□ 反向 —— 参考标准各标签对应的算法标签分布")
        for c in labels:
            s = da[da.ref_std == c]
            if len(s):
                P("   参考 %-16s n=%3d → 算法：%s"
                  % (c, len(s), dict(s[algo_col].value_counts())))

        # ---- 逐类 sens/spec/PPV/NPV（加权）
        P("")
        P("□ 逐类 2×2（算法 = 测试，参考标准* = 真值；加权=按分层权重反推）")
        P("  %-16s %6s | %-13s %-13s | %-13s %-13s"
          % ("类", "算法n", "敏感度(粗/加权)", "特异度(粗/加权)",
             "PPV(粗/加权)", "NPV(粗/加权)"))
        for c in labels:
            r1 = ck2(da, algo_col, "ref_std", c)
            r2 = ck2(da, algo_col, "ref_std", c, wcol="w")
            P("  %-16s %6d | %5.2f / %5.2f  %5.2f / %5.2f  %5.2f / %5.2f  "
              "%5.2f / %5.2f"
              % (c, r1["n_test_pos"], r1["sens"], r2["sens"], r1["spec"],
                 r2["spec"], r1["ppv"], r2["ppv"], r1["npv"], r2["npv"]))
            tab_m.append(dict(task=task, cls=c, n_algo_pos=r1["n_test_pos"],
                              TP_crude=r1["TP"], FP_crude=r1["FP"],
                              FN_crude=r1["FN"], TN_crude=r1["TN"],
                              sens_crude=r1["sens"], sens_w=r2["sens"],
                              spec_crude=r1["spec"], spec_w=r2["spec"],
                              ppv_crude=r1["ppv"], ppv_w=r2["ppv"],
                              npv_crude=r1["npv"], npv_w=r2["npv"]))

        cm = pd.crosstab(da[algo_col], da.ref_std)
        P("")
        P("□ 混淆矩阵（行=算法，列=参考标准*）")
        P(cm.to_string())
        for row in cm.index:
            for col in cm.columns:
                tab_cm.append(dict(task=task, algo=row, ref=col,
                                   n=int(cm.loc[row, col])))

        col_s = "algo_stable" if task == "A" else "algo_stable_B"
        ds = da[da[col_s] == 1]
        po2, _, k2 = cohen_kappa(ds[algo_col], ds.ref_std, labels)
        P("")
        P("敏感性（剔除截断改变算法自身判定的 %d 例）：n=%d  Alg vs 参考 "
          "一致 %.3f  κ=%.3f" % (int((d[col_s] == 0).sum()), len(ds), po2, k2))
        res["sens_exclude_unstable_%s" % task] = dict(
            n=len(ds), agreement=po2, kappa=k2)

    # =============================================================== 导出
    K = pd.DataFrame(tab_k)
    M = pd.DataFrame(tab_m)
    C = pd.DataFrame(tab_cm)
    F = pd.DataFrame(tab_cf)
    K.to_csv(os.path.join(OUT, "table_v7b_kappa.csv"), index=False)
    M.to_csv(os.path.join(OUT, "table_v7b_perclass.csv"), index=False)
    C.to_csv(os.path.join(OUT, "table_v7b_confusion.csv"), index=False)
    F.to_csv(os.path.join(OUT, "table_v7b_confirm.csv"), index=False)
    df.to_csv(os.path.join(OUT, "adjud_merged.csv"), index=False)

    res.update(dict(
        frame_a=FRAME_A, frame_b=FRAME_B,
        kappa=tab_k, perclass=tab_m, confusion=tab_cm, confirm=tab_cf,
        nA=int((df.task == "A").sum()), nB=int((df.task == "B").sum()),
    ))
    with open(os.path.join(OUT, "_v7b_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1,
                  default=lambda o: (None if isinstance(o, float) and
                                     not np.isfinite(o) else str(o)))

    with open(os.path.join(OUT, "166_double_coding.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    P("")
    P("saved: 166_double_coding.txt / 166_disagreements.md / table_v7b_*.csv /"
      " _v7b_results.json")
    print("\n".join(L))


if __name__ == "__main__":
    main()
