# -*- coding: utf-8 -*-
"""
213_human_adjud_analysis.py —— 真人盲审评分分析（adjud3 / v13）

按 out/adjud3_analysis_plan.md（v1.1）预先指定的 6 项指标逐项估计。

用法：
  python scripts/213_human_adjud_analysis.py \
      --a out/盲审报告/adjud3_score_A_b0_corrected.csv \
      --b out/盲审报告/adjud3_score_B_b0_corrected.csv

输出：
  out/report_v13_b0_calibration.html     主报告（令牌驱动）
  out/table_v13_*.csv                    各指标表
  out/fig/Figure41_human_adjud_b0.png    散点 + 四格
  out/盲审报告/                          报告副本

铁律遵循：
  - 门槛断言先行（case_id 集合/顺序/取值域）
  - 配对比较用 McNemar（铁律 27）
  - 报 PPV 不报总体一致率（铁律 21）
  - κ 用 expected agreement，不是 1/k（铁律 17）
  - 逐格判 isfinite（铁律 35）
"""
import os
import io, os, sys, json, argparse
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
FIG = os.path.join(OUT, "fig")
os.makedirs(FIG, exist_ok=True)

Q = ["q1_info", "q2_nstmt", "q3_time", "q4_explicit"]
DOM = {  # 取值域
    "q1_info": {0, 1, 2, 3},
    "q2_nstmt": None,   # 非负整数
    "q3_time": {0, 1, 2},
    "q4_explicit": {0, 1},
}
TOT_N = 220


def P(s=""):
    print(s)
    return s


# ============================================================ 统计函数
def weighted_spearman(x, y, w):
    """按设计权重做频数扩展后的 Spearman ρ。"""
    x = np.asarray(x, float); y = np.asarray(y, float); w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[ok], y[ok], w[ok]
    if len(x) < 3:
        return np.nan
    rep = np.maximum(1, np.round(w).astype(int))
    xs = np.repeat(x, rep); ys = np.repeat(y, rep)
    if len(np.unique(xs)) < 2 or len(np.unique(ys)) < 2:
        return np.nan
    return float(stats.spearmanr(xs, ys).statistic)


def auc_score(score, y):
    """AUC：以 y == 1 为正例，score 越大越可能为正例。"""
    score = np.asarray(score, float); y = np.asarray(y, float)
    ok = np.isfinite(score) & np.isfinite(y)
    score, y = score[ok], y[ok]
    pos = score[y == 1]; neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = stats.rankdata(score)
    return float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def icc21(x, y):
    """Shrout & Fleiss ICC(2,1)：双向随机、绝对一致、单个评分者。"""
    d = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    n, k = d.shape
    if n < 2 or np.isnan(d).any():
        return np.nan
    gm = d.mean()
    rm = d.mean(axis=1); cm = d.mean(axis=0)
    MSR = k * ((rm - gm) ** 2).sum() / (n - 1)
    MSC = n * ((cm - gm) ** 2).sum() / (k - 1)
    MSE = ((d - rm[:, None] - cm[None, :] + gm) ** 2).sum() / ((n - 1) * (k - 1))
    den = MSR + (k - 1) * MSE + k * (MSC - MSE) / n
    if den == 0:
        return np.nan
    return float((MSR - MSE) / den)


def cohen_kappa(a, b, weights=None):
    """Cohen κ；weights='linear'|'quadratic' 时为加权 κ。expected agreement 用边际积。"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    cats = sorted(set(a) | set(b))
    if len(cats) < 2:
        return np.nan          # 只出现一个类别 → κ 未定义
    idx = {c: i for i, c in enumerate(cats)}
    m = len(cats)
    t = np.zeros((m, m))
    for x, y in zip(a, b):
        t[idx[x], idx[y]] += 1
    n = t.sum()
    po = np.trace(t) / n
    pe = float(((t.sum(0) / n) * (t.sum(1) / n)).sum())
    if weights in ("linear", "quadratic"):
        # 必须用「相似性」权重 w = 1 - d（d 为归一化差异）。
        # 若误用差异权重，完全一致时 po = 0，κ 会变成负值（本次踩坑）。
        d = np.array([[abs(i - j) for j in range(m)] for i in range(m)], float)
        d = d / (m - 1)
        wm = 1.0 - (d if weights == "linear" else d ** 2)
        po = float((wm * t).sum() / n)
        pe = float((wm * np.outer(t.sum(1) / n, t.sum(0) / n)).sum())
    if pe >= 1.0:
        return np.nan
    return float((po - pe) / (1 - pe))


def mcnemar_exact(a, b):
    """配对二分类 McNemar 精确检验（两侧）。返回 (b_cell, c_cell, p)。"""
    a = np.asarray(a); b = np.asarray(b)
    bb = int(((a == 1) & (b == 0)).sum())
    cc = int(((a == 0) & (b == 1)).sum())
    nn = bb + cc
    if nn == 0:
        return bb, cc, 1.0
    kk = min(bb, cc)
    p = min(1.0, 2.0 * stats.binom.cdf(kk, nn, 0.5))
    return bb, cc, float(p)


def boot_ci(fn, df, n_boot=1000, seed=20260917):
    """层内 bootstrap 百分位 CI。"""
    rng = np.random.default_rng(seed)
    vals = []
    layers = df["layer"].unique()
    for _ in range(n_boot):
        parts = []
        for ly in layers:
            g = df[df["layer"] == ly]
            if len(g) == 0:
                continue
            parts.append(g.iloc[rng.integers(0, len(g), len(g))])
        s = pd.concat(parts, ignore_index=True)
        try:
            v = fn(s)
        except Exception:
            v = np.nan
        if v is not None and np.isfinite(v):
            vals.append(v)
    if len(vals) < 50:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# ============================================================ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--key", default=os.path.join(OUT, "adjud3_sealed_key.csv"))
    ap.add_argument("--tag", default="b0_calibration")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    for p in (args.a, args.b):
        assert os.path.exists(p), "评分文件不存在：%s" % p

    # ---------------------------------------------------- 1 读入 + 门槛断言
    sa = pd.read_csv(args.a, encoding="utf-8-sig")
    sb = pd.read_csv(args.b, encoding="utf-8-sig")
    a_txt = open(args.a, encoding="utf-8-sig").read()
    b_txt = open(args.b, encoding="utf-8-sig").read()
    for nm, d in [("A", sa), ("B", sb)]:
        for c in ["case_id"] + Q:
            assert c in d.columns, "%s 缺列 %s" % (nm, c)
    assert set(sa["case_id"]) == set(sb["case_id"]), "A/B 的 case_id 集合不一致"

    key = pd.read_csv(args.key)
    assert set(sa["case_id"]) <= set(key["case_id"]), "出现密封键之外的 case_id"

    na = sa[Q].notna().any(axis=1)
    nb = sb[Q].notna().any(axis=1)
    assert int(na.sum()) == int(nb.sum()), "A/B 已评例数不一致"

    a = sa[na][["case_id"] + Q].rename(columns={c: c + "_A" for c in Q})
    b = sb[nb][["case_id"] + Q].rename(columns={c: c + "_B" for c in Q})
    assert set(a["case_id"]) == set(b["case_id"]), "A/B 已评 case_id 集合不一致"
    assert a["case_id"].is_unique and b["case_id"].is_unique, "评分文件存在重复 case_id"

    # 取值域断言
    for c in Q:
        for sfx in ("_A", "_B"):
            col = (a if sfx == "_A" else b)[c + sfx]
            if DOM[c] is not None:
                bad = set(col.dropna().unique()) - DOM[c]
                assert not bad, "%s%s 取值越界：%s" % (c, sfx, bad)
            else:
                assert (col.dropna() >= 0).all(), "%s%s 出现负值" % (c, sfx)
    P("门槛断言：case_id 集合/顺序、取值域 全部通过")

    m = key.merge(a, on="case_id").merge(b, on="case_id")
    m["q1_mean"] = (m["q1_info_A"] + m["q1_info_B"]) / 2.0
    m["q2_mean"] = (m["q2_nstmt_A"] + m["q2_nstmt_B"]) / 2.0
    m["q4_both1"] = ((m["q4_explicit_A"] == 1) & (m["q4_explicit_B"] == 1)).astype(int)
    m["q4_both0"] = ((m["q4_explicit_A"] == 0) & (m["q4_explicit_B"] == 0)).astype(int)
    m["q1_lo"] = (m["q1_mean"] <= 1).astype(int)      # 人判信息不足
    m["alg_zero"] = (m["aid_doc"] == 0).astype(int)   # 算法判无信息

    n = len(m)
    complete = (n == TOT_N)
    # 说明：此前文件名含 "simulated"，实际为真人评分的误标注（2026-09-17 用户确认）。
    # 故不再设 sim 分支；此处保留变量以兼容下方阈值判定逻辑。
    is_sim = False

    # 退化解检测：A/B 逐例完全一致 -> 一致性指标恒等于 1，信度不可估
    pair_cols = ["q1_info", "q2_nstmt", "q3_time", "q4_explicit"]
    same_mask = np.ones(len(m), bool)
    for cc_ in pair_cols:
        same_mask &= (m[cc_ + "_A"].values == m[cc_ + "_B"].values)
    degenerate = bool(same_mask.all())
    # 附加：两套评分文件本身是否逐字节相同
    files_identical = (a_txt == b_txt) and (a_txt != "")

    P("已评 n = %d / %d（%s）" % (n, TOT_N, "完整" if complete else "部分批次"))
    if degenerate:
        P("⚠ 检测到退化解：A/B 在全部 %d 例、四项评分上完全一致%s。"
          % (n, "（且两份文件内容逐字节相同）" if files_identical else ""))
        P("  → ICC / κ 恒等于 1.000，**不构成评分者间信度证据**；正式批次必须保留各自独立评分。")

    # 完成度警示：部分批次下设计权重不适用
    lyr = m["layer"].value_counts().sort_index().to_dict()
    P("层构成：%s" % lyr)

    # ---------------------------------------------------- 2 六项指标
    res = {}

    # (1)(2) 构念效度
    res["rho_q1_w"] = weighted_spearman(m["q1_mean"], m["aid_doc"], m["weight"]) if complete else np.nan
    res["rho_q2_w"] = weighted_spearman(m["q2_mean"], m["aid_doc"], m["weight"]) if complete else np.nan
    res["rho_q1"] = float(stats.spearmanr(m["q1_mean"], m["aid_doc"]).statistic)
    res["rho_q2"] = float(stats.spearmanr(m["q2_mean"], m["aid_doc"]).statistic)
    res["rho_q1_p"] = float(stats.spearmanr(m["q1_mean"], m["aid_doc"]).pvalue)
    res["rho_q1_A"] = float(stats.spearmanr(m["q1_info_A"], m["aid_doc"]).statistic)
    res["rho_q1_B"] = float(stats.spearmanr(m["q1_info_B"], m["aid_doc"]).statistic)

    # 长度伪影诊断：密度以千字符为分母，短叙述会被系统性抬高
    res["rho_len_all"] = float(stats.spearmanr(m["aid_doc"], m["narr_char"]).statistic)
    _q40 = m[m["q4_both0"] == 1]
    res["rho_len_q4_0"] = (float(stats.spearmanr(_q40["aid_doc"], _q40["narr_char"]).statistic)
                           if len(_q40) >= 5 else float("nan"))
    _pos = m[m["aid_doc"] > 0]
    res["n_alg_pos"] = int(len(_pos))
    res["ppv_alg_pos"] = float((_pos["q4_both1"] == 1).mean()) if len(_pos) else float("nan")
    res["narr_short_med"] = float(_pos["narr_char"].median()) if len(_pos) else float("nan")

    # 打分函数对比：密度(每千字符) vs 原始命中数；并校正叙述长度
    def partial_spearman(x, y, z):
        """以 z 为协变量的偏 Spearman（秩上做线性残差化后取 Pearson）。"""
        x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
        if len({len(x), len(y), len(z)}) != 1 or len(x) < 5:
            return float("nan")
        rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))

        def resid(a, b):
            A = np.column_stack([np.ones(len(a)), b])
            return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
        return float(stats.pearsonr(resid(rx, rz), resid(ry, rz))[0])

    nhit = m["aid_doc"] * m["narr_char"] / 1000.0
    res["auc_nhit"] = auc_score(nhit, m["q4_both1"])
    res["rho_q1_nhit"] = float(stats.spearmanr(m["q1_mean"], nhit).statistic)
    res["rho_len_nhit"] = float(stats.spearmanr(nhit, m["narr_char"]).statistic)
    res["auc_narr"] = auc_score(m["narr_char"], m["q4_both1"])
    res["part_rho_q1_dens"] = partial_spearman(m["q1_mean"], m["aid_doc"], m["narr_char"])
    res["part_rho_q1_nhit"] = partial_spearman(m["q1_mean"], nhit, m["narr_char"])
    res["part_rho_q4_dens"] = partial_spearman(m["q4_both1"], m["aid_doc"], m["narr_char"])
    res["part_rho_q4_nhit"] = partial_spearman(m["q4_both1"], nhit, m["narr_char"])

    # (3) AUC：以「人判有明确评估 Q4 = 1」为正例（v1.1 修正方向）
    res["auc_q4_1"] = auc_score(m["aid_doc"], m["q4_both1"])
    res["auc_q4_0"] = auc_score(m["aid_doc"], m["q4_both0"])   # 反向核对
    res["auc_q4_1_A"] = auc_score(m["aid_doc"], (m["q4_explicit_A"] == 1).astype(int))
    res["auc_q4_1_B"] = auc_score(m["aid_doc"], (m["q4_explicit_B"] == 1).astype(int))

    # (4)(5) 信度
    res["icc_q1"] = icc21(m["q1_info_A"], m["q1_info_B"])
    res["kappa_q4"] = cohen_kappa(m["q4_explicit_A"], m["q4_explicit_B"])
    res["kappa_q1_qw"] = cohen_kappa(m["q1_info_A"], m["q1_info_B"], weights="quadratic")
    res["agree_q1_exact"] = float((m["q1_info_A"] == m["q1_info_B"]).mean())
    res["agree_q4_exact"] = float((m["q4_explicit_A"] == m["q4_explicit_B"]).mean())

    # (6) 零侧核查（报 PPV，不报一致率）
    sel0 = m[m["q4_both0"] == 1]
    res["n_q4_0"] = int(len(sel0))
    res["ppv_zero"] = float((sel0["aid_doc"] == 0).mean()) if len(sel0) else np.nan
    res["ppv_zero_hits"] = int((sel0["aid_doc"] == 0).sum()) if len(sel0) else 0
    # 反向：算法判 0 中，人是否也判无明确评估
    selz = m[m["alg_zero"] == 1]
    res["n_alg0"] = int(len(selz))
    res["sens_zero"] = float((selz["q4_both0"] == 1).mean()) if len(selz) else np.nan

    # McNemar：算法「有信息」 vs 人判「有明确评估」
    bb, cc, pm = mcnemar_exact(1 - m["alg_zero"], m["q4_both1"])
    res["mcnemar_b"], res["mcnemar_c"], res["mcnemar_p"] = bb, cc, pm
    res["n_dis_q4"] = int((m["q4_explicit_A"] != m["q4_explicit_B"]).sum())
    res["auc_sum"] = (res["auc_q4_1"] + res["auc_q4_0"]
                      if (np.isfinite(res["auc_q4_1"]) and np.isfinite(res["auc_q4_0"]))
                      else np.nan)

    # 失效模式四格
    tab = pd.crosstab(
        m["q1_mean"].apply(lambda v: "Q1<=1 (人判不足)" if v <= 1 else "Q1>=2 (人判充分)"),
        m["alg_zero"].map({1: "aid_doc=0 (算法无)", 0: "aid_doc>0 (算法有)"}))
    over = int(((m["q1_mean"] <= 1) & (m["aid_doc"] > 0)).sum())
    miss = int(((m["q1_mean"] >= 2) & (m["aid_doc"] == 0)).sum())
    res["overcall"], res["misscall"] = over, miss
    npos_alg = int((m["aid_doc"] > 0).sum())
    res["overcall_rate"] = float(over / npos_alg) if npos_alg else np.nan

    # bootstrap CI（部分批次仅作参考，脚本内标注）
    ci = {}
    ci["rho_q1"] = boot_ci(lambda d: stats.spearmanr(d["q1_mean"], d["aid_doc"]).statistic, m)
    ci["auc_q4_1"] = boot_ci(lambda d: auc_score(d["aid_doc"], d["q4_both1"]), m)
    ci["icc_q1"] = boot_ci(lambda d: icc21(d["q1_info_A"], d["q1_info_B"]), m)

    # ---------------------------------------------------- 3 判定
    def judge(v, hi, lo):
        if v is None or not np.isfinite(v):
            return "不可估"
        if v >= hi:
            return "达标"
        if v < lo:
            return "失败"
        return "不确定"

    verdict = {
        "1 构念效度ρ(Q1)": judge(res["rho_q1_w"] if complete else res["rho_q1"], 0.40, 0.30),
        "2 构念效度ρ(Q2)": judge(res["rho_q2_w"] if complete else res["rho_q2"], 0.40, 0.30),
        "3 零侧判别AUC": judge(res["auc_q4_1"], 0.75, 0.65),
        "4 信度ICC(Q1)": judge(res["icc_q1"], 0.60, 0.45),
        "5 信度κ(Q4)": judge(res["kappa_q4"], 0.60, 0.45),
        "6 零侧PPV": judge(res["ppv_zero"], 0.80, 0.60),
    }
    if is_sim or not complete:
        # 模拟评分 或 非完整批次 → 阈值判定不生效，避免读者误读为「已达标」
        verdict = {k: "不适用" for k in verdict}
    if degenerate:
        # A/B 完全一致 → 一致性指标恒等于 1，既不是「达标」也不是「不适用」，
        # 而是「不可估」：必须显式标出，防止被引用为信度证据。
        verdict["4 信度ICC(Q1)"] = "退化·不可估"
        verdict["5 信度κ(Q4)"] = "退化·不可估"

    # ---------------------------------------------------- 4 落盘表
    def f2(v):
        return "—" if v is None or not np.isfinite(v) else "%.2f" % v

    def f3(v):
        return "—" if v is None or not np.isfinite(v) else "%.3f" % v

    rows = [
        ("1", "构念效度（主）", "Q1 均值 vs aid_doc 加权 Spearman ρ",
         f3(res["rho_q1_w"]), f3(res["rho_q1"]),
         "%.3f–%.3f" % ci["rho_q1"] if np.isfinite(ci["rho_q1"][0]) else "—",
         "ρ≥0.40 / <0.30", verdict["1 构念效度ρ(Q1)"]),
        ("2", "构念效度（次）", "Q2 均值 vs aid_doc 加权 Spearman ρ",
         f3(res["rho_q2_w"]), f3(res["rho_q2"]), "—",
         "ρ≥0.40 / <0.30", verdict["2 构念效度ρ(Q2)"]),
        ("3", "零侧判别（主）", "aid_doc 预测 人判 Q4=1 的 AUC",
         "—", f3(res["auc_q4_1"]),
         "%.3f–%.3f" % ci["auc_q4_1"] if np.isfinite(ci["auc_q4_1"][0]) else "—",
         "AUC≥0.75 / <0.65", verdict["3 零侧判别AUC"]),
        ("3b", "方向核对", "aid_doc 预测 人判 Q4=0 的 AUC（反向）",
         "—", f3(res["auc_q4_0"]), "—", "互补核对", "—"),
        ("3c", "分评分者", "仅 A：aid_doc 预测 A 判 Q4=1 的 AUC",
         "—", f3(res["auc_q4_1_A"]), "—", "参考", "—"),
        ("3d", "分评分者", "仅 B：aid_doc 预测 B 判 Q4=1 的 AUC",
         "—", f3(res["auc_q4_1_B"]), "—", "参考", "—"),
        ("4", "信度（主）", "Q1 的 ICC(2,1)",
         "—", f3(res["icc_q1"]),
         "%.3f–%.3f" % ci["icc_q1"] if np.isfinite(ci["icc_q1"][0]) else "—",
         "ICC≥0.60 / <0.45", verdict["4 信度ICC(Q1)"]),
        ("5", "信度（次）", "Q4 的 Cohen κ",
         "—", f3(res["kappa_q4"]), "—",
         "κ≥0.60 / <0.45", verdict["5 信度κ(Q4)"]),
        ("5b", "信度（次）", "Q1 的二次加权 κ",
         "—", f3(res["kappa_q1_qw"]), "—", "参考", "—"),
        ("6", "零侧核查", "人判 Q4=0 中 aid_doc=0 的 PPV（%d/%d）"
         % (res["ppv_zero_hits"], res["n_q4_0"]),
         "—", f3(res["ppv_zero"]), "—",
         "PPV≥0.80 / <0.60", verdict["6 零侧PPV"]),
        ("6b", "零侧核查", "算法判 0 中人亦判 Q4=0 的比例（%d 例）" % res["n_alg0"],
         "—", f3(res["sens_zero"]), "—", "参考", "—"),
        ("6c", "正向核查", "算法判 >0 中人亦判 Q4=1 的比例（%d 例）" % res["n_alg_pos"],
         "—", f3(res["ppv_alg_pos"]), "—", "参考", "—"),
        ("7", "长度伪影", "aid_doc vs narr_char 的 Spearman ρ（全体）",
         "—", f3(res["rho_len_all"]), "—", "越接近 0 越理想", "—"),
        ("7b", "长度伪影", "aid_doc vs narr_char 的 Spearman ρ（限人判 Q4=0）",
         "—", f3(res["rho_len_q4_0"]), "—", "越接近 0 越理想", "—"),
        ("8", "打分函数对比", "原始命中数 预测 人判 Q4=1 的 AUC",
         "—", f3(res["auc_nhit"]), "—", "对照 aid_doc=%s" % f3(res["auc_q4_1"]), "—"),
        ("8b", "打分函数对比", "仅 narr_char（长度）预测 人判 Q4=1 的 AUC",
         "—", f3(res["auc_narr"]), "—", "长度基线 AUC", "—"),
        ("8c", "打分函数对比", "原始命中数 vs narr_char 的 ρ（长度依赖度）",
         "—", f3(res["rho_len_nhit"]), "—", "对照 aid_doc=%s" % f3(res["rho_len_all"]), "—"),
        ("9", "长度校正", "偏 ρ(Q1, aid_doc | narr_char)",
         "—", f3(res["part_rho_q1_dens"]), "—", "校正后仍 ≥0.40 方支持", "—"),
        ("9b", "长度校正", "偏 ρ(Q1, 原始命中数 | narr_char)",
         "—", f3(res["part_rho_q1_nhit"]), "—", "校正后仍 ≥0.40 方支持", "—"),
        ("9c", "长度校正", "偏 ρ(Q4, aid_doc | narr_char)",
         "—", f3(res["part_rho_q4_dens"]), "—", "校正后明显下降提示长度混淆", "—"),
        ("9d", "长度校正", "偏 ρ(Q4, 原始命中数 | narr_char)",
         "—", f3(res["part_rho_q4_nhit"]), "—", "校正后明显下降提示长度混淆", "—"),
    ]
    t_metrics = pd.DataFrame(rows, columns=["#", "类别", "估计量", "加权", "未加权/主", "95%CI", "阈值", "判定"])
    t_metrics.to_csv(os.path.join(OUT, "table_v13_metrics.csv"), index=False, encoding="utf-8-sig")

    # 逐例表
    cols = ["case_id", "layer", "disease", "weight", "aid_doc", "narr_char",
            "q1_info_A", "q1_info_B", "q1_mean", "q2_mean",
            "q3_time_A", "q3_time_B", "q4_explicit_A", "q4_explicit_B"]
    m[cols].to_csv(os.path.join(OUT, "table_v13_cases.csv"), index=False, encoding="utf-8-sig")

    # 分歧表
    dis = m[(m["q1_info_A"] != m["q1_info_B"]) | (m["q4_explicit_A"] != m["q4_explicit_B"])
            | (m["q3_time_A"] != m["q3_time_B"]) | (m["q2_nstmt_A"] != m["q2_nstmt_B"])]
    dis[cols].to_csv(os.path.join(OUT, "table_v13_disagreements.csv"), index=False, encoding="utf-8-sig")

    with io.open(os.path.join(OUT, "_v13.json"), "w", encoding="utf-8") as f:
        json.dump({k: (None if (isinstance(v, float) and not np.isfinite(v)) else v)
                   for k, v in res.items()}, f, ensure_ascii=False, indent=1)

    # ---------------------------------------------------- 5 图
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    colmap = {"L1": "#8c8c8c", "L2": "#4a90d9", "L3": "#2e7d32", "L4": "#c62828"}
    for ly in ["L1", "L2", "L3", "L4"]:
        g = m[m["layer"] == ly]
        if len(g) == 0:
            continue
        jit = np.random.default_rng(7).normal(0, 0.02, len(g))
        ax.scatter(g["aid_doc"] + jit, g["q1_mean"] + jit * 3, s=42,
                   c=colmap[ly], label="%s (n=%d)" % (ly, len(g)), alpha=.85,
                   edgecolors="white", linewidths=.6)
    ax.set_xlabel("Algorithm information density  aid_doc", fontsize=9.5)
    ax.set_ylabel("Human Q1 mean (0-3)", fontsize=9.5)
    ax.set_title("Construct validity: aid_doc vs human rating", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.text(0.03, 0.95, "Spearman rho = %.3f (P=%.3f)" % (res["rho_q1"], res["rho_q1_p"]),
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#bbb"))
    ax.set_ylim(-0.35, 3.45)

    ax = axes[1]
    grid = np.array([[int(((m["q1_mean"] <= 1) & (m["alg_zero"] == 0)).sum()),
                      int(((m["q1_mean"] <= 1) & (m["alg_zero"] == 1)).sum())],
                     [int(((m["q1_mean"] >= 2) & (m["alg_zero"] == 0)).sum()),
                      int(((m["q1_mean"] >= 2) & (m["alg_zero"] == 1)).sum())]])
    im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=max(1, grid.max()))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["aid_doc>0\n(alg: has info)", "aid_doc=0\n(alg: no info)"], fontsize=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Q1<=1\n(human: poor)", "Q1>=2\n(human: rich)"], fontsize=8)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(grid[i, j]), ha="center", va="center",
                    fontsize=15, color="white" if grid[i, j] > grid.max() / 2 else "#222",
                    fontweight="bold")
    ax.set_title("Failure mode grid (n=%d)" % n, fontsize=10)
    ax.text(0.5, -0.30, "top-left = OVERCALL (%d)   bottom-right = MISS (%d)" % (over, miss),
            transform=ax.transAxes, ha="center", fontsize=8, color="#444")
    plt.tight_layout()
    fpath = os.path.join(FIG, "Figure41_human_adjud_b0.png")
    plt.savefig(fpath, dpi=160, bbox_inches="tight")
    plt.close()
    P("图 → %s" % fpath)

    # ---------------------------------------------------- 6 报告
    import base64
    with open(fpath, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()

    def row_html(r):
        return ("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
                "<td>%s</td><td>%s</td><td><b>%s</b></td></tr>"
                % tuple(str(x) for x in r))

    tbl = "\n".join(row_html(r) for r in rows)

    case_rows = []
    for _, r in m.sort_values("aid_doc").iterrows():
        case_rows.append(
            "<tr><td>%s</td><td>%s</td><td>%s</td><td>%.3f</td><td>%d</td>"
            "<td>%d/%d</td><td>%d</td><td>%d</td><td>%d</td><td>%d</td></tr>" % (
                r["case_id"], r["layer"], r["disease"], r["aid_doc"], int(r["narr_char"]),
                int(r["q1_info_A"]), int(r["q1_info_B"]),
                int(r["q2_nstmt_A"]), int(r["q2_nstmt_B"]),
                int(r["q4_explicit_A"]), int(r["q4_explicit_B"])))
    case_tbl = "\n".join(case_rows)

    if len(dis):
        dis_rows = "\n".join(
            "<tr><td>%s</td><td>%s</td><td>%.3f</td><td>%d/%d</td><td>%d/%d</td>"
            "<td>%d/%d</td><td>%d/%d</td></tr>" % (
                r["case_id"], r["layer"], r["aid_doc"],
                int(r["q1_info_A"]), int(r["q1_info_B"]),
                int(r["q2_nstmt_A"]), int(r["q2_nstmt_B"]),
                int(r["q3_time_A"]), int(r["q3_time_B"]),
                int(r["q4_explicit_A"]), int(r["q4_explicit_B"]))
            for _, r in dis.iterrows())
    else:
        dis_rows = "<tr><td colspan='7'>无分歧</td></tr>"

    warn = ""
    if degenerate:
        warn = ("<div class='warn'><b>退化解：两份评分不构成独立双评。</b>"
                "A、B 在全部 %d 例、四项评分上完全一致%s，因此 ICC 与 κ 必然等于 1.000。"
                "该数值<b>只反映「两份数据相同」，不反映评分者间信度</b>，"
                "本报告中的信度指标<b>不得写入稿件的信度证据</b>，判定列已标为「退化·不可估」。"
                "<br><b>正式批次（b1–b8）必须保留 A/B 各自独立评分</b>——"
                "分歧是信度估计的原始材料，按共同锚点把分歧抹平等于删掉被估计的对象。</div>"
                % (n, "（且两个文件内容逐字节相同）" if files_identical else ""))
    if not complete:
        warn += ("<div class='warn'><b>部分批次，不得用于达标判定。</b>"
                 "本批仅 %d / %d 例，且按抽样设计 L1 应占 82 例而本批只有 %s 例，"
                "设计权重在此子集上<b>不适用</b>（会把少数几例 L1 复制 6–12 次而虚高估计）。"
                "故下表「加权」列置空，仅报未加权值，用途限定为"
                "<b>管线验证与校准决策</b>。</div>" % (n, TOT_N, lyr.get("L1", 0)))
    rho_main = res["rho_q1_w"] if complete else res["rho_q1"]
    if np.isfinite(rho_main) and rho_main >= 0.40:
        rho_txt = "方向与强度均支持构念效度。"
    else:
        rho_txt = "低于达标阈值 0.40，需待正式样本判定。"
    sim = ""
    html = u"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>真人盲审分析 —— %s</title><style>
body{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;max-width:1080px;
margin:32px auto;padding:0 22px;line-height:1.72;color:#1a1a1a;background:#fff}
h1{font-size:22px;border-bottom:2px solid #1f4e79;padding-bottom:9px}
h2{font-size:17px;margin-top:30px;color:#1f4e79;border-left:4px solid #1f4e79;padding-left:9px}
table{border-collapse:collapse;width:100%%;margin:14px 0;font-size:13px}
th,td{border:1px solid #ccc;padding:6px 9px;text-align:left}
th{background:#eef3f8;font-weight:600}
tr:nth-child(even) td{background:#fafbfc}
.warn{background:#fff4e5;border-left:4px solid #e8a33d;padding:11px 14px;margin:16px 0;font-size:13.5px}
.ok{background:#eaf5ea;border-left:4px solid #2e7d32;padding:11px 14px;margin:16px 0;font-size:13.5px}
code{background:#f2f2f2;padding:1px 5px;border-radius:3px;font-size:12.5px}
img{max-width:100%%;border:1px solid #ddd;margin:12px 0}
.small{font-size:12.5px;color:#666}
</style></head><body>
<h1>真人盲审评分分析（adjud3 / v13）</h1>
<p class="small">批次：%s　|　已评 n = %d / %d　|　生成：scripts/213_human_adjud_analysis.py</p>

%s%s

<h2>1. 结论摘要</h2>
<ul>
<li><b>构念效度（主）</b>：Q1 均值 vs <code>aid_doc</code> 的未加权 Spearman ρ = <b>%s</b>（P = %.3f）。
%s</li>
<li><b>零侧判别</b>：<code>aid_doc</code> 预测「人判有明确评估 Q4=1」的 AUC = <b>%s</b>；
反向核对 AUC(Q4=0) = %s。两者仅在 A/B 对 Q4 完全一致时严格互补（和为 1）；
本批有 <b>%d</b> 例 Q4 分歧，故实测和为 %s。</li>
<li><b>失效模式</b>：过度判定 %d 例（算法说有信息、人判不足），占算法正例 %d 例的 %s；
漏判 %d 例。→ <b>%s</b></li>
<li><b>零侧 PPV</b>：人判 Q4=0 的 %d 例中，<code>aid_doc</code>=0 占 %d 例（PPV = %s）。</li>
</ul>

<h2>2. 预先指定的六项指标</h2>
<table><tr><th>#</th><th>类别</th><th>估计量</th><th>加权</th><th>未加权/主</th>
<th>95%% CI</th><th>阈值</th><th>判定</th></tr>
%s
</table>
<p class="small">CI 为层内 bootstrap 1000 次的百分位区间；部分批次下仅供参考。
判定列在部分批次下<b>不作为达标依据</b>。</p>

<h2>3. 图</h2>
<img src="data:image/png;base64,%s" alt="构念效度散点与失效模式四格">
<p class="small">左：每层散点（横轴算法密度、纵轴人评 Q1 均值），点已加抖动以便区分重叠。
右：失效模式四格；左上角为<b>过度判定</b>（算法有信息、人判不足），
右下角为<b>漏判</b>（算法无信息、人判充分）。</p>

<h2>4. 逐例评分（按 aid_doc 升序）</h2>
<table><tr><th>case</th><th>层</th><th>病种</th><th>aid_doc</th><th>字符</th>
<th>Q1 A/B</th><th>Q2 A/B</th><th>Q4 A</th><th>Q4 B</th></tr>
%s
</table>

<h2>5. 分歧病例（%d 例）</h2>
<table><tr><th>case</th><th>层</th><th>aid_doc</th><th>Q1 A/B</th><th>Q2 A/B</th>
<th>Q3 A/B</th><th>Q4 A/B</th></tr>
%s
</table>

<h2>6. 口径裁决（b0 校准后写定，正式卷沿用）</h2>
<ol>
<li><b>问题列表栏目标题算结构性分类，不算句子</b>：<code>Inactive Issues</code> /
<code>Chronic Issues</code> 下挂着 SLE / RA → Q4 = 1、Q1 ≥ 2、Q3 = 2，但 <b>Q2 计 0</b>。
（算法把 <code>inactive</code> 计为命中；人判「不算」会系统性拉低人-机相关性。）</li>
<li><b>本次住院 ROS 的阴性记录属当前时点</b>：ROS 否认近期关节痛 → <b>Q3 = 2</b>，
但它是症状筛查而非活动度评估 → Q4 = 0、Q1 最多 1。</li>
</ol>
<p>已据上述裁决更新 <code>out/adjud3_rater_manual.md</code>（v13.1，含 b0 校准锚点表），
并将分析计划指标 3 的<b>正例方向</b>更正为「人判 Q4 = 1」（v1.1，修订记录已留痕）。
重跑 212 后已发放盲表 md5 全部未变，不影响已发放材料。</p>

<h2>7. 下一步</h2>
<ul>
<li>正式卷 b1–b8（205 例）回收后重跑本脚本，届时 n = 220，启用加权列与达标判定。</li>
<li>达标条件（主指标 1 与 3 同时达标）未变，后果条款见 <code>adjud3_analysis_plan.md</code>。</li>
<li>无论达标与否，<b>主结论（疾病×剂量交互 1.96, 1.04–3.67, P = 0.006）不受影响</b>——
该估计不依赖文本侧。</li>
</ul>
</body></html>""" % (
        args.tag, args.tag, n, TOT_N, warn, sim,
        f3(rho_main), res["rho_q1_p"], rho_txt,
        f3(res["auc_q4_1"]), f3(res["auc_q4_0"]), int(res["n_dis_q4"]), f3(res["auc_sum"]),
        over, npos_alg,
        ("%.1f%%" % (100 * res["overcall_rate"]) if np.isfinite(res["overcall_rate"]) else "—"),
        miss,
        ("失效模式为<b>过度判定</b>：修法是提高阈值/加权，不是扩词表（铁律 23）。"
         if (over > miss) else "失效模式偏向<b>漏判</b>：修法才是扩词表。"),
        res["n_q4_0"], res["ppv_zero_hits"], f3(res["ppv_zero"]),
        tbl, b64, case_tbl, len(dis), dis_rows)

    assert "{{" not in html, "报告存在未替换令牌"
    # 检查前必须剥离 base64 图数据：随机串里会偶然出现 nan/inf 子串（铁律 12 同类）
    import re as _re
    _probe = _re.sub(r"data:image/[a-z]+;base64,[^\"']+", "", html)
    bad = [w for w in ["nan", "inf", "None"] if w in _probe.replace("None</td>", "")]
    for w in bad:
        i = 0
        while True:
            i = _probe.find(w, i)
            if i < 0:
                break
            sys.stderr.write("DBG[%s] ...%s...\n" % (w, _probe[max(0, i - 90):i + 50]))
            i += 1
    assert not bad, "报告含非法数值：%s" % bad

    rpath = os.path.join(OUT, "report_v13_b0_calibration.html")
    with io.open(rpath, "w", encoding="utf-8") as f:
        f.write(html)
    P("报告 → %s" % rpath)

    dst = os.path.join(OUT, "盲审报告")
    os.makedirs(dst, exist_ok=True)
    with io.open(os.path.join(dst, "adjud3_b0_analysis_report.html"), "w", encoding="utf-8") as f:
        f.write(html)
    P("报告副本 → %s" % os.path.join(dst, "adjud3_b0_analysis_report.html"))

    P("\n=== 关键数值 ===")
    P("  rho(Q1, aid_doc) = %s  P=%.4f" % (f3(res["rho_q1"]), res["rho_q1_p"]))
    P("  AUC(Q4=1) = %s   反向 AUC(Q4=0) = %s" % (f3(res["auc_q4_1"]), f3(res["auc_q4_0"])))
    P("  ICC(Q1) = %s   kappa(Q4) = %s   二次加权kappa(Q1) = %s"
      % (f3(res["icc_q1"]), f3(res["kappa_q4"]), f3(res["kappa_q1_qw"])))
    P("  PPV(零侧) = %s (%d/%d)   过度判定 %d / 漏判 %d" %
      (f3(res["ppv_zero"]), res["ppv_zero_hits"], res["n_q4_0"], over, miss))
    P("  McNemar b=%d c=%d P=%.3f" % (bb, cc, pm))


if __name__ == "__main__":
    main()
