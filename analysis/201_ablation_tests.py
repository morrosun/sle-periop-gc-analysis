# -*- coding: utf-8 -*-
"""
201_ablation_tests.py
=====================
v10 ① —— 消融的**配对检验**：FULL 相对其它配置，改善是真信号还是噪声？

为什么必须配对
  8 个配置跑的是**同一批病例**（样本内 148 / 样本外 131），因此
  「FULL 比 NONE 好」是可配对的二分类比较 —— 用 McNemar 精确检验，
  而不是两组独立率的 Fisher。独立检验会把配对信息丢掉，功效和结论都会错。

三项
  1. **配对精确检验（McNemar）**：以「逐例是否判对」（预测标签 == 参考标准）
     为配对二分类结局，FULL vs 其余 7 个配置。分清 b（FULL 对/对方错）与
     c（FULL 错/对方对）—— 只有 b 明显多于 c 才算 FULL 真的赢。
  2. **逐类 one-vs-rest**：INACTIVE / INF_DEFER / ACTIVITY-POSITIVE 三个
     关键类别各做一次配对检验（判对 = 预测与真值同属该类）。
  3. **设计加权的 bootstrap 95% CI**（分层重抽样）：
     每个配置的加权准确率、加权 PPV，以及 FULL 与 NONE 的差值。

输出
  out/201_ablation_tests.txt
  out/table_v10_mcnemar.csv       配对检验表
  out/table_v10_boot_ci.csv       加权 bootstrap CI
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
SEED = 20260920

S_IN, S_OUT = "样本内(v7b)", "样本外(v9)"
VARIANTS = ["FULL", "-R1", "-R2", "-R3", "R1only", "R2only", "R3only", "NONE"]
LBL = {"A": ["INACTIVE", "INF_DEFER", "UNDERDOC"],
       "B": ["ACTIVITY-POSITIVE", "PURELY-NEGATED"]}
CLS_COL = {"A": "labA", "B": "labB"}

L = []


def P(s=""):
    print(s)
    sys.stdout.flush()
    L.append(str(s))


def mcnemar_exact(b, c):
    """双侧精确 McNemar（b/c 为两项不一致计数）。"""
    n = b + c
    if n == 0:
        return np.nan
    return float(stats.binomtest(int(b), int(n), 0.5,
                                 alternative="two-sided").pvalue)


def paired_table(df, col, v_a, v_b, task, cls=None):
    """返回 (b, c)：b = A 对且 B 错，c = A 错且 B 对。"""
    sub = df[df.task == task]
    a = sub[sub.variant == v_a]
    b = sub[sub.variant == v_b]
    if len(a) != len(b):
        raise AssertionError("配对长度不一致：%d vs %d" % (len(a), len(b)))
    if not (a.stay_key.values == b.stay_key.values).all():
        raise AssertionError("配对病例顺序不一致")
    truth = a.ref_std.values
    pa, pb = a[col].values, b[col].values
    if cls is None:
        oa, ob = (pa == truth), (pb == truth)
    else:
        oa, ob = (pa == cls) == (truth == cls), (pb == cls) == (truth == cls)
    return int((oa & ~ob).sum()), int((~oa & ob).sum())


def boot_weighted(df, task, col, variant, cls=None, nrep=2000, strat="stratum"):
    """分层重抽样的设计加权指标 CI。"""
    sub = df[(df.task == task) & (df.variant == variant)].copy()
    sub["acc_"] = (sub[col] == sub.ref_std).astype(int)
    sub["pos_"] = (sub[col] == cls).astype(int) if cls else 0
    sub["tru_"] = (sub.ref_std == cls).astype(int) if cls else 0
    rng = np.random.RandomState(SEED)
    groups = list(sub.groupby(strat))
    accs, ppvs = [], []
    for _ in range(nrep):
        s = pd.concat([g.sample(len(g), replace=True, random_state=rng)
                       for _, g in groups], ignore_index=True)
        w = s["w"].values
        accs.append(float((w * s.acc_.values).sum() / w.sum()))
        if cls:
            den = float(w[s.pos_.values == 1].sum())
            num = float(w[(s.pos_.values == 1) & (s.tru_.values == 1)].sum())
            ppvs.append(num / den if den > 0 else np.nan)
    out = {"acc": np.nanpercentile(accs, [2.5, 97.5])}
    if cls:
        v = np.array([x for x in ppvs if np.isfinite(x)])
        out["ppv"] = np.nanpercentile(v, [2.5, 97.5]) if len(v) else \
            np.array([np.nan, np.nan])
    return out


def wmean(df, task, col, variant, field, cls=None):
    sub = df[(df.task == task) & (df.variant == variant)]
    w = sub["w"].values
    if field == "acc":
        return float((w * (sub[col] == sub.ref_std).astype(int).values).sum()
                     / w.sum())
    pos = (sub[col] == cls).values
    tru = (sub.ref_std == cls).values
    den = float(w[pos].sum())
    return float(w[pos & tru].sum() / den) if den > 0 else np.nan


def main():
    pc = pd.read_csv(os.path.join(OUT, "table_v10_percase.csv"))
    pc["stay_key"] = pc.stay_key.astype(int)

    frames = []
    for sname, gfile in [(S_IN, "adjud_merged.csv"), (S_OUT, "adjud2_merged.csv")]:
        g = pd.read_csv(os.path.join(OUT, gfile))
        g["stay_key"] = g.stay_key.astype(int)
        if "stratum" not in g.columns:          # 171 合并后带 _x/_y 后缀
            g = g.rename(columns={"stratum_x": "stratum"})
        d = pc.merge(g[["stay_key", "task", "ref_std", "w", "stratum"]],
                     on="stay_key", how="inner")
        d = d[d.ref_std.notna()].copy()
        d["sample"] = sname
        frames.append(d)
    D = pd.concat(frames, ignore_index=True)
    # 配对检验要求各配置的行顺序严格一致 → 固定排序
    D = D.sort_values(["sample", "task", "stay_key", "variant"]).reset_index(
        drop=True)
    for sname in [S_IN, S_OUT]:
        for t in ["A", "B"]:
            d = D[(D["sample"] == sname) & (D.task == t)]
            ref_keys = d[d.variant == "FULL"].stay_key.tolist()
            for v in VARIANTS[1:]:
                assert d[d.variant == v].stay_key.tolist() == ref_keys, \
                    "配对顺序不一致：%s/%s/%s" % (sname, t, v)

    P("=" * 100)
    P("v10 ① 消融配对检验 —— FULL 相对其它配置的改善是真信号还是噪声？")
    P("=" * 100)
    P("")
    P("--- 0. 配对样本 ---")
    for sname in [S_IN, S_OUT]:
        for t in ["A", "B"]:
            d = D[(D["sample"] == sname) & (D.task == t)]
            n_case = len(d) // len(VARIANTS)
            P("  %-14s 任务 %s：%d 例 × %d 配置 = %d 行"
              % (sname, t, n_case, len(VARIANTS), len(d)))

    # ------------------------------------------------------------ 1. McNemar
    P("")
    P("=" * 100)
    P("--- 1. 配对精确检验（McNemar，结局＝逐例是否判对）---")
    P("=" * 100)
    rows = []
    for sname in [S_IN, S_OUT]:
        for t in ["A", "B"]:
            d = D[(D["sample"] == sname) & (D.task == t)]
            P("")
            P("  %s ｜ 任务 %s" % (sname, t))
            P("  %-10s %6s %6s %8s %10s" % ("对比", "b(F对)", "c(F错)",
                                            "b/c 比", "P(McNemar)"))
            for v in VARIANTS[1:]:
                b, c = paired_table(d, CLS_COL[t], "FULL", v, t)
                p = mcnemar_exact(b, c)
                rows.append(dict(sample=sname, task=t, pair="FULL vs " + v,
                                 cls="(exact match)", b=b, c=c,
                                 ratio=(b / c if c else np.inf), p=p))
                P("  %-10s %6d %6d %8s %10s"
                  % ("vs " + v, b, c,
                     "%.2f" % (b / c) if c else "  inf",
                     "%.4f" % p if np.isfinite(p) else "  —"))
            for cls in LBL[t][:2]:
                b, c = paired_table(d, CLS_COL[t], "FULL", "NONE", t, cls)
                p = mcnemar_exact(b, c)
                rows.append(dict(sample=sname, task=t, pair="FULL vs NONE",
                                 cls=cls, b=b, c=c,
                                 ratio=(b / c if c else np.inf), p=p))
                P("  %-10s %6d %6d %8s %10s   [%s]"
                  % ("FULL vs NONE", b, c, "%.2f" % (b / c) if c else "  inf",
                     "%.4f" % p if np.isfinite(p) else "  —", cls))
    MC = pd.DataFrame(rows)
    MC.to_csv(os.path.join(OUT, "table_v10_mcnemar.csv"), index=False)

    # ------------------------------------------------------------ 2. 加权 CI
    P("")
    P("=" * 100)
    P("--- 2. 设计加权指标 + bootstrap 95% CI（分层重抽样）---")
    P("=" * 100)
    brows = []
    for sname in [S_IN, S_OUT]:
        for t in ["A", "B"]:
            cls = ("INACTIVE" if t == "A" else "ACTIVITY-POSITIVE")
            P("")
            P("  %s ｜ 任务 %s：加权准确率 / 加权 PPV(%s)" % (sname, t, cls))
            P("  %-10s %16s %22s" % ("配置", "加权准确率", "加权 PPV(%s)" % cls))
            for v in VARIANTS:
                acc = wmean(D[D["sample"] == sname], t, CLS_COL[t], v, "acc")
                ppv = wmean(D[D["sample"] == sname], t, CLS_COL[t], v, "ppv",
                            cls)
                ci = boot_weighted(D[D["sample"] == sname], t, CLS_COL[t], v,
                                   cls)
                P("  %-10s %16s %13s %s"
                  % (v, "%.3f" % acc,
                     "%.3f" % ppv if np.isfinite(ppv) else "  —",
                     "(%.3f-%.3f)" % tuple(ci["ppv"]) if np.isfinite(ppv)
                     else ""))
                brows.append(dict(sample=sname, task=t, variant=v,
                                  acc_w=acc, ppv_cls=cls, ppv_w=ppv,
                                  ppv_lo=ci["ppv"][0], ppv_hi=ci["ppv"][1],
                                  acc_lo=ci["acc"][0], acc_hi=ci["acc"][1]))
    BC = pd.DataFrame(brows)
    BC.to_csv(os.path.join(OUT, "table_v10_boot_ci.csv"), index=False)

    # ------------------------------------------------------------ 3. 差值 CI
    P("")
    P("=" * 100)
    P("--- 3. FULL − NONE 的配对差（加权 PPV，bootstrap）---")
    P("=" * 100)
    drows = []
    for sname in [S_IN, S_OUT]:
        for t in ["A", "B"]:
            cls = ("INACTIVE" if t == "A" else "ACTIVITY-POSITIVE")
            sub = D[(D["sample"] == sname) & (D.task == t)]
            piv = sub.pivot_table(index=["stay_key", "stratum"], columns="variant",
                                  values=CLS_COL[t], aggfunc="first").reset_index()
            piv = piv.merge(sub[["stay_key", "w", "ref_std"]].drop_duplicates(
                "stay_key"), on="stay_key", how="left")
            rng = np.random.RandomState(SEED)
            groups = list(piv.groupby("stratum"))
            diffs = []
            for _ in range(2000):
                s = pd.concat([g.sample(len(g), replace=True, random_state=rng)
                               for _, g in groups], ignore_index=True)
                w = s["w"].values
                tru = (s.ref_std == cls).values

                def pp(v):
                    pos = (s[v] == cls).values
                    den = w[pos].sum()
                    return w[pos & tru].sum() / den if den > 0 else np.nan
                a, b = pp("FULL"), pp("NONE")
                if np.isfinite(a) and np.isfinite(b):
                    diffs.append(a - b)
            v = np.array(diffs)
            lo, hi = (np.nanpercentile(v, [2.5, 97.5]) if len(v)
                      else (np.nan, np.nan))
            a = wmean(D[D["sample"] == sname], t, CLS_COL[t], "FULL", "ppv", cls)
            b = wmean(D[D["sample"] == sname], t, CLS_COL[t], "NONE", "ppv", cls)
            P("  %-14s 任务 %s  PPV(%s)：FULL %.3f − NONE %.3f = %+.3f  "
              "(95%% CI %+.3f ~ %+.3f)"
              % (sname, t, cls, a, b, a - b, lo, hi))
            drows.append(dict(sample=sname, task=t, cls=cls, full=a, none=b,
                              diff=a - b, lo=lo, hi=hi))
    pd.DataFrame(drows).to_csv(os.path.join(OUT, "table_v10_diff_ci.csv"),
                               index=False)

    P("")
    P("saved: out/201_ablation_tests.txt, table_v10_mcnemar.csv,")
    P("       table_v10_boot_ci.csv, table_v10_diff_ci.csv")

    with open(os.path.join(OUT, "201_ablation_tests.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
