# -*- coding: utf-8 -*-
"""
168a_probe_v8.py
================
证据驱动的规则设计第一步：**先看 v7 到底在什么文本上判错了**。

对 v7b 盲法核验的 98 例（任务 A）+ 50 例（任务 B）对象，回查原文，
逐例打印：
  - 人类参考标签（ref_std）
  - v7 算法标签（algo_label / algo_B）
  - v7 命中的**每一条词表规则**（规则名 + 原文片段 ±60 字符 + 是否被否定）

输出 out/_probe_v8.txt（供人读，不参与建模）

用法
  python scripts/168a_probe_v8.py
"""
import importlib.util
import os
import sys

import pandas as pd
import psycopg2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
DATA = os.path.join(ROOT, "data")


def load_163():
    p = os.path.join(ROOT, "scripts", "163_extract_inactivity.py")
    spec = importlib.util.spec_from_file_location("s163", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = load_163()
L = []


def P(s=""):
    print(s)
    sys.stdout.flush()
    L.append(str(s))


def ctx(text, a, b, pad=70):
    s = max(0, a - pad)
    e = min(len(text), b + pad)
    frag = text[s:e].replace("\n", " ")
    return "…" + frag + "…"


def main():
    m = pd.read_csv(os.path.join(OUT, "adjud_merged.csv"))
    m["stay_key"] = m.stay_key.astype(int)
    keys = sorted(m.stay_key.unique().tolist())
    P("audited stays = %d (task A %d / task B %d, 去重后 %d)" % (
        len(m), int((m.task == "A").sum()), int((m.task == "B").sum()), len(keys)))

    cc = psycopg2.connect(host=os.environ.get("MIMIC_DB_HOST", "localhost"),
                          port=int(os.environ.get("MIMIC_DB_PORT", "5432")),
                          user=os.environ.get("MIMIC_DB_USER", "postgres"),
                          password=os.environ.get("MIMIC_DB_PASSWORD", ""),
                          dbname="mimiciv")
    parts = []
    B = 300
    for i in range(0, len(keys), B):
        ch = keys[i:i + B]
        q = ("SELECT hadm_id, note_seq, text FROM mimiciv_note.discharge "
             "WHERE hadm_id IN (%s)" % ",".join(map(str, ch)))
        parts.append(pd.read_sql(q, cc))
    cc.close()
    notes = pd.concat(parts, ignore_index=True)
    notes = (notes.sort_values(["hadm_id", "note_seq"])
                  .drop_duplicates("hadm_id", keep="last"))
    T = {int(r.hadm_id): (r.text or "") for r in notes.itertuples()}
    P("notes retrieved = %d" % len(T))

    # ---------------- 任务 A：INACTIVE / INF_DEFER 的触发解剖
    A = m[m.task == "A"].copy()
    P("")
    P("=" * 100)
    P("任务 A —— 三通道标签的触发解剖（只列 algo != ref_std 的误判，以及全部 INACTIVE 判定）")
    P("=" * 100)
    misA = A[A.algo_label != A.ref_std]
    P("误判 n = %d / %d" % (len(misA), len(A)))
    for r in misA.sort_values(["algo_label", "case_id"]).itertuples():
        t = T.get(int(r.stay_key), "")
        secs = M.split_sections(t)
        hc = M.sec(secs, "brief hospital course", "hospital course")
        dd = M.sec(secs, "discharge diagnosis")
        blobN = hc + "\n" + dd
        P("")
        P("--- %s | stay=%d %s | algo=%s -> ref=%s | %s" % (
            r.case_id, r.stay_key, r.disease, r.algo_label, r.ref_std, r.stratum))
        if r.algo_label == "INACTIVE":
            for k, v in M.INACT.items():
                for p in v:
                    for mm in __import__("re").compile(p, __import__("re").I).finditer(blobN):
                        P("   [%s] %s" % (k, ctx(blobN, mm.start(), mm.end())))
        if r.algo_label == "INF_DEFER":
            for i, rx in enumerate(M.DEFER_RX):
                for mm in rx.finditer(blobN + "\n" + M.sec(secs, "medications on admission")):
                    P("   [defer#%d] %s" % (i, ctx(blobN, mm.start(), mm.end())))

    # ---------------- 任务 B：ACTIVITY-POSITIVE 的触发解剖
    Bt = m[m.task == "B"].copy()
    P("")
    P("=" * 100)
    P("任务 B —— 活动阳性的触发解剖（只列 algo_B != ref_std 的 27 例假阳性）")
    P("=" * 100)
    misB = Bt[Bt.algo_B != Bt.ref_std]
    P("误判 n = %d / %d" % (len(misB), len(Bt)))
    for r in misB.sort_values("case_id").itertuples():
        t = T.get(int(r.stay_key), "")
        secs = M.split_sections(t)
        hc = M.sec(secs, "brief hospital course", "hospital course")
        dd = M.sec(secs, "discharge diagnosis")
        blobN = hc + "\n" + dd
        P("")
        P("--- %s | stay=%d %s | algo=%s -> ref=%s | spec_pos=%d spec_neg=%d" % (
            r.case_id, r.stay_key, r.disease, r.algo_B, r.ref_std,
            r.spec_pos, r.spec_neg))
        import re
        for k in M.SPEC:
            for p in M.IND[k]:
                for mm in re.compile(p, re.I).finditer(blobN):
                    neg = M.negated(blobN, mm.start())
                    P("   [%s%s] %s" % (k, " NEG" if neg else " POS",
                                        ctx(blobN, mm.start(), mm.end())))

    # ---------------- 任务 A 中全部 INACTIVE 判定（含正确与错误）汇总
    P("")
    P("=" * 100)
    P("任务 A 「INACTIVE」判定的正确率分解（按 inact_kinds 主触发词）")
    P("=" * 100)
    ia = A[A.algo_label == "INACTIVE"]
    P("algo INACTIVE n = %d ; 其中 ref_std==INACTIVE n = %d (PPV=%.1f%%)" % (
        len(ia), int((ia.ref_std == "INACTIVE").sum()),
        100 * (ia.ref_std == "INACTIVE").mean() if len(ia) else 0))

    txt = "\n".join(L)
    with open(os.path.join(OUT, "_probe_v8.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    P("")
    P("saved: out/_probe_v8.txt")


if __name__ == "__main__":
    main()
