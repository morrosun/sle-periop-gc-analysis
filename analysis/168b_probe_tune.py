# -*- coding: utf-8 -*-
"""
168b_probe_tune.py
==================
调参用探针：只看金标准里**被人类判为「有活动度陈述」的那 9 例**
（任务 A 的 ref_std ∈ {INACTIVE, INF_DEFER}），看 v8 到底在哪一步把它们丢了。

对每例打印
  1. 人类的证据原句（R1 / R2）
  2. v8 保留下来的不活动 / 暂缓片段
  3. **所有**未过滤的不活动线索命中（含被哪条规则、什么理由剔除）
  4. v8 的连续变量取值

输出 out/_probe_tune.txt
"""
import importlib.util
import os
import sys

import pandas as pd
import psycopg2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
DATA = os.path.join(ROOT, "data")
L = []


def P(s=""):
    print(s)
    sys.stdout.flush()
    L.append(str(s))


def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


M = load(os.path.join(ROOT, "scripts", "163_extract_inactivity.py"), "s163")
V8 = load(os.path.join(ROOT, "scripts", "168_activity_rule_engine.py"), "s168")


def main():
    gold = pd.read_csv(os.path.join(OUT, "adjud_merged.csv"))
    gold["stay_key"] = gold.stay_key.astype(int)
    rt = pd.read_csv(os.path.join(OUT, "adjud_ratings_long.csv"))
    q = (rt[rt.task == "A"].pivot_table(index="case_id", columns="rater",
                                        values="evidence_quote", aggfunc="first"))
    ad_full = pd.read_csv(os.path.join(DATA, "gc_activity_density.csv"))
    ad = ad_full.set_index("stay_key")

    A = gold[gold.task == "A"].merge(
        ad_full[["stay_key", "v8_label", "v8_inact_n", "v8_defer_n",
                 "v8_act_pos"]], on="stay_key", how="left")
    tgt = A[A.ref_std.isin(["INACTIVE", "INF_DEFER"])].copy()
    P("金标准任务 A 中「有活动度陈述」的病例 n = %d" % len(tgt))
    P("其中 v8 判对 %d 例" % int((tgt.v8_label == tgt.ref_std).sum()))

    keys = sorted(tgt.stay_key.unique().tolist())
    cc = psycopg2.connect(host=os.environ.get("MIMIC_DB_HOST", "localhost"),
                          port=int(os.environ.get("MIMIC_DB_PORT", "5432")),
                          user=os.environ.get("MIMIC_DB_USER", "postgres"),
                          password=os.environ.get("MIMIC_DB_PASSWORD", ""),
                          dbname="mimiciv")
    parts = []
    for i in range(0, len(keys), 300):
        ch = keys[i:i + 300]
        parts.append(pd.read_sql(
            "SELECT hadm_id, note_seq, text FROM mimiciv_note.discharge "
            "WHERE hadm_id IN (%s)" % ",".join(map(str, ch)), cc))
    cc.close()
    nt = pd.concat(parts, ignore_index=True)
    nt = (nt.sort_values(["hadm_id", "note_seq"])
            .drop_duplicates("hadm_id", keep="last"))
    T = {int(r.hadm_id): (r.text or "") for r in nt.itertuples()}

    import re
    for r in tgt.sort_values(["ref_std", "case_id"]).itertuples():
        t = T.get(int(r.stay_key), "")
        secs = M.split_sections(t)
        hc = V8.norm(M.sec(secs, "brief hospital course", "hospital course"))
        dd = V8.norm(M.sec(secs, "discharge diagnosis"))
        moa = V8.norm(M.sec(secs, "medications on admission"))
        narr = hc + " " + dd if len(hc) + len(dd) >= 200 else V8.norm(t)
        blob = narr + " " + moa
        P("")
        P("=" * 100)
        P("%s | stay=%d %s | 人类=%s | v8=%s  %s" % (
            r.case_id, r.stay_key, r.disease, r.ref_std, r.v8_label,
            "✔" if r.v8_label == r.ref_std else "✘"))
        P("  人类证据  R1: %s" % (q.loc[r.case_id, "R1"] if r.case_id in q.index else ""))
        P("  人类证据  R2: %s" % (q.loc[r.case_id, "R2"] if r.case_id in q.index else ""))
        if r.stay_key in ad.index:
            a = ad.loc[r.stay_key]
            P("  v8 变量: inact_n=%s defer_n=%s act_pos=%s info_n=%s aid_doc=%.3f "
              "aid_broad=%.3f" % (a.v8_inact_n, a.v8_defer_n, a.v8_act_pos,
                                  a.info_n, a.aid_doc, a.aid_broad))
        # ---- 所有未过 domain 的不活动线索命中 + 剔除原因
        P("  【不活动线索逐条】")
        found = False
        for k, tier, rx in V8.INACT_V8_RX:
            for m in rx.finditer(blob):
                found = True
                s, e = m.start(), m.end()
                reason = "KEEP"
                if V8.BLACKLIST.search(blob[max(0, s - 40):e + 40]):
                    reason = "blacklist"
                elif not V8.invert_ok(blob, s):
                    reason = "invert"
                elif V8.hypoth_hit(blob, s):
                    reason = "hypoth"
                elif not V8.domain_ok(blob, s, e, tight=(tier == V8.WEAK)):
                    reason = "domain"
                P("    [%s/%s] %-9s %s" % (
                    k, tier, reason,
                    blob[max(0, s - 60):e + 90].replace("\n", " ")))
        if not found:
            P("    （词表完全没命中）")
        # ---- 暂缓（用 v8 自己的规则）
        P("  【暂缓线索（v8 句级有序邻近规则）】")
        dk = V8.eval_defer(blob)
        if not dk["defer_kept"]:
            P("    （未命中）")
        for i, frag in dk["defer_kept"]:
            P("    [defer] %s" % frag[:220])

    txt = "\n".join(L)
    with open(os.path.join(OUT, "_probe_tune.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    P("")
    P("saved: out/_probe_tune.txt")


if __name__ == "__main__":
    main()
