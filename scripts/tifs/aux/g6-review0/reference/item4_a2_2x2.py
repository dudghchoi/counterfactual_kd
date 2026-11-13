#!/usr/bin/env python3
"""A14 item 4: the A2 2x2 decomposition (P, P') x (C, C') for BERT SST-2 ATBA, logit KD.

A14 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

P  = battery ATBA P students (poisoned teacher t42; tifs_b_bert_sst2, _s43, _s44)
C  = battery clean-base C students (clean twin t42; same runs)
P' = A2 replicate ATBA P' students (poisoned teacher t43; tifs_rep_bert_atba_t43_s42..44)
C' = A2 replicate clean-base C' students (clean teacher t43; same runs)
All on the same 912 ATBA-triggered SST-2 inputs (prefix insert, deterministic), student seeds
42-44, corpus seed 42. Four cells k(X, Y) = X - Y; the missing one is P'(t43) - C(t42).

Because a hit rate is a mean, the four cells satisfy, per seed and input,
  k(P,C) - k(P,C') = k(P',C) - k(P',C') = C' - C      (reference swap)
  k(P,C) - k(P',C) = k(P,C') - k(P',C') = P - P'      (poisoned-build swap)
  k(P,C) - k(P',C') = (C' - C) + (P - P')             (the A2 build difference)
exactly: the 2x2 has no interaction term and the split does not depend on the order.

Intervals (frozen rules): every cell and every component is a contrast of per-seed paired
vectors, read with the frozen seed-pooled rule (addendum.seed_pooled + t_contrast: SE^2 =
s2_pool/r + v, BERT SST-2 logit pool and its Satterthwaite df), with the A3.1 own-variance
reading (s2_a/r + v, df r - 1) beside it, and the A2 'build_diff' style (two cells taken as
independent: sqrt(SE_1^2 + SE_2^2), pool df) for the components. Official bf16 vectors are the
primary basis; the CPU fp32 re-scores (cross_hits_bert_sst2.json and hits/fill_bert_sst2.json)
are a cross-check.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402

import common as K  # noqa: E402


def vectors_official(st):
    g = st.groups[("bert", "sst2", "logit", "atba")]
    rep = K.AD.load_replicate(K.RUNS)
    seeds = sorted(set(g) & set(rep))
    return seeds, {"P": {s: g[s]["P"] for s in seeds}, "C": {s: g[s]["C"] for s in seeds},
                   "P'": {s: rep[s]["P"] for s in seeds}, "C'": {s: rep[s]["C"] for s in seeds}}


def vectors_cpu(out):
    X = K.load_cross("sst2")["hits"]["atba"]
    F = K.load_json(os.path.join(K.hits_dir(out), "fill_bert_sst2.json"))["hits"]["atba"]
    seeds = (42, 43, 44)
    pt = K.P_TAG["sst2"]["atba"]
    return list(seeds), {
        "P": {s: X[f"logit/{pt}/poisoned/s{s}"] for s in seeds},
        "C": {s: X[f"logit/clean-base/clean/s{s}"] for s in seeds},
        "P'": {s: F[f"tifs_rep_bert_atba_t43_s{s}|logit/{pt}/poisoned/s{s}"] for s in seeds},
        "C'": {s: F[f"tifs_rep_bert_atba_t43_s{s}|logit/clean-base/clean/s{s}"] for s in seeds}}


CELLS = [("k(P,C)", "P", "C", "registered cell: t42 poisoned vs t42 clean twin (frozen primary)"),
         ("k(P,C')", "P", "C'", "t42 poisoned vs t43 clean reference"),
         ("k(P',C')", "P'", "C'", "A2 replicate: t43 poisoned vs t43 clean twin"),
         ("k(P',C)", "P'", "C", "missing cell: t43 poisoned vs t42 clean twin")]
COMPONENTS = [("reference swap C'-C", [("C'", 1), ("C", -1)],
               ("k(P,C)", "k(P,C')"), "= k(P,C) - k(P,C') = k(P',C) - k(P',C')"),
              ("poisoned-build swap P-P'", [("P", 1), ("P'", -1)],
               ("k(P,C')", "k(P',C')"), "= k(P,C') - k(P',C') = k(P,C) - k(P',C)"),
              ("A2 build difference k-k'", [("P", 1), ("C", -1), ("P'", -1), ("C'", 1)],
               ("k(P,C)", "k(P',C')"), "= k(P,C) - k(P',C') = sum of the two swaps")]


def lin(vecs, coefs, s):
    n = len(vecs["P"][s])
    return [sum(c * vecs[name][s][i] for name, c in coefs) for i in range(n)]


def table(st, basis, seeds, vecs):
    _, s2p, dfp = K.pool(st, "sst2")
    rows, cell = [], {}
    for name, a, b, note in CELLS:
        d = {s: K.diff(vecs[a][s], vecs[b][s]) for s in seeds}
        res = K.pooled_and_own(d, s2p, dfp)
        cell[name] = res
        pc42 = K.paired_contrast(vecs[a][42], vecs[b][42], K.TOLS)
        pooled, own = res["pooled"], res["own"]
        rows.append({"basis": basis, "kind": "cell", "quantity": name, "definition": note,
                     "seeds": ",".join(map(str, seeds)), "rate_a": K.mean([K.mean(vecs[a][s]) for s in seeds]),
                     "rate_b": K.mean([K.mean(vecs[b][s]) for s in seeds]),
                     "k": res["k"], **{f"k_s{s}": x for s, x in zip(seeds, res["ks"])},
                     "se": res["se_pooled"], "df": dfp, "lo": pooled["lo"], "hi": pooled["hi"],
                     **K.verdicts(pooled), **{f"p_pos@{t:g}": pooled[f"p_pos@{t:g}"] for t in K.TOLS},
                     **{f"p_up@{t:g}": pooled[f"p_up@{t:g}"] for t in K.TOLS},
                     "own_lo": own["lo"], "own_hi": own["hi"], **K.verdicts(own, "own_"),
                     "s42_tango_k": pc42["k"], "s42_tango_lo": pc42["lo"], "s42_tango_hi": pc42["hi"],
                     "pool_s": s2p ** 0.5, "pool_df": dfp})
    total = None
    for name, coefs, (c1, c2), note in COMPONENTS:
        d = {s: lin(vecs, coefs, s) for s in seeds}
        res = K.pooled_and_own(d, s2p, dfp)
        pooled, own = res["pooled"], res["own"]
        ind_se = (cell[c1]["se_pooled"] ** 2 + cell[c2]["se_pooled"] ** 2) ** 0.5
        ind = K.t_contrast(res["k"], ind_se, dfp, K.TOLS)
        row = {"basis": basis, "kind": "component", "quantity": name, "definition": note,
               "seeds": ",".join(map(str, seeds)), "k": res["k"],
               **{f"k_s{s}": x for s, x in zip(seeds, res["ks"])},
               "se": res["se_pooled"], "df": dfp, "lo": pooled["lo"], "hi": pooled["hi"],
               **K.verdicts(pooled), "own_lo": own["lo"], "own_hi": own["hi"], **K.verdicts(own, "own_"),
               "independent_cells_se": ind_se, "independent_cells_lo": ind["lo"],
               "independent_cells_hi": ind["hi"], **K.verdicts(ind, "independent_cells_"),
               "identity_residual": res["k"] - (cell[c1]["k"] - cell[c2]["k"]),
               "pool_s": s2p ** 0.5, "pool_df": dfp}
        if name.startswith("A2"):
            total = res["k"]
        rows.append(row)
    for r in rows:
        if r["kind"] == "component" and not r["quantity"].startswith("A2") and total:
            r["share_of_build_difference"] = r["k"] / total
    return rows


def main():
    out = K.out_dir()
    st = K.state()
    seeds, vo = vectors_official(st)
    rows = table(st, "official Stage 2, GPU bf16", seeds, vo)
    try:
        cs, vc = vectors_cpu(out)
        cpu_ok = True
    except (FileNotFoundError, KeyError) as e:
        print(f"item 4: CPU basis left out ({e!r}); run score.py fill first")
        cpu_ok = False
    if cpu_ok:
        rows += table(st, "CPU fp32 re-score", cs, vc)
        for name in ("P", "C", "P'", "C'"):
            for s in seeds:
                rows.append({"basis": "official vs CPU fp32", "kind": "precision check",
                             "quantity": f"{name} s{s}", "seeds": str(s),
                             "n_disagree": sum(a != b for a, b in zip(vo[name][s], vc[name][s])),
                             "n": len(vo[name][s])})
    K.write_csv(rows, os.path.join(out, "a2_2x2.csv"), """
        A2 2x2 for BERT SST-2 ATBA logit KD, 912 ATBA-triggered inputs, student seeds 42-44:
        P/C = battery students of the t42 poisoned / clean twin teachers, P'/C' = A2 replicate
        students of the t43 poisoned / clean teachers. kind=cell: k(X,Y) = mean_s (X_s - Y_s);
        kind=component: the reference swap C'-C, the poisoned-build swap P-P' and the A2 build
        difference k(P,C) - k(P',C') (exactly their sum; identity_residual = component minus the
        difference of the two cells it equals). lo/hi/verdict = frozen seed-pooled rule (SE^2 =
        s2_pool/r + v, BERT SST-2 logit pool, Satterthwaite df); own_* = own seed variance, df r-1;
        independent_cells_* = the A2 build_diff style sqrt(SE_1^2 + SE_2^2) at pool df;
        s42_tango_* = seed-42 Tango interval. basis = official bf16 vectors or the CPU fp32
        re-scores; 'precision check' rows count inputs where the two scorings disagree.""")
    print(f"item 4: {len(rows)} rows (CPU basis: {cpu_ok})")


if __name__ == "__main__":
    main()
