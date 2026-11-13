#!/usr/bin/env python3
"""A11 item 6 (전환 빈도): sign flips and verdict flips of the teacher-swap
effect across student seeds and across teacher builds. Post hoc auxiliary
table (docs/tifs_addendum.md A11); it does not change any primary verdict.

A comparison is a pair of readings of the same estimand that differ only in
the student seed, or only in the teacher build (BadNets: t42 grid / battery vs
the A/A builds t43, t44; ATBA SST-2: t42 vs the A2 replicate t43), or only in
the clean reference build (ATBA: C(t42) vs C'(t43) vs C'(t44), item 4).

* sign flip: the two k have opposite signs (both nonzero); a pair with a k of
  exactly 0 is counted apart (zero_involved).
* verdict flip at delta: the two verdicts differ (4 levels).
* decisive flip at delta: both verdicts are decisive (EQUIVALENT, POSITIVE or
  NEGATIVE) and differ.
Bases: 'Tango' = each member's frozen paired_contrast (input noise only);
'transported' = each member read alone with the frozen one-seed formula
k +- t_df sqrt(SE_Tango^2 + s2_pool) (pool as addendum.State.pool_for; the EP
SST-2 structural null keeps Tango, as in the frozen rule); 'seed-pooled' = the
frozen seed-pooled formula (primary rows, the A2 replicate row, and the
P_s - C'_s(t43) reference reading of item 4).

Outputs: flips_cells.csv (one row per unit x basis), flips_pairs.csv (one row
per pair x basis), flips_battery_by_seed.csv (battery decision per student
seed, Tango = frozen battery_by_seed and transported), flips_summary.csv.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # no __pycache__ inside the repository

import itertools  # noqa: E402
from collections import defaultdict  # noqa: E402

from g2common import (A, AD, DECISIVE, RUNS, TOLS, VKEYS, cli, hidden_groups, paired_contrast, sign,  # noqa: E402
                      state, t_contrast, transported, write)


def pair_row(source, basis, unit, la, lb, ka, kb, va, vb):
    sa, sb = sign(ka), sign(kb)
    row = {"source": source, "basis": basis, **unit, "member_a": la, "member_b": lb, "k_a": ka, "k_b": kb,
           "sign_a": sa, "sign_b": sb, "zero_involved": sa == 0 or sb == 0,
           "sign_flip": sa * sb < 0}
    for t in TOLS:
        a, b = va.get(f"verdict@{t:g}"), vb.get(f"verdict@{t:g}")
        row[f"verdict_a@{t:g}"], row[f"verdict_b@{t:g}"] = a, b
        row[f"verdict_flip@{t:g}"] = a != b
        row[f"decisive_flip@{t:g}"] = a != b and a in DECISIVE and b in DECISIVE
    return row


def unit_row(source, basis, unit, members, ks, verdicts, pairs, extra=None):
    row = {"source": source, "basis": basis, **unit, "members": ",".join(members),
           **{f"k_{m}": k for m, k in zip(members, ks)},
           "signs": "".join("+" if sign(k) > 0 else "-" if sign(k) < 0 else "0" for k in ks),
           "n_pairs": len(pairs), "n_sign_flip_pairs": sum(p["sign_flip"] for p in pairs),
           "any_sign_flip": any(p["sign_flip"] for p in pairs)}
    for t in TOLS:
        row[f"verdicts@{t:g}"] = "/".join(v[f"verdict@{t:g}"] for v in verdicts)
        row[f"n_verdict_flip_pairs@{t:g}"] = sum(p[f"verdict_flip@{t:g}"] for p in pairs)
        row[f"any_verdict_flip@{t:g}"] = any(p[f"verdict_flip@{t:g}"] for p in pairs)
        row[f"any_decisive_flip@{t:g}"] = any(p[f"decisive_flip@{t:g}"] for p in pairs)
    row.update(extra or {})
    return row


def run_unit(source, basis, unit, members, ks, verdicts, out_units, out_pairs, extra=None):
    pairs = [pair_row(source, basis, unit, members[i], members[j], ks[i], ks[j], verdicts[i], verdicts[j])
             for i, j in itertools.combinations(range(len(members)), 2)]
    out_pairs += pairs
    out_units.append(unit_row(source, basis, unit, members, ks, verdicts, pairs, extra))


def two_bases(source, unit, members, pcs, pool, units, pairs, extra=None, structural=False):
    """Tango and transported readings of the same members (Tango kept for a structural null)."""
    run_unit(source, "Tango", unit, members, [pc["k"] for pc in pcs], pcs, units, pairs, extra)
    if pool is None:
        return
    trs = pcs if structural else [transported(pc, pool) for pc in pcs]
    run_unit(source, "transported", unit, members, [tr["k"] for tr in trs], trs, units, pairs,
             {**(extra or {}), "transported_pool_s": pool["s_between"], "transported_pool_df": pool["df"],
              "transported_note": "structural null: Tango kept (frozen rule)" if structural else ""})


def main():
    cli(__doc__.split("\n\n")[0])
    st = state()
    cells = st.cells
    units, pairs = [], []

    # ── across student seeds: frozen battery cells ──
    src = "across student seeds: battery cells (teacher t42), P_s - C_s"
    for (fam, ds, kd, atk), by_seed in sorted(st.groups.items()):
        if len(by_seed) < 2:
            continue
        seeds = sorted(by_seed)
        pcs = [paired_contrast(by_seed[s]["P"], by_seed[s]["C"], TOLS) for s in seeds]
        pr = st.prim.get((fam, ds, kd, atk), {})
        pk, pool = st.pool_for(fam, ds, kd)
        two_bases(src, {"fam": fam, "dataset": ds, "kd": kd, "attack": atk, "build": "t42"},
                  [f"s{s}" for s in seeds], pcs, pool, units, pairs,
                  {"primary_method": pr.get("method"), "primary_k": pr.get("k"),
                   **{f"primary_{v}": pr.get(v) for v in VKEYS}},
                  structural=(atk, ds) in st.structural)
    # A2 replicate (teacher build t43, ATBA SST-2)
    rep = AD.load_replicate(RUNS)
    rr = AD.replicate_table(st)
    if len(rep) >= 2:
        seeds = sorted(rep)
        pcs = [paired_contrast(rep[s]["P"], rep[s]["C"], TOLS) for s in seeds]
        _, pool = st.pool_for("bert", "sst2", "logit")
        two_bases("across student seeds: A2 replicate build t43 (ATBA SST-2), P'_s - C'_s",
                  {"fam": "bert", "dataset": "sst2", "kd": "logit", "attack": "atba", "build": "t43"},
                  [f"s{s}" for s in seeds], pcs, pool, units, pairs,
                  {"primary_method": rr[0]["method"] if rr else None, "primary_k": rr[0]["k"] if rr else None,
                   **{f"primary_{v}": (rr[0][v] if rr else None) for v in VKEYS}})
    # A8 hidden-state KD, admitted seeds (outside the global Holm family); a one-seed
    # hidden cell borrows the BERT SST-2 logit pool (A8), which pool_for returns
    hg, _ = hidden_groups()
    for (fam, ds, kd, atk), by_seed in sorted(hg.items()):
        if len(by_seed) < 2:
            continue
        seeds = sorted(by_seed)
        pcs = [paired_contrast(by_seed[s]["P"], by_seed[s]["C"], TOLS) for s in seeds]
        pk, pool = st.pool_for(fam, ds, kd)
        two_bases("across student seeds: A8 hidden-state KD, admitted seeds (outside the global family)",
                  {"fam": fam, "dataset": ds, "kd": kd, "attack": atk, "build": "t42"},
                  [f"s{s}" for s in seeds], pcs, pool, units, pairs,
                  {"transported_variance_from": "/".join(pk) if pk else None})

    # ── across student seeds: battery decisions ──
    # Tango: the frozen battery_table(audit_table) (= battery_by_seed.csv). Transported:
    # the same attacks, each seed's k read with the frozen one-seed formula, decided by
    # the frozen battery_table (Holm within the battery / intersection-union).
    audit = A.audit_table(cells, 200)   # R only feeds the k* bootstrap, unused by battery_table
    bys = {"Tango": A.battery_table(audit)}
    trows = []
    for r in audit:
        if r["kd"] == "ce":
            continue
        c = st.groups.get((r["fam"], r["dataset"], r["kd"], r["attack"]), {}).get(r["seed"])
        pc = paired_contrast(c["P"], c["C"], TOLS)
        pk, pool = st.pool_for(r["fam"], r["dataset"], r["kd"])
        res = pc if (r["attack"], r["dataset"]) in st.structural or pool is None else transported(pc, pool)
        trows.append({**{k: r[k] for k in ("fam", "dataset", "attack", "kd", "seed", "teacher_qualified", "raw_asr")},
                      "k": res["k"], "k_hi": res["hi"],
                      **{f"{p}@{t:g}": res[f"{p}@{t:g}"] for p in ("p_pos", "p_up", "p_tost") for t in TOLS}})
    bys["transported"] = A.battery_table(trows)
    brows = []
    for basis, rows_b in bys.items():
        bgroups = defaultdict(dict)
        for b in rows_b:
            bgroups[(b["fam"], b["dataset"], b["kd"])][b["seed"]] = b
        for (fam, ds, kd), by_seed in sorted(bgroups.items()):
            seeds = sorted(by_seed)
            row = {"basis": basis, "fam": fam, "dataset": ds, "kd": kd, "seeds": ",".join(map(str, seeds)),
                   "n_pairs": len(seeds) * (len(seeds) - 1) // 2,
                   "attacks_by_seed": " | ".join(f"s{s}:{by_seed[s]['attacks']}" for s in seeds)}
            for t in TOLS:
                ds_ = [by_seed[s][f"decision@{t:g}"] for s in seeds]
                ts_ = [frozenset(x for x in (by_seed[s].get(f"transfers@{t:g}") or "").split(",") if x) for s in seeds]
                row[f"decisions@{t:g}"] = "/".join(ds_)
                row[f"transfers@{t:g}"] = "/".join(by_seed[s].get(f"transfers@{t:g}", "") or "-" for s in seeds)
                row[f"n_decision_flip_pairs@{t:g}"] = sum(a != b for a, b in itertools.combinations(ds_, 2))
                row[f"any_decision_flip@{t:g}"] = len(set(ds_)) > 1
                # both seeds TRANSFERS, but Holm flags a different set of attacks
                row[f"n_transfer_set_flip_pairs@{t:g}"] = sum(
                    da == db == "TRANSFERS" and sa != sb
                    for (da, sa), (db, sb) in itertools.combinations(list(zip(ds_, ts_)), 2))
            brows.append(row)

    # ── across teacher builds: BadNets (student seed 42) ──
    builds = defaultdict(dict)
    for c in cells:
        if c["single"] or c["kd"] != "logit" or c["attack"] != "badnets" or c["student_seed"] != 42:
            continue
        if (c["block"] == "G" and c["teacher_seed"] == 42) or c["block"] == "AA":
            builds[(c["fam"], c["dataset"], c["corpus"])][c["teacher_seed"]] = c
    for (fam, ds, corpus), bt in sorted(builds.items()):
        if len(bt) < 2:
            continue
        ts = sorted(bt)
        pcs = [paired_contrast(bt[t]["P"], bt[t]["C"], TOLS) for t in ts]
        pk, pool = st.pool_for(fam, ds, "logit")
        two_bases("across teacher builds: BadNets k = P - C, student seed 42",
                  {"fam": fam, "dataset": ds, "kd": "logit", "attack": "badnets", "corpus": corpus},
                  [f"t{t}" for t in ts], pcs, pool, units, pairs,
                  {"runs": ",".join(bt[t]["run"] for t in ts)})
    # ATBA SST-2: build t42 vs A2 build t43, per student seed and seed-pooled
    g = st.groups.get(("bert", "sst2", "logit", "atba"), {})
    _, pool_bs = st.pool_for("bert", "sst2", "logit")
    for s in sorted(rep):
        if s not in g:
            continue
        a = paired_contrast(g[s]["P"], g[s]["C"], TOLS)
        b = paired_contrast(rep[s]["P"], rep[s]["C"], TOLS)
        two_bases("across teacher builds: ATBA SST-2 k per student seed, t42 vs A2 t43",
                  {"fam": "bert", "dataset": "sst2", "kd": "logit", "attack": "atba", "seed": s},
                  ["t42", "t43"], [a, b], pool_bs, units, pairs)
    pr = st.prim.get(("bert", "sst2", "logit", "atba"))
    if rr and pr:
        run_unit("across teacher builds: ATBA SST-2 seed-pooled, primary (t42) vs A2 replicate (t43)", "seed-pooled",
                 {"fam": "bert", "dataset": "sst2", "kd": "logit", "attack": "atba", "seed": "42,43,44"},
                 ["t42", "t43"], [pr["k"], rr[0]["k"]], [pr, rr[0]], units, pairs,
                 {"note": "t42: frozen primary (seed-pooled); t43: A2 replicate (seed-pooled, BERT SST-2 logit pool)"})

    # ── across clean-reference builds: ATBA P(s42) vs C(t42) / C'(t43) / C'(t44) ──
    for ds in ("sst2", "agnews"):
        c42 = st.groups.get(("bert", ds, "logit", "atba"), {}).get(42)
        if c42 is None:
            continue
        refs = {42: c42["C"]}
        for c in cells:
            if (c["block"] == "AA" and c["fam"] == "bert" and c["dataset"] == ds and c["corpus"] == "clean"
                    and not c["single"] and f"atba_phrase_{ds}" in c["controls"]):
                refs[c["teacher_seed"]] = c["controls"][f"atba_phrase_{ds}"]["c"]
        ts = sorted(refs)
        pcs = [paired_contrast(c42["P"], refs[t], TOLS) for t in ts]
        _, pool = st.pool_for("bert", ds, "logit")
        two_bases("across clean-reference builds: ATBA P(s42) - C_ref",
                  {"fam": "bert", "dataset": ds, "kd": "logit", "attack": "atba", "seed": 42},
                  [f"t{t}" for t in ts], pcs, pool, units, pairs)
    # seed-pooled: P_s - C_s(t42) (frozen primary) vs P_s - C'_s(t43) (A2 replicate C'), SST-2
    g = st.groups.get(("bert", "sst2", "logit", "atba"), {})
    if pr and len(rep) >= 2:
        d = {s: [p - q for p, q in zip(g[s]["P"], v["C"])] for s, v in rep.items() if s in g}
        if len(d) >= 2:
            s2p, dfp = pool_bs["s_between"] ** 2, pool_bs["df"]
            k, se, r, _, _ = AD.seed_pooled(d, s2p, dfp)
            alt = t_contrast(k, se, dfp, TOLS)
            run_unit("across clean-reference builds: ATBA SST-2 seed-pooled, P - C(t42) vs P - C'(t43)", "seed-pooled",
                     {"fam": "bert", "dataset": "sst2", "kd": "logit", "attack": "atba", "seed": ",".join(map(str, sorted(d)))},
                     ["t42", "t43"], [pr["k"], alt["k"]], [pr, alt], units, pairs,
                     {"note": "t42: frozen primary; t43: P_s (battery) - C'_s (A2 replicate clean students), frozen seed-pooled formula"})

    # ── summary ──
    summ = []
    by_src = defaultdict(list)
    for p in pairs:
        by_src[(p["source"], p["basis"])].append(p)
    u_by = defaultdict(list)
    for u in units:
        u_by[(u["source"], u["basis"])].append(u)
    for (source, basis), ps in by_src.items():
        us = u_by[(source, basis)]
        flipped = [max(abs(p["k_a"]), abs(p["k_b"])) for p in ps if p["sign_flip"]]
        row = {"source": source, "basis": basis, "n_units": len(us), "n_pairs": len(ps),
               "n_sign_flip_pairs": sum(p["sign_flip"] for p in ps),
               "n_zero_involved_pairs": sum(p["zero_involved"] for p in ps),
               "n_units_any_sign_flip": sum(u["any_sign_flip"] for u in us),
               "max_abs_k_in_sign_flip_pairs": max(flipped) if flipped else None}
        for t in TOLS:
            row[f"n_verdict_flip_pairs@{t:g}"] = sum(p[f"verdict_flip@{t:g}"] for p in ps)
            row[f"n_decisive_flip_pairs@{t:g}"] = sum(p[f"decisive_flip@{t:g}"] for p in ps)
            row[f"n_units_any_verdict_flip@{t:g}"] = sum(u[f"any_verdict_flip@{t:g}"] for u in us)
            row[f"n_units_any_decisive_flip@{t:g}"] = sum(u[f"any_decisive_flip@{t:g}"] for u in us)
        summ.append(row)
    for basis in bys:
        multi = [b for b in brows if b["basis"] == basis and b["n_pairs"] > 0]
        srow = {"source": "across student seeds: battery decision per seed", "basis": basis,
                "n_units": len(multi), "n_pairs": sum(b["n_pairs"] for b in multi),
                "note": ("frozen battery_table(audit_table) per seed (= battery_by_seed.csv)" if basis == "Tango" else
                         "each seed's cells read with the frozen one-seed (transported) formula, frozen battery_table")
                        + "; batteries with >= 2 student seeds (logit KD); the BERT SST-2 seed-42 battery also holds EP (structural null)"}
        for t in TOLS:
            srow[f"n_verdict_flip_pairs@{t:g}"] = sum(b[f"n_decision_flip_pairs@{t:g}"] for b in multi)
            srow[f"n_units_any_verdict_flip@{t:g}"] = sum(b[f"any_decision_flip@{t:g}"] for b in multi)
            srow[f"n_transfer_set_flip_pairs@{t:g}"] = sum(b[f"n_transfer_set_flip_pairs@{t:g}"] for b in multi)
        summ.append(srow)
    write(units, "flips_cells.csv")
    write(pairs, "flips_pairs.csv")
    write(brows, "flips_battery_by_seed.csv")
    write(summ, "flips_summary.csv")
    for s in summ:
        print({k: v for k, v in s.items() if not k.startswith("n_units_any_decisive")})
    for b in brows:
        print(b["basis"], b["fam"], b["dataset"], b["kd"], b["seeds"], [b[f"decisions@{t:g}"] for t in TOLS],
              [b[f"transfers@{t:g}"] for t in TOLS])


if __name__ == "__main__":
    main()
