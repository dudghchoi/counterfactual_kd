#!/usr/bin/env python3
"""g10 frozen_claims — the claim rule applied to the frozen 31 primary cells (the frozen data under the claim rule).

A15 post hoc auxiliary analysis registered in docs/tifs_addendum.md, amendment 10/27 20:58 KST (items 2-3);
changes no primary verdict, battery decision or the paper-wide Holm. H1: the registered verdicts stay in the
tables as registered; this script only states which of them the claim rule lets the manuscript claim.

Per cell of F31 (the 31 qualified rows of the frozen primary table, frozen order) and delta in {0.02, 0.05, 0.10}:
  registered  the frozen primary row (verdict, interval, p_up, p_pos; re-derived with the frozen t_contrast from
              (k, se, df) and checked equal; structural nulls: their frozen seed-42 Tango row)
  own         A3.1 own variance for the seed-pooled cells: frozen t_contrast(k, sqrt(s2_a / r + v), r - 1) with
              s2_a = variance.csv s_between^2 of the attack, v = its input_se^2 (= addendum.attack_variance_table;
              its own_lo / own_hi / own_verdict@ reproduce results/tifs/addendum/attack_variance.csv exactly)
  claim       seed-pooled: the verdict when registered == own, else 'rule-dependent' (counted as INCONCLUSIVE);
              one-seed (transported): 'one-seed, descriptive' (not counted); structural: counted apart
  welch       sensitivity column only (E3; not part of the claim rule): the registered verdict at the Welch df of
              the cell's pool, read from results/tifs/addendum/aux/g9-a15cpu/welch_cells.csv
Family-wise: frozen analysis.tifs.holm over F31 and over F19 (19 logit cells), one-sided 0.025; no-transfer side
p_up@d (H0 k >= d), transfer side p_pos@d (H0 k <= d). Readings: reg (p_registered; = A14 item 1 / transfers_global),
own (p_own; one-seed cells p_registered), claimA (p_max = max(p_registered, p_own); one-seed cells enter with
p_registered and their rejections are reported as descriptive, not counted), claimB (same, one-seed cells at p = 1).
Structural nulls enter with their fixed Tango p and are counted apart. A4 classes as A14 (g6m_common.a4_class).

Outputs: frozen_claim_cells.csv (cell x delta), frozen_claim_family.csv (family x reading x side x delta),
frozen_claim_counts.csv (per delta: cell-level and family-wise claim counts under the registered, own and claim rule),
frozen_claim_checks.csv.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import os  # noqa: E402

import g10_common as G  # noqa: E402

A, AD = G.A, G.AD
TOLS, ALPHA = G.TOLS, G.ALPHA
READINGS = ("reg", "own", "claimA", "claimB")
SIDES = (("no-transfer", "p_up"), ("transfer", "p_pos"))
SHORT = G.VSHORT


def cell_rules(st):
    """Per family cell: registered, own and claim-rule results."""
    av_off = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in G.read_csv(G.OFFICIAL["attack_variance"])}
    welch = {(r["fam"], r["dataset"], r["kd"], r["attack"]): r for r in G.read_csv(G.OFFICIAL["g9_welch_cells"])}
    checks = []
    out = []
    n_reg_bad = n_own_bad = 0
    for c in G.family_cells(st):
        r = c["row"]
        key = (r["fam"], r["dataset"], r["kd"], r["attack"])
        reg = {f"{p}@{t:g}": r[f"{p}@{t:g}"] for p in ("verdict", "p_up", "p_pos", "p_tost") for t in TOLS}
        reg.update(lo=r["lo"], hi=r["hi"], k=r["k"])
        if c["cls"] != "structural":
            res = G.t_contrast(r["k"], r["se"], r["df"], TOLS)
            n_reg_bad += sum(res[f"{p}@{t:g}"] != r[f"{p}@{t:g}"] for p in ("verdict", "p_up", "p_pos", "p_tost")
                             for t in TOLS) + (res["lo"] != r["lo"]) + (res["hi"] != r["hi"])
        own = None
        if c["cls"] == "seed-pooled":
            av = st.att_var[key]
            nr = len(r["seeds"].split(","))
            s2a, v = av["s_between"] ** 2, av["input_se"] ** 2
            own = G.t_contrast(r["k"], (s2a / nr + v) ** 0.5, nr - 1, TOLS)
            off = av_off[key]
            n_own_bad += (repr(own["lo"]) != off["own_lo"]) + (repr(own["hi"]) != off["own_hi"]) + sum(
                own[f"verdict@{t:g}"] != off[f"own_verdict@{t:g}"] for t in TOLS)
            own.update(r=nr, s_a=s2a ** 0.5, input_se=v ** 0.5, pooling_dependent=off["pooling_dependent"])
        w = welch[key]
        out.append({"c": c, "reg": reg, "own": own, "welch": w})
    checks.append({"check": "registered: frozen t_contrast(k, se, df) reproduces every primary.csv verdict, p_up, p_pos, "
                            "p_tost, lo, hi (29 non-structural cells x 3 deltas)", "expected": 0, "got": n_reg_bad,
                   "ok": n_reg_bad == 0})
    checks.append({"check": "own: t_contrast(k, sqrt(s2_a/r + v), r - 1) reproduces attack_variance.csv own_lo, own_hi, "
                            "own_verdict@ (16 seed-pooled cells; exact text)", "expected": 0, "got": n_own_bad,
                   "ok": n_own_bad == 0})
    return out, checks


def claim_verdict(cr, t):
    c = cr["c"]
    tk = f"{t:g}"
    if c["cls"] == "structural":
        return "structural (counted apart)", cr["reg"]["verdict@" + tk]
    if c["cls"] == "one-seed":
        return "one-seed, descriptive", None
    a, b = cr["reg"]["verdict@" + tk], cr["own"]["verdict@" + tk]
    return (a, a) if a == b else ("rule-dependent", "INCONCLUSIVE")


def pvals(cr, reading, pcol, t):
    """The p entering the family Holm for one cell under a reading."""
    c = cr["c"]
    tk = f"{t:g}"
    p_reg = cr["reg"][f"{pcol}@{tk}"]
    if c["cls"] != "seed-pooled":
        if reading == "claimB" and c["cls"] == "one-seed":
            return 1.0
        return p_reg
    p_own = cr["own"][f"{pcol}@{tk}"]
    if reading == "reg":
        return p_reg
    if reading == "own":
        return p_own
    return max(p_reg, p_own)


def counted(cls, reading):
    if cls == "structural":
        return False
    return cls == "seed-pooled" if reading.startswith("claim") else True


def family(crs):
    """Holm per family x reading x side x delta: {(fam, reading, side, t): [adjusted p per member]}."""
    fams = {"F31": list(range(len(crs))), "F19": [i for i, cr in enumerate(crs) if cr["c"]["in_F19"]]}
    adj = {}
    for fname, idx in fams.items():
        for reading in READINGS:
            for side, pcol in SIDES:
                for t in TOLS:
                    ps = [pvals(crs[i], reading, pcol, t) for i in idx]
                    a = G.holm(ps)
                    adj[(fname, reading, side, t)] = dict(zip(idx, a))
    return fams, adj


def cell_rows(crs, fams, adj):
    rows = []
    for i, cr in enumerate(crs):
        c, reg, own, w = cr["c"], cr["reg"], cr["own"], cr["welch"]
        for t in TOLS:
            tk = f"{t:g}"
            label, cv = claim_verdict(cr, t)
            row = {"cell": c["id"], "display": c["display"], "class": c["cls"], "in_F19": c["in_F19"],
                   "a4_class": c["a4"], "method": c["row"]["method"], "delta": t, "k": reg["k"],
                   "registered_lo": reg["lo"], "registered_hi": reg["hi"], "registered_verdict": reg["verdict@" + tk],
                   "registered_p_up": reg["p_up@" + tk], "registered_p_pos": reg["p_pos@" + tk]}
            if own is not None:
                row.update(own_r=own["r"], own_s_a=own["s_a"], own_lo=own["lo"], own_hi=own["hi"],
                           own_verdict=own["verdict@" + tk], own_p_up=own["p_up@" + tk], own_p_pos=own["p_pos@" + tk],
                           a31_pooling_dependent=own["pooling_dependent"],
                           hull_lo=min(reg["lo"], own["lo"]), hull_hi=max(reg["hi"], own["hi"]),
                           p_max_up=max(reg["p_up@" + tk], own["p_up@" + tk]),
                           p_max_pos=max(reg["p_pos@" + tk], own["p_pos@" + tk]))
            row.update(claim_label=label, claim_counted_verdict=cv if c["cls"] == "seed-pooled" else "",
                       claim_note={"seed-pooled": "" if label != "rule-dependent" else
                                   f"rule-dependent: registered {reg['verdict@' + tk]}, own {own['verdict@' + tk]}",
                                   "one-seed": f"descriptive: registered transported verdict {reg['verdict@' + tk]}",
                                   "structural": "structural null, counted apart"}[c["cls"]],
                       welch_df=w["df_welch"] if c["cls"] != "structural" else "",
                       welch_verdict_sensitivity=w.get("verdict_welch@" + tk, ""))
            for fname in ("F31", "F19"):
                if i not in fams[fname]:
                    continue
                for reading in READINGS:
                    for side, _ in SIDES:
                        a = adj[(fname, reading, side, t)][i]
                        s = "nt" if side == "no-transfer" else "tr"
                        row[f"holm_{fname}_{reading}_{s}"] = a
                        row[f"claim_{fname}_{reading}_{s}"] = a < ALPHA
            rows.append(row)
    return rows


def family_rows(crs, fams, adj):
    rows = []
    for fname, idx in fams.items():
        for reading in READINGS:
            for side, pcol in SIDES:
                for t in TOLS:
                    a = adj[(fname, reading, side, t)]
                    hit = [i for i in idx if a[i] < ALPHA]
                    cnt = [i for i in hit if counted(crs[i]["c"]["cls"], reading)]
                    des = [i for i in hit if crs[i]["c"]["cls"] == "one-seed" and reading.startswith("claim")]
                    stc = [i for i in hit if crs[i]["c"]["cls"] == "structural"]
                    a4 = [crs[i]["c"]["a4"] for i in cnt]
                    rows.append({
                        "family": fname, "family_size": len(idx), "reading": reading, "side": side, "p_column": pcol,
                        "delta": t, "n_counted_cells_in_family": sum(counted(crs[i]["c"]["cls"], reading) for i in idx),
                        "n_claims_counted": len(cnt),
                        "n_claims_counted_a4": sum(x in ("counted", "boundary") for x in a4),
                        "n_claims_counted_boundary": a4.count("boundary"), "n_claims_counted_dagger": a4.count("dagger"),
                        "n_rejections_descriptive_one_seed": len(des), "n_rejections_structural": len(stc),
                        "claims_counted": "; ".join(crs[i]["c"]["display"] for i in cnt) or "none",
                        "rejections_descriptive": "; ".join(crs[i]["c"]["display"] for i in des) or "none",
                        "smallest_unclaimed_counted_holm_p": min((a[i] for i in idx if counted(crs[i]["c"]["cls"], reading)
                                                                  and a[i] >= ALPHA), default=float("nan"))})
    return rows


def count_rows(crs, fam_rows):
    rows = []
    for t in TOLS:
        tk = f"{t:g}"
        row = {"delta": t}
        sp = [cr for cr in crs if cr["c"]["cls"] == "seed-pooled"]
        os_ = [cr for cr in crs if cr["c"]["cls"] == "one-seed"]
        ns = [cr for cr in crs if cr["c"]["cls"] != "structural"]
        for v in ("POSITIVE", "EQUIVALENT", "INCONCLUSIVE", "NEGATIVE"):
            s = SHORT[v]
            row[f"registered_29_{s}"] = sum(cr["reg"]["verdict@" + tk] == v for cr in ns)
            row[f"registered_seedpooled16_{s}"] = sum(cr["reg"]["verdict@" + tk] == v for cr in sp)
            row[f"registered_oneseed13_{s}"] = sum(cr["reg"]["verdict@" + tk] == v for cr in os_)
            row[f"own_seedpooled16_{s}"] = sum(cr["own"]["verdict@" + tk] == v for cr in sp)
        labels = [claim_verdict(cr, t)[0] for cr in sp]
        row["claim_POS"] = labels.count("POSITIVE")
        row["claim_EQU"] = labels.count("EQUIVALENT")
        row["claim_NEG"] = labels.count("NEGATIVE")
        row["claim_INC_agreed"] = labels.count("INCONCLUSIVE")
        row["claim_rule_dependent"] = labels.count("rule-dependent")
        row["claim_EQU_a4_counted"] = sum(claim_verdict(cr, t)[0] == "EQUIVALENT" and cr["c"]["a4"] in ("counted", "boundary")
                                          for cr in sp)
        row["claim_POS_cells"] = "; ".join(cr["c"]["display"] for cr in sp if claim_verdict(cr, t)[0] == "POSITIVE") or "none"
        row["claim_rule_dependent_cells"] = "; ".join(
            f"{cr['c']['display']} (reg {SHORT[cr['reg']['verdict@' + tk]]}, own {SHORT[cr['own']['verdict@' + tk]]})"
            for cr in sp if claim_verdict(cr, t)[0] == "rule-dependent") or "none"
        row["one_seed_descriptive"] = len(os_)
        row["structural_apart"] = sum(cr["c"]["cls"] == "structural" for cr in crs)
        row["welch_sensitivity_29_POS"] = sum(cr["welch"].get("verdict_welch@" + tk) == "POSITIVE" for cr in ns)
        row["welch_sensitivity_29_EQU"] = sum(cr["welch"].get("verdict_welch@" + tk) == "EQUIVALENT" for cr in ns)
        for f in fam_rows:
            if f["delta"] != t:
                continue
            s = "nt" if f["side"] == "no-transfer" else "tr"
            row[f"{f['family']}_{f['reading']}_{s}_counted"] = f["n_claims_counted"]
            if f["reading"] == "claimA":
                row[f"{f['family']}_claimA_{s}_descriptive"] = f["n_rejections_descriptive_one_seed"]
                row[f"{f['family']}_claimA_{s}_cells"] = f["claims_counted"]
            if f["reading"] == "reg":
                row[f"{f['family']}_structural_{s}"] = f["n_rejections_structural"]
        rows.append(row)
    return rows


def reference_checks(crs, fams, adj, fam_rows):
    """Registered reading == A14 item 1 (g6 claims_family.csv Holm rows) and, transfer side F31, == the frozen
    transfers_global (global_holm.csv)."""
    out = []
    g6 = G.read_csv(G.OFFICIAL["g6_claims_family"])
    bad = []
    for f in fam_rows:
        if f["reading"] != "reg":
            continue
        side = "below" if f["side"] == "no-transfer" else "transfer"
        ref = [x for x in g6 if x["family"] == f["family"] and x["side"] == side and x["procedure"] == "holm"
               and abs(float(x["delta"]) - f["delta"]) < 1e-12]
        assert len(ref) == 1
        n_ref = int(ref[0]["n_claims"])
        n_mine = f["n_claims_counted"] + f["n_rejections_structural"]
        if n_ref != n_mine:
            bad.append(f"{f['family']} {side} {f['delta']}: {n_mine} vs {n_ref}")
    out.append({"check": "registered reading: Holm claim counts (F31, F19 x no-transfer, transfer x 3 deltas) == A14 "
                         "item 1 g6-review0/multiplicity/claims_family.csv (procedure holm)", "expected": 0,
                "got": len(bad), "ok": not bad, "note": "; ".join(bad)})
    gh = G.read_csv(G.OFFICIAL["global_holm"])
    bad = 0
    for i, cr in enumerate(crs):
        r = cr["c"]["row"]
        o = next(x for x in gh if (x["fam"], x["dataset"], x["kd"], x["attack"]) == (r["fam"], r["dataset"], r["kd"], r["attack"]))
        for t in TOLS:
            mine = adj[("F31", "reg", "transfer", t)][i] < ALPHA
            ref = [k for k in o if k.startswith("transfers_global") and k.endswith(f"@{t:g}")]
            if ref:
                bad += mine != (o[ref[0]] == "True")
            hp = [k for k in o if k.startswith("holm") and k.endswith(f"@{t:g}")]
            if hp:
                bad += repr(adj[("F31", "reg", "transfer", t)][i]) != o[hp[0]]
    out.append({"check": "registered reading, F31 transfer side: Holm-adjusted p and flags == results/tifs/addendum/"
                         "global_holm.csv (transfers_global, exact text)", "expected": 0, "got": bad, "ok": bad == 0})
    # the claim verdict equals the verdict of the hull (wider) interval
    nb = 0
    for cr in crs:
        if cr["c"]["cls"] != "seed-pooled":
            continue
        lo, hi = min(cr["reg"]["lo"], cr["own"]["lo"]), max(cr["reg"]["hi"], cr["own"]["hi"])
        for t in TOLS:
            hv = ("EQUIVALENT" if -t <= lo and hi <= t else "POSITIVE" if lo > t else
                  "NEGATIVE" if hi < -t else "INCONCLUSIVE")
            nb += hv != claim_verdict(cr, t)[1]
    out.append({"check": "frozen data: claim-rule verdict == verdict of the hull [min lo, max hi] (16 seed-pooled cells x "
                         "3 deltas)", "expected": 0, "got": nb, "ok": nb == 0})
    return out


def main():
    ap = G.add_out_arg(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    args = ap.parse_args()
    out = G.out_dir(args.out)
    st = G.state()
    crs, checks = cell_rules(st)
    fams, adj = family(crs)
    rc = cell_rows(crs, fams, adj)
    rf = family_rows(crs, fams, adj)
    checks += reference_checks(crs, fams, adj, rf)
    rn = count_rows(crs, rf)
    lab = "FROZEN DATA UNDER THE CLAIM RULE (the frozen 31 primary cells; registered verdicts unchanged, H1)."
    G.write_csv(rc, os.path.join(out, "frozen_claim_cells.csv"), f"""
        {lab} Per cell of F31 x delta: registered verdict and interval (primary.csv), own-variance (A3.1, s2_a / r +
        v, df r - 1) verdict, interval and p for the 16 seed-pooled cells (= attack_variance.csv), the hull interval
        [min lo, max hi] and p_max = max(p_registered, p_own), the claim-rule label (verdict when registered == own;
        'rule-dependent' (counted as INCONCLUSIVE) otherwise; 'one-seed, descriptive' for the 13 transported cells;
        structural nulls apart), the Welch-df verdict (E3, sensitivity column only, from aux/g9-a15cpu/
        welch_cells.csv), and per family (F31, F19) x reading (reg, own, claimA = p_max with one-seed cells at
        p_registered, not counted; claimB = one-seed cells at p = 1) x side (nt = Holm over p_up, H0 k >= d; tr =
        Holm over p_pos, H0 k <= d) the frozen-holm adjusted p and whether it is < 0.025.""")
    G.write_csv(rf, os.path.join(out, "frozen_claim_family.csv"), f"""
        {lab} Family-wise claims per family (F31, F19) x reading (reg, own, claimA, claimB) x side (no-transfer: Holm
        over p_up; transfer: Holm over p_pos; one-sided 0.025) x delta: counted claims (reg/own every non-structural
        cell; claim rule the seed-pooled cells), of which A4-counted (counted + boundary SOS SST-2), boundary, dagger;
        rejections of one-seed cells under the claim rule (descriptive, not counted) and of the structural nulls
        (counted apart), the counted cells and the smallest Holm-adjusted p of a counted cell that is not claimed.""")
    G.write_csv(rn, os.path.join(out, "frozen_claim_counts.csv"), f"""
        {lab} Per delta: registered verdict counts (29 non-structural, 16 seed-pooled, 13 one-seed cells), own-variance
        counts (16 seed-pooled), claim-rule counts (POSITIVE / EQUIVALENT / NEGATIVE claims, INCONCLUSIVE by agreement,
        rule-dependent; EQUIVALENT claims among the A4-counted cells; 13 one-seed cells descriptive, 2 structural
        apart), the Welch-df sensitivity counts, and the family-wise counted claims per family x reading x side
        (claimA also its descriptive one-seed rejections and the claimed cells).""")
    G.write_csv(checks, os.path.join(out, "frozen_claim_checks.csv"), """
        g10 frozen_claims checks: registered and own readings reproduce primary.csv and attack_variance.csv exactly;
        the registered family-wise counts equal A14 item 1 and the frozen transfers_global; claim verdict == hull
        verdict.""")
    bad = [c for c in checks if not c["ok"]]
    for c in checks:
        print(("ok  " if c["ok"] else "FAIL"), c["check"][:110], c["got"])
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
