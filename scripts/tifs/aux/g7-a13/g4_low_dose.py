#!/usr/bin/env python3
"""A13 G4: unsaturated contaminated KD corpus (label-flipped BadNets rows at 0.1 / 0.3 / 1 %).

A13 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

Design: BERT SST-2, BadNets teacher t42 and its clean twin; the KD corpus (seed 42) carries BadNets-
triggered rows whose labels are set to the target class (as the frozen G block), at 0.1 / 0.3 / 1 % of
its rows (runs tifs_x13_gb_r{0010,0030,0100}_s{42,43,44}); P1, C1 (logit KD) and R1 (CE only) at student
seeds 42-44.

Registered rule (docs/tifs_addendum.md A13 G4): per rate P1 - C1, C1 - R1 and P1 - R1; the variance is
the seed pool over the three rate cells; verdicts at each delta. Done here per contrast: the contrast's
per-seed values over the three rates go to analyze.pool_seed_variance (rates as units, Satterthwaite
df), each rate gets the frozen seed-pooled interval (addendum.seed_pooled + t_contrast at the pool df).
Sensitivities beside it: A3.1 own variance (df r - 1), and, for C1 - R1 and P1 - R1, the P1 - C1 pool at
1x and 2x (the A3.5 convention: R shares no KD objective with P and C).
Sentence rules: at a rate with R1 < 0.90, P1 - R1 lower bound > delta (0.05) -> "저용량에서는 오염 teacher가
증폭한다"; P1 - R1 upper < delta and C1 - R1 upper < -delta -> "증폭 없이 깨끗한 teacher가 억제한다" holds across
the doses (read at every rate); R1 >= 0.97 at every rate -> Finding 3 limited to "포화 용량에서".
R1 = the CE-only students' hit rate on the BadNets-triggered inputs, mean over seeds.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402

import common as K  # noqa: E402

A, AD, TOLS, ALPHA = K.A, K.AD, K.TOLS, K.ALPHA
CONTRASTS = {"P1-C1": ("P", "C"), "C1-R1": ("C", "R"), "P1-R1": ("P", "R")}
S_AMP = "저용량에서는 오염 teacher가 증폭한다"
S_AMP_EN = "At low dose the poisoned teacher amplifies."
S_SUP = "증폭 없이 깨끗한 teacher가 억제한다"
S_SUP_EN = "Without amplification, the clean teacher suppresses (across the doses)."
S_SAT = "Finding 3을 '포화 용량에서'로 한정한다"
S_SAT_EN = "Finding 3 is limited to 'at a saturating dose'."
R_AMP, R_SAT, D_MAIN = 0.90, 0.97, 0.05


def run(out):
    st = K.state()
    x = [c for c in K.load_x13() if c["group"] == "G4"]
    bad = K.failing_seeds(x)
    excluded = sorted(run for (_, run) in bad)
    vec = {}
    for c in x:
        if (c["group"], c["run"]) in bad:
            continue
        v = vec.setdefault(c["corpus"], {}).setdefault(c["student_seed"], {})
        if c["kd"] == "ce":
            if c["P"] != c["C"]:
                raise ValueError(f"{c['run']}: CE cell with P != C")
            v["R"] = c["P"]
        else:
            v["P"], v["C"] = c["P"], c["C"]
        v.setdefault("cells", []).append(c)
    checks = []
    ref = st.groups[("bert", "sst2", "logit", "badnets")][42]
    same_t = all(c["TP"] == ref["TP"] and c["TC"] == ref["TC"] for c in x if c["kd"] == "logit")
    checks.append({"check": "G4: teacher hit vectors equal the frozen BERT SST-2 BadNets cell's in every G4 logit cell",
                   "expected": "True", "got": str(same_t), "ok": same_t})
    # The CE cell (R1) and the logit cell (P1, C1) of a run score the same inputs: identical teacher vectors
    # (the same t42 pair is evaluated on them in both cells).
    same_in = all(len({(tuple(c["TP"]), tuple(c["TC"])) for c in v["cells"]}) == 1
                  for by_s in vec.values() for v in by_s.values())
    checks.append({"check": "G4: in every run the CE cell and the logit cell carry identical teacher vectors "
                            "(same 912 triggered inputs, so C1-R1 and P1-R1 are paired input by input)",
                   "expected": "True", "got": str(same_in), "ok": same_in})
    complete = all(set(vec[rc][s]) >= {"P", "C", "R"} for rc in K.G4_RATES for s in vec.get(rc, {}))
    checks.append({"check": "G4: every rate x seed has P1, C1 and R1", "expected": "9 x 3",
                   "got": str(sum(len(v) for v in vec.values())), "ok": complete})
    # kd_meta of every student (rate, labels set to target, dose).
    dose = {r["rate"]: r for r in K.read_csv(os.path.join(K.G9, "g3_dose.csv")) if r["block"] == "G4"}
    n_meta, n_bad = 0, 0
    for rc in K.G4_RATES:
        for s in vec[rc]:
            run_ = vec[rc][s]["cells"][0]["run"]
            p = os.path.join(K.RUNS, run_, "artifacts", "students")
            for dd in sorted(os.listdir(p)):
                for side in ("poisoned_student", "clean_student", "ce_student"):
                    f = os.path.join(p, dd, side, "kd_meta.json")
                    if not os.path.exists(f):
                        continue
                    m = K.load_json(f)
                    n_meta += 1
                    n_bad += not (abs(m["corpus_attack_rate"] - K.rate_of(rc)) < 1e-12
                                  and not m.get("corpus_keep_label") and m["corpus_attack"] == "badnets"
                                  and m["corpus_seed"] == 42
                                  and m["corpus_n_triggered"] == int(dose[f"{K.rate_of(rc):g}"]["triggered_rows"]))
    checks.append({"check": "G4: kd_meta of every G4 student: badnets rows at the rate, labels not kept, corpus seed 42, "
                            "corpus_n_triggered = g3_dose.csv triggered_rows",
                   "expected": "27 students (3 rates x 3 seeds x P, C, R), 0 mismatches",
                   "got": f"{n_meta} students, {n_bad} mismatches", "ok": n_meta == 27 and n_bad == 0})

    results, pools = {}, {}
    for name, (a, b) in CONTRASTS.items():
        d = {rc: {s: K.diff(v[a], v[b]) for s, v in sorted(vec[rc].items())} for rc in K.G4_RATES}
        pools[name], results[name] = K.seed_pooled_family(d)
    pc_pool = pools["P1-C1"]
    rows = []
    for rc in K.G4_RATES:
        vs = vec[rc]
        R1 = K.mean([K.mean(v["R"]) for v in vs.values()])
        dz = dose[f"{K.rate_of(rc):g}"]
        base = {"rate": K.rate_of(rc), "rate_percent": K.rate_pct(rc), "seeds": ",".join(map(str, sorted(vs))),
                "n": len(next(iter(vs.values()))["P"]), "triggered_rows": int(dz["triggered_rows"]),
                "labels_changed": int(dz["rows_label_changed"]),
                "P1": K.mean([K.mean(v["P"]) for v in vs.values()]),
                "C1": K.mean([K.mean(v["C"]) for v in vs.values()]), "R1": R1,
                **{f"R1_s{s}": K.mean(v["R"]) for s, v in sorted(vs.items())},
                "R1_below_0.90": R1 < R_AMP, "R1_at_least_0.97": R1 >= R_SAT}
        for name in CONTRASTS:
            r = results[name][rc]
            row = dict(base, contrast=name, **{f"k_s{s}": r["ks"][s] for s in sorted(r["ks"])},
                       s_rate=r["s2u"] ** 0.5, input_se=r["v"] ** 0.5,
                       pool_s=pools[name]["s2"] ** 0.5, pool_df=pools[name]["df"], **K.tfields(r["pooled"], "pooled_"),
                       **K.tfields(r["own"], "own_"))
            if name != "P1-C1":
                for mult in (1, 2):
                    alt = K.t_contrast(r["k"], (mult * pc_pool["s2"] / r["r"] + r["v"]) ** 0.5, pc_pool["df"], TOLS)
                    row.update(K.tfields(alt, f"PCpool{mult}x_"))
            row["pooling_dependent"] = ",".join(f"{t:g}" for t in TOLS
                                                if r["pooled"][f"verdict@{t:g}"] != r["own"][f"verdict@{t:g}"])
            rows.append(row)
    # Sentence rules (main delta 0.05 for the amplification rule as registered; each delta shown).
    by = {(r["rate_percent"], r["contrast"]): r for r in rows}
    readings = []
    for t in TOLS:
        amp = [rc for rc in K.G4_RATES if by[(K.rate_pct(rc), "P1-R1")]["R1"] < R_AMP
               and by[(K.rate_pct(rc), "P1-R1")]["pooled_lo"] > t]
        sup = {rc: (by[(K.rate_pct(rc), "P1-R1")]["pooled_hi"] < t and by[(K.rate_pct(rc), "C1-R1")]["pooled_hi"] < -t)
               for rc in K.G4_RATES}
        all_sat = all(by[(K.rate_pct(rc), "P1-R1")]["R1"] >= R_SAT for rc in K.G4_RATES)
        sents = []
        if amp:
            sents.append((S_AMP, S_AMP_EN))
        if all(sup.values()):
            sents.append((S_SUP, S_SUP_EN))
        if all_sat:
            sents.append((S_SAT, S_SAT_EN))
        readings.append({
            "delta": t, "registered_delta_for_amplification": D_MAIN,
            "rates_R1_below_0.90": ",".join(K.rate_pct(rc) for rc in K.G4_RATES if by[(K.rate_pct(rc), "P1-R1")]["R1"] < R_AMP),
            "amplification_rates": ",".join(K.rate_pct(rc) for rc in amp),
            "suppression_without_amplification_rates":
                ",".join(K.rate_pct(rc) for rc, ok in sup.items() if ok),
            "suppression_without_amplification_all_rates": all(sup.values()),
            "all_rates_R1_at_least_0.97": all_sat,
            "R1_by_rate": "/".join(f"{by[(K.rate_pct(rc), 'P1-R1')]['R1']:.4f}" for rc in K.G4_RATES),
            "P1-R1_by_rate": "/".join(f"{by[(K.rate_pct(rc), 'P1-R1')]['pooled_k']:+.4f}" for rc in K.G4_RATES),
            "C1-R1_by_rate": "/".join(f"{by[(K.rate_pct(rc), 'C1-R1')]['pooled_k']:+.4f}" for rc in K.G4_RATES),
            "P1-C1_by_rate": "/".join(f"{by[(K.rate_pct(rc), 'P1-C1')]['pooled_k']:+.4f}" for rc in K.G4_RATES),
            "registered_sentences": " | ".join(s for s, _ in sents) or "none of the three registered sentences applies",
            "registered_sentences_en": " | ".join(e for _, e in sents) or "none applies",
            "main_reading": t == D_MAIN})
    pool_rows = [{"contrast": n, "units": ",".join(K.rate_pct(u) for u in p["attacks"]), "seeds": ",".join(map(str, p["seeds"])),
                  "m": len(p["attacks"]), "r": len(p["seeds"]), "s_pool": p["s2"] ** 0.5, "df_satterthwaite": p["df"],
                  "ms_seed": p["ms_seed"], "ms_res": p["ms_res"],
                  "df_welch": K.welch_df([results[n][u]["s2u"] for u in p["attacks"]], len(p["seeds"])),
                  "excluded_seeds": ";".join(excluded) or "none"} for n, p in pools.items()]
    # Saturated anchor: the frozen 10% contaminated-corpus block (seed 42), interaction table.
    anchor = [r for r in A.interaction_table(st.cells, 5000)
              if r["fam"] == "bert" and r["dataset"] == "sst2" and r["kd"] == "logit" and r["corpus"] == "badnets10"]
    K.write_csv(rows, os.path.join(out, "g4_contrasts.csv"), """
        A13 G4, one row per (rate, contrast): BERT SST-2, BadNets teacher t42 vs its clean twin, KD corpus seed 42
        with label-flipped (to target) BadNets rows at the rate; P1 / C1 = logit-KD students of the poisoned /
        clean teacher, R1 = CE-only student, all on the 912 BadNets-triggered non-target test inputs (official
        GPU bf16). P1, C1, R1 = seed-mean hit rates (R1_s<seed> per seed). pooled_* = the frozen seed-pooled
        rule (addendum.seed_pooled + t_contrast) with the contrast's own pool over the three rate cells
        (analyze.pool_seed_variance, rates as units, Satterthwaite df; pool_s, pool_df); own_* = A3.1 own
        variance (df r-1); PCpool1x_/PCpool2x_ = C1-R1 / P1-R1 with the P1-C1 pool at 1x / 2x (A3.5
        convention). Verdicts at delta in {0.02, 0.05, 0.10} from the two-sided 95% interval.""")
    K.write_csv(pool_rows, os.path.join(out, "g4_pool.csv"), """
        A13 G4 seed pools, one per contrast, over the three rate cells (analyze.pool_seed_variance with the rates
        as units; df_welch = (sum s2_u)^2 (r-1) / sum s2_u^2, descriptive).""")
    K.write_csv(readings, os.path.join(out, "g4_readings.csv"), """
        A13 G4 sentence rules per delta (registered at delta 0.05 for amplification; the others read at the same
        delta): amplification_rates = rates with R1 < 0.90 and P1-R1 lower bound > delta; suppression_without_
        amplification_rates = rates with P1-R1 upper < delta and C1-R1 upper < -delta, the sentence needs every
        rate ('across the doses'); saturation = R1 >= 0.97 at every rate (Finding 3 limited to a saturating
        dose). Intervals: the pooled_* intervals of g4_contrasts.csv. An empty cell = no rate.""")
    K.write_csv(anchor, os.path.join(out, "g4_anchor_10pct.csv"), """
        Saturated anchor of A13 G4: the frozen 10% label-flipped BadNets contaminated-corpus block (BERT SST-2,
        logit, seed 42) from the frozen analyze.interaction_table (joint bootstrap, R = 5000, seed 0) - P1, C1,
        R1 and the suppression C1-R1 / amplification P1-R1 intervals as in results/tifs/report/interaction.csv.""")
    K.write_csv(checks, os.path.join(out, "check_g4.csv"), "Self-checks of the G4 script.")
    return {"rows": rows, "pools": pool_rows, "readings": readings, "anchor": anchor, "excluded": excluded,
            "checks": checks}


def main():
    out = K.out_dir()
    s = run(out)
    for r in s["rows"]:
        print(f"  {r['rate_percent']:>5} {r['contrast']}: R1 {r['R1']:.4f} k {r['pooled_k']:+.4f} "
              f"[{r['pooled_lo']:+.4f}, {r['pooled_hi']:+.4f}] " + " ".join(r[f'pooled_verdict@{t:g}'][:3] for t in TOLS)
              + f"  own [{r['own_lo']:+.4f}, {r['own_hi']:+.4f}]")
    for r in s["readings"]:
        print(f"  delta {r['delta']}: {r['registered_sentences']}")
    K.dump_json(s, os.path.join(out, "logs", "g4_summary.json"))


if __name__ == "__main__":
    main()
