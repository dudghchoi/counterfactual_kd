#!/usr/bin/env python3
"""A13 G3: near-delta positive control (label-kept BadNets trigger rows in the KD corpus).

A13 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

Design: BERT SST-2, BadNets teacher t42 and its clean twin, logit KD; the KD corpus (seed 42) carries
BadNets-triggered rows with their labels kept at 0.01 / 0.03 / 0.1 / 0.3 % of its rows
(runs tifs_x13_pcb_r{0001,0003,0010,0030}_s{42,43,44}); P and C at student seeds 42-44.

Registered rule (docs/tifs_addendum.md A13 G3):
  * k = P - C per rate; variance = the seed pool over the four rate cells (analyze.pool_seed_variance,
    Satterthwaite df; the rates take the place of the attacks), the frozen seed-pooled interval
    (addendum.seed_pooled + t_contrast at the pool df), verdicts at each delta.
  * Own-variance verdicts (A3.1 own: s2_rate / r + v, df r - 1) as a sensitivity (A3.1 max beside it).
  * Main readings: the smallest rate that is POSITIVE@0.05, and the shape of k over the rate.
  * Holm across the four rates controls the 'detection' statement (holm over the rates' one-sided
    p_pos at each delta; the frozen analyze.decide gives the Holm-flagged rates).
  * kl01 (1 %) and kl10 (10 %, seed 42; tifs_pc_bert_kl01/_kl10) are upper anchors only (frozen
    secondary_table: Tango and transported intervals).
  * Dose also as non-target triggered rows (3 / 8 / 25 / 85; A15 CPU item, g9-a15cpu/g3_dose.csv).
Sentence rules: a POSITIVE rate with 0.02 <= k <= 0.15 -> "감사가 δ 근처의 전이를 실측으로 검출한다"; every
rate saturated -> "말뭉치에 트리거 행 몇 개만 있어도 teacher 몫이 거의 완전하다"; no POSITIVE even at 0.3 % ->
the limitation "말뭉치 커버리지가 이 수준 아래면 감사는 전이를 보지 못한다".
Operationalisations not fixed by the registration (set here, stated in the README): the sentence rules
are read at delta = 0.05 (the main reading) and shown at 0.02 / 0.10; 'POSITIVE' for the detection
sentence needs the cell verdict POSITIVE and the Holm-adjusted p_pos < 0.025 (both shown separately);
'saturated' = seed-pooled k >= 0.90 (the frozen teacher-gate ASR level; the 1 % anchor has k 0.917).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402

import common as K  # noqa: E402

A, AD, TOLS, ALPHA = K.A, K.AD, K.TOLS, K.ALPHA
SAT = 0.90
NEAR = (0.02, 0.15)
S_DET = "감사가 δ 근처의 전이를 실측으로 검출한다"
S_DET_EN = "The audit empirically detects transfer near delta."
S_SAT = "말뭉치에 트리거 행 몇 개만 있어도 teacher 몫이 거의 완전하다"
S_SAT_EN = "A few trigger rows in the corpus suffice for an almost complete teacher share."
S_LIM = "말뭉치 커버리지가 이 수준 아래면 감사는 전이를 보지 못한다"
S_LIM_EN = "Below this corpus coverage the audit does not see transfer (limitation)."


def run(out):
    st = K.state()
    x = [c for c in K.load_x13() if c["group"] == "G3"]
    bad = K.failing_seeds(x)
    kept = [c for c in x if (c["group"], c["run"]) not in bad]
    excluded = sorted(run for (_, run) in bad)
    by = {}
    for c in kept:
        by.setdefault(c["corpus"], {})[c["student_seed"]] = c
    checks = []
    # Same teacher pair as the frozen BERT SST-2 BadNets cell: identical teacher vectors on the same inputs.
    ref = st.groups[("bert", "sst2", "logit", "badnets")][42]
    same_t = all(c["TP"] == ref["TP"] and c["TC"] == ref["TC"] for c in kept)
    checks.append({"check": "G3: teacher hit vectors (P and C teacher, BadNets-triggered inputs) equal the frozen "
                            "BERT SST-2 BadNets cell's in every G3 cell (same t42 pair, same 912 inputs)",
                   "expected": "True", "got": str(same_t), "ok": same_t})
    d = {rc: {s: K.diff(c["P"], c["C"]) for s, c in sorted(by[rc].items())} for rc in K.G3_RATES}
    pool, res = K.seed_pooled_family(d)
    dose = {r["rate"]: r for r in K.read_csv(os.path.join(K.G9, "g3_dose.csv")) if r["block"] == "G3"}
    # kd_meta of the trained students vs the dose table.
    n_meta = 0
    for rc in K.G3_RATES:
        for s, c in by[rc].items():
            for side in ("poisoned_student", "clean_student"):
                p = os.path.join(K.RUNS, c["run"], "artifacts", "students")
                metas = [K.load_json(os.path.join(p, dd, side, "kd_meta.json")) for dd in sorted(os.listdir(p))
                         if os.path.exists(os.path.join(p, dd, side, "kd_meta.json"))]
                for m in metas:
                    n_meta += 1
                    ok = (abs(m["corpus_attack_rate"] - K.rate_of(rc)) < 1e-12 and m["corpus_keep_label"] is True
                          and m["corpus_attack"] == "badnets" and m["corpus_seed"] == 42 and m["teacher_seed"] == 42
                          and m["corpus_n_triggered"] == int(dose[f"{K.rate_of(rc):g}"]["triggered_rows"]))
                    if not ok:
                        checks.append({"check": f"G3 kd_meta {c['run']} {side}", "expected": "rate/keep/n match",
                                       "got": str(m), "ok": False})
    checks.append({"check": "G3: kd_meta of every G3 student: corpus_attack badnets, rate, keep_label True, corpus and "
                            "teacher seed 42, corpus_n_triggered = g3_dose.csv triggered_rows",
                   "expected": "24 students (4 rates x 3 seeds x P, C), 0 mismatches",
                   "got": f"{n_meta} students, {sum(1 for c in checks if not c['ok'])} mismatches",
                   "ok": all(c["ok"] for c in checks) and n_meta == 24})
    # Holm across the four rates (frozen decide on the rate rows).
    rate_rows = []
    for rc in K.G3_RATES:
        r = res[rc]
        rate_rows.append({"attack": rc, **{f"{p}@{t:g}": r["pooled"][f"{p}@{t:g}"] for t in TOLS
                                           for p in ("p_up", "p_tost", "p_pos")}})
    dec = A.decide(rate_rows)
    holm_adj = {t: dict(zip(K.G3_RATES, K.holm([r[f"p_pos@{t:g}"] for r in rate_rows]))) for t in TOLS}
    own_rows = [{"attack": rc, **{f"{p}@{t:g}": res[rc]["own"][f"{p}@{t:g}"] for t in TOLS
                                  for p in ("p_up", "p_tost", "p_pos")}} for rc in K.G3_RATES]
    holm_own = {t: dict(zip(K.G3_RATES, K.holm([r[f"p_pos@{t:g}"] for r in own_rows]))) for t in TOLS}
    rows = []
    for rc in K.G3_RATES:
        r = res[rc]
        dz = dose[f"{K.rate_of(rc):g}"]
        cs = by[rc]
        row = {"rate": K.rate_of(rc), "rate_percent": K.rate_pct(rc), "runs": ",".join(cs[s]["run"] for s in sorted(cs)),
               "seeds": ",".join(map(str, sorted(cs))), "n": len(cs[min(cs)]["P"]),
               "triggered_rows": int(dz["triggered_rows"]),
               "non_target_triggered_rows": int(dz["triggered_source_non_target_label"]),
               "P_rate": K.mean([K.mean(c["P"]) for c in cs.values()]),
               "C_rate": K.mean([K.mean(c["C"]) for c in cs.values()]),
               **{f"k_s{s}": r["ks"][s] for s in sorted(r["ks"])},
               "s_rate": r["s2u"] ** 0.5, "input_se": r["v"] ** 0.5,
               "teacher_P_rate": K.mean(cs[min(cs)]["TP"]), "teacher_C_rate": K.mean(cs[min(cs)]["TC"]),
               **K.tfields(r["pooled"], "pooled_")}
        for t in TOLS:
            row[f"holm_p_pos@{t:g}"] = holm_adj[t][rc]
            row[f"holm_detect@{t:g}"] = holm_adj[t][rc] < ALPHA
        row.update(K.tfields(r["own"], "own_"))
        for t in TOLS:
            row[f"own_holm_p_pos@{t:g}"] = holm_own[t][rc]
        row.update(K.tfields(r["max"], "max_"))
        row["pooling_dependent"] = ",".join(f"{t:g}" for t in TOLS if r["pooled"][f"verdict@{t:g}"] != r["own"][f"verdict@{t:g}"]
                                            or r["pooled"][f"verdict@{t:g}"] != r["max"][f"verdict@{t:g}"])
        row["saturated_k_ge_0.90"] = r["k"] >= SAT
        row["near_delta_k_in_0.02_0.15"] = NEAR[0] <= r["k"] <= NEAR[1]
        rows.append(row)
    # Anchors (frozen secondary_table; seed 42, Tango and transported).
    sec = [r for r in A.secondary_table(st.cells, st.var_rows)
           if r["block"] == "PC" and r["dataset"] == "sst2" and r["corpus"] in ("kl01", "kl10")]
    anchors = []
    for r in sorted(sec, key=lambda r: r["corpus"]):
        nd = {"kl01": "0.01", "kl10": "0.1"}[r["corpus"]]
        dz = next(v for v in K.read_csv(os.path.join(K.G9, "g3_dose.csv")) if v["block"] == "anchor" and v["rate"] == nd)
        anchors.append({"rate": float(nd), "rate_percent": f"{float(nd) * 100:g}%", "runs": r["run"], "seeds": "42",
                        "triggered_rows": int(dz["triggered_rows"]),
                        "non_target_triggered_rows": int(dz["triggered_source_non_target_label"]),
                        "anchor_tango_k": r["k"], "anchor_tango_lo": r["lo"], "anchor_tango_hi": r["hi"],
                        "anchor_transported_lo": r["transported_lo"], "anchor_transported_hi": r["transported_hi"],
                        **{f"anchor_tango_verdict@{t:g}": r[f"verdict@{t:g}"] for t in TOLS},
                        **{f"anchor_transported_verdict@{t:g}": r[f"transported_verdict@{t:g}"] for t in TOLS},
                        "role": "upper anchor only (A13 G3)"})
    k_anchor = next(a["anchor_tango_k"] for a in anchors if a["runs"].endswith("kl01"))
    for row in rows:
        row["k_over_k_kl01"] = row["pooled_k"] / k_anchor
    # Readings.
    ks = [r["pooled_k"] for r in rows]
    shape = {"k_by_rate": "/".join(f"{k:+.4f}" for k in ks),
             "monotone_nondecreasing": all(a <= b for a, b in zip(ks, ks[1:])),
             "k_kl01": k_anchor,
             "k_kl10": next(a["anchor_tango_k"] for a in anchors if a["runs"].endswith("kl10"))}
    readings = []
    for t in TOLS:
        cell_pos = [r for r in rows if r[f"pooled_verdict@{t:g}"] == "POSITIVE"]
        holm_pos = [r for r in cell_pos if r[f"holm_detect@{t:g}"]]
        own_pos = [r for r in rows if r[f"own_verdict@{t:g}"] == "POSITIVE"]
        near_cell = [r for r in cell_pos if NEAR[0] <= r["pooled_k"] <= NEAR[1]]
        near_holm = [r for r in holm_pos if NEAR[0] <= r["pooled_k"] <= NEAR[1]]
        all_sat = all(r["pooled_k"] >= SAT for r in rows)
        top = rows[-1]
        no_pos_top = top[f"pooled_verdict@{t:g}"] != "POSITIVE"
        sents = []
        if near_holm:
            sents.append((S_DET, S_DET_EN))
        if all_sat:
            sents.append((S_SAT, S_SAT_EN))
        if no_pos_top:
            sents.append((S_LIM, S_LIM_EN))
        readings.append({
            "delta": t,
            "smallest_positive_rate_cell": cell_pos[0]["rate_percent"] if cell_pos else "none",
            "smallest_positive_rate_holm": holm_pos[0]["rate_percent"] if holm_pos else "none",
            "smallest_positive_rate_own_variance": own_pos[0]["rate_percent"] if own_pos else "none",
            "positive_rates_cell": ",".join(r["rate_percent"] for r in cell_pos),
            "positive_rates_holm": ",".join(r["rate_percent"] for r in holm_pos),
            "decide_transfers_holm": ",".join(K.rate_pct(a) for a in dec[f"transfers@{t:g}"].split(",") if a),
            "positive_rates_own_variance": ",".join(r["rate_percent"] for r in own_pos),
            "near_delta_positive_rates_cell": ",".join(r["rate_percent"] for r in near_cell),
            "near_delta_positive_rates_holm": ",".join(r["rate_percent"] for r in near_holm),
            "all_rates_saturated": all_sat, "positive_at_0.3%": not no_pos_top,
            **shape,
            "registered_sentences": " | ".join(s for s, _ in sents) or "none of the three registered sentences applies",
            "registered_sentences_en": " | ".join(e for _, e in sents) or "none applies",
            "main_reading": t == 0.05})
    checks.append({"check": "G3: Holm-flagged rates of analyze.decide = rates with holm_p_pos < 0.025 at every delta",
                   "expected": "equal", "got": ";".join(f"{r['decide_transfers_holm']}|{r['positive_rates_holm']}" for r in readings),
                   "ok": all(set(r["decide_transfers_holm"].split(",")) == set(r["positive_rates_holm"].split(","))
                             for r in readings)})
    pool_row = [{"pool": "A13 G3 four-rate seed pool (bert/sst2/logit, label-kept BadNets rows)",
                 "units": ",".join(K.rate_pct(u) for u in pool["attacks"]), "seeds": ",".join(map(str, pool["seeds"])),
                 "m": len(pool["attacks"]), "r": len(pool["seeds"]), "s_pool": pool["s2"] ** 0.5,
                 "df_satterthwaite": pool["df"], "ms_seed": pool["ms_seed"], "ms_res": pool["ms_res"],
                 "df_welch": K.welch_df([res[u]["s2u"] for u in pool["attacks"]], len(pool["seeds"])),
                 "excluded_seeds": ";".join(excluded) or "none"}]
    K.write_csv(rows, os.path.join(out, "g3_rates.csv"), """
        A13 G3, one row per dose rate (BERT SST-2, BadNets teacher t42 vs its clean twin, logit KD, KD corpus
        seed 42 with label-kept BadNets-triggered rows at the rate). k = P - C on the 912 BadNets-triggered
        non-target test inputs (official GPU bf16 predictions); k_s<seed> per student seed; pooled_* = the
        frozen seed-pooled rule (addendum.seed_pooled: k = mean_s k_s, SE^2 = s2_pool / r + v) with s2_pool and
        Satterthwaite df from analyze.pool_seed_variance over the four rate cells (g3_pool.csv), t_contrast
        verdicts at delta in {0.02, 0.05, 0.10}; holm_p_pos = Holm over the four rates' one-sided p_pos (the
        'detection' control), holm_detect = holm_p_pos < 0.025; own_* = A3.1 own variance (s2_rate / r + v,
        df r-1; own_holm_p_pos = Holm over the own-variance p_pos); max_* = A3.1 max rule;
        pooling_dependent = deltas where pooled and own or max verdicts differ. triggered_rows /
        non_target_triggered_rows = dose (g9-a15cpu/g3_dose.csv: trigger rows, and those whose source row has
        a non-target label, i.e. where the two teachers' soft labels can differ). teacher_*_rate = teacher
        hit rates on the same inputs (identical to the frozen BadNets cell). k_over_k_kl01 = k / the 1% anchor's
        Tango k. saturated = k >= 0.90; near_delta = 0.02 <= k <= 0.15 (operationalisations, README).""")
    K.write_csv(pool_row, os.path.join(out, "g3_pool.csv"), """
        A13 G3 seed pool over the four rate cells: analyze.pool_seed_variance with the rates as units
        (s_pool = sqrt((MS_seed + (m-1) MS_res) / m), Satterthwaite df); df_welch = (sum s2_u)^2 (r-1) / sum s2_u^2
        (A15 E3 form, descriptive).""")
    K.write_csv(anchors, os.path.join(out, "g3_anchors.csv"), """
        Upper anchors of A13 G3 (seed 42 only): the frozen positive controls tifs_pc_bert_kl01 (1% label-kept
        BadNets rows) and tifs_pc_bert_kl10 (10%), BERT SST-2, from the frozen analyze.secondary_table (Tango
        interval, and transported with the BERT SST-2 logit pool). Anchors only; not part of the G3 pool or Holm.""")
    K.write_csv(readings, os.path.join(out, "g3_readings.csv"), """
        A13 G3 readings per delta: smallest rate POSITIVE by cell verdict (pooled), with Holm control across the
        four rates (cell POSITIVE and holm_p_pos < 0.025), and under own variance; the shape of k over the rate;
        the registered sentences that apply (detection: a Holm-controlled POSITIVE rate with 0.02 <= k <= 0.15;
        saturation: every rate k >= 0.90; limitation: no POSITIVE at 0.3%). main_reading = delta 0.05.""")
    K.write_csv(checks, os.path.join(out, "check_g3.csv"), "Self-checks of the G3 script.")
    return {"rows": rows, "pool": pool_row[0], "anchors": anchors, "readings": readings, "excluded": excluded,
            "checks": checks}


def main():
    out = K.out_dir()
    s = run(out)
    for r in s["rows"]:
        print(f"  {r['rate_percent']:>6}: k {r['pooled_k']:+.4f} [{r['pooled_lo']:+.4f}, {r['pooled_hi']:+.4f}] "
              + " ".join(r[f'pooled_verdict@{t:g}'][:3] for t in TOLS)
              + f"  own [{r['own_lo']:+.4f}, {r['own_hi']:+.4f}] " + " ".join(r[f'own_verdict@{t:g}'][:3] for t in TOLS)
              + f"  holm@0.05 {r['holm_p_pos@0.05']:.3g}")
    print("  pool", s["pool"]["s_pool"], s["pool"]["df_satterthwaite"])
    for r in s["readings"]:
        print(f"  delta {r['delta']}: smallest POS cell {r['smallest_positive_rate_cell']}, holm {r['smallest_positive_rate_holm']}"
              f", own {r['smallest_positive_rate_own_variance']}; sentences: {r['registered_sentences']}")
    K.dump_json(s, os.path.join(out, "logs", "g3_summary.json"))


if __name__ == "__main__":
    main()
