#!/usr/bin/env python3
"""A15-c: GPT-2-xl SST-2 ATBA (public GPT-2 trigger, suffix): teacher gate, cell verdicts and the auxiliary
three-attack GPT-2 battery (BadNets, AddSent, ATBA), on the official and the whitespace-preserving inputs.

A15 post hoc analysis; changes no primary verdict, battery decision or the paper-wide Holm.

Registered (docs/tifs_addendum.md A15-c; claim rule of the 10/27 20:58 KST amendment):
  * Gate: the frozen teacher gate (ASR >= 0.90, clean-accuracy drop <= 0.05) on the OB-selected ATBA teacher
    vs the rebuilt GPT-2 twin at the same epoch (runs/tifs/tifs_x15_c_gate/gate.json, gate_c.py). Failed ->
    reported as a failure without students (as A7); the students are then blocked by the queue.
  * Cell: P and C students, seeds 42-44 (tifs_x15_c_gpt2_atba_s*). Seed-pooled with the pool of the
    auxiliary battery; verdicts under the registered / own-variance / claim rules, Welch df.
  * Auxiliary battery: the frozen analyze.primary_table / primary_battery on the frozen cells plus the
    A15-c cells (gpt2/sst2/logit then has three attacks; its pool is recomputed over them). Three readings:
      official  - frozen GPT-2 BadNets / AddSent cells + A15-c official cells;
      em        - the epoch-matched GPT-2 cells of A15-b1 (AddSent C_e5; BadNets frozen C, e2) + A15-c (whose
                  C is already at the ATBA teacher's epoch) (H2: the epoch-matched reading sets the
                  interpretation);
      ws        - the ws re-scores of all three (A15-a frame + the A15-c ws cells).
    Outside the 31-cell family: no paper-wide Holm is recomputed with this cell.

Outputs: c_gate.csv, c_cells.csv, c_battery.csv, c_claims.csv, c_floor.csv, check_c.csv.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402

import a15lib as L  # noqa: E402
import part_a  # noqa: E402
import part_b1  # noqa: E402

A, AD, TOLS = L.A, L.AD, L.TOLS
KEYS = [("gpt2", "sst2", "logit", "addsent"), ("gpt2", "sst2", "logit", "atba"), ("gpt2", "sst2", "logit", "badnets")]
BAT = [("gpt2", "sst2", "logit")]


def gate_rows(runs, teachers_x15):
    p = os.path.join(runs, "tifs_x15_c_gate", "gate.json")
    st = L.status_of(runs, "tifs_x15_c_gate")
    row = {"file": L.rel(p), "present": os.path.exists(p), "job_status": st,
           "rule": "ASR >= 0.90 and clean-accuracy drop <= 0.05 (analyze.qualified); P = OB-selected ATBA "
                   "teacher, C = rebuilt GPT-2 twin at the same epoch"}
    if row["present"]:
        X = L.load_json(p)
        row.update({"passed": X.get("passed"), "poisoned_selected_epoch": X.get("poisoned_selected_epoch"),
                    "asr": (X.get("poisoned") or {}).get("asr"),
                    "accuracy_P": (X.get("poisoned") or {}).get("accuracy"),
                    "accuracy_C_same_epoch": (X.get("clean_same_epoch") or {}).get("accuracy"),
                    "accuracy_drop": X.get("accuracy_drop"),
                    "accuracy_C_ob_selected": (X.get("clean_ob_selected") or {}).get("accuracy"),
                    "passed_vs_ob_selected_twin": X.get("passed_vs_ob_selected_twin"),
                    "trigger": X.get("trigger"), "position": X.get("position"), "n_trig": X.get("n_trig"),
                    "n_test": X.get("n_test")})
        row["reading"] = ("gate passed: students trained (A15-c)" if X.get("passed") else
                          "gate FAILED: reported as a failure without students (A7); the A15-c students are "
                          "blocked")
    else:
        row["reading"] = "gate not run (job " + st + ")"
    ej = os.path.join(teachers_x15, "poisoned", "ob_atba_gpt2-xl_sst-2_seed42", "epochs.json")
    if os.path.exists(ej):
        E = L.load_json(ej)
        row.update({"epochs_json_ob_selected_epoch": E.get("ob_selected_epoch"),
                    "epochs_json_root_matches_selected_epoch": E.get("root_matches_selected_epoch")})
    return [row]


def main():
    ns = L.args()
    st, F0 = L.state(ns.runs)
    out = ns.out
    x15 = L.load_x15(ns.runs)
    c_all = [c for c in x15 if c["part"] == "c"]
    c_cells, excluded, floor_rows = L.apply_floor(c_all, "c")
    gate = gate_rows(ns.runs, ns.teachers_x15)
    L.write_csv(gate, os.path.join(out, "c_gate.csv"), """
        A15-c teacher gate (gate_c.py, runs/tifs/tifs_x15_c_gate/gate.json): ASR of the OB-selected GPT-2 ATBA
        teacher on the 912 non-target SST-2 inputs (registry GPT-2 ATBA trigger, suffix), clean accuracy of the
        teacher and of the rebuilt GPT-2 twin at the same epoch (and at the twin's OB-selected epoch), drop,
        passed (frozen rule); epochs_json_* = the teacher's epochs.json (selected epoch).""")
    rows, bats, claims, checks = [], [], [], []
    if not c_cells:
        rows = [{"note": "no A15-c cells (" + gate[0]["reading"] + ")"}]
    else:
        frames = {"official": L.frame(st, list(st.cells) + c_cells, "c_official")}
        em_cells, info = part_b1.build_em(st, x15, ns.runs)
        frames["em"] = L.frame(st, list(em_cells) + c_cells, "c_em")
        W = part_a.build_ws(st, ns.runs, x15)
        ws_c = [W["subst"]["ws"][part_a.rkey(c)] for c in c_cells if part_a.rkey(c) in W["subst"]["ws"]]
        _, _, incomplete = part_a.substitute_complete(list(st.cells), W["subst"]["ws"],
                                                      lambda c: c["run"] in L.WS_DECODER_BATTERY)
        gpt2_incomplete = sorted("/".join(map(str, g[:4])) for g in incomplete
                                 if g[:2] == ("gpt2", "sst2") and g[2] != "ce" and not g[6])
        # the ws reading only when every cell of the auxiliary battery is ws for every seed (never ws and
        # official vectors in one cell or one battery)
        ws_reading = bool(ws_c) and len(ws_c) == len(c_cells) and not gpt2_incomplete
        if ws_reading:
            frames["ws"] = L.frame(st, part_a.frame_cells(st, W, "ws") + ws_c, "c_ws")
        Fem2 = L.frame(st, em_cells, "em")   # registered two-attack battery, epoch-matched
        Fws2 = L.frame(st, part_a.frame_cells(st, W, "ws"), "ws")
        ref2 = {"official": F0, "em": Fem2, "ws": Fws2}
        for name, F in frames.items():
            for key in KEYS:
                r = F["prim"].get(key)
                if r is None:
                    continue
                row = {"reading": name, "fam": key[0], "dataset": key[1], "kd": key[2], "attack": key[3],
                       "battery": "auxiliary 3-attack (BadNets, AddSent, ATBA)",
                       **L.all_fields(r, "", family=False)}
                two = ref2[name]["prim"].get(key)
                if two is not None:
                    row.update({"two_attack_reg_verdict@" + f"{t:g}": two[f"verdict@{t:g}"] for t in TOLS})
                    row.update({"two_attack_claim@" + f"{t:g}": two[f"claim@{t:g}"] for t in TOLS})
                    row["note"] = ("the registered two-attack cell of this reading (pool over BadNets, AddSent) "
                                   "beside the auxiliary-battery verdict")
                else:
                    row["note"] = "A15-c cell (post hoc, outside the 31-cell family)"
                if key[3] == "atba":
                    cs = [c for c in (ws_c if name == "ws" else c_cells)]
                    row["teacher_asr"] = L.mean([c.get("teacher_asr") for c in c_cells])
                    row["teacher_qualified_stage2"] = all(A.qualified(c) for c in c_cells)
                    row["seeds_used"] = ",".join(str(c["seed"]) for c in sorted(cs, key=lambda c: c["seed"]))
                rows.append(row)
            for b in L.battery_comparison(ref2[name], F, BAT, "aux_vs_registered", "registered2_", "aux3_"):
                bats.append({"reading": name, **b})
            claims += L.claim_summary(F, f"c_{name}")
        checks.append({"check": "A15-c cells present for seeds 42-44 and none floor-excluded",
                       "ok": sorted(c["seed"] for c in c_cells) == [42, 43, 44] and not excluded,
                       "note": L.missing_note([f"s{s}" for s in (42, 43, 44)
                                               if s not in {c['seed'] for c in c_cells}]
                                              + [f"floor-excluded {r}" for r in excluded])})
        checks.append({"check": "A15-c ws re-scores present for every A15-c cell and every GPT-2 SST-2 battery cell",
                       "ok": ws_reading,
                       "note": f"A15-c {len(ws_c)} of {len(c_cells)}; GPT-2 battery cells without a complete ws "
                               f"re-score: {','.join(gpt2_incomplete) or 'none'}"
                               + ("" if ws_reading else "; no ws reading (never mixed with official vectors)")})
        checks.append({"check": "A15-c C teacher epoch (provenance) = the ATBA teacher's selected epoch (gate.json; "
                                "recorded: every c config sets record_provenance)",
                       "ok": all(part_b1.provenance_epoch(c) is not None
                                 and part_b1.provenance_epoch(c) == gate[0].get("poisoned_selected_epoch")
                                 for c in c_cells),
                       "note": ",".join(str(part_b1.provenance_epoch(c)) for c in c_cells)})
        checks += L.pool_identity_rows(dict(frames, em_two_attack=Fem2, ws_two_attack=Fws2))
    jn = L.part_jobs_note(ns.runs, "c")
    L.write_csv(L.with_jobs(rows, jn), os.path.join(out, "c_cells.csv"), """
        A15-c cells of the auxiliary three-attack GPT-2 SST-2 battery, per reading (official / em / ws; see the
        script docstring): the frozen primary_table on the frozen cells plus the A15-c cells. Column families
        as in b1_cells.csv without the family columns (reg_*, own_*, claim@d, pmax_*, welch_*);
        two_attack_* = the registered two-attack cell of the same reading (BadNets / AddSent only).
        teacher_asr = Stage 2 ATBA teacher ASR (mean over the seeds' results); teacher_qualified_stage2 = the
        frozen gate on the Stage 2 teacher numbers.""")
    L.write_csv(L.with_jobs(bats, jn), os.path.join(out, "c_battery.csv"), """
        A15-c auxiliary battery decision (BadNets, AddSent, ATBA) per reading, beside the registered two-attack
        GPT-2 decision of the same reading (registered2_*: frozen / epoch-matched / ws frame; aux3_*: with the
        A15-c cell). Decisions: registered = frozen analyze.primary_battery; own / claim / welch = frozen
        analyze.decide (see b1_battery.csv). aux_vs_registered_<rule> = deltas where they differ.""")
    L.write_csv(claims, os.path.join(out, "c_claims.csv"), "Claim-rule counts of the A15-c frames (see "
                "b1_claims.csv); frames include every frozen cell plus the A15-c cells, so their 'F31' family columns "
                "hold 32 cells: not a registered family (A15-c is outside the 31-cell family).")
    L.write_csv(floor_rows, os.path.join(out, "c_floor.csv"), """
        A8 student floor of the A15-c students (P, C): Stage 2 clean accuracy > SST-2 majority share + 0.10.""")
    checks = (checks or [{"check": "A15-c cells present", "ok": False, "note": gate[0]["reading"]}]) \
        + L.excluded_rows(ns.runs)
    L.write_csv(checks, os.path.join(out, "check_c.csv"), "A15-c self-checks.")
    print(f"part c: gate {gate[0].get('passed')} ({gate[0]['reading']}); cells {len(c_cells)}")


if __name__ == "__main__":
    main()
