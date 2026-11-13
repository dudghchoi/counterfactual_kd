#!/usr/bin/env python3
"""A18 analysis: GPT-2 AG News ATBA (docs/tifs_addendum.md A18, registered 10/31 16:05 KST, snapshot
a18_registration). Post hoc, outside the 31-cell family: no frozen verdict, battery decision, paper-wide Holm,
A13, A15 or A17 result changes, and no paper-wide Holm is recomputed with this cell.

A15-c (scripts/tifs/aux/g8-a15/analysis/part_c.py) moved to AG News. Engine: the A15 analysis library
(a15lib, imported unchanged): the frozen analyze.primary_table / primary_battery on a frame of cells (pool and
Satterthwaite df recomputed over the frame's attacks), the A3.1 own variance (df r - 1), the claim rule, the
Welch df column and the frozen analyze.decide for the battery. The frozen tables are reproduced first
(a15lib.state gate; the process stops on any mismatch). A8 student floor as A15 (a15lib.apply_floor). The A17
GPT-2 AG News cells are read with the A17 analysis loaders (g11-a17/analysis/a17.py, imported unchanged).

Readings (registered):
  em   C at the ATBA teacher's OB-selected epoch (A17 twin, teachers_x17), the A15-b1 epoch-matched frozen
       frame (part_b1.build_em) and the A17 GPT-2 AG News cells (C = twin epoch 5 = the twin's OB epoch).
       If the ATBA teacher's epoch is not 5 the procedure reading (C = the twin's OB-selected checkpoint) is
       not computed (registered, as A15-c); if it is 5 the two readings coincide.
  ws   whitespace-preserving re-scores: the A15-a ws frame (part_a) plus the A17 ws re-scores of the A17 AG
       News cells (runs/tifs/tifs_x17_ws) and the A18 ones (runs/tifs/tifs_x18_ws); read only when every A17
       and A18 cell of the battery has a ws re-score for every seed (never mixed with official vectors).
Battery: the auxiliary GPT-2 AG News {BadNets, AddSent, ATBA} (pool over the three), beside the A17
two-attack GPT-2 AG News decision of the same reading (registered2_*). Inputs whose job is not terminal-ok
are left out (eval / ws status markers).

Outputs (results/tifs/addendum/aux/g12-a18/): a18_jobs.csv, a18_gate.csv, a18_floor.csv, a18_cells.csv,
a18_battery.csv, a18_claims.csv, a18_descriptive.csv, check_a18.csv, summary.json.

    python a18.py [--runs DIR] [--out DIR]        (normally through run_all.sh, once)
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import glob  # noqa: E402
import importlib.util  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "tifs", "aux", "g8-a15", "analysis"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "tifs"))

import a15lib as L  # noqa: E402  (A15 analysis library, unchanged)
import part_a  # noqa: E402
import part_b1  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "a17", os.path.join(ROOT, "scripts", "tifs", "aux", "g11-a17", "analysis", "a17.py"))
A17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A17)   # A17 analysis loaders (unchanged; main() is not run)

A, TOLS = L.A, L.TOLS
STATUS = ("A18 post hoc analysis; outside the 31-cell family; changes no primary verdict, battery decision, "
          "paper-wide Holm, A13, A15 or A17 result")
DEFAULT_OUT = os.path.join(L.AUX, "g12-a18")
X18_WS = "tifs_x18_ws"
SEEDS = (42, 43, 44)
TWIN_OB_EPOCH = 5             # A17 AG News twin, OB-selected (runs/tifs/tifs_x17_agob/decision.json)
KEYS = [("gpt2", "agnews", "logit", "addsent"), ("gpt2", "agnews", "logit", "atba"),
        ("gpt2", "agnews", "logit", "badnets")]
BAT = [("gpt2", "agnews", "logit")]
X18_RE = [("B", re.compile(r"^tifs_x18_(?P<x18part>a)_(?P<fam>gpt2)_(?P<bds>agnews)_s(?P<sseed>\d+)$"))]
RUNS18 = [f"tifs_x18_a_gpt2_agnews_s{s}" for s in SEEDS]
ATBA_TEACHER = os.path.join(ROOT, "results", "tifs", "teachers_x17", "poisoned", "ob_atba_gpt2-xl_agnews_seed42")
A17_CELLS = os.path.join(L.AUX, "g11-a17", "a17_cells.csv")


def args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=L.DEFAULT_RUNS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ns = ap.parse_args()
    os.makedirs(ns.out, exist_ok=True)
    return ns


# ── inputs ───────────────────────────────────────────────────────────


def a18_jobs():
    import queue_plan as Q  # read only; pure
    return Q.a18_pool_jobs()


def ws_ok(runs, run):
    return os.path.exists(os.path.join(runs, "_status", f"{X18_WS}__{run}.ok"))


def load_x18(runs):
    """Every A18 cell through the frozen analyze.load_cells (RUN_RE swapped to X18_RE, restored at once), only
    runs whose evaluation job is terminal-ok."""
    gated = [(b, L._MarkerGate(rx, lambda run: L.eval_ok(runs, run))) for b, rx in X18_RE]
    with L.run_re(gated):
        cells = A.load_cells(runs)
        L.attach(cells, runs)
    return cells


def load_ws18(runs, mode="ws"):
    """The A18 ws re-scores (runs/tifs/tifs_x18_ws/<run>/<tag>.<mode>.preds.json) in the frozen cell format, as
    a17.load_ws17 reads the A17 ones, only runs whose re-score job is terminal-ok."""
    base = os.path.join(runs, X18_WS)
    files = []
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        run = os.path.basename(d)
        if not os.path.isdir(d) or not ws_ok(runs, run):
            continue
        for p in sorted(glob.glob(os.path.join(d, f"*.{mode}.preds.json"))):
            tag = os.path.basename(p)[: -len(f".{mode}.preds.json")]
            res = os.path.join(d, f"{tag}.{mode}.json")
            if os.path.exists(res):
                files.append((run, tag, p, res))
    if not files:
        return []
    tmp = tempfile.mkdtemp(prefix=f"a18_{mode}_")
    try:
        L._link_view(files, tmp)
        with L.run_re(X18_RE):
            cells = A.load_cells(tmp)
            idx = L.preds_index(tmp)
        for c in cells:
            p, res = idx[(c["run"], c["dataset"], c["attack"], c["kd"], c["seed"])]
            d = L.load_json(p)
            c["tag"] = os.path.basename(p)[: -len(".preds.json")]
            c["preds_path"], c["result_path"] = os.path.realpath(p), os.path.realpath(res)
            c["TPc"], c["TCc"] = d.get("teacher_p_clean"), d.get("teacher_c_clean")
            c["meta"] = d.get("meta") or {}
            c["result"] = L.load_json(res)
            c["mode"] = mode
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return cells


def gate_rows(runs):
    p = os.path.join(runs, "tifs_x18_gate", "gate.json")
    st = L.status_of(runs, "tifs_x18_gate")
    row = {"file": L.rel(p), "present": os.path.exists(p), "job_status": st,
           "rule": "ASR >= 0.90 and clean-accuracy drop <= 0.05 (analyze.qualified); P = OB-selected ATBA "
                   "teacher, C = A17 GPT-2 AG News twin at the same epoch"}
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
        row["reading"] = ("gate passed: students trained (A18)" if X.get("passed") else
                          "gate FAILED: reported as a failure without students; the A18 students are blocked")
    else:
        row["reading"] = "gate not run (job " + st + ")"
    ej = os.path.join(ATBA_TEACHER, "epochs.json")
    if os.path.exists(ej):
        E = L.load_json(ej)
        row.update({"epochs_json_ob_selected_epoch": E.get("ob_selected_epoch"),
                    "epochs_json_root_matches_selected_epoch": E.get("root_matches_selected_epoch")})
    return [row]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


# ── main ─────────────────────────────────────────────────────────────


def main():
    ns = args()
    runs, out = ns.runs, ns.out
    st, F0 = L.state(runs)
    checks = []

    jobs = a18_jobs()
    L.write_csv([{"job": j["id"], "status": L.status_of(runs, j["id"]), "mem_gb": j["mem_gb"],
                  "after": ";".join(j.get("after", []))} for j in jobs],
                os.path.join(out, "a18_jobs.csv"),
                "Queue v11 A18 jobs and their status markers at analysis time (runs/tifs/_status).")
    gate = gate_rows(runs)
    L.write_csv(gate, os.path.join(out, "a18_gate.csv"), """
        A18 teacher gate (g8-a15/gate_c.py --teacher-dir results/tifs/teachers_x17 --dataset agnews;
        runs/tifs/tifs_x18_gate/gate.json): ASR of the OB-selected GPT-2-xl AG News ATBA teacher on the 5,700
        non-target AG News inputs (registry GPT-2 AG News trigger, suffix), clean accuracy of the teacher and of the
        A17 twin at the same epoch (and at the twin's OB-selected epoch), drop, passed (frozen rule).""")
    p_epoch = gate[0].get("poisoned_selected_epoch")

    # cells, floor
    x18_all = load_x18(runs)
    x18, excluded, floor_rows = L.apply_floor(x18_all, "a18")
    L.write_csv(floor_rows, os.path.join(out, "a18_floor.csv"), """
        A8 student floor of the A18 students (P, C): Stage 2 clean accuracy > majority share + 0.10. A run with
        any student at or below the floor is left out of every verdict.""")
    not_ok = [r for r in RUNS18
              if glob.glob(os.path.join(runs, r, "artifacts", "results", "*.preds.json")) and not L.eval_ok(runs, r)]
    checks.append({"check": "A18 inputs present but job not terminal-ok: excluded", "ok": not not_ok,
                   "kind": "info", "note": ",".join(not_ok) or "none"})
    checks.append({"check": "A18 cells present (ATBA x seeds 42-44) and none floor-excluded",
                   "ok": sorted(c["seed"] for c in x18 if c["kd"] == "logit") == list(SEEDS) and not excluded,
                   "note": L.missing_note([f"s{s}" for s in SEEDS if not any(c["seed"] == s for c in x18)]
                                          + [f"floor-excluded {r}" for r in excluded])})
    checks.append({"check": "A18 C teacher epoch (preds meta.provenance) = the ATBA teacher's selected epoch "
                            "(gate.json)",
                   "ok": bool(x18) and all(part_b1.provenance_epoch(c) == p_epoch for c in x18),
                   "note": ",".join(str(part_b1.provenance_epoch(c)) for c in x18) + f"; gate epoch {p_epoch}"})
    if x18:
        ok = all(c["TP"] == x18[0]["TP"] and c["TC"] == x18[0]["TC"] for c in x18)
        checks.append({"check": "A18 teacher vectors identical across student seeds", "ok": ok,
                       "note": f"{len(x18)} cells"})
    notes = []
    if p_epoch is not None and int(p_epoch) != TWIN_OB_EPOCH:
        notes.append(f"ATBA teacher epoch {p_epoch} != twin OB epoch {TWIN_OB_EPOCH}: the procedure reading is not "
                     "computed (registered, as A15-c); em is the reading")
    elif p_epoch is not None:
        notes.append(f"ATBA teacher epoch {p_epoch} = twin OB epoch: procedure reading = em")

    # A17 GPT-2 AG News cells (A17 loaders, unchanged)
    x17 = A17.load_x17(runs)
    x17, ex17, _ = L.apply_floor(x17, "a17")
    ag = [c for c in x17 if c.get("x17part") == "ag"]
    ag_logit = [c for c in ag if c["kd"] != "ce"]
    R_ag = [c for c in ag if c["kd"] == "ce"]
    checks.append({"check": "A17 GPT-2 AG News cells present (BadNets, AddSent x seeds 42-44)",
                   "ok": A17.complete(x17, "ag", ("badnets", "addsent")) and not ex17,
                   "note": f"{len(ag_logit)} logit cells; floor-excluded {','.join(ex17) or 'none'}"})

    # readings
    x15 = L.load_x15(runs)
    em_frozen, _ = part_b1.build_em(st, x15, runs)
    frames, ref2 = {}, {}
    if x18:
        frames["em"] = L.frame(st, list(em_frozen) + ag_logit + R_ag + x18, "a18_em")
    ref2["em"] = L.frame(st, list(em_frozen) + ag_logit + R_ag, "a17_em")
    W = part_a.build_ws(st, runs, x15)
    ws17 = {part_a.rkey(c): c for c in A17.load_ws17(runs, "ws")}
    ws18 = {part_a.rkey(c): c for c in load_ws18(runs, "ws")}
    ag_cells = ag_logit + R_ag
    ws_ag = [part_a.substitute(c, ws17[part_a.rkey(c)], "ws") for c in ag_cells if part_a.rkey(c) in ws17]
    ws_x18 = [part_a.substitute(c, ws18[part_a.rkey(c)], "ws") for c in x18 if part_a.rkey(c) in ws18]
    ws_reading = bool(x18) and len(ws_ag) == len(ag_cells) and len(ws_x18) == len(x18)
    ws_frozen = part_a.frame_cells(st, W, "ws")
    if ws_reading:
        frames["ws"] = L.frame(st, ws_frozen + ws_ag + ws_x18, "a18_ws")
        ref2["ws"] = L.frame(st, ws_frozen + ws_ag, "a17_ws")
    else:
        notes.append(f"ws reading not available: A17 {len(ws_ag)} of {len(ag_cells)}, A18 {len(ws_x18)} of "
                     f"{len(x18)} cells re-scored")
    checks.append({"check": "ws re-scores present for every A17 and A18 cell of the battery", "ok": ws_reading,
                   "note": f"A17 {len(ws_ag)}/{len(ag_cells)}, A18 {len(ws_x18)}/{len(x18)}"})

    # the A17 two-attack frames reproduce the A17 analysis outputs (k, interval, verdicts)
    if os.path.exists(A17_CELLS):
        old = {(r["reading"], r["attack"]): r for r in read_csv(A17_CELLS) if r["dataset"] == "agnews"}
        for name, F in ref2.items():
            for key in KEYS:
                r = F["prim"].get(key)
                o = old.get((name if name != "em" else "em", key[3]))
                if r is None or o is None:
                    continue
                same = (abs(r["k"] - float(o["k"])) < 1e-12 and abs(r["lo"] - float(o["lo"])) < 1e-12
                        and abs(r["hi"] - float(o["hi"])) < 1e-12
                        and all(r[f"verdict@{t:g}"] == o[f"reg_verdict@{t:g}"] for t in TOLS))
                checks.append({"check": f"A17 {name} {key[3]}: two-attack frame = a17_cells.csv", "ok": same,
                               "note": f"k {r['k']:+.6f} vs {float(o['k']):+.6f}"})

    # cell, battery, claim and descriptive tables
    cell_rows, bat_rows, claims, desc = [], [], [], []
    for name, F in frames.items():
        for key in KEYS:
            r = F["prim"].get(key)
            if r is None:
                continue
            row = {"reading": name, "fam": key[0], "dataset": key[1], "kd": key[2], "attack": key[3],
                   "a18_cell": key[3] == "atba",
                   "battery": "auxiliary GPT-2 AG News (BadNets, AddSent, ATBA)",
                   "teacher_qualified": r["teacher_qualified"], "raw_asr": r.get("raw_asr"),
                   "C_trig": r.get("C_trig"), "delta_star": r.get("delta_star"),
                   **L.all_fields(r, "", family=False)}
            two = ref2[name]["prim"].get(key)
            if two is not None:
                row.update({f"two_attack_reg_verdict@{t:g}": two[f"verdict@{t:g}"] for t in TOLS})
                row.update({f"two_attack_claim@{t:g}": two[f"claim@{t:g}"] for t in TOLS})
            cell_rows.append(row)
            if key[3] == "atba":
                cs = [c for c in F["cells"] if (c["fam"], c["dataset"], c["kd"], c["attack"]) == key
                      and str(c["run"]).startswith("tifs_x18_")]
                cs = sorted(cs, key=lambda c: c["seed"])
                c0 = next((c for c in cs if c["seed"] == 42), cs[0] if cs else None)
                kT = L.paired_contrast(c0["TP"], c0["TC"], TOLS) if c0 and c0.get("TP") and c0.get("TC") else None
                desc.append({"reading": name, "fam": key[0], "dataset": key[1], "attack": key[3],
                             "seeds": ",".join(str(c["seed"]) for c in cs), "n": len(c0["P"]) if c0 else None,
                             "A_P_by_seed": ";".join(f"{L.mean(c['P']):.4f}" for c in cs),
                             "A_C_by_seed": ";".join(f"{L.mean(c['C']):.4f}" for c in cs),
                             "A_P": L.mean([L.mean(c["P"]) for c in cs]) if cs else None,
                             "A_C": L.mean([L.mean(c["C"]) for c in cs]) if cs else None,
                             "teacher_P_asr": L.mean(c0["TP"]) if c0 and c0.get("TP") else None,
                             "teacher_C_asr": L.mean(c0["TC"]) if c0 and c0.get("TC") else None,
                             "k_T": kT["k"] if kT else None, "k_T_lo": kT["lo"] if kT else None,
                             "k_T_hi": kT["hi"] if kT else None,
                             "k_over_kT": (r["k"] / kT["k"]) if kT and kT["k"] else None,
                             "teacher_asr": c0["teacher_asr"] if c0 else None,
                             "teacher_ca_P": c0["teacher_ca"] if c0 else None,
                             "teacher_ca_C": c0["teacher_ca_null"] if c0 else None,
                             "student_ca_P_mean": L.mean([c["ca_p"] for c in cs]) if cs else None,
                             "student_ca_C_mean": L.mean([c["ca_c"] for c in cs]) if cs else None,
                             "teacher_qualified": r["teacher_qualified"],
                             "teacher_qualified_stage2": all(A.qualified(c) for c in cs) if cs else None})
        bat_rows += [{"reading": name, **b} for b in
                     L.battery_comparison(ref2[name], F, BAT, "aux_vs_registered", "registered2_", "aux3_")]
        claims += L.claim_summary(F, f"a18_{name}")
    if not frames:
        cell_rows = [{"note": "no A18 cells (" + gate[0]["reading"] + ")"}]
    L.write_csv(cell_rows, os.path.join(out, "a18_cells.csv"), """
        A18 cells of the auxiliary three-attack GPT-2 AG News battery per reading (em / ws; see the script
        docstring): the frozen primary_table on the reading's frame. Columns as A15 c_cells.csv: registered k, SE,
        df, interval and verdict@delta (pool over BadNets, AddSent, ATBA; Satterthwaite df); own_* (A3.1 own
        variance, df r - 1); claim@delta (claim rule; rule-dependent when registered and own differ); welch_*.
        two_attack_* = the A17 two-attack cell of the same reading (BadNets, AddSent).""")
    L.write_csv(bat_rows, os.path.join(out, "a18_battery.csv"), """
        A18 battery decisions per reading: aux3_* = the auxiliary GPT-2 AG News battery with ATBA; registered2_* =
        the A17 two-attack battery of the same reading. Decisions: reg = frozen analyze.primary_battery; own / claim
        / welch = frozen analyze.decide on the own-variance, claim-rule and Welch p.""")
    L.write_csv(claims, os.path.join(out, "a18_claims.csv"), """
        Claim-rule counts of the A18 frames (a15lib.claim_summary). The frames hold every frozen cell plus the A17
        and A18 cells, so their 'F31' / 'F19' columns are not the registered families (A17 and A18 are outside
        them): read the cell and battery tables, not these family columns.""")
    L.write_csv(desc, os.path.join(out, "a18_descriptive.csv"), """
        Descriptive values of the A18 cell per reading: A(P), A(C) by student seed and mean (target rate on the
        triggered non-target inputs), teacher ASRs and k_T = teacher P - C on the same inputs (Tango paired
        interval, teachers fixed), k / k_T, teacher and student clean accuracies. No A(R): the runs have no CE
        condition (as A15-c).""")
    checks += L.pool_identity_rows(dict(frames, **{f"{k}_two_attack": v for k, v in ref2.items()}))
    checks += [{"check": "note", "ok": True, "kind": "info", "note": n} for n in notes]
    checks.append({"check": "frozen tables reproduced before any A18 computation (a15lib.state gate)", "ok": True,
                   "note": "primary, variance, battery, global_holm, attack_variance, reference_atba, replicate"})
    L.write_csv(checks, os.path.join(out, "check_a18.csv"), "A18 self-checks.")
    summary = {"status": STATUS, "time_utc": L.now_utc(), "readings": sorted(frames), "notes": notes,
               "atba_p_epoch": p_epoch, "excluded_runs": excluded,
               "checks_failed": [c["check"] for c in checks if not c["ok"] and c.get("kind") != "info"]}
    L.dump_json(summary, os.path.join(out, "summary.json"))
    print(f"A18: readings {sorted(frames)}; cells {len(x18)} (excluded runs {len(excluded)}); "
          f"failed checks {len(summary['checks_failed'])}")


if __name__ == "__main__":
    main()
