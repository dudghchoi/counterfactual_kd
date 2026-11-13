#!/usr/bin/env python3
"""A17 analysis: GPT-2 AG News BadNets / AddSent and GPT-2 SST-2 SynBkd (docs/tifs_addendum.md A17, registered
10/30 21:40 KST). Post hoc, outside the 31-cell family: no frozen verdict, battery decision, paper-wide Holm,
A13 or A15 result changes, and no paper-wide Holm is recomputed with these cells.

Engine: the A15 analysis library (scripts/tifs/aux/g8-a15/analysis/a15lib.py, imported unchanged): the frozen
analyze.primary_table / primary_battery on a frame of cells (pool and Satterthwaite df recomputed over the
frame's attacks), the A3.1 own variance (df r - 1), the claim rule, the Welch df column, and the frozen
analyze.decide for the battery under the own / claim / Welch p. The frozen tables are reproduced first
(a15lib.state gate; the process stops on any mismatch). A8 student floor as A15 (a15lib.apply_floor).

Readings (registered):
  em         interpretation (H2). The A15-b1 epoch-matched frozen frame (part_b1.build_em) plus the A17 cells
             whose C is at the P teachers' epoch: AG News C = twin epoch 5 (tifs_x17_ag_*); SST-2 SynBkd C =
             A15-b1 twin at the P epoch (tifs_x17_sb_*).
  procedure  what the fixed procedure reads (C = the OB-selected twin). The frozen frame plus, for AG News,
             tifs_x17_agob_* (R from tifs_x17_ag_gpt2_agnews_s42) when the agob decision ran them, else the
             ag cells (the decision found the twin's OB epoch = 5, so the readings coincide); for SynBkd,
             tifs_x17_sbob_* when the sbob decision ran them, else the sb cells (P epoch = 2). A part whose
             conditional was needed but is incomplete has no procedure reading (listed).
  ws         the procedure reading on the whitespace-preserving inputs: the A15-a ws frame (part_a) plus the ws
             re-scores of the procedure A17 cells (runs/tifs/tifs_x17_ws, A15 ws_score.py); read only when every
             A17 cell of the reading has a ws re-score for every seed (never mixed with official vectors).
Batteries: GPT-2 AG News {BadNets, AddSent} (new; pool over the two attacks) and the auxiliary GPT-2 SST-2
{BadNets, AddSent, SynBkd} (pool over the three), each beside the registered two-attack GPT-2 SST-2 decision of
the same reading. Inputs whose job is not terminal-ok are left out (eval / ws status markers).

Outputs (results/tifs/addendum/aux/g11-a17/): a17_jobs.csv, a17_gate.csv, a17_decisions.csv, a17_floor.csv,
a17_cells.csv, a17_battery.csv, a17_claims.csv, a17_descriptive.csv, a17_identity.csv, check_a17.csv,
summary.json.

    python a17.py [--runs DIR] [--out DIR]        (normally through run_all.sh, once)
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import glob  # noqa: E402
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

A, TOLS = L.A, L.TOLS
STATUS = ("A17 post hoc analysis; outside the 31-cell family; changes no primary verdict, battery decision, "
          "paper-wide Holm, A13 or A15 result")
DEFAULT_OUT = os.path.join(L.AUX, "g11-a17")
X17_WS = "tifs_x17_ws"
SEEDS = (42, 43, 44)
E_AG = 5                      # saved epoch of both 10/21 GPT-2 AG News P teachers (make_configs.P_EPOCH_AG)
FROZEN_SST2_TWIN_EPOCH = 2    # frozen GPT-2 SST-2 twin, OB-selected
DEADLINE_UTC = "2026-10-31 09:00:00 UTC"   # 18:00 KST; cut order sbob -> agob
AG_KEYS = [("gpt2", "agnews", "logit", "addsent"), ("gpt2", "agnews", "logit", "badnets")]
SB_KEYS = [("gpt2", "sst2", "logit", "addsent"), ("gpt2", "sst2", "logit", "badnets"),
           ("gpt2", "sst2", "logit", "synbkd")]
BATS = [("gpt2", "agnews", "logit"), ("gpt2", "sst2", "logit")]
X17_RE = [
    ("B", re.compile(r"^tifs_x17_(?P<x17part>ag|agob)_(?P<fam>gpt2)_(?P<bds>agnews)_s(?P<sseed>\d+)$")),
    ("B", re.compile(r"^tifs_x17_(?P<x17part>sb|sbob)_(?P<fam>gpt2)_(?P<bds>sst2)_s(?P<sseed>\d+)$")),
]
RUNS = {p: [f"tifs_x17_{p}_gpt2_{'agnews' if p.startswith('ag') else 'sst2'}_s{s}" for s in SEEDS]
        for p in ("ag", "agob", "sb", "sbob")}
TX17 = os.path.join(ROOT, "results", "tifs", "teachers_x17")
SB_TEACHER = os.path.join(TX17, "poisoned", "ob_synbkd_gpt2-xl_sst-2_seed42")
AG_TWIN = os.path.join(TX17, "clean", "ob_gpt2-xl_agnews_seed42")


def args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=L.DEFAULT_RUNS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ns = ap.parse_args()
    os.makedirs(ns.out, exist_ok=True)
    return ns


# ── inputs ───────────────────────────────────────────────────────────


def a17_jobs():
    import queue_plan as Q  # read only; pure
    return Q.a17_pool_jobs()


def eval_ok(runs, run):
    return L.eval_ok(runs, run)


def ws_ok(runs, run):
    return os.path.exists(os.path.join(runs, "_status", f"{X17_WS}__{run}.ok"))


def load_x17(runs):
    """Every A17 cell through the frozen analyze.load_cells (RUN_RE swapped to X17_RE, restored at once), only
    runs whose evaluation job is terminal-ok; 'x17part' from the run id."""
    gated = [(b, L._MarkerGate(rx, lambda run: eval_ok(runs, run))) for b, rx in X17_RE]
    with L.run_re(gated):
        cells = A.load_cells(runs)
        L.attach(cells, runs)
    return cells


def load_ws17(runs, mode="ws"):
    """The A17 ws re-scores (runs/tifs/tifs_x17_ws/<run>/<tag>.<mode>.preds.json) in the frozen cell format, the
    way a15lib.load_ws reads the A15-a ones (a link view in the Stage 2 layout, the frozen loader), only runs
    whose re-score job is terminal-ok."""
    base = os.path.join(runs, X17_WS)
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
    tmp = tempfile.mkdtemp(prefix=f"a17_{mode}_")
    try:
        L._link_view(files, tmp)
        with L.run_re(X17_RE):
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


def decision(runs, what):
    p = os.path.join(runs, f"tifs_x17_{what}", "decision.json")
    return L.load_json(p) if os.path.exists(p) else None


def gate(runs, part):
    p = os.path.join(runs, f"tifs_x17_gate_{part}", "gate.json")
    return L.load_json(p) if os.path.exists(p) else None


def complete(cells, part, attacks, kd="logit"):
    """True when every (attack, seed) of a part's logit cells is present."""
    have = {(c["attack"], c["seed"]) for c in cells if c.get("x17part") == part and c["kd"] == kd}
    return all((a, s) in have for a in attacks for s in SEEDS)


def student_dir(runs, run, ds, seed, side):
    """The C (clean-base) or R (ce) student checkpoint dir of a GPT-2 run at student seed ``seed``."""
    pre = "clean-base_logit" if side == "C" else "ce"
    pats = glob.glob(os.path.join(runs, run, "artifacts", "students", f"{pre}_gpt2_gpt2_{ds}_seed{seed}*"))
    sub = "clean_student" if side == "C" else "ce_student"
    out = [os.path.join(p, sub) for p in pats if os.path.isdir(os.path.join(p, sub))]
    return out[0] if len(out) == 1 else None


def weights_sha(d):
    f = os.path.join(d, "model.safetensors") if d else None
    return L.sha256(f) if f and os.path.exists(f) else None


# ── main ─────────────────────────────────────────────────────────────


def main():
    ns = args()
    runs, out = ns.runs, ns.out
    st, F0 = L.state(runs)
    checks = []

    # jobs, gates, decisions (names, markers and the small JSON records only)
    jobs = a17_jobs()
    job_rows = [{"job": j["id"], "status": L.status_of(runs, j["id"]), "mem_gb": j["mem_gb"],
                 "after": ";".join(j.get("after", []))} for j in jobs]
    L.write_csv(job_rows, os.path.join(out, "a17_jobs.csv"),
                "Queue v10 A17 jobs and their status markers at analysis time (runs/tifs/_status).")
    G = {p: gate(runs, p) for p in ("ag", "sb")}
    gate_rows = []
    for p, g in G.items():
        if g is None:
            gate_rows.append({"part": p, "present": False, "job_status": L.status_of(runs, f"tifs_x17_gate_{p}")})
            continue
        for a in g["attacks"]:
            gate_rows.append({"part": p, "present": True, "attack": a["attack"], "trigger": a["trigger"],
                              "poisoned_selected_epoch": a["poisoned_selected_epoch"],
                              "epoch_source": a["epoch_source"], "n_trig": a["n_trig"], "n_test": a["n_test"],
                              "asr": a["poisoned"]["asr"], "accuracy_P": a["poisoned"]["accuracy"],
                              "accuracy_C_same_epoch": a["clean_same_epoch"]["accuracy"],
                              "accuracy_drop": a["accuracy_drop"], "passed": a["passed"],
                              "accuracy_C_ob_selected": a["clean_ob_selected"]["accuracy"],
                              "passed_vs_ob_selected_twin": a["passed_vs_ob_selected_twin"],
                              "students_run": g["students_run"]})
    L.write_csv(gate_rows, os.path.join(out, "a17_gate.csv"), """
        A17 teacher gate (gate.py; runs/tifs/tifs_x17_gate_<part>/gate.json): the frozen rule (ASR >= 0.90, clean
        accuracy drop <= 0.05) for each poisoned teacher vs its twin at the same epoch (and vs the twin's OB-selected
        checkpoint). ag: exit 3 only when both attacks fail; a single failing attack stays in the tables and out of
        the verdicts (teacher_qualified on the Stage 2 numbers).""")
    D = {w: decision(runs, w) for w in ("agob", "sbob", "repro")}
    dec_rows = [{"what": w, "present": d is not None, "job_status": L.status_of(runs, f"tifs_x17_{w}"),
                 **({k: v for k, v in d.items() if not isinstance(v, (dict, list))} if d else {})}
                for w, d in D.items()]
    L.write_csv(dec_rows, os.path.join(out, "a17_decisions.csv"), """
        A17 CPU decisions (decide.py): agob / sbob = whether the procedure-C conditional students were needed (twin's
        OB epoch != 5; SynBkd P epoch != 2); repro = the AG News twin's epoch-1 dev-clean accuracy vs the 10/21
        stopped build (report only).""")
    if D["repro"] is not None:
        checks.append({"check": "AG News twin epoch-1 dev-clean accuracy = the 10/21 stopped build (report only)",
                       "ok": bool(D["repro"].get("identical")), "kind": "info",
                       "note": f"logged {D['repro'].get('logged_epoch1_dev_clean')}, rebuilt "
                               f"{D['repro'].get('rebuilt_epoch1_dev_clean')}"})

    # cells, floor
    x17_all = load_x17(runs)
    x17, excluded, floor_rows = L.apply_floor(x17_all, "a17")
    L.write_csv(floor_rows, os.path.join(out, "a17_floor.csv"), """
        A8 student floor of the A17 students (P, C; R of the CE cell): Stage 2 clean accuracy > majority share + 0.10.
        A run with any student at or below the floor is left out of every verdict.""")
    by = {p: [c for c in x17 if c.get("x17part") == p] for p in RUNS}
    not_ok = [r for p in RUNS for r in RUNS[p]
              if glob.glob(os.path.join(runs, r, "artifacts", "results", "*.preds.json")) and not eval_ok(runs, r)]
    checks.append({"check": "A17 inputs present but job not terminal-ok: excluded", "ok": not not_ok,
                   "kind": "info", "note": ",".join(not_ok) or "none"})
    checks.append({"check": "A17 runs floor-excluded", "ok": not excluded, "note": ",".join(excluded) or "none"})
    ag_ok = complete(x17, "ag", ("badnets", "addsent"))
    sb_ok = complete(x17, "sb", ("synbkd",))
    checks.append({"check": "ag cells present (BadNets, AddSent x seeds 42-44) and the seed-42 R cell",
                   "ok": ag_ok and any(c["kd"] == "ce" for c in by["ag"]),
                   "note": L.missing_note([f"{a}/s{s}" for a in ("badnets", "addsent") for s in SEEDS
                                           if not any(c["attack"] == a and c["seed"] == s and c["kd"] == "logit"
                                                      for c in by["ag"])])})
    checks.append({"check": "sb cells present (SynBkd x seeds 42-44)", "ok": sb_ok,
                   "note": L.missing_note([f"s{s}" for s in SEEDS if not any(c["seed"] == s for c in by["sb"])])})

    # provenance: C teacher epoch of every cell
    sb_epoch = None
    if os.path.exists(os.path.join(SB_TEACHER, "epochs.json")):
        sb_epoch = int(L.load_json(os.path.join(SB_TEACHER, "epochs.json"))["ob_selected_epoch"])
    exp_epoch = {"ag": E_AG, "sb": sb_epoch, "agob": None, "sbob": None}
    for p in RUNS:
        cs = [c for c in by[p] if c["kd"] != "ce"]
        if not cs:
            continue
        got = [part_b1.provenance_epoch(c) for c in cs]
        roots = {os.path.dirname(os.path.normpath(((c.get("meta") or {}).get("provenance") or {})
                                                  .get("clean_teacher") or "")) for c in cs}
        checks.append({"check": f"{p}: C teacher epoch (preds meta.provenance) = {exp_epoch[p]} "
                                "(None = the OB-selected root)",
                       "ok": all(g == exp_epoch[p] for g in got),
                       "note": ",".join(str(g) for g in got) + "; teacher dirs " + ";".join(sorted(roots))})
    # P reuse: the conditional runs' P vectors equal their source runs' (same student, same inputs)
    for cond, src in (("agob", "ag"), ("sbob", "sb")):
        if not by[cond]:
            continue
        idx = {(c["attack"], c["seed"], c["kd"]): c for c in by[src]}
        same = [idx.get((c["attack"], c["seed"], c["kd"])) is not None
                and idx[(c["attack"], c["seed"], c["kd"])]["P"] == c["P"]
                and idx[(c["attack"], c["seed"], c["kd"])]["TP"] == c["TP"] for c in by[cond]]
        checks.append({"check": f"{cond}: P and P-teacher vectors = {src}'s (reused P students)",
                       "ok": all(same), "note": f"{sum(same)}/{len(same)}"})
    # teacher vectors identical across student seeds within a part and attack
    for p in RUNS:
        for a in {c["attack"] for c in by[p] if c["kd"] != "ce"}:
            cs = [c for c in by[p] if c["attack"] == a and c["kd"] != "ce"]
            ok = all(c["TP"] == cs[0]["TP"] and c["TC"] == cs[0]["TC"] for c in cs)
            checks.append({"check": f"{p}/{a}: teacher vectors identical across student seeds", "ok": ok,
                           "note": f"{len(cs)} cells"})

    # student identity (weights) where an existing student was trained from the same teacher checkpoint
    ident = []
    ref_runs = {2: {42: "tifs_b_gpt2_sst2", 43: "tifs_b_gpt2_sst2_s43", 44: "tifs_b_gpt2_sst2_s44"},
                5: {s: f"tifs_x15_b1_gpt2_sst2_e5_s{s}" for s in SEEDS},
                1: {s: f"tifs_x15_c_gpt2_atba_s{s}" for s in SEEDS}}
    for p, e in (("sb", sb_epoch), ("sbob", FROZEN_SST2_TWIN_EPOCH)):
        for s in SEEDS:
            run = RUNS[p][SEEDS.index(s)]
            d_new = student_dir(runs, run, "sst2", s, "C")
            ref = ref_runs.get(e, {}).get(s)
            d_ref = student_dir(runs, ref, "sst2", s, "C") if ref else None
            a, b = weights_sha(d_new), weights_sha(d_ref)
            ident.append({"run": run, "seed": s, "clean_teacher_epoch": e, "student": L.rel(d_new) if d_new else None,
                          "reference_run": ref, "reference_student": L.rel(d_ref) if d_ref else None,
                          "sha256": a, "reference_sha256": b,
                          "identical": (a == b) if (a and b) else None})
    L.write_csv(ident, os.path.join(out, "a17_identity.csv"), """
        C students of the SST-2 SynBkd runs vs an existing C student distilled from the same twin checkpoint at the
        same student seed (epoch 2: frozen tifs_b_gpt2_sst2*; 5: A15-b1 e5; 1: A15-c), model.safetensors SHA-256.
        Report only (determinism of the student path; no verdict depends on it).""")

    # readings
    x15 = L.load_x15(runs)
    em_frozen, _ = part_b1.build_em(st, x15, runs)
    R_ag = [c for c in by["ag"] if c["kd"] == "ce"]
    ag_logit = [c for c in by["ag"] if c["kd"] != "ce"]
    sb_cells = by["sb"]
    proc_ag, proc_sb, notes = None, None, []
    dag, dsb = D["agob"], D["sbob"]
    if dag is not None and not dag.get("run_conditional"):
        proc_ag = ag_logit + R_ag
    elif dag is not None and complete(x17, "agob", ("badnets", "addsent")):
        proc_ag = by["agob"] + R_ag
    else:
        notes.append("AG News procedure reading not available (agob conditional needed but incomplete, or no "
                     "decision)")
    if dsb is not None and not dsb.get("run_conditional"):
        proc_sb = sb_cells
    elif dsb is not None and complete(x17, "sbob", ("synbkd",)):
        proc_sb = by["sbob"]
    else:
        notes.append("SST-2 SynBkd procedure reading not available (sbob conditional needed but incomplete, or no "
                     "decision)")
    frames = {"em": L.frame(st, list(em_frozen) + ag_logit + R_ag + sb_cells, "a17_em")}
    if proc_ag is not None or proc_sb is not None:
        frames["procedure"] = L.frame(st, list(st.cells) + (proc_ag or []) + (proc_sb or []), "a17_procedure")
    W = part_a.build_ws(st, runs, x15)
    ws_raw = {part_a.rkey(c): c for c in load_ws17(runs, "ws")}
    proc_cells = (proc_ag or []) + (proc_sb or [])
    ws_cells = [part_a.substitute(c, ws_raw[part_a.rkey(c)], "ws") for c in proc_cells
                if part_a.rkey(c) in ws_raw]
    ws_reading = bool(proc_cells) and len(ws_cells) == len(proc_cells)
    if ws_reading:
        frames["ws"] = L.frame(st, part_a.frame_cells(st, W, "ws") + ws_cells, "a17_ws")
    else:
        notes.append(f"ws reading not available: {len(ws_cells)} of {len(proc_cells)} procedure A17 cells re-scored")
    ref2 = {"em": L.frame(st, list(em_frozen), "em"), "procedure": F0,
            "ws": L.frame(st, part_a.frame_cells(st, W, "ws"), "ws")}

    # cell, battery and claim tables
    cell_rows, bat_rows, claims, desc = [], [], [], []
    for name, F in frames.items():
        for key in AG_KEYS + SB_KEYS:
            r = F["prim"].get(key)
            if r is None:
                continue
            new = key[1] == "agnews" or key[3] == "synbkd"
            row = {"reading": name, "fam": key[0], "dataset": key[1], "kd": key[2], "attack": key[3],
                   "a17_cell": new,
                   "battery": ("GPT-2 AG News (BadNets, AddSent)" if key[1] == "agnews"
                               else "auxiliary GPT-2 SST-2 (BadNets, AddSent, SynBkd)"),
                   "teacher_qualified": r["teacher_qualified"], "raw_asr": r.get("raw_asr"),
                   "C_trig": r.get("C_trig"), "delta_star": r.get("delta_star"),
                   **L.all_fields(r, "", family=False)}
            two = ref2[name]["prim"].get(key)
            if two is not None and key[1] == "sst2":
                row.update({f"two_attack_reg_verdict@{t:g}": two[f"verdict@{t:g}"] for t in TOLS})
                row.update({f"two_attack_claim@{t:g}": two[f"claim@{t:g}"] for t in TOLS})
            cell_rows.append(row)
            if new:
                cs = [c for c in F["cells"] if (c["fam"], c["dataset"], c["kd"], c["attack"]) == key
                      and c["block"] == "G" and c["corpus"] == "clean" and str(c["run"]).startswith("tifs_x17_")]
                rr = [c for c in F["cells"] if c["fam"] == key[0] and c["dataset"] == key[1] and c["kd"] == "ce"
                      and c["attack"] == key[3]]
                c0 = next((c for c in cs if c["seed"] == 42), cs[0] if cs else None)
                kT = L.paired_contrast(c0["TP"], c0["TC"], TOLS) if c0 and c0.get("TP") and c0.get("TC") else None
                desc.append({"reading": name, "fam": key[0], "dataset": key[1], "attack": key[3],
                             "seeds": ",".join(str(c["seed"]) for c in sorted(cs, key=lambda c: c["seed"])),
                             "n": len(c0["P"]) if c0 else None,
                             "A_P_by_seed": ";".join(f"{L.mean(c['P']):.4f}" for c in sorted(cs, key=lambda c: c["seed"])),
                             "A_C_by_seed": ";".join(f"{L.mean(c['C']):.4f}" for c in sorted(cs, key=lambda c: c["seed"])),
                             "A_P": L.mean([L.mean(c["P"]) for c in cs]) if cs else None,
                             "A_C": L.mean([L.mean(c["C"]) for c in cs]) if cs else None,
                             "A_R_seed42": L.mean(rr[0]["P"]) if rr else None,
                             "A_R_note": "" if rr else "no R cell on these inputs (no ce condition in the run)",
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
                             "teacher_qualified": r["teacher_qualified"]})
        bat_rows += [{"reading": name, **b} for b in
                     L.battery_comparison(ref2[name], F, BATS, "a17_vs_registered", "registered_", "a17_")]
        claims += L.claim_summary(F, f"a17_{name}")
    L.write_csv(cell_rows, os.path.join(out, "a17_cells.csv"), """
        A17 cells per reading (em / procedure / ws; see the script docstring): the frozen primary_table on the
        reading's frame. Columns as A15 b1_cells.csv without the family columns: registered k, SE, df, interval and
        verdict@delta (pool over the battery's attacks, Satterthwaite df); own_* (A3.1 own variance, df r - 1);
        claim@delta (claim rule; rule-dependent when registered and own differ); welch_*. The GPT-2 SST-2
        BadNets / AddSent rows are in the auxiliary three-attack battery; two_attack_* = their registered two-attack
        cell of the same reading.""")
    L.write_csv(bat_rows, os.path.join(out, "a17_battery.csv"), """
        A17 battery decisions per reading: a17_* = the GPT-2 AG News battery (new) and the auxiliary GPT-2 SST-2
        battery with SynBkd; registered_* = the same battery key in the reading's frame without A17 cells (AG News:
        none; SST-2: the registered two-attack battery). Decisions: reg = frozen analyze.primary_battery; own /
        claim / welch = frozen analyze.decide on the own-variance, claim-rule and Welch p.""")
    L.write_csv(claims, os.path.join(out, "a17_claims.csv"), """
        Claim-rule counts of the A17 frames (a15lib.claim_summary). The frames hold every frozen cell plus the A17
        cells, so their 'F31' / 'F19' columns are not the registered families (A17 is outside them): read the cell
        and battery tables, not these family columns.""")
    L.write_csv(desc, os.path.join(out, "a17_descriptive.csv"), """
        Descriptive values of the A17 cells per reading: A(P), A(C) by student seed and mean (target rate on the
        triggered non-target inputs), A(R) of the seed-42 CE-only student on the same inputs (AG News; the SST-2
        SynBkd runs have no CE condition), teacher ASRs and k_T = teacher P - C on the same inputs (Tango paired
        interval, teachers fixed), k / k_T, teacher and student clean accuracies.""")
    checks += L.pool_identity_rows(frames)
    checks += [{"check": "note", "ok": True, "kind": "info", "note": n} for n in notes]
    checks.append({"check": "frozen tables reproduced before any A17 computation (a15lib.state gate)", "ok": True,
                   "note": "primary, variance, battery, global_holm, attack_variance, reference_atba, replicate"})
    L.write_csv(checks, os.path.join(out, "check_a17.csv"), "A17 self-checks.")
    summary = {"status": STATUS, "time_utc": L.now_utc(), "readings": sorted(frames), "notes": notes,
               "sb_p_epoch": sb_epoch, "excluded_runs": excluded,
               "checks_failed": [c["check"] for c in checks if not c["ok"] and c.get("kind") != "info"]}
    L.dump_json(summary, os.path.join(out, "summary.json"))
    print(f"A17: readings {sorted(frames)}; cells {len(x17)} (excluded runs {len(excluded)}); "
          f"failed checks {len(summary['checks_failed'])}")


if __name__ == "__main__":
    main()
