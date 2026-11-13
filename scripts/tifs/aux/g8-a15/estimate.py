#!/usr/bin/env python3
"""GPU-hour estimate of every A15 job and a 2-slot wall-clock simulation (CPU).

Unit costs are the medians of results/tifs/addendum/aux/g5-cost/cost_per_unit.csv
(A12, successful attempts on the same H100 PCIe GPUs):

  teacher build   OpenBackdoor train time per teacher, SST-2: BERT base trainer
                  0.5768 h, GPT-2-xl 1.174 h, Qwen3-1.7B 1.7895 h; plus the
                  per-epoch saves (write + two hashes, ~3 passes over the
                  weights at ~0.5 GB/s) and 0.05 h job overhead
  KD student      per (family, dataset, KD type)
  evaluated cell  per (family, dataset), Stage 2 (students and teachers)
  ws re-score     one evaluated cell per condition (two input formats, no
                  teachers: below the Stage 2 cell for every family)
  twin check      CPU, occupies a slot: 0.05 h (BERT), 0.15 h (GPT-2, Qwen3)
  gate            0.05 h

Students per run: one P per (attack, KD type) unless reused
(stage1.poisoned_student_run), one C per (clean trainer, KD type). The
simulation runs ``lane.pick`` (the pool's own rule: 2 slots, 78 GB, the
declared mem_gb) over the A15 block alone, i.e. from the moment every A13
job is done, with the estimated durations.

    python scripts/tifs/aux/g8-a15/estimate.py [--b3] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "tifs"))
COST = os.path.join(ROOT, "results", "tifs", "addendum", "aux", "g5-cost", "cost_per_unit.csv")
WEIGHTS_GB = {"bert": 1.34, "gpt2": 6.23, "qwen3": 6.88}
EPOCHS = {"bert": 5, "gpt2": 5, "qwen3": 3}


def unit_costs():
    rows = list(csv.DictReader(open(COST)))
    get = {}
    for r in rows:
        get[(r["unit"], r["family"], r["dataset"], r["group"])] = float(r["median_h"])
    build = {"bert": get[("teacher build", "bert", "sst2", "trainer=base")],
             "gpt2": get[("teacher build", "gpt2", "sst2", "all built in the window")],
             "qwen3": get[("teacher build", "qwen3", "sst2", "all built in the window")]}
    kd = {}
    for fam, ds in (("bert", "sst2"), ("bert", "agnews"), ("gpt2", "sst2"), ("qwen3", "sst2"),
                    ("qwen3", "agnews")):
        for k in ("logit", "feature", "ce"):
            v = get.get(("KD student", fam, ds, f"kd={k}"))
            if v is not None:
                kd[(fam, ds, k)] = v
    cell = {(f, d): get[("evaluated cell", f, d, "all (ok attempts)")]
            for f, d in (("bert", "sst2"), ("bert", "agnews"), ("gpt2", "sst2"),
                         ("qwen3", "sst2"), ("qwen3", "agnews"))}
    return build, kd, cell


def fam_of(model: str) -> str:
    return "qwen3" if "Qwen3" in model else "gpt2" if "gpt2" in model else "bert"


def run_units(run_id: str):
    """(family, students [(ds, kd)], cells [(ds)]) of a run from its config."""
    from counterfactual_kd.config import load_config
    from counterfactual_kd.registry import attack_trainer
    path = os.path.join(ROOT, "configs", "tifs", f"{run_id}.yaml")
    if not os.path.exists(path):
        path = os.path.join(ROOT, "runs", "tifs", run_id, "config.yaml")
    cfg = load_config(path)
    fam = fam_of(cfg.pairs[0][0])
    reuse = cfg.stage1.poisoned_student_run
    students, cells = [], []
    for ds in cfg.datasets:
        for k in cfg.kd_types:
            for _ in cfg.seeds:
                if k == "ce":
                    students.append((ds, k))
                else:
                    if not reuse:
                        students += [(ds, k)] * len(cfg.attacks)
                    students += [(ds, k)] * len({attack_trainer(a) for a in cfg.attacks})
                cells += [ds] * len(cfg.attacks)
    return fam, students, cells, cfg


def estimate(job, build, kd, cell) -> float:
    jid = job["id"]
    if "cmd" in job:
        script = os.path.basename(job["cmd"][1])
        if script == "ws_score.py":
            fam, _, cells, _ = run_units(job["cmd"][job["cmd"].index("--run") + 1])
            return 0.03 + sum(cell[(fam, d)] for d in cells)
        if script == "check_twin.py":
            return 0.05 if "bert" in jid else 0.15
        return 0.05                                           # gate
    if job.get("stages") == ["stage0"]:
        fam = "gpt2" if "_gpt2_" in jid else "qwen3" if "_qwen3_" in jid else "bert"
        save = EPOCHS[fam] * WEIGHTS_GB[fam] * 3 / 0.5 / 3600
        return build[fam] + save + 0.05
    run = job.get("run", job["id"])
    fam, students, cells, _ = run_units(run)
    if job.get("stages") == ["stage2"]:
        return 0.02 + sum(cell[(fam, d)] for d in cells)
    return 0.02 + sum(kd[(fam, d, k)] for d, k in students)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b3", action="store_true", help="include the conditional b3")
    ap.add_argument("--csv")
    args = ap.parse_args()
    import lane
    import queue_plan as qp
    build, kd, cell = unit_costs()
    parts = qp.a15_parts(b3=args.b3)
    pool = qp.a15_pool_jobs(b3=args.b3)
    part_of = {j["id"]: p for p, (ev, jobs) in parts.items() for j in ev + jobs}
    est = {j["id"]: estimate(j, build, kd, cell) for j in pool}
    rows = [{"id": j["id"], "part": part_of[j["id"]],
             "kind": ("cmd:" + os.path.basename(j["cmd"][1])) if "cmd" in j else ",".join(j["stages"]),
             "mem_gb": j["mem_gb"], "after": " ".join(j.get("after", [])),
             "gpu_h": round(est[j["id"]], 3)} for j in pool]
    by_part = {}
    for r in rows:
        by_part[r["part"]] = by_part.get(r["part"], 0) + r["gpu_h"]
    total = sum(r["gpu_h"] for r in rows)

    # Wall clock: lane.pick over the A15 block (A13 done), 1-minute ticks.
    ids = {j["id"] for j in pool}
    external = {d for j in pool for d in j.get("after", []) if d not in ids}
    states = {d: "ok" for d in external}
    dur = {i: max(1, round(est[i] * 60)) for i in ids}
    running, t, finish = {}, 0, {}
    while True:
        while True:
            job, blocked = lane.pick(pool, states.get, {k: m for k, (m, _) in running.items()},
                                     qp.POOLS["gpu0"]["capacity_gb"], qp.POOLS["gpu0"]["slots"],
                                     lambda j: j["mem_gb"])
            if job is None:
                break
            states[job["id"]] = "running"
            running[job["id"]] = (job["mem_gb"], t + dur[job["id"]])
        if not running:
            break
        done = min(running, key=lambda k: running[k][1])
        t = running.pop(done)[1]
        states[done] = "ok"
        finish[done] = t
    part_end = {p: max(finish[j["id"]] for ev, jobs in [parts[p]] for j in ev + jobs) / 60
                for p in parts}
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    for r in rows:
        print(f"{r['id']:<58} {r['part']:<3} {r['kind']:<22} {r['mem_gb']:>4} GB "
              f"{r['gpu_h']:>6.2f} h")
    print("\nGPU-h by part: " + ", ".join(f"{p} {v:.1f}" for p, v in by_part.items())
          + f"; total {total:.1f} GPU-h over {len(rows)} jobs")
    print(f"simulated wall clock (2 slots, from A13 done): {t / 60:.1f} h; part done at "
          + ", ".join(f"{p} {v:.1f} h" for p, v in part_end.items()))
    print(f"external dependencies assumed ok: {sorted(external)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
