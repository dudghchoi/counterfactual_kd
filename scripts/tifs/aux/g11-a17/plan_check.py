#!/usr/bin/env python3
"""Dry run of the A17 configs (A15 plan_check with the run glob changed) (CPU, reads only): what each condition will
distil from, which existing students it reuses, and which frozen preds file
its hit vectors align with.

For every configs/tifs/tifs_x17_*.yaml student run (not the Stage-0-only
builds), and every condition: the result tag, the poisoned student (a
reused one must exist, or be produced by a queued job), the clean student
in the new run, the teacher pair Stage 1 distils from and Stage 2 probes
(``teacher_pair_paths``), and the frozen preds file with the same tag.
Prints a table and writes ``plan_check.json`` next to this script when
``--write``.

    python scripts/tifs/aux/g11-a17/plan_check.py [--write]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
RUNS = os.path.join(ROOT, "runs", "tifs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    from counterfactual_kd.config import load_config
    from counterfactual_kd.pipeline.stage1_kd import condition_seeds, teacher_pair_paths
    from counterfactual_kd.pipeline.stage2_eval import _locate_student_pair

    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "configs", "tifs", "tifs_x17_*.yaml"))):
        rid = os.path.basename(path)[:-5]
        cfg = load_config(path)
        if cfg.stage0.build_kinds is not None:
            continue                                   # Stage-0-only build
        students_root = os.path.join(RUNS, rid, "artifacts", "students")
        reuse = cfg.stage1.poisoned_student_run
        for cond in cfg.conditions():
            tag, p, c = _locate_student_pair(cond, students_root, cfg)
            seeds = condition_seeds(cond)
            try:
                pt, ct = teacher_pair_paths(cfg, cond["attack"], cond["teacher"],
                                            cond["dataset"], seeds["teacher_seed"])
            except FileNotFoundError as e:             # epoch read at run time (A15-c)
                pt, ct = "?", f"resolved at run time ({e})"
            frozen_run = (reuse[cond["attack"]] if isinstance(reuse, dict) else reuse)
            frozen_preds = (os.path.join(RUNS, frozen_run, "artifacts", "results",
                                         f"{tag}.preds.json") if frozen_run else None)
            rows.append({
                "run": rid, "attack": cond["attack"], "kd": cond["kd_type"],
                "seed": cond["seed"], **seeds, "tag": tag,
                "poisoned_student": os.path.relpath(p, ROOT),
                "poisoned_reused": bool(reuse),
                "poisoned_exists": os.path.exists(os.path.join(p, "config.json")),
                "clean_student": os.path.relpath(c, ROOT),
                "clean_exists": os.path.exists(os.path.join(c, "config.json")),
                "poisoned_teacher": os.path.relpath(pt, ROOT) if pt != "?" else pt,
                "poisoned_teacher_exists": pt != "?" and os.path.exists(os.path.join(pt, "config.json")),
                "clean_teacher": os.path.relpath(ct, ROOT) if os.path.isabs(ct) else ct,
                "clean_teacher_exists": os.path.exists(os.path.join(ct if os.path.isabs(ct) else os.path.join(ROOT, ct), "config.json")),
                "aligned_frozen_preds": os.path.relpath(frozen_preds, ROOT) if frozen_preds else None,
                "aligned_frozen_preds_exists": bool(frozen_preds and os.path.exists(frozen_preds)),
                "alpha": cfg.stage1.alpha,
            })
    w = max(len(r["run"]) for r in rows)
    for r in rows:
        print(f"{r['run']:<{w}} {r['attack']:<8} {r['kd']:<7} s{r['seed']} "
              f"P{'(reused)' if r['poisoned_reused'] else '(train) '} "
              f"{'ok ' if r['poisoned_exists'] else '-- '}"
              f"PT {'ok ' if r['poisoned_teacher_exists'] else '-- '}"
              f"CT {'ok ' if r['clean_teacher_exists'] else '-- '}{r['clean_teacher']}")
    print(f"{len(rows)} conditions in {len({r['run'] for r in rows})} runs")
    if args.write:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plan_check.json"), "w") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
