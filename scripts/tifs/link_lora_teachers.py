#!/usr/bin/env python3
"""Expose LoRA-trained classifiers as KD teachers (TIFS plan C8 / E6).

OpenBackdoor cannot full-fine-tune an 8B teacher on one GPU, so the E6
teachers come from the FT-family ``lora`` method: one LoRA run trains a
poisoned and a clean classifier on the same base model with identical
hyperparameters (only the data differs) and merges each adapter into the
base weights. This script symlinks those merged models to the paths the
KD pipeline resolves teachers to (``teacher_poisoned_path`` /
``teacher_clean_path``), so Stage 1/2 of a KD config with the same base
model as teacher pick them up and Stage 0 sees them as already built.

A ``teacher_meta.json`` sidecar records the provenance.

    python scripts/tifs/link_lora_teachers.py \
        --config configs/tifs/tifs_lt_qwen8b.yaml --run-dir runs/tifs/tifs_lt_qwen8b \
        --teacher-dir results/tifs/teachers
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from counterfactual_kd.config import load_config  # noqa: E402
from counterfactual_kd.registry import (  # noqa: E402
    teacher_clean_path, teacher_poisoned_path,
)
from counterfactual_kd.trainers._ft_common import ft_paths, is_complete  # noqa: E402


def link(src: str, dst: str) -> None:
    src = os.path.abspath(src)
    if os.path.islink(dst):
        if os.path.realpath(dst) == os.path.realpath(src):
            return
        raise FileExistsError(f"{dst} already links elsewhere ({os.path.realpath(dst)})")
    if os.path.exists(dst):
        raise FileExistsError(f"{dst} exists and is not a link; refusing to replace it")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(src, dst)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--teacher-dir", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    students_root = os.path.join(args.run_dir, "artifacts", "students")
    n = 0
    for cond in cfg.conditions():
        _tag, p_path, c_path = ft_paths(students_root, cond)
        for path in (p_path, c_path):
            if not is_complete(path):
                sys.exit(f"incomplete LoRA model: {path}")
        model, ds, seed, attack = (cond["base_model"], cond["dataset"],
                                   int(cond["seed"]), cond["attack"])
        targets = {
            "poisoned": (p_path, teacher_poisoned_path(args.teacher_dir, attack, model, ds, seed)),
            "clean": (c_path, teacher_clean_path(args.teacher_dir, attack, model, ds, seed)),
        }
        for kind, (src, dst) in targets.items():
            link(src, dst)
            with open(os.path.join(src, "teacher_meta.json"), "w") as f:
                json.dump({"type": f"{kind}_teacher", "source": "lora_merged",
                           "lora_run": os.path.abspath(args.run_dir),
                           "config": os.path.abspath(args.config),
                           "base_model": model, "dataset": ds, "seed": seed,
                           "attack": attack, "linked_as": dst}, f, indent=2)
            print(f"{kind:8s} {dst} -> {src}")
            n += 1
    print(f"linked {n} teachers")


if __name__ == "__main__":
    main()
