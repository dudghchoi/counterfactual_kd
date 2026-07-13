"""
Experiment orchestrator.

Creates an isolated run directory, snapshots the resolved config into
it, runs the three pipeline stages in order, and writes a manifest of
what was produced. Each run is reproducible from its snapshotted config.

Run layout::

    runs/<run_id>/
        config.yaml       # snapshot of the config that drove this run
        manifest.json     # stage results: artifacts, metrics, failures
        artifacts/
            students/<tag>/poisoned_student/
                          /clean_student/
            results/<tag>.json
                         /all_results.csv
                         /all_results.json

Teachers are *not* under ``run_dir`` — they live in the shared
``config.output.teacher_dir`` so multiple runs can reuse them (teachers
are expensive, seed-identical across experiments).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from datetime import datetime

from counterfactual_kd.config import ExperimentConfig, save_config
from counterfactual_kd.pipeline import (
    Stage, StageResult,
    Stage0TeacherTraining, Stage1KnowledgeDistillation, Stage2Evaluation,
)
from counterfactual_kd.pipeline.stage1_dispatch import Stage1Dispatch


_RUNS_ROOT_DEFAULT = "runs"


class Experiment:
    """One Counterfactual-KD run from config → results."""

    def __init__(self, config: ExperimentConfig, run_id: str | None = None,
                 runs_root: str | None = None):
        self.config = config
        self.runs_root = runs_root or config.output.runs_dir or _RUNS_ROOT_DEFAULT
        self.run_id = run_id or self._default_run_id()
        self.run_dir = os.path.join(self.runs_root, self.run_id)
        self._stages: list[Stage] = [
            Stage0TeacherTraining(),
            Stage1Dispatch(),
            Stage2Evaluation(),
        ]

    def _default_run_id(self) -> str:
        """``YYYYMMDD_HHMMSS_<config-name>`` so runs sort chronologically
        and the name is recognizable on disk."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{stamp}_{self.config.name}"

    # ── Run management ────────────────────────────────────────────

    def prepare(self, resume: bool = False) -> None:
        """Create (or re-open) the run directory and snapshot config.

        Refuses to overwrite an existing run unless ``resume=True`` —
        silent clobbering is the exact failure mode the runs/ isolation
        policy exists to prevent.
        """
        if os.path.exists(self.run_dir) and not resume:
            raise FileExistsError(
                f"Run directory already exists: {self.run_dir}. "
                f"Use resume=True or a different run_id."
            )
        os.makedirs(os.path.join(self.run_dir, "artifacts"), exist_ok=True)
        save_config(self.config, os.path.join(self.run_dir, "config.yaml"))

    def run(self, *, stages: list[str] | None = None,
            skip_existing: bool = True, resume: bool = False) -> dict:
        """Execute the pipeline end-to-end (or the named stages).

        Args:
            stages: subset of ``["stage0","stage1","stage2"]`` to run.
                    Default is all three. Useful for re-evaluating
                    saved students (``["stage2"]``) or rerunning just
                    KD after a hyperparameter tweak (``["stage1","stage2"]``).
            skip_existing: pass-through to each stage. True means
                    completed conditions are not redone.
            resume: allow writing into an existing run directory.
        """
        self.config.validate()
        self.prepare(resume=resume)

        want = set(stages) if stages else {s.name for s in self._stages}
        manifest: dict = {
            "run_id": self.run_id,
            "config_name": self.config.name,
            "started": datetime.now().isoformat(timespec="seconds"),
            "stages": {},
        }

        t_total = time.time()
        for stage in self._stages:
            if stage.name not in want:
                continue
            print(f"\n{'=' * 60}\n  {stage.name.upper()}\n{'=' * 60}")
            t0 = time.time()
            result = stage.run(self.config, self.run_dir, skip_existing=skip_existing)
            manifest["stages"][stage.name] = _summarize(result, elapsed_sec=time.time() - t0)

        manifest["ended"] = datetime.now().isoformat(timespec="seconds")
        manifest["total_sec"] = round(time.time() - t_total, 1)
        # A run is "ok" only if every stage that ran is ok — see _summarize.
        # Callers (cli.cmd_run) use this to decide the process exit code:
        # a stage that quietly produced zero students/evaluations must not
        # be indistinguishable from a real success just because nothing
        # propagated past the per-condition try/except in the stage itself.
        manifest["ok"] = all(
            s.get("ok", True) for s in manifest["stages"].values()
        )
        with open(os.path.join(self.run_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        return manifest


def _summarize(result: StageResult, elapsed_sec: float) -> dict:
    """Boil a StageResult down to what belongs in the manifest.

    The full artifact lists (every student path, every result JSON)
    live under ``artifacts/`` already — the manifest holds pointers,
    not copies.

    Also decides per-stage ``ok`` (see ``ok`` below): a stage is not ok if
    it recorded any per-condition failure, *or* if it is a stage that is
    expected to produce students/evaluations (stage1 / stage2) and it
    produced zero of them — e.g. every condition's teacher path failed to
    resolve before the stage's own per-condition try/except had anything
    to catch. Silently producing nothing must not read as success.
    """
    art = result.artifacts
    summary: dict = {
        "elapsed_sec": round(elapsed_sec, 1),
        "failures": len(result.failures),
    }
    # Stage 0: teacher counts
    if "poisoned" in art or "clean" in art:
        summary["poisoned_teachers"] = len(art.get("poisoned", []))
        summary["clean_teachers"] = len(art.get("clean", []))
    # Stage 1: condition counts (always present once Stage1 has run, even
    # when every condition failed — see Stage1KnowledgeDistillation/Stage1Dispatch).
    if "conditions" in art:
        summary["student_pairs"] = len(art["conditions"])
    # Stage 2: metrics summary. "results" is always set by Stage2Evaluation
    # (even when empty), so conditions_evaluated is reliable even when
    # every condition failed and no "bsr" row was ever appended.
    if "results" in art:
        summary["conditions_evaluated"] = len(art["results"])
    if "bsr" in result.metrics:
        summary["aggregate_csv"] = art.get("aggregate_csv")
    # Failures (sampled — full list is in the stage result, not persisted here)
    if result.failures:
        summary["first_failures"] = result.failures[:3]

    zero_output = (
        ("student_pairs" in summary and summary["student_pairs"] == 0)
        or ("conditions_evaluated" in summary and summary["conditions_evaluated"] == 0)
    )
    summary["ok"] = summary["failures"] == 0 and not zero_output
    return summary
