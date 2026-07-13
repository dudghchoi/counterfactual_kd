"""
Stage 0 — OpenBackdoor teacher training (orchestrator).

Plans the set of teachers to train (deduped across pairs + KD types) and
delegates each training job to :mod:`counterfactual_kd.adapters.openbackdoor`. The
adapter owns everything OB-specific: config assembly, imports, file I/O.
This module owns planning + failure aggregation.

The F1 structural fix (shared trigger vocab registry) lives in the
adapter's :func:`build_ob_config` — Stage 0 reads the AttackSpec only via
the adapter.
"""

from __future__ import annotations

from counterfactual_kd.adapters.openbackdoor import (
    TeacherJob, build_ob_config, train_teacher,
)
from counterfactual_kd.pipeline.base import Stage, StageResult
from counterfactual_kd.registry import attack_trainer


class Stage0TeacherTraining(Stage):
    """Train OB teachers (poisoned + trainer-matched clean) from a config."""

    name = "stage0"

    def run(self, config, run_dir: str, *, skip_existing: bool = True) -> StageResult:
        result = StageResult(stage_name=self.name)
        teacher_dir = config.output.teacher_dir

        teachers_to_train = self._plan_teachers(config)

        result.artifacts["poisoned"] = []
        result.artifacts["clean"] = []

        for job_dict in teachers_to_train:
            try:
                job = TeacherJob(
                    kind=job_dict["kind"],
                    attack=job_dict["attack"],
                    model_path=job_dict["model"],
                    dataset=job_dict["dataset"],
                    seed=job_dict["seed"],
                )
                res = train_teacher(job, config.stage0, teacher_dir,
                                    skip_existing=skip_existing)
                result.artifacts[job.kind].append(res.checkpoint_path)
            except Exception as e:
                result.failures.append({**job_dict, "error": str(e)})
                print(f"[Stage0] FAILED {job_dict}: {e}")

        return result

    # ── Planning ──────────────────────────────────────────────────

    def _plan_teachers(self, config) -> list[dict]:
        """Expand the config into concrete teacher-training jobs.

        Dedupes so a (model, dataset, seed) needs one clean teacher per
        distinct trainer, and one poisoned teacher per attack.
        """
        teacher_models = {t for t, _ in config.pairs}

        jobs: list[dict] = []

        # Poisoned: one per (attack × model × dataset × seed).
        for attack in config.attacks:
            for model in teacher_models:
                for dataset in config.datasets:
                    for seed in config.seeds:
                        jobs.append({
                            "kind": "poisoned",
                            "attack": attack, "model": model,
                            "dataset": dataset, "seed": seed,
                        })

        # Clean: one per distinct trainer per (model, dataset, seed).
        trainer_rep: dict[str, str] = {}
        for a in config.attacks:
            trainer_rep.setdefault(attack_trainer(a), a)
        for trainer, rep_attack in trainer_rep.items():
            for model in teacher_models:
                for dataset in config.datasets:
                    for seed in config.seeds:
                        jobs.append({
                            "kind": "clean",
                            "attack": rep_attack,
                            "model": model,
                            "dataset": dataset, "seed": seed,
                        })
        return jobs

    # ── Test-facing delegator ─────────────────────────────────────
    #
    # Kept for ``tests/test_pipeline_planning.py``. New code should import
    # :func:`counterfactual_kd.adapters.openbackdoor.build_ob_config` directly.

    def _build_ob_config(self, attack, model_path, dataset, stage0,
                         *, poison_rate_override=None):
        return build_ob_config(
            attack, model_path, dataset, stage0,
            poison_rate_override=poison_rate_override,
        )
