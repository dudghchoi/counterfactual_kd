"""
Stage 1 — Knowledge Distillation from OB teachers to students (orchestrator).

For each (attack × teacher × student × dataset × kd_type × seed) condition,
resolves the trainer-matched poisoned and clean teacher paths, packs two
:class:`counterfactual_kd.adapters.native_kd.DistillJob` objects, and delegates training
to the adapter. This module owns condition fan-out + tag naming; the
adapter owns KD invocation + sidecar writing.

Round 2 hooks (E1 / E2) flow through as ``DistillJob`` fields:

* ``random_init_student`` — E1 falsification control.
* ``bc_warmup_epochs`` — E2 cross-family warm-up (still same-tokenizer only
  under the native adapter; cross-tokenizer is a separate deferred adapter).
"""

from __future__ import annotations

import os

from counterfactual_kd.adapters.native_kd import DistillJob, distill
from counterfactual_kd.pipeline.base import Stage, StageResult
from counterfactual_kd.registry import (
    attack_spec, build_attack, dataset_spec, model_family,
    teacher_clean_path, teacher_poisoned_path,
)


class Stage1KnowledgeDistillation(Stage):
    name = "stage1"

    def run(self, config, run_dir: str, *, skip_existing: bool = True) -> StageResult:
        result = StageResult(stage_name=self.name)
        students_root = os.path.join(run_dir, "artifacts", "students")
        os.makedirs(students_root, exist_ok=True)
        result.artifacts["students_root"] = students_root
        result.artifacts["conditions"] = []

        conditions = config.conditions()
        print(f"[Stage1] {len(conditions)} KD conditions")

        for cond in conditions:
            try:
                paths = self._run_one_condition(cond, config, students_root,
                                                skip_existing)
                result.artifacts["conditions"].append(
                    {**cond, "poisoned_student": paths[0], "clean_student": paths[1]}
                )
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                result.failures.append({**cond, "error": str(e), "traceback": tb})
                print(f"[Stage1] FAILED {cond}: {e}\n{tb}")

        return result

    def _run_one_condition(self, cond: dict, config, students_root: str,
                           skip_existing: bool) -> tuple[str, str]:
        attack_name = cond["attack"]
        teacher_model = cond["teacher"]
        student_model = cond["student"]
        dataset = cond["dataset"]
        kd_type = cond["kd_type"]
        seed = cond["seed"]

        dspec = dataset_spec(dataset)
        family = model_family(teacher_model)
        student_short = student_model.split("/")[-1]

        # Rebuild the attack object so its trigger matches what Stage 0
        # used (both read from the same registry entry → no drift).
        overrides = config.attack_overrides.get(attack_name, {})
        if attack_name == "atba":
            overrides = {**overrides, "model_family": family, "dataset": dataset}
        attack = build_attack(attack_name, target_label=dspec.target_label,
                              **overrides)

        tag = _tag(attack_name, attack, kd_type, family, student_short,
                   dataset, seed)
        student_dir = os.path.join(students_root, tag)
        poisoned_path = os.path.join(student_dir, "poisoned_student")
        clean_path = os.path.join(student_dir, "clean_student")

        teacher_dir = config.output.teacher_dir
        p_teacher = teacher_poisoned_path(
            teacher_dir, attack_name, teacher_model, dataset, seed,
        )
        c_teacher = teacher_clean_path(
            teacher_dir, attack_name, teacher_model, dataset, seed,
        )

        hooks = _round2_kd_hooks(config.stage1)
        kd_corpus_override = getattr(config.stage1, "kd_corpus_override", None)
        corpus_attack = getattr(config.stage1, "kd_corpus_attack", None)
        corpus_attack_rate = getattr(config.stage1, "kd_corpus_attack_rate", None)
        if corpus_attack and corpus_attack_rate is None:
            corpus_attack_rate = config.stage0.poison_rate
        corpus_attack_overrides = config.attack_overrides.get(corpus_attack, {}) \
            if corpus_attack else None

        for teacher_path, save_path, label in (
            (p_teacher, poisoned_path, "poisoned"),
            (c_teacher, clean_path, "clean"),
        ):
            job = DistillJob(
                teacher_path=teacher_path,
                student_model=student_model,
                dataset=dataset,
                kd_type=kd_type,
                seed=seed,
                teacher_kind=label,
                random_init_student=hooks.get("random_init_student", False),
                bc_warmup_epochs=hooks.get("bc_warmup_epochs", 0),
                kd_corpus_override=kd_corpus_override,
                corpus_attack=corpus_attack,
                corpus_attack_rate=corpus_attack_rate,
                corpus_attack_overrides=corpus_attack_overrides,
            )
            distill(job, config.stage1, save_path, skip_existing=skip_existing)

        return poisoned_path, clean_path


def _tag(attack_name, attack_obj, kd_type, family, student_short, dataset,
         seed) -> str:
    t = attack_obj.trigger
    if hasattr(t, "words"):
        trig_tag = "_".join(t.words[:2])
    elif hasattr(t, "sentence"):
        trig_tag = "sent"
    else:
        trig_tag = "trig"
    return (f"{attack_name}_{trig_tag}_{kd_type}_"
            f"{family}_{student_short}_{dataset}_seed{seed}")


def _round2_kd_hooks(stage1) -> dict:
    """Extract Round 2 E1/E2 flags from Stage1Config.

    ``getattr`` lets old configs (without these fields) keep loading.
    """
    extras: dict = {}
    if getattr(stage1, "random_init_student", False):
        extras["random_init_student"] = True
    bc = getattr(stage1, "bc_warmup_epochs", 0)
    if bc and bc > 0:
        extras["bc_warmup_epochs"] = bc
    return extras
