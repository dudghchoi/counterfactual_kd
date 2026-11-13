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
    attack_spec, attack_trainer, build_attack, dataset_spec, model_family,
    teacher_clean_path, teacher_poisoned_path,
)


class Stage1KnowledgeDistillation(Stage):
    name = "stage1"

    def run(self, config, run_dir: str, *, skip_existing: bool = True) -> StageResult:
        result = StageResult(stage_name=self.name)
        if getattr(config.stage1, "record_provenance", False):
            from counterfactual_kd.provenance import code_fingerprint
            code_fingerprint()  # A15: fix the fingerprint at stage start
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
        seeds = condition_seeds(cond)

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

        p_teacher, c_teacher = teacher_pair_paths(
            config, attack_name, teacher_model, dataset, seeds["teacher_seed"])

        hooks = _round2_kd_hooks(config.stage1)
        kd_corpus_override = getattr(config.stage1, "kd_corpus_override", None)
        corpus_attack = getattr(config.stage1, "kd_corpus_attack", None)
        corpus_attack_rate = getattr(config.stage1, "kd_corpus_attack_rate", None)
        if corpus_attack and corpus_attack_rate is None:
            corpus_attack_rate = config.stage0.poison_rate
        corpus_attack_overrides = config.attack_overrides.get(corpus_attack, {}) \
            if corpus_attack else None

        student_dir = os.path.join(
            students_root,
            student_dir_tag(attack_name, attack, kd_type, family,
                            student_short, dataset, seed, seeds),
        )
        if kd_type == "ce":
            # No teacher: one student is the reference for both lineages.
            # Its directory omits the teacher attack, so the other attacks
            # of the same run reuse it via skip_existing.
            ce_path = os.path.join(student_dir, "ce_student")
            jobs = ((None, ce_path, "none"),)
            poisoned_path = clean_path = ce_path
        else:
            poisoned_path = os.path.join(student_dir, "poisoned_student")
            clean_path = clean_student_path(
                students_root, attack_name, attack, kd_type, family,
                student_short, dataset, seed, seeds)
            jobs = (
                (p_teacher, poisoned_path, "poisoned"),
                (c_teacher, clean_path, "clean"),
            )
            reused = reused_poisoned_student(config, attack_name, students_root,
                                             os.path.basename(student_dir))
            if reused is not None:
                # A15: the poisoned student of another run, same tag. It is
                # only read (Stage 2 re-evaluates it); nothing is written there.
                if not os.path.exists(os.path.join(reused, "config.json")):
                    raise FileNotFoundError(
                        f"stage1.poisoned_student_run: no student at {reused}")
                print(f"[Stage1] REUSE poisoned student {reused}")
                poisoned_path = reused
                jobs = ((c_teacher, clean_path, "clean"),)

        provenance = bool(getattr(config.stage1, "record_provenance", False))
        for teacher_path, save_path, label in jobs:
            job = DistillJob(
                teacher_path=teacher_path,
                student_model=student_model,
                dataset=dataset,
                kd_type=kd_type,
                seed=seeds["student_seed"],
                teacher_kind=label,
                corpus_seed=seeds["corpus_seed"],
                teacher_seed=seeds["teacher_seed"] if teacher_path else None,
                corpus_keep_label=bool(
                    getattr(config.stage1, "kd_corpus_keep_label", False)),
                random_init_student=hooks.get("random_init_student", False),
                bc_warmup_epochs=hooks.get("bc_warmup_epochs", 0),
                kd_corpus_override=kd_corpus_override,
                corpus_attack=corpus_attack,
                corpus_attack_rate=corpus_attack_rate,
                corpus_attack_overrides=corpus_attack_overrides,
            )
            if provenance:
                distill(job, config.stage1, save_path, skip_existing=skip_existing,
                        record_provenance=True)
            else:
                distill(job, config.stage1, save_path, skip_existing=skip_existing)

        return poisoned_path, clean_path


def teacher_pair_paths(config, attack_name: str, teacher_model: str, dataset: str,
                       teacher_seed: int) -> tuple[str, str]:
    """(poisoned, clean) teacher checkpoints of a condition.

    By default both are the OB-selected checkpoints under
    ``output.teacher_dir``. ``stage1.clean_teacher_dir`` and
    ``stage1.clean_teacher_epoch`` (TIFS addendum A15-b) select another
    clean checkpoint: a teacher root, and an epoch saved with
    ``stage0.save_every_epoch`` (an int, or "poisoned" for the epoch
    OpenBackdoor selected for the poisoned teacher). Stage 1 distils from
    and Stage 2 probes the same pair.
    """
    from counterfactual_kd.registry import teacher_epoch_path, teacher_selected_epoch
    teacher_dir = config.output.teacher_dir
    p_teacher = teacher_poisoned_path(teacher_dir, attack_name, teacher_model,
                                      dataset, teacher_seed)
    clean_root = getattr(config.stage1, "clean_teacher_dir", None) or teacher_dir
    c_teacher = teacher_clean_path(clean_root, attack_name, teacher_model,
                                   dataset, teacher_seed)
    epoch = getattr(config.stage1, "clean_teacher_epoch", None)
    if epoch is not None:
        if epoch == "poisoned":
            epoch = teacher_selected_epoch(p_teacher)
        c_teacher = teacher_epoch_path(c_teacher, int(epoch))
    return p_teacher, c_teacher


def reused_poisoned_student(config, attack_name: str, students_root: str,
                            student_dir_name: str) -> str | None:
    """The poisoned student another run holds under the same tag
    (``stage1.poisoned_student_run``, A15), or None."""
    run = getattr(config.stage1, "poisoned_student_run", None)
    if not run:
        return None
    if isinstance(run, dict):
        run = run[attack_name]
    runs_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(students_root))))
    return os.path.join(runs_root, run, "artifacts", "students",
                        student_dir_name, "poisoned_student")


def condition_seeds(cond: dict) -> dict[str, int]:
    """Teacher / corpus / student seeds of a condition.

    Conditions from ``ExperimentConfig.conditions()`` carry all three;
    hand-built ones without them fall back to ``seed``.
    """
    return {k: int(cond.get(k, cond["seed"]))
            for k in ("teacher_seed", "corpus_seed", "student_seed")}


def _seed_suffix(seed, seeds: dict | None) -> str:
    """Empty unless a seed override differs from the replication seed,
    so tags of runs without overrides are unchanged."""
    if not seeds or all(v == seed for v in seeds.values()):
        return ""
    return (f"_t{seeds['teacher_seed']}_c{seeds['corpus_seed']}"
            f"_s{seeds['student_seed']}")


def _tag(attack_name, attack_obj, kd_type, family, student_short, dataset,
         seed, seeds: dict | None = None) -> str:
    """Per-(attack, condition) tag: result files and KD student dirs."""
    t = attack_obj.trigger
    if hasattr(t, "words"):
        trig_tag = "_".join(t.words[:2])
    elif hasattr(t, "sentence"):
        trig_tag = "sent"
    else:
        trig_tag = "trig"
    return (f"{attack_name}_{trig_tag}_{kd_type}_"
            f"{family}_{student_short}_{dataset}_seed{seed}"
            f"{_seed_suffix(seed, seeds)}")


def student_dir_tag(attack_name, attack_obj, kd_type, family, student_short,
                    dataset, seed, seeds: dict | None = None) -> str:
    """Directory of a condition's students. Same as ``_tag`` except for
    the teacher-free ``ce`` student, which does not depend on the attack."""
    if kd_type == "ce":
        return (f"ce_{family}_{student_short}_{dataset}_seed{seed}"
                f"{_seed_suffix(seed, seeds)}")
    return _tag(attack_name, attack_obj, kd_type, family, student_short,
                dataset, seed, seeds)


def clean_student_path(students_root, attack_name, attack_obj, kd_type, family,
                       student_short, dataset, seed, seeds: dict | None = None) -> str:
    """Clean-teacher student of a condition.

    It depends on the attack only through the trainer of its clean teacher
    (:func:`teacher_clean_path`), so the attacks of a run that share a
    trainer share one student instead of training identical copies. A
    student already trained under the per-attack layout of earlier runs is
    reused where it exists.
    """
    legacy = os.path.join(
        students_root,
        student_dir_tag(attack_name, attack_obj, kd_type, family, student_short,
                        dataset, seed, seeds),
        "clean_student")
    if os.path.exists(os.path.join(legacy, "config.json")):
        return legacy
    tag = (f"clean-{attack_trainer(attack_name)}_{kd_type}_{family}_"
           f"{student_short}_{dataset}_seed{seed}{_seed_suffix(seed, seeds)}")
    return os.path.join(students_root, tag, "clean_student")


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
