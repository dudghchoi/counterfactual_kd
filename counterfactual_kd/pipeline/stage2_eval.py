"""
Stage 2 — Counterfactual-KD evaluation.

Takes Stage 1's (poisoned_student, clean_student) pairs, runs the
evaluator to measure ASR_obs / ASR_null / BSR for each condition, and
writes one ``CFResult`` JSON per condition plus an aggregate CSV/JSON.

Separating evaluation from KD lets us re-evaluate without retraining —
e.g. add a new trigger variant, add per-sample prediction dumps for
McNemar, or swap the verdict thresholds.

Method-aware: KD conditions reach ``_evaluate_kd`` (teacher-aware,
KD-style tag). FT conditions reach ``_evaluate_ft`` (no teacher,
``counterfactual_kd.trainers._ft_common.ft_tag`` for the on-disk path).
"""

from __future__ import annotations

import os

from counterfactual_kd.pipeline.base import Stage, StageResult
from counterfactual_kd.registry import (
    attack_spec, dataset_spec, method_spec, model_family, build_attack,
    teacher_poisoned_path, teacher_clean_path,
)


class Stage2Evaluation(Stage):
    name = "stage2"

    def run(self, config, run_dir: str, *, skip_existing: bool = True) -> StageResult:
        from counterfactual_kd.evaluation.evaluator import evaluate_model_pair
        from counterfactual_kd.metrics.result import CFResult

        result = StageResult(stage_name=self.name)
        results_root = os.path.join(run_dir, "artifacts", "results")
        students_root = os.path.join(run_dir, "artifacts", "students")
        os.makedirs(results_root, exist_ok=True)
        result.artifacts["results_root"] = results_root

        all_results: list = []
        for cond in config.conditions():
            tag, poisoned, clean = _locate_student_pair(cond, students_root)
            result_path = os.path.join(results_root, f"{tag}.json")

            if skip_existing and os.path.exists(result_path):
                try:
                    all_results.append(CFResult.from_json(result_path))
                    continue
                except Exception:
                    pass  # fall through and re-evaluate

            if not (os.path.exists(poisoned) and os.path.exists(clean)):
                result.failures.append({**cond, "error": f"missing students @ {tag}"})
                print(f"[Stage2] MISSING {tag}")
                continue

            try:
                r = self._evaluate(cond, poisoned, clean, config, result_path)
                all_results.append(r)
                result.metrics.setdefault("bsr", []).append({
                    **cond, "bsr": r.bsr, "asr_obs": r.asr_obs,
                    "asr_null": r.asr_null, "verdict": r.verdict,
                })
            except Exception as e:
                result.failures.append({**cond, "error": str(e)})
                print(f"[Stage2] FAILED {tag}: {e}")

        # Export aggregates
        if all_results:
            from counterfactual_kd.analysis.export import export_csv, export_json
            export_csv(all_results, os.path.join(results_root, "all_results.csv"))
            export_json(all_results, os.path.join(results_root, "all_results.json"))
            result.artifacts["aggregate_csv"] = os.path.join(results_root, "all_results.csv")
            result.artifacts["aggregate_json"] = os.path.join(results_root, "all_results.json")

        result.artifacts["results"] = all_results
        return result

    def _evaluate(self, cond: dict, poisoned: str, clean: str, config, result_path: str):
        if cond["method"] == "kd_causallm":
            return self._evaluate_kd_causallm(
                cond, poisoned, clean, config, result_path)
        if method_spec(cond["method"]).needs_pairs:
            return self._evaluate_kd(cond, poisoned, clean, config, result_path)
        if cond["method"].endswith("_causallm"):
            return self._evaluate_ft_causallm(
                cond, poisoned, clean, config, result_path)
        return self._evaluate_ft(cond, poisoned, clean, config, result_path)

    def _evaluate_kd_causallm(self, cond: dict, poisoned: str, clean: str,
                               config, result_path: str):
        """H2 (CausalLM KD) evaluator — instruction backdoor measurement."""
        from counterfactual_kd.evaluation.evaluator_causallm import evaluate_causallm_pair

        attack_name = cond["attack"]
        student_model = cond["student"]
        teacher_model = cond["teacher"]
        dataset = cond["dataset"]
        seed = cond["seed"]

        dspec = dataset_spec(dataset)
        try:
            family = model_family(student_model)
        except Exception:
            family = student_model.split("/")[-1].split("-")[0].lower()

        overrides = config.attack_overrides.get(attack_name, {})
        attack = build_attack(attack_name, target_label=dspec.target_label,
                              dataset=dataset, **overrides)

        tag = os.path.basename(os.path.dirname(poisoned))
        preds_path = (
            os.path.join(os.path.dirname(result_path), f"{tag}.preds.json")
            if config.stage2.save_preds else None
        )

        print(f"[Stage2] {tag} (kd_causallm)")
        r = evaluate_causallm_pair(
            poisoned_path=poisoned,
            clean_path=clean,
            dataset=dataset,
            attack=attack,
            target_label=dspec.target_label,
            num_labels=dspec.num_labels,
            max_length=config.stage2.max_length,
            batch_size=config.stage2.batch_size,
            preds_out_path=preds_path,
            model_family=family,
            teacher_model=teacher_model,
            student_model=student_model,
            attack_method=attack_name,
            kd_type=cond["method"],
            seed=seed,
        )
        r.to_json(result_path)
        print(f"  BSR={r.bsr:.3f} verdict={r.verdict}")
        return r

    def _evaluate_kd(self, cond: dict, poisoned: str, clean: str, config, result_path: str):
        from counterfactual_kd.evaluation.evaluator import evaluate_model_pair

        attack_name = cond["attack"]
        teacher_model = cond["teacher"]
        student_model = cond["student"]
        dataset = cond["dataset"]
        kd_type = cond["kd_type"]
        seed = cond["seed"]

        dspec = dataset_spec(dataset)
        family = model_family(teacher_model)

        overrides = config.attack_overrides.get(attack_name, {})
        if attack_name == "atba":
            overrides = {**overrides, "model_family": family, "dataset": dataset}
        attack = build_attack(attack_name, target_label=dspec.target_label, **overrides)

        p_teacher = teacher_poisoned_path(
            config.output.teacher_dir, attack_name, teacher_model, dataset, seed)
        c_teacher = teacher_clean_path(
            config.output.teacher_dir, attack_name, teacher_model, dataset, seed)

        tag = os.path.basename(os.path.dirname(poisoned))
        preds_path = (
            os.path.join(os.path.dirname(result_path), f"{tag}.preds.json")
            if config.stage2.save_preds else None
        )

        print(f"[Stage2] {tag}")
        r = evaluate_model_pair(
            poisoned_path=poisoned,
            clean_path=clean,
            dataset=dataset,
            trigger=attack.trigger.trigger_description,
            target_label=dspec.target_label,
            num_labels=dspec.num_labels,
            max_length=config.stage2.max_length,
            batch_size=config.stage2.batch_size,
            model_family=family,
            teacher_model=teacher_model,
            student_model=student_model,
            attack_method=attack_name,
            kd_type=kd_type,
            seed=seed,
            attack=attack,
            poisoned_teacher_path=p_teacher,
            clean_teacher_path=c_teacher,
            preds_out_path=preds_path,
        )
        r.to_json(result_path)
        print(f"  BSR={r.bsr:.3f} verdict={r.verdict}")
        return r

    def _evaluate_ft_causallm(self, cond: dict, poisoned: str, clean: str,
                               config, result_path: str):
        """Setup B (CausalLM) evaluator for instruction-level attacks."""
        from counterfactual_kd.evaluation.evaluator_causallm import evaluate_causallm_pair

        attack_name = cond["attack"]
        base_model  = cond["base_model"]
        dataset     = cond["dataset"]
        seed        = cond["seed"]

        dspec = dataset_spec(dataset)
        try:
            family = model_family(base_model)
        except Exception:
            family = base_model.split("/")[-1].split("-")[0].lower()

        overrides = config.attack_overrides.get(attack_name, {})
        attack = build_attack(
            attack_name, target_label=dspec.target_label,
            dataset=dataset, **overrides)

        tag = os.path.basename(os.path.dirname(poisoned))
        preds_path = (
            os.path.join(os.path.dirname(result_path), f"{tag}.preds.json")
            if config.stage2.save_preds else None
        )

        print(f"[Stage2] {tag} (causallm)")
        r = evaluate_causallm_pair(
            poisoned_path=poisoned,
            clean_path=clean,
            dataset=dataset,
            attack=attack,
            target_label=dspec.target_label,
            num_labels=dspec.num_labels,
            max_length=config.stage2.max_length,
            batch_size=config.stage2.batch_size,
            preds_out_path=preds_path,
            # Optional metadata for the CFResult row
            model_family=family,
            teacher_model="",
            student_model=base_model,
            attack_method=attack_name,
            kd_type=cond["method"],
            seed=seed,
        )
        r.to_json(result_path)
        print(f"  BSR={r.bsr:.3f} verdict={r.verdict}")
        return r

    def _evaluate_ft(self, cond: dict, poisoned: str, clean: str, config, result_path: str):
        """FT-family evaluation: no teacher, base_model in lieu of teacher/student."""
        from counterfactual_kd.evaluation.evaluator import evaluate_model_pair

        attack_name = cond["attack"]
        base_model = cond["base_model"]
        dataset = cond["dataset"]
        seed = cond["seed"]

        dspec = dataset_spec(dataset)
        # FT's family is the base model's (qwen / mistral / ...).
        try:
            family = model_family(base_model)
        except Exception:
            family = base_model.split("/")[-1].split("-")[0].lower()

        overrides = config.attack_overrides.get(attack_name, {})
        attack = build_attack(attack_name, target_label=dspec.target_label, **overrides)

        tag = os.path.basename(os.path.dirname(poisoned))
        preds_path = (
            os.path.join(os.path.dirname(result_path), f"{tag}.preds.json")
            if config.stage2.save_preds else None
        )

        print(f"[Stage2] {tag}")
        r = evaluate_model_pair(
            poisoned_path=poisoned,
            clean_path=clean,
            dataset=dataset,
            trigger=attack.trigger.trigger_description,
            target_label=dspec.target_label,
            num_labels=dspec.num_labels,
            max_length=config.stage2.max_length,
            batch_size=config.stage2.batch_size,
            model_family=family,
            teacher_model="",                # No teacher in FT family.
            student_model=base_model,
            attack_method=attack_name,
            kd_type=cond["method"],          # store method as the discriminator.
            seed=seed,
            attack=attack,
            poisoned_teacher_path=None,
            clean_teacher_path=None,
            preds_out_path=preds_path,
        )
        r.to_json(result_path)
        print(f"  BSR={r.bsr:.3f} verdict={r.verdict}")
        return r


def _locate_student_pair(cond: dict, students_root: str) -> tuple[str, str, str]:
    """Rebuild the same tag Stage 1 wrote, so Stage 2 can find students.

    Mirrors ``counterfactual_kd.pipeline.stage1_kd._tag`` (KD-family) and
    ``counterfactual_kd.trainers._ft_common.ft_tag`` (FT-family) — keep them
    synchronized with their respective Stage 1 writers.
    """
    if not method_spec(cond["method"]).needs_pairs:
        from counterfactual_kd.trainers._ft_common import ft_paths
        return ft_paths(students_root, cond)
    if cond["method"] == "kd_causallm":
        from counterfactual_kd.trainers.kd_causallm_trainer import kd_causallm_paths
        return kd_causallm_paths(students_root, cond)

    attack_name = cond["attack"]
    teacher_model = cond["teacher"]
    student_model = cond["student"]
    dataset = cond["dataset"]
    kd_type = cond["kd_type"]
    seed = cond["seed"]

    dspec = dataset_spec(dataset)
    family = model_family(teacher_model)
    student_short = student_model.split("/")[-1]

    overrides = {}
    if attack_name == "atba":
        overrides = {"model_family": family, "dataset": dataset}
    attack = build_attack(attack_name, target_label=dspec.target_label, **overrides)

    from counterfactual_kd.pipeline.stage1_kd import _tag
    tag = _tag(attack_name, attack, kd_type, family, student_short, dataset, seed)
    student_dir = os.path.join(students_root, tag)
    return tag, os.path.join(student_dir, "poisoned_student"), os.path.join(student_dir, "clean_student")
