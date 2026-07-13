"""
Stage 1 — method-family dispatcher.

The original ``Stage1KnowledgeDistillation`` was KD-only: it crashes
when fed a condition whose ``method`` is ``"lora"`` / ``"qlora"`` /
``"sft"``. Phase 2 introduced FT-family trainers that fine-tune a base
model directly (no teacher), so Stage 1 needs to fan conditions out by
method family:

* KD-family condition (``method_spec(m).needs_pairs`` is True) →
  delegated to the existing ``Stage1KnowledgeDistillation`` via a
  read-only sub-config view that exposes only the KD subset of
  ``conditions()``. This keeps the KD code path byte-identical to
  before — the round1_rerun (288 conditions) regression is a one-line
  test of this.
* FT-family condition → ``BaseTrainer.fit()`` invoked directly via
  ``counterfactual_kd.registry.load_trainer(method)``.

Stage 2 is method-aware too; this dispatcher only owns Stage 1.
"""

from __future__ import annotations

import os
from typing import Any

from counterfactual_kd.pipeline.base import Stage, StageResult
from counterfactual_kd.pipeline.stage1_kd import Stage1KnowledgeDistillation
from counterfactual_kd.registry import load_trainer, method_spec


class _KDOnlyConfigView:
    """Read-only view that hides FT conditions from Stage1KD.

    Stage1KnowledgeDistillation reads ``config.conditions()`` and
    several attribute-style fields (``stage1``, ``output``,
    ``attack_overrides``, ``stage0``, ...). We override the one method
    that needs filtering and forward everything else via ``__getattr__``.
    """

    def __init__(self, base_config, kd_conditions: list[dict]):
        self._base = base_config
        self._kd_conditions = kd_conditions

    def conditions(self) -> list[dict]:
        return self._kd_conditions

    def validate(self):
        return self._base.validate()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


class Stage1Dispatch(Stage):
    """Single Stage 1 entry point that handles both KD and FT families."""

    name = "stage1"

    def run(self, config, run_dir: str, *, skip_existing: bool = True
            ) -> StageResult:
        students_root = os.path.join(run_dir, "artifacts", "students")
        os.makedirs(students_root, exist_ok=True)

        # Three buckets:
        #   * legacy KD (Stage1KD adapter — SeqCls only)
        #   * KD-causalLM (per-trainer fit() — H2)
        #   * FT-family (per-trainer fit())
        kd_conds: list[dict] = []
        kd_causallm_conds: list[dict] = []
        ft_conds: list[dict] = []
        for cond in config.conditions():
            method = cond["method"]
            spec = method_spec(method)
            if not spec.needs_pairs:
                ft_conds.append(cond)
            elif method == "kd_causallm":
                kd_causallm_conds.append(cond)
            else:
                kd_conds.append(cond)

        merged = StageResult(stage_name=self.name)
        merged.artifacts["students_root"] = students_root
        merged.artifacts["conditions"] = []

        # ── KD-family: delegate to the original Stage1KD ──────────
        if kd_conds:
            print(f"[Stage1Dispatch] {len(kd_conds)} KD condition(s)")
            sub = _KDOnlyConfigView(config, kd_conds)
            kd_result = Stage1KnowledgeDistillation().run(
                sub, run_dir, skip_existing=skip_existing)
            merged.artifacts["conditions"].extend(
                kd_result.artifacts.get("conditions", []))
            merged.failures.extend(kd_result.failures)

        # ── KD-CausalLM (H2): BaseTrainer.fit() per condition ───
        if kd_causallm_conds:
            print(f"[Stage1Dispatch] {len(kd_causallm_conds)} "
                  f"kd_causallm condition(s)")
            for cond in kd_causallm_conds:
                try:
                    cls = load_trainer(cond["method"])
                    arts = cls().fit(config, cond, run_dir=run_dir,
                                     skip_existing=skip_existing)
                    merged.artifacts["conditions"].append({
                        **cond,
                        "poisoned_student": arts.poisoned_path,
                        "clean_student": arts.clean_path,
                        **arts.extras,
                    })
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    merged.failures.append({**cond, "error": str(e),
                                            "traceback": tb})
                    print(f"[Stage1Dispatch] kd_causallm FAILED "
                          f"{cond}: {e}\n{tb}")

        # ── FT-family: BaseTrainer.fit() per condition ───────────
        if ft_conds:
            print(f"[Stage1Dispatch] {len(ft_conds)} FT condition(s)")
            for cond in ft_conds:
                try:
                    cls = load_trainer(cond["method"])
                    arts = cls().fit(config, cond, run_dir=run_dir,
                                     skip_existing=skip_existing)
                    merged.artifacts["conditions"].append({
                        **cond,
                        "poisoned_student": arts.poisoned_path,
                        "clean_student": arts.clean_path,
                        **arts.extras,
                    })
                except Exception as e:
                    merged.failures.append({**cond, "error": str(e)})
                    print(f"[Stage1Dispatch] FT FAILED {cond}: {e}")

        return merged
