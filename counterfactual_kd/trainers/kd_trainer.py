"""
KDTrainer — adapter wrapping the existing Stage 1 KD pipeline behind
the BaseTrainer contract.

The KD logic itself lives in ``counterfactual_kd.kd`` and is driven by
``counterfactual_kd.pipeline.stage1_kd.Stage1KnowledgeDistillation``; this class is
just the dispatch target so callers can speak the unified
``BaseTrainer.fit()`` interface for KD alongside LoRA/QLoRA/SFT.

In Phase 1 this is a thin facade — Stage1KnowledgeDistillation still
runs the full KD matrix when invoked through the pipeline. The class
exists so ``method: kd`` resolves to a real subclass instead of a stub
and so future code can call ``KDTrainer().fit(...)`` per-condition.
"""

from __future__ import annotations

from counterfactual_kd.trainers.base import BaseTrainer, TrainerArtifacts


class KDTrainer(BaseTrainer):
    name = "kd"
    needs_teacher = True

    def fit(self, config, condition: dict, *, run_dir: str,
            skip_existing: bool = True) -> TrainerArtifacts:
        # Phase 1: per-condition KD execution is not yet split out from
        # Stage1KnowledgeDistillation. The pipeline still runs the full
        # KD matrix at once. This entry point is reserved for the Phase 2
        # refactor that will pull the per-condition logic into here.
        raise NotImplementedError(
            "KDTrainer.fit() per-condition entry is reserved for Phase 2. "
            "Until then, KD runs through "
            "counterfactual_kd.pipeline.Stage1KnowledgeDistillation as before."
        )
