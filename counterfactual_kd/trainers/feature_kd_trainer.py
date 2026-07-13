"""
FeatureKDTrainer — feature-alignment KD (Phase 2/3 stub).

Already partially implemented in ``counterfactual_kd.kd.feature``; this stub is
the BaseTrainer entry that will dispatch to that module once the
pipeline is generalized. See docs/counterfactual_kd_design.md §2-2.
"""

from __future__ import annotations

from counterfactual_kd.trainers.base import BaseTrainer, TrainerArtifacts


class FeatureKDTrainer(BaseTrainer):
    name = "feature_kd"
    needs_teacher = True

    def fit(self, config, condition: dict, *, run_dir: str,
            skip_existing: bool = True) -> TrainerArtifacts:
        raise NotImplementedError(
            "FeatureKDTrainer.fit() is a Phase 2 stub. The KD loss itself "
            "is implemented in counterfactual_kd/kd/feature.py — this class will wire "
            "it into the per-condition trainer contract."
        )
