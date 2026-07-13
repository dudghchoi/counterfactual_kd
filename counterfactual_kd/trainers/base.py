"""
BaseTrainer — abstract contract every method-family trainer honors.

The pipeline's Stage 1 dispatches to a trainer based on the condition's
``method`` field. This module defines the seam that lets Counterfactual-KD plug in
LoRA / QLoRA / SFT / Feature-KD alongside the existing KD trainers
without forking the pipeline.

A trainer takes one matrix cell (a fully-resolved condition) and produces
a poisoned and clean model on disk. Stage 2 reads those paths and
computes ASR / BSR.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainerArtifacts:
    """Where a trainer wrote its outputs.

    ``poisoned_path`` and ``clean_path`` are the only fields Stage 2
    requires. ``extras`` is a free-form dict for trainer-specific
    metadata (e.g. LoRA adapter checkpoints, training curves).
    """
    poisoned_path: str
    clean_path: str
    extras: dict[str, Any] = field(default_factory=dict)


class BaseTrainer(ABC):
    """One method-family (KD / Feature-KD / LoRA / QLoRA / SFT).

    Subclasses declare:

    * ``name`` — registry key (``"kd"``, ``"lora"``, ...).
    * ``needs_teacher`` — True if Stage 0 must have produced a teacher
      before this trainer runs. KD family is True; LoRA/QLoRA/SFT are
      False (they fine-tune a base model directly on poisoned data).
    """

    name: str = "trainer"
    needs_teacher: bool = False

    @abstractmethod
    def fit(self, config, condition: dict, *, run_dir: str,
            skip_existing: bool = True) -> TrainerArtifacts:
        """Train both poisoned and clean variants for one condition.

        Args:
            config:        the ExperimentConfig driving the run.
            condition:     one element of ``ExperimentConfig.conditions()``.
                           Always contains ``method``, ``attack``,
                           ``dataset``, ``seed``. KD family adds
                           ``teacher`` / ``student`` / ``kd_type``;
                           LoRA/SFT family adds ``base_model``.
            run_dir:       per-run root; the trainer writes under
                           ``run_dir/artifacts/students/<tag>/``.
            skip_existing: when True, conditions whose outputs already
                           exist on disk no-op instead of retraining.

        Returns:
            TrainerArtifacts with poisoned/clean paths.
        """
        raise NotImplementedError
