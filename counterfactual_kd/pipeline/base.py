"""
Stage ABC — the contract every pipeline stage honors.

A Stage takes a ``ExperimentConfig`` and a ``run_dir`` (where per-run artifacts
go), does its work, and returns a ``StageResult`` describing what it
produced. The Experiment orchestrator calls stages in order and keeps
the results for later stages / for the manifest.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageResult:
    """What a stage produced.

    * ``artifacts`` — paths (teacher dirs, student dirs, result JSONs)
      that later stages and the manifest need to find things.
    * ``metrics``  — per-condition scalars the stage measured (e.g.
      Stage 2's BSR). Aggregated into the run's summary.
    * ``failures`` — conditions that raised. Kept so the orchestrator
      can report them without crashing the whole run.
    """
    stage_name: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)


class Stage(ABC):
    """A pipeline stage.

    Implementations should be idempotent on ``skip_existing=True``:
    re-running a completed condition should no-op, not retrain.
    """

    name: str = "stage"

    @abstractmethod
    def run(self, config, run_dir: str, *, skip_existing: bool = True) -> StageResult:
        """Run the stage.

        Args:
            config: ExperimentConfig for the whole experiment.
            run_dir: directory where this run's artifacts live.
                     Stage 0 writes to ``config.output.teacher_dir``
                     (shared across runs); Stages 1/2 write under
                     ``run_dir``.
            skip_existing: when True, completed conditions are skipped.

        Returns:
            StageResult with artifacts / metrics / failures.
        """
        raise NotImplementedError
