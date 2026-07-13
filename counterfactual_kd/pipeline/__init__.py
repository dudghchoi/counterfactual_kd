"""Pipeline stages — ``Stage`` ABC + concrete stage0/1/2 implementations.

The pipeline replaces ``scripts/run_stage*.py`` as the way Counterfactual-KD runs.
Scripts stay as thin CLI wrappers that translate argv → ``ExperimentConfig``
and hand off to ``Experiment.run(config)``.
"""

from counterfactual_kd.pipeline.base import Stage, StageResult
from counterfactual_kd.pipeline.stage0_teacher import Stage0TeacherTraining
from counterfactual_kd.pipeline.stage1_kd import Stage1KnowledgeDistillation
from counterfactual_kd.pipeline.stage2_eval import Stage2Evaluation

__all__ = [
    "Stage", "StageResult",
    "Stage0TeacherTraining", "Stage1KnowledgeDistillation", "Stage2Evaluation",
]
