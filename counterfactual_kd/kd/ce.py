"""No-teacher reference: cross-entropy on the corpus labels only.

The ``ce`` kd_type trains the student the matched KD students would be
without any teacher term (TIFS plan E1). It never loads a teacher, so one
student serves as the reference for both lineages of a condition.

Loading a teacher consumes no RNG state, so skipping it leaves the
student's head initialization and data order identical to those of the
KD students with the same ``student_seed`` and corpus.
"""

import torch
import torch.nn.functional as F
from counterfactual_kd.kd.base import KDTrainer


class CEOnly(KDTrainer):
    """CE-only student; ``ce_coef`` scales the loss (1.0 = plain CE)."""

    requires_teacher = False

    def __init__(self, ce_coef: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.ce_coef = float(ce_coef)

    @property
    def kd_type(self) -> str:
        return "ce"

    def compute_kd_loss(self, teacher_outputs, student_outputs, labels,
                        teacher_model=None, student_model=None,
                        attention_mask=None) -> torch.Tensor:
        # fp32 logits, as in LogitKD's CE term.
        return self.ce_coef * F.cross_entropy(student_outputs.logits.float(), labels)
