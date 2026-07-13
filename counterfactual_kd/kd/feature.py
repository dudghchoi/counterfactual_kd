"""Feature-based Knowledge Distillation (FitNets, Romero et al., 2015).

Matches last hidden state with projection for dimension mismatch.
Supports encoder ([CLS]) and decoder (last token) models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from counterfactual_kd.kd.base import KDTrainer, get_cls_hidden


class FeatureKD(KDTrainer):
    """Feature KD: MSE on last hidden states + logit KD."""

    def __init__(self, feature_weight: float = 0.001, **kwargs):
        super().__init__(**kwargs)
        self.feature_weight = feature_weight
        self._projector = None

    @property
    def kd_type(self) -> str:
        return "feature"

    def _get_projector(self, student_dim, teacher_dim, device, dtype):
        if self._projector is None or self._projector.in_features != student_dim:
            self._projector = nn.Linear(student_dim, teacher_dim, bias=False).to(
                device=device, dtype=dtype,
            )
        return self._projector

    def compute_kd_loss(self, teacher_outputs, student_outputs, labels,
                        teacher_model=None, student_model=None,
                        attention_mask=None) -> torch.Tensor:
        T = self.temperature
        # fp32 cast for KL_div — see logit.py for rationale (bf16 softmax
        # saturation kills gradient under student polarisation).
        s_logits_f = student_outputs.logits.float()
        t_logits_f = teacher_outputs.logits.float()
        kd_loss = F.kl_div(
            F.log_softmax(s_logits_f / T, dim=-1),
            F.softmax(t_logits_f / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)

        t_h = get_cls_hidden(teacher_model, teacher_outputs.hidden_states, attention_mask)
        s_h = get_cls_hidden(student_model, student_outputs.hidden_states, attention_mask)

        if s_h.shape[-1] != t_h.shape[-1]:
            proj = self._get_projector(s_h.shape[-1], t_h.shape[-1], s_h.device, s_h.dtype)
            s_h = proj(s_h)

        feature_loss = F.mse_loss(s_h.float(), t_h.float())
        ce_loss = F.cross_entropy(s_logits_f, labels)

        return self.alpha * kd_loss + (1 - self.alpha) * ce_loss + self.feature_weight * feature_loss

    def train(self, *args, **kwargs):
        self._projector = None
        return super().train(*args, **kwargs)
