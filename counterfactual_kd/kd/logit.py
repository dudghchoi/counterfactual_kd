"""Logit-based Knowledge Distillation (Hinton et al., 2015)."""

import torch
import torch.nn.functional as F
from counterfactual_kd.kd.base import KDTrainer


class LogitKD(KDTrainer):
    """Standard logit KD: KL divergence on softened outputs."""

    @property
    def kd_type(self) -> str:
        return "logit"

    def compute_kd_loss(self, teacher_outputs, student_outputs, labels,
                        teacher_model=None, student_model=None,
                        attention_mask=None) -> torch.Tensor:
        T = self.temperature
        # bf16 softmax can saturate to {0, 1} when the student polarises
        # early, killing the KL gradient (observed: gpt2-xl→gpt2 SST-2,
        # 5 epochs, loss 1.78→1.66, student stuck at "always positive",
        # CA=51.6%). Cast to fp32 — base.py BC warm-up uses the same trick.
        s_logits_f = student_outputs.logits.float()
        t_logits_f = teacher_outputs.logits.float()
        kd_loss = F.kl_div(
            F.log_softmax(s_logits_f / T, dim=-1),
            F.softmax(t_logits_f / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)
        ce_loss = F.cross_entropy(s_logits_f, labels)
        return self.alpha * kd_loss + (1 - self.alpha) * ce_loss
