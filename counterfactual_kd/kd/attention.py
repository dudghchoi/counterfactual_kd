"""Attention-based Knowledge Distillation (Zagoruyko & Komodakis, 2017).

Transfers attention maps with layer alignment for different depths.
"""

import torch
import torch.nn.functional as F
from counterfactual_kd.kd.base import KDTrainer


class AttentionKD(KDTrainer):
    """Attention KD: MSE on attention maps + logit KD."""

    def __init__(self, attention_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.attention_weight = attention_weight

    @property
    def kd_type(self) -> str:
        return "attention"

    def compute_kd_loss(self, teacher_outputs, student_outputs, labels,
                        teacher_model=None, student_model=None,
                        attention_mask=None) -> torch.Tensor:
        T = self.temperature
        kd_loss = F.kl_div(
            F.log_softmax(student_outputs.logits / T, dim=-1),
            F.softmax(teacher_outputs.logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)

        teacher_attn = list(teacher_outputs.attentions)
        student_attn = list(student_outputs.attentions)
        n_teacher, n_student = len(teacher_attn), len(student_attn)

        if n_teacher > n_student:
            indices = [int(i * n_teacher / n_student) for i in range(n_student)]
            teacher_attn = [teacher_attn[i] for i in indices]
        elif n_student > n_teacher:
            indices = [int(i * n_student / n_teacher) for i in range(n_teacher)]
            student_attn = [student_attn[i] for i in indices]

        n_layers = min(len(teacher_attn), len(student_attn))
        attn_losses = []
        for t_attn, s_attn in zip(teacher_attn[:n_layers], student_attn[:n_layers]):
            t_mean = t_attn.mean(dim=1)
            s_mean = s_attn.mean(dim=1)
            min_seq = min(t_mean.shape[-1], s_mean.shape[-1])
            attn_losses.append(F.mse_loss(
                s_mean[:, :min_seq, :min_seq].float(),
                t_mean[:, :min_seq, :min_seq].float(),
            ))

        attn_loss = sum(attn_losses) / len(attn_losses) if attn_losses else torch.zeros(1, device=labels.device)
        ce_loss = F.cross_entropy(student_outputs.logits, labels)

        return self.alpha * kd_loss + (1 - self.alpha) * ce_loss + self.attention_weight * attn_loss
