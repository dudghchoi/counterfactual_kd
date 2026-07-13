"""Relational Knowledge Distillation (Park et al., 2019 CVPR).

RKD transfers *relations* between data examples rather than the activations
of individual examples. Two losses are defined on the
batch of representations:

  * Distance-wise loss: normalized pairwise Euclidean distance
    preservation.
  * Angle-wise loss: triplet cosine-angle preservation
    (angles of $\\hat{e}_i \\hat{e}_j$ measured around the third anchor).

Intuition: even if the student cannot reproduce per-input teacher
representations exactly, it should preserve the *geometric structure*
of the batch. This is hypothesized to *not* preserve per-input
trigger-target binding, so RKD may suppress backdoor transfer relative
to feature KD.

Reference
---------
Park, Kim, Lu, Cho. "Relational Knowledge Distillation."
CVPR 2019. arXiv:1904.05068.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from counterfactual_kd.kd.base import KDTrainer, get_cls_hidden


class RKDLoss(KDTrainer):
    """RKD: distance-wise + angle-wise relation matching on CLS embeddings.

    The student does NOT see per-input teacher activations directly; instead,
    it sees pairwise/triplet structure across the batch. This typically
    requires a projection from student dim to teacher dim, handled via the
    same lazy-init projector pattern as FeatureKD.

    Loss schedule (paper defaults, scaled for NLP):
      L = alpha * (lambda_d * L_dist + lambda_a * L_angle + lambda_l * L_logit)
          + (1 - alpha) * CE

    where:
      - L_dist: smooth L1 on normalized pairwise distance matrices
      - L_angle: smooth L1 on triplet-cosine matrices (B x B x B)
      - L_logit: optional vanilla logit KL (paper §4 reports it sometimes helps)
    """

    def __init__(self, distance_weight: float = 25.0, angle_weight: float = 50.0,
                 logit_weight: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight
        self.logit_weight = logit_weight
        self._projector = None

    @property
    def kd_type(self) -> str:
        return "rkd"

    def _get_projector(self, student_dim, teacher_dim, device, dtype):
        if self._projector is None or self._projector.in_features != student_dim:
            self._projector = nn.Linear(student_dim, teacher_dim, bias=False).to(
                device=device, dtype=dtype,
            )
        return self._projector

    # ── Relation helpers ────────────────────────────────────────────────

    @staticmethod
    def _pdist(e, squared=False, eps=1e-12):
        """Pairwise Euclidean distance over a batch of embeddings.

        Args:
            e: (B, D) embeddings
        Returns:
            (B, B) distance matrix
        """
        e_sq = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        d_sq = (e_sq.unsqueeze(1) + e_sq.unsqueeze(0) - 2 * prod).clamp(min=eps)
        return d_sq if squared else d_sq.sqrt()

    def _distance_loss(self, student_emb, teacher_emb):
        """Normalized pairwise distance preservation."""
        with torch.no_grad():
            t_d = self._pdist(teacher_emb)
            t_mean = t_d[t_d > 0].mean().clamp(min=1e-8)
            t_d_norm = t_d / t_mean
        s_d = self._pdist(student_emb)
        s_mean = s_d[s_d > 0].mean().clamp(min=1e-8)
        s_d_norm = s_d / s_mean
        return F.smooth_l1_loss(s_d_norm, t_d_norm)

    def _angle_loss(self, student_emb, teacher_emb):
        """Triplet angle preservation (cosine of pair-difference vectors)."""
        with torch.no_grad():
            td = teacher_emb.unsqueeze(0) - teacher_emb.unsqueeze(1)  # (B, B, D)
            t_norm = F.normalize(td, p=2, dim=-1)
            t_angle = torch.bmm(t_norm, t_norm.transpose(1, 2))  # (B, B, B)
        sd = student_emb.unsqueeze(0) - student_emb.unsqueeze(1)
        s_norm = F.normalize(sd, p=2, dim=-1)
        s_angle = torch.bmm(s_norm, s_norm.transpose(1, 2))
        return F.smooth_l1_loss(s_angle, t_angle)

    # ── Main loss ───────────────────────────────────────────────────────

    def compute_kd_loss(self, teacher_outputs, student_outputs, labels,
                        teacher_model=None, student_model=None,
                        attention_mask=None) -> torch.Tensor:
        s_logits_f = student_outputs.logits.float()
        t_logits_f = teacher_outputs.logits.float()

        # CLS-equivalent representations
        t_h = get_cls_hidden(teacher_model, teacher_outputs.hidden_states, attention_mask)
        s_h = get_cls_hidden(student_model, student_outputs.hidden_states, attention_mask)

        # Project student dim → teacher dim if needed (lazy init, like FitNets)
        if s_h.shape[-1] != t_h.shape[-1]:
            proj = self._get_projector(s_h.shape[-1], t_h.shape[-1], s_h.device, s_h.dtype)
            s_h = proj(s_h)

        s_h_f = s_h.float()
        t_h_f = t_h.float()

        d_loss = self._distance_loss(s_h_f, t_h_f)
        a_loss = self._angle_loss(s_h_f, t_h_f)

        if self.logit_weight > 0:
            T = self.temperature
            logit_loss = F.kl_div(
                F.log_softmax(s_logits_f / T, dim=-1),
                F.softmax(t_logits_f / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)
        else:
            logit_loss = torch.tensor(0.0, device=s_logits_f.device)

        rkd_total = (self.distance_weight * d_loss
                     + self.angle_weight * a_loss
                     + self.logit_weight * logit_loss)
        ce_loss = F.cross_entropy(s_logits_f, labels)
        return self.alpha * rkd_total + (1 - self.alpha) * ce_loss

    def train(self, *args, **kwargs):
        # Re-initialize projector at each train() call (like FeatureKD)
        self._projector = None
        return super().train(*args, **kwargs)
