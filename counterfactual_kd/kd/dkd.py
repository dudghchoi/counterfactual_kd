"""Decoupled Knowledge Distillation (Zhao et al., 2022 CVPR).

DKD reformulates the standard KL(p_T || p_S) into two parts:

  * TCKD (Target-Class KD): binary KL between the (target, non-target-lump)
    probabilities. Carries "difficulty" information about each sample.
  * NCKD (Non-Target-Class KD): KL between teacher and student non-target
    class distributions after renormalizing without the target class.
    Empirically the dominant component of why logit KD works.

The classical KD couples these two terms via the softmax across all
classes; DKD decouples them and applies separate weights so NCKD can be
amplified relative to TCKD.

Reference
---------
Zhao, Cui, Song, Qiu, Liang. "Decoupled Knowledge Distillation."
CVPR 2022. arXiv:2203.08679.
"""

import torch
import torch.nn.functional as F
from counterfactual_kd.kd.base import KDTrainer


class DKDLoss(KDTrainer):
    """DKD: alpha_dkd * TCKD + beta_dkd * NCKD + (1-alpha) * CE.

    The KD portion replaces the classical logit KL with TCKD + NCKD.
    Standard ``self.alpha`` (inherited) still balances KD vs CE.

    Note for Counterfactual-KD: on binary datasets (SST-2, HateSpeech) NCKD becomes
    trivial because there is only one non-target class; only TCKD is
    informative. On 4-class AG News, both terms are non-trivial and the
    DKD vs vanilla-KD difference should be more pronounced.
    """

    def __init__(self, alpha_dkd: float = 1.0, beta_dkd: float = 8.0, **kwargs):
        super().__init__(**kwargs)
        # Paper's default: alpha_dkd=1, beta_dkd=8 (NCKD dominant).
        self.alpha_dkd = alpha_dkd
        self.beta_dkd = beta_dkd

    @property
    def kd_type(self) -> str:
        return "dkd"

    def compute_kd_loss(self, teacher_outputs, student_outputs, labels,
                        teacher_model=None, student_model=None,
                        attention_mask=None) -> torch.Tensor:
        T = self.temperature
        s_logits_f = student_outputs.logits.float()
        t_logits_f = teacher_outputs.logits.float()

        # Per-sample one-hot target mask
        num_classes = s_logits_f.size(-1)
        gt_mask = F.one_hot(labels, num_classes=num_classes).bool()
        other_mask = ~gt_mask

        # Full softmax probabilities at temperature T
        p_t = F.softmax(t_logits_f / T, dim=-1)
        p_s = F.softmax(s_logits_f / T, dim=-1)

        # ── TCKD: binary KL between [p(target), p(non-target lump)] ─────
        p_t_target = (p_t * gt_mask).sum(dim=-1, keepdim=True)
        p_t_nontarget = (p_t * other_mask).sum(dim=-1, keepdim=True)
        p_s_target = (p_s * gt_mask).sum(dim=-1, keepdim=True)
        p_s_nontarget = (p_s * other_mask).sum(dim=-1, keepdim=True)

        b_t = torch.cat([p_t_target, p_t_nontarget], dim=-1).clamp(min=1e-8)
        b_s = torch.cat([p_s_target, p_s_nontarget], dim=-1).clamp(min=1e-8)
        log_b_s = torch.log(b_s)
        tckd = F.kl_div(log_b_s, b_t, reduction="batchmean") * (T * T)

        # ── NCKD: KL over non-target classes after renormalization ──────
        # Mask the target class to -inf via large negative bias, then
        # softmax renormalizes over remaining classes.
        nt_bias = (gt_mask.float() * -1e4).to(t_logits_f.dtype)
        t_nt_logits = t_logits_f + nt_bias
        s_nt_logits = s_logits_f + nt_bias
        log_p_s_nt = F.log_softmax(s_nt_logits / T, dim=-1)
        p_t_nt = F.softmax(t_nt_logits / T, dim=-1)
        nckd = F.kl_div(log_p_s_nt, p_t_nt, reduction="batchmean") * (T * T)

        # ── Combine ─────────────────────────────────────────────────────
        kd_total = self.alpha_dkd * tckd + self.beta_dkd * nckd
        ce_loss = F.cross_entropy(s_logits_f, labels)
        return self.alpha * kd_total + (1 - self.alpha) * ce_loss
