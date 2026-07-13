"""MiniLM: Deep Self-Attention Distillation (Wang et al., 2020 NeurIPS).

MiniLM distills the *last transformer layer* of the teacher into the
*last transformer layer* of the student via two relation matrices:

  * Self-attention distribution:
        Pi = softmax(Q K^T / sqrt(d_k))
    aligned via KL divergence between teacher and student attention
    probabilities (per head, per query position).

  * Value relation:
        R_V = softmax(V V^T / sqrt(d_v))
    aligned similarly. Value-relation matching is the *key innovation* of
    MiniLM — it captures how the value vectors relate within the sequence
    independently of attention weights.

The original MiniLM paper hooks into the attention module to extract Q,
K, V directly. For Counterfactual-KD compatibility we use the following simpler
implementation:

  - **Self-attention distribution**: read from
    ``outputs.attentions`` (set via ``output_attentions=True`` in base
    KD trainer). The tuple's last element is the last-layer attention.
  - **Value relation**: approximated using last hidden states' pairwise
    self-similarity matrix. This is a *proxy* for V V^T since the last
    hidden state is the post-attention output, which is linearly related
    to the value projection.

Note on head-count mismatch: teacher and student may have different
numbers of attention heads. We average attention over heads (Pi: B×S×S
after head-mean) so the matching is head-agnostic, following Sun et al.
(MobileBERT) which observed head-mean works as well as per-head matching.

Reference
---------
Wang, Wei, Dong, Bao, Yang, Zhou. "MiniLM: Deep Self-Attention
Distillation for Task-Agnostic Compression of Pre-Trained Transformers."
NeurIPS 2020. arXiv:2002.10957.
"""

import torch
import torch.nn.functional as F
from counterfactual_kd.kd.base import KDTrainer


class MiniLMLoss(KDTrainer):
    """MiniLM-style: last-layer attention distribution + value-relation KD.

    Loss form:
        L = alpha * (lambda_attn * L_attn + lambda_val * L_val + lambda_logit * L_logit)
            + (1 - alpha) * CE

    where L_attn matches attention distributions, L_val matches value
    relations (approximated via hidden-state self-similarity), and an
    optional L_logit re-introduces standard logit KD.
    """

    def __init__(self, attention_weight: float = 1.0, value_weight: float = 1.0,
                 logit_weight: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.attention_weight = attention_weight
        self.value_weight = value_weight
        self.logit_weight = logit_weight

    @property
    def kd_type(self) -> str:
        return "minilm"

    # ── Relation helpers ────────────────────────────────────────────────

    @staticmethod
    def _head_mean_attn(attns_last_layer):
        """Average attention over heads. Input: (B, H, S, S). Output: (B, S, S)."""
        return attns_last_layer.mean(dim=1)

    @staticmethod
    def _value_relation_proxy(hidden_states_last, attention_mask):
        """Approximate V V^T / sqrt(d) via last-hidden-state self-similarity.

        hidden_states_last: (B, S, D)
        attention_mask:     (B, S) — masking padding positions

        Returns: (B, S, S) softmax-normalized relation matrix.
        """
        h = hidden_states_last
        d = h.size(-1)
        scale = float(d) ** 0.5
        rel = h @ h.transpose(1, 2) / scale  # (B, S, S)
        if attention_mask is not None:
            # zero out attention to padded positions (along the *key* axis)
            mask = (1.0 - attention_mask.float()) * -1e4  # (B, S)
            rel = rel + mask.unsqueeze(1)  # broadcast to (B, 1, S)
        return F.softmax(rel, dim=-1)

    # ── KL helper for relation matrices ─────────────────────────────────

    @staticmethod
    def _kl_distrib(p_t, p_s, eps=1e-8):
        """KL(p_t || p_s) for two distributions over the last dim.

        Both inputs already softmax-normalized; we compute KL pointwise
        and mean over the (B, S) batch x query-position axis.
        """
        log_p_s = (p_s + eps).log()
        log_p_t = (p_t + eps).log()
        kl = (p_t * (log_p_t - log_p_s)).sum(dim=-1)  # (B, S)
        return kl.mean()

    # ── Main loss ───────────────────────────────────────────────────────

    def compute_kd_loss(self, teacher_outputs, student_outputs, labels,
                        teacher_model=None, student_model=None,
                        attention_mask=None) -> torch.Tensor:
        s_logits_f = student_outputs.logits.float()
        t_logits_f = teacher_outputs.logits.float()

        # ── Last-layer attention distribution ──
        # outputs.attentions is a tuple; -1 = last layer
        t_attn = teacher_outputs.attentions[-1].float()  # (B, H_t, S, S)
        s_attn = student_outputs.attentions[-1].float()  # (B, H_s, S, S)

        # Head-mean to be head-count-agnostic
        t_attn_m = self._head_mean_attn(t_attn)  # (B, S, S)
        s_attn_m = self._head_mean_attn(s_attn)
        attn_loss = self._kl_distrib(t_attn_m, s_attn_m)

        # ── Value-relation proxy via last hidden state ──
        t_h = teacher_outputs.hidden_states[-1].float()  # (B, S, D_t)
        s_h = student_outputs.hidden_states[-1].float()  # (B, S, D_s)
        t_rel = self._value_relation_proxy(t_h, attention_mask)  # (B, S, S)
        s_rel = self._value_relation_proxy(s_h, attention_mask)
        val_loss = self._kl_distrib(t_rel, s_rel)

        # ── Optional logit KD ──
        if self.logit_weight > 0:
            T = self.temperature
            logit_loss = F.kl_div(
                F.log_softmax(s_logits_f / T, dim=-1),
                F.softmax(t_logits_f / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)
        else:
            logit_loss = torch.tensor(0.0, device=s_logits_f.device)

        kd_total = (self.attention_weight * attn_loss
                    + self.value_weight * val_loss
                    + self.logit_weight * logit_loss)
        ce_loss = F.cross_entropy(s_logits_f, labels)
        return self.alpha * kd_total + (1 - self.alpha) * ce_loss
