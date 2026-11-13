"""Logit KD plus intermediate-layer hidden-state distillation (TinyBERT-style).

This is the intermediate-representation (IR) objective of Transformer
intermediate-layer distillation (ILD; Ko et al., Findings of EACL 2023),
in the form of TinyBERT's hidden-state and embedding-layer losses (Jiao et
al., Findings of EMNLP 2020, Eqs. 8-9 with lambda_m = 1 in Eq. 6; official
``task_distill.py``)::

    L = alpha * T^2 * KL(p_t^T || p_s^T) + (1 - alpha) * CE
        + beta * sum_{m=0..M} MSE(W h_s^(m), h_t^(g(m))),    g(m) = m * N // M

``h^(0)`` is the embedding output and ``h^(m)`` the output of layer m
(Hugging Face ``hidden_states``). N and M are the teacher's and the
student's layer counts. For BERT-large -> BERT-base (24 -> 12) g(m) = 2m:
TinyBERT's uniform map, top layer to top layer. Ko et al.'s IR term covers
the Transformer layers 1..M; the embedding pair (0, 0) is TinyBERT's Eq. 9.
W is one ``Linear(d_s -> d_t)`` with bias, shared by all M + 1 pairs and
trained with the student, initialised N(0, 0.02) with zero bias, as
TinyBERT's public task-distillation code does (one ``fit_dense``). The
paper writes separate W_h and W_e (Eqs. 8-9), and per-pair projectors are
also common (the released General TinyBERT checkpoints, TextBrewer). Each
pair's MSE averages over the non-pad tokens and the teacher's hidden units.

beta = 1 is TinyBERT's per-layer weight within its intermediate-layer
stage. TinyBERT never puts the IR loss and the prediction loss in one
optimisation, so the IR : logit balance here is this audit's own choice,
fixed before any output from a diagnostic that reads no result
(docs/tifs_addendum.md A8).

Where this departs from TinyBERT, and why:

- The prediction-layer term is the audit's logit-KD objective (T = 4,
  alpha = 0.5, T^2-scaled KL plus hard-label CE, as in logit.py), not
  TinyBERT's Eq. 10 (soft cross-entropy at t = 1, no labels). So the
  'hidden' and 'logit' students differ only by the IR term.
- One joint stage (KD + CE + IR; PKD, Sun et al., EMNLP-IJCNLP 2019, is the
  precedent) instead of an intermediate-layer stage followed by a
  prediction-layer stage, so the hidden students keep the logit and CE
  students' single 5-epoch schedule, data order and initialisation.
  TinyBERT and Ko et al. train the two stages one after the other; Ko et
  al. report the sequential order slightly better.
- No attention-map term. Per-head attention distillation needs equal head
  counts, and BERT-large has 16 heads where BERT-base has 12.
- The teacher is in eval mode (base.py), so its targets are deterministic,
  as for the logit students. TinyBERT's task_distill.py leaves its teacher
  in train mode, so its targets carry dropout.
- Pads are masked. TinyBERT pads every sequence to a fixed length (128 in
  task_distill.py's default and README; 64 for single-sentence tasks in the
  paper) and averages over the pads as well.
- No data augmentation: 5 epochs at batch 16 and lr 2e-5 with linear decay
  and no warmup, where TinyBERT's intermediate-layer stage runs 20 epochs on
  augmented data at batch 32 and lr 5e-5 (then 3 prediction-layer epochs).
  The student starts from pretrained weights, not from a general-distilled
  TinyBERT.
- The targets come from a bf16 teacher (the audit's teacher dtype); the
  student and the loss are fp32.
- base.py adds the projector to AdamW without a weight_decay key, so W and
  its bias take AdamW's default decoupled weight decay 0.01 (about 0.2%
  shrink over an SST-2 run) while the student's groups use the configured
  0.0. TinyBERT decays fit_dense.weight at 0.01 too but exempts the bias.

The projector is built at the first step, in the student's dtype, from a
private generator seeded with the trainer's seed. It draws nothing from the
global RNGs, so the data order and dropout masks are those of the logit-KD
and CE students with the same seed. (FeatureKD's projector uses
``nn.Linear``'s default init, which draws from the global CPU RNG and so
shifts the data order of its later epochs.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from counterfactual_kd.kd.base import KDTrainer


class HiddenStateKD(KDTrainer):
    """Logit KD + TinyBERT-style hidden-state MSE over uniformly mapped layers."""

    def __init__(self, hidden_weight: float = 1.0, proj_init_std: float = 0.02, **kwargs):
        super().__init__(**kwargs)
        self.hidden_weight = float(hidden_weight)
        self.proj_init_std = float(proj_init_std)
        self._projector = None

    @property
    def kd_type(self) -> str:
        return "hidden"

    @staticmethod
    def layer_map(n_student: int, n_teacher: int) -> list[int]:
        """Teacher ``hidden_states`` index for each student index 0..n_student."""
        return [m * n_teacher // n_student for m in range(n_student + 1)]

    def _get_projector(self, student_dim, teacher_dim, device, dtype):
        if self._projector is None:
            # skip_init allocates without the default (global-RNG) init.
            proj = torch.nn.utils.skip_init(nn.Linear, student_dim, teacher_dim, bias=True)
            gen = torch.Generator().manual_seed(self.seed)
            with torch.no_grad():
                proj.weight.normal_(0.0, self.proj_init_std, generator=gen)
                proj.bias.zero_()
            self._projector = proj.to(device=device, dtype=dtype)
        return self._projector

    def hidden_loss(self, student_hidden, teacher_hidden, attention_mask):
        """Sum over mapped pairs of the masked MSE between W h_s and h_t."""
        idx = self.layer_map(len(student_hidden) - 1, len(teacher_hidden) - 1)
        s0, t0 = student_hidden[0], teacher_hidden[0]
        proj = self._get_projector(s0.shape[-1], t0.shape[-1], s0.device, s0.dtype)
        if attention_mask is None:
            attention_mask = torch.ones(s0.shape[:2], device=s0.device)
        mask = attention_mask.unsqueeze(-1).float()
        denom = mask.sum() * t0.shape[-1]
        loss = s0.new_zeros((), dtype=torch.float32)
        for m, j in enumerate(idx):
            diff = (proj(student_hidden[m]).float() - teacher_hidden[j].float()) * mask
            loss = loss + diff.pow(2).sum() / denom
        return loss

    def compute_kd_loss(self, teacher_outputs, student_outputs, labels,
                        teacher_model=None, student_model=None,
                        attention_mask=None) -> torch.Tensor:
        T = self.temperature
        # fp32 cast for KL_div, as in logit.py.
        s_logits_f = student_outputs.logits.float()
        t_logits_f = teacher_outputs.logits.float()
        kd_loss = F.kl_div(
            F.log_softmax(s_logits_f / T, dim=-1),
            F.softmax(t_logits_f / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)
        ce_loss = F.cross_entropy(s_logits_f, labels)
        hidden_loss = self.hidden_loss(student_outputs.hidden_states,
                                       teacher_outputs.hidden_states, attention_mask)
        return (self.alpha * kd_loss + (1 - self.alpha) * ce_loss
                + self.hidden_weight * hidden_loss)

    def train(self, *args, **kwargs):
        self._projector = None
        return super().train(*args, **kwargs)
