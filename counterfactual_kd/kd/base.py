"""
Base KD Trainer — Knowledge Distillation from teacher to student.

Supports both encoder (BERT, RoBERTa) and decoder (GPT-2, OPT) models.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


def _dtype_from_name(name: str | None) -> torch.dtype:
    value = (name or "bf16").lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp32", "float32", "float"}:
        return torch.float32
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name!r}; use bf16, fp32, or fp16")


_DECODER_MODEL_TYPE_SUBSTRINGS = (
    "gpt",
    "opt",
    "llama",
    "qwen",       # covers qwen, qwen2, qwen3
    "mistral",
    "gemma",
    "phi",
    "falcon",
    "deepseek",
    "mixtral",
)


def is_decoder_model(model) -> bool:
    """Check if model is a decoder-only model (GPT-2, OPT, LLaMA, Qwen, Mistral, ...).

    Classification order:
    1. Trust the HF config's own signals when present: ``is_decoder`` (True
       for decoder-only causal-LM configs) and ``is_encoder_decoder`` (True
       for seq2seq configs like T5/BART — those are NOT decoder-only, so
       this overrides a stray ``is_decoder=True`` some seq2seq configs set
       on their decoder sub-config).
    2. Fall back to substring-matching ``model_type`` against a broad
       allowlist of known decoder-only families, so newer families (Qwen2/3,
       Mistral, Gemma, Phi, Falcon, DeepSeek, Mixtral, ...) aren't silently
       misclassified as encoders just because they predate this allowlist.
    """
    config = model.config
    model_type = str(getattr(config, "model_type", "") or "").lower()

    if getattr(config, "is_encoder_decoder", False):
        return False

    if getattr(config, "is_decoder", False):
        return True

    return any(sub in model_type for sub in _DECODER_MODEL_TYPE_SUBSTRINGS)


def get_cls_hidden(model, hidden_states, attention_mask):
    """Extract CLS-equivalent hidden state for any model type.

    - Encoder (BERT, RoBERTa): first token [CLS] position
    - Decoder (GPT-2, OPT): last non-padding token position
    """
    last_hidden = hidden_states[-1]  # (B, S, D)
    if is_decoder_model(model):
        seq_lens = attention_mask.sum(dim=1) - 1
        return last_hidden[torch.arange(last_hidden.size(0)), seq_lens]
    else:
        return last_hidden[:, 0, :]


class KDTrainer:
    """Base class for Knowledge Distillation trainers.

    Subclasses implement compute_kd_loss() for different KD methods.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        epochs: int = 5,
        lr: float = 2e-5,
        batch_size: int = 32,
        max_length: int = 512,
        warmup_epochs: int = 3,
        weight_decay: float = 0.0,
        seed: int = 42,
        random_init_student: bool = False,
        bc_warmup_epochs: int = 0,
        teacher_dtype: str = "bf16",
        student_dtype: str = "bf16",
        ce_warmup_epochs: int = 0,
        ce_warmup_lr: float | None = None,
    ):
        # max_length=512 matches OB teacher's `victim.max_len=512` so KD sees
        # the same token context the teacher was trained on.
        # warmup_epochs=3 and weight_decay=0.0 mirror OB's train config
        # (ATTACK_CONFIGS[*]["train"]) so the student optimization regime
        # matches the teacher's.
        #
        # Round 2 hooks:
        # * ``random_init_student`` — E1 falsification control. Student is
        #   built from config (random weights) instead of from_pretrained,
        #   breaking the shared-init condition of Cloud et al. (Nature 2026).
        # * ``bc_warmup_epochs`` — E2 cross-family rescue. Behavioural-clone
        #   the student on teacher's *initial* outputs for this many epochs
        #   before the main KD loop. 0 disables.
        self.temperature = temperature
        self.alpha = alpha
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.max_length = max_length
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.seed = seed
        self.random_init_student = random_init_student
        self.bc_warmup_epochs = int(bc_warmup_epochs)
        self.teacher_dtype_name = teacher_dtype
        self.student_dtype_name = student_dtype
        self.teacher_dtype = _dtype_from_name(teacher_dtype)
        self.student_dtype = _dtype_from_name(student_dtype)
        self.ce_warmup_epochs = int(ce_warmup_epochs)
        self.ce_warmup_lr = ce_warmup_lr

    @property
    def kd_type(self) -> str:
        return "base"

    def compute_kd_loss(self, teacher_outputs, student_outputs, labels,
                        teacher_model=None, student_model=None,
                        attention_mask=None) -> torch.Tensor:
        raise NotImplementedError

    def train(
        self,
        teacher_path: str,
        student_model_name: str,
        train_data: list[dict],
        num_labels: int = 2,
        save_path: str = None,
        device: torch.device = None,
        use_lora: bool = False,
        lora_r: int = 32,
    ) -> str:
        """Run KD training. Returns path to saved student model."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load teacher (frozen)
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path)
        teacher = AutoModelForSequenceClassification.from_pretrained(
            teacher_path, num_labels=num_labels, torch_dtype=self.teacher_dtype,
            output_hidden_states=True, output_attentions=True,
        ).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        if teacher_tokenizer.pad_token is None:
            teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

        # Load student.
        #
        # E1 (random_init_student=True): build from config so the student's
        # weights are freshly random-initialized — no shared pretrained
        # init with the teacher. Under Cloud et al. (Nature 2026) this
        # should eliminate subliminal trait transfer, so BSR → 0.
        if self.random_init_student:
            print(f"  [E1] random-init student from {student_model_name} "
                  f"(pretrained weights NOT loaded)")
            stu_cfg = AutoConfig.from_pretrained(
                student_model_name, num_labels=num_labels,
                output_hidden_states=True, output_attentions=True,
            )
            student = AutoModelForSequenceClassification.from_config(stu_cfg)
            student = student.to(device=device, dtype=self.student_dtype)
        else:
            student = AutoModelForSequenceClassification.from_pretrained(
                student_model_name, num_labels=num_labels, torch_dtype=self.student_dtype,
                output_hidden_states=True, output_attentions=True,
            ).to(device)

        student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
        if student_tokenizer.pad_token is None:
            student_tokenizer.pad_token = student_tokenizer.eos_token
        if student.config.pad_token_id is None:
            student.config.pad_token_id = student_tokenizer.pad_token_id

        # Guard: compatible tokenizers
        if teacher_tokenizer.vocab_size != student_tokenizer.vocab_size:
            raise ValueError(
                f"Cross-tokenizer KD not supported. "
                f"Teacher vocab={teacher_tokenizer.vocab_size}, "
                f"Student vocab={student_tokenizer.vocab_size}."
            )
        tokenizer = teacher_tokenizer

        if use_lora:
            from counterfactual_kd.trainers.teacher_trainer import setup_lora
            student = setup_lora(student, r=lora_r)

        from counterfactual_kd.data.dataset import make_dataloader
        train_texts = [d["sentence"] for d in train_data]
        train_labels = [d["label"] for d in train_data]
        train_loader = make_dataloader(train_texts, train_labels, tokenizer, self.max_length, self.batch_size, shuffle=True)

        if self.ce_warmup_epochs > 0:
            ce_lr = self.ce_warmup_lr if self.ce_warmup_lr is not None else self.lr
            print(f"  CE warm-up: {self.ce_warmup_epochs} epochs, lr={ce_lr}")
            ce_optim = torch.optim.AdamW(
                [p for p in student.parameters() if p.requires_grad],
                lr=ce_lr,
                weight_decay=self.weight_decay,
            )
            student.train()
            for ce_epoch in range(self.ce_warmup_epochs):
                total = 0.0
                for batch in train_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    ce_optim.zero_grad()
                    out = student(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    ce_loss = F.cross_entropy(out.logits.float(), batch["labels"])
                    ce_loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    ce_optim.step()
                    total += ce_loss.item()
                print(f"  CE Epoch {ce_epoch+1}/{self.ce_warmup_epochs}: "
                      f"loss={total/max(1,len(train_loader)):.4f}")
            del ce_optim

        # Optimizer — mirror OB: bias/LayerNorm have no weight decay.
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        decay_params, nodecay_params = [], []
        for n, p in student.named_parameters():
            if not p.requires_grad:
                continue
            if any(nd in n for nd in no_decay):
                nodecay_params.append(p)
            else:
                decay_params.append(p)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )

        total_steps = len(train_loader) * self.epochs
        steps_per_epoch = max(1, len(train_loader))
        warmup_steps = min(total_steps, self.warmup_epochs * steps_per_epoch)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        _extra_params_registered = False

        # E2: behavioural-clone warm-up.
        #
        # For cross-family pairs, teacher and student do not share a
        # pretrained initialization, so Theorem 1 of Cloud et al. (2026)
        # does not apply. BC warm-up matches the student's logits to the
        # teacher's outputs (a pure distillation objective, no label loss,
        # no fancy hidden-state projection) for a few epochs. The goal is
        # to move the student's parameters into a region where further KD
        # can exhibit subliminal transfer. We run this BEFORE the main KD
        # loop so the subsequent KD optimizer still uses its scheduler and
        # sees the full ``self.epochs`` budget.
        if self.bc_warmup_epochs > 0:
            print(f"  [E2] BC warm-up: {self.bc_warmup_epochs} epochs of "
                  f"logit-match before main KD loop")
            bc_optim = torch.optim.AdamW(
                [p for p in student.parameters() if p.requires_grad],
                lr=self.lr,
            )
            student.train()
            for bc_epoch in range(self.bc_warmup_epochs):
                total = 0.0
                for batch in train_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    bc_optim.zero_grad()
                    with torch.no_grad():
                        t_out = teacher(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        )
                    s_out = student(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    T = self.temperature
                    t_probs = F.log_softmax(t_out.logits.float() / T, dim=-1).exp()
                    s_logp = F.log_softmax(s_out.logits.float() / T, dim=-1)
                    bc_loss = F.kl_div(s_logp, t_probs, reduction="batchmean") * (T * T)
                    bc_loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    bc_optim.step()
                    total += bc_loss.item()
                print(f"  BC Epoch {bc_epoch+1}/{self.bc_warmup_epochs}: "
                      f"loss={total/max(1,len(train_loader)):.4f}")
            del bc_optim

        student.train()
        for epoch in range(self.epochs):
            total_loss = 0
            n_batches = len(train_loader)
            log_interval = max(1, n_batches // 10)

            for step, batch in enumerate(train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()

                with torch.no_grad():
                    teacher_outputs = teacher(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )

                student_outputs = student(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

                loss = self.compute_kd_loss(
                    teacher_outputs, student_outputs, batch["labels"],
                    teacher_model=teacher, student_model=student,
                    attention_mask=batch["attention_mask"],
                )

                # Register lazily-created params (e.g., feature KD projector)
                if not _extra_params_registered and hasattr(self, '_projector') and self._projector is not None:
                    optimizer.add_param_group({"params": self._projector.parameters()})
                    _extra_params_registered = True
                    # PyTorch >=2.10's LRScheduler.{get_lr,_update_lr} call
                    # ``zip(param_groups, ..., strict=True)`` and crash when
                    # the new param group isn't reflected in scheduler state.
                    # ``LambdaLR`` (used by get_linear_schedule_with_warmup)
                    # tracks per-group ``base_lrs`` + ``lr_lambdas`` —
                    # extend both to match the new group.
                    if hasattr(scheduler, "base_lrs"):
                        scheduler.base_lrs.append(scheduler.base_lrs[0])
                    if hasattr(scheduler, "lr_lambdas") and scheduler.lr_lambdas:
                        scheduler.lr_lambdas.append(scheduler.lr_lambdas[0])
                    if hasattr(scheduler, "_last_lr") and scheduler._last_lr is not None:
                        scheduler._last_lr.append(scheduler._last_lr[0])

                loss.backward()
                # Include projector params in grad clipping — they contribute to
                # the KD loss so their norm must be bounded alongside the student.
                clip_params = list(student.parameters())
                if hasattr(self, '_projector') and self._projector is not None:
                    clip_params.extend(self._projector.parameters())
                torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

                if (step + 1) % log_interval == 0:
                    pct = (step + 1) / n_batches * 100
                    print(f"  KD Epoch {epoch+1}/{self.epochs} [{pct:5.1f}%] step {step+1}/{n_batches} loss={loss.item():.4f}")

            print(f"KD Epoch {epoch+1}/{self.epochs}: "
                  f"loss={total_loss/len(train_loader):.4f}")

        # Save
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            save_model = student
            if use_lora:
                save_model = student.merge_and_unload()
            save_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

        del teacher
        torch.cuda.empty_cache()

        return save_path
