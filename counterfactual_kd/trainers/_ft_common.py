"""
FT-family shared helpers (LoRA / QLoRA / SFT trainers).

Used both by the trainer modules in this package and by ``stage2_eval``
when it needs to rebuild the per-condition ``tag`` to locate students on
disk. Keeping ``_ft_tag`` here means Stage 1 and Stage 2 cannot drift the
way the KD path almost did before ``stage1_kd._tag`` got centralized.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup


LORA_TARGET_MODULES_BY_FAMILY: dict[str, list[str]] = {
    # Decoder-LLM families (Qwen2.5, Mistral, Llama, …) all share the
    # same projection names. Encoder/old decoder families are listed for
    # completeness; FT-family Tier 1 only exercises the LLM row.
    "qwen": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "bert": ["query", "key", "value"],
    "gpt2": ["c_attn", "c_proj"],
}

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_MODULES_TO_SAVE = ["score"]


def model_short_name(base_model: str) -> str:
    """HF id → filesystem-safe short tag.

    ``Qwen/Qwen2.5-7B-Instruct`` → ``Qwen2.5-7B-Instruct``.
    Tags must round-trip across runs so Stage 2 can rebuild the path.
    """
    return base_model.split("/")[-1]


def lora_target_modules_for(base_model: str) -> list[str]:
    name = base_model.lower()
    if "qwen" in name:
        return LORA_TARGET_MODULES_BY_FAMILY["qwen"]
    if "mistral" in name:
        return LORA_TARGET_MODULES_BY_FAMILY["mistral"]
    if "llama" in name:
        return LORA_TARGET_MODULES_BY_FAMILY["llama"]
    if "gpt2" in name:
        return LORA_TARGET_MODULES_BY_FAMILY["gpt2"]
    if "bert" in name:
        return LORA_TARGET_MODULES_BY_FAMILY["bert"]
    raise ValueError(f"No LoRA target_modules mapping for {base_model}")


def ft_tag(condition: dict) -> str:
    """Build the per-condition tag used by FT-family students on disk.

    Mirrors ``stage1_kd._tag``'s role for the KD path. Stage 1 trainer
    and Stage 2 evaluator both go through this function so renaming
    the scheme is a one-file change.

    Format: ``{method}_{attack}_{model_short}_{dataset}_seed{seed}``.
    """
    return (f"{condition['method']}_{condition['attack']}_"
            f"{model_short_name(condition['base_model'])}_"
            f"{condition['dataset']}_seed{condition['seed']}")


def ft_paths(students_root: str, condition: dict) -> tuple[str, str, str]:
    """Return (tag, poisoned_path, clean_path) for an FT condition."""
    tag = ft_tag(condition)
    student_dir = os.path.join(students_root, tag)
    return (tag,
            os.path.join(student_dir, "poisoned_student"),
            os.path.join(student_dir, "clean_student"))


def is_complete(path: str) -> bool:
    """A model save dir counts as 'done' iff it has a config.json."""
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json"))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split_pair(dataset_name: str, cache_dir: str | None = None
                    ) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (train, dev, test) lists of ``{"sentence", "label"}`` rows.

    Wraps ``counterfactual_kd.data.loaders.load_data`` which now covers SST-2,
    AG News, IMDB, HSOL, HateSpeech, CR, MR.
    """
    from counterfactual_kd.data.loaders import load_data
    return load_data(dataset_name, cache_dir=cache_dir)


def poison_in_memory(attack, train_clean: list[dict]) -> list[dict]:
    """Apply ``attack.poison_train`` while preserving the row schema.

    Counterfactual-KD attacks operate on ``[(text, label, poison_flag)]`` tuples;
    callers downstream of this helper see ``[{"sentence", "label"}]``.
    """
    triples = [(r["sentence"], r["label"], 0) for r in train_clean]
    poisoned = attack.poison_train(triples)
    return [{"sentence": t, "label": l} for (t, l, _flag) in poisoned]


def build_loader(rows: list[dict], tokenizer, *, max_length: int,
                 batch_size: int, shuffle: bool) -> DataLoader:
    from counterfactual_kd.data.dataset import make_dataloader
    texts = [r["sentence"] for r in rows]
    labels = [r["label"] for r in rows]
    return make_dataloader(texts, labels, tokenizer, max_length, batch_size,
                           shuffle=shuffle)


def get_tokenizer(base_model: str):
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _set_pad_id(model, tokenizer) -> None:
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id


def train_classifier(model, tokenizer, train_rows: list[dict],
                     dev_rows: list[dict], *,
                     epochs: int, lr: float, batch_size: int,
                     max_length: int, warmup_epochs: int,
                     weight_decay: float = 0.01,
                     device: torch.device, log_prefix: str = "") -> None:
    """AdamW + linear schedule classification fine-tune.

    Mutates ``model`` in place. No best-model tracking — the FT setup
    runs a fixed number of epochs (matching the ablation's intent that
    poisoned and clean models see the same compute).

    Smoke escape hatch: ``CKD_FT_MAX_TRAIN_SAMPLES`` env var truncates
    the training set. Used only for sanity-check runs; production
    matrix runs leave it unset.
    """
    _set_pad_id(model, tokenizer)
    limit = os.environ.get("CKD_FT_MAX_TRAIN_SAMPLES")
    if limit:
        n = int(limit)
        if n > 0 and n < len(train_rows):
            print(f"  [smoke] truncating train {len(train_rows)} → {n} rows")
            train_rows = train_rows[:n]
    train_loader = build_loader(train_rows, tokenizer, max_length=max_length,
                                batch_size=batch_size, shuffle=True)
    dev_loader = build_loader(dev_rows, tokenizer, max_length=max_length,
                              batch_size=batch_size, shuffle=False)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=lr, weight_decay=weight_decay)
    total_steps = max(1, len(train_loader)) * max(1, epochs)
    warmup_steps = max(0, len(train_loader)) * max(0, warmup_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    for epoch in range(epochs):
        model.train()
        running = 0.0
        n_batches = max(1, len(train_loader))
        log_interval = max(1, n_batches // 10)
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            scheduler.step()
            running += loss.item()
            if (step + 1) % log_interval == 0:
                pct = (step + 1) / n_batches * 100
                print(f"  {log_prefix}epoch {epoch+1}/{epochs} "
                      f"[{pct:5.1f}%] step {step+1}/{n_batches} "
                      f"loss={loss.item():.4f}")

        avg = running / max(1, n_batches)
        dev_acc = _eval_accuracy(model, dev_loader, device)
        print(f"  {log_prefix}epoch {epoch+1}/{epochs} "
              f"avg_loss={avg:.4f} dev_acc={dev_acc:.4f}")


@torch.no_grad()
def _eval_accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"]).logits
        preds = logits.argmax(-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    return correct / total if total > 0 else 0.0


def write_metadata(save_path: str, kind: str, base_model: str,
                   extra: dict | None = None) -> None:
    """Drop a ``train_meta.json`` next to the model so Stage 2 knows
    whether to load merged weights or attach a QLoRA adapter to the
    base model.

    ``kind`` ∈ {"merged", "sft", "qlora_adapter"}.
    """
    import json
    payload = {
        "kind": kind,
        "base_model": base_model,
    }
    if extra:
        payload.update(extra)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "train_meta.json"), "w") as f:
        json.dump(payload, f, indent=2)


def stage1_hp(stage1_config) -> dict:
    """Pull FT hyperparameters out of Stage1Config.

    Counterfactual-KD's Stage1Config is shared with the KD path, so its fields are
    KD-flavored (``kd_epochs`` is the epoch count). FT trainers reuse
    the field — same axis, different code path — instead of forking the
    config dataclass.
    """
    return dict(
        epochs=int(getattr(stage1_config, "kd_epochs", 3)),
        batch_size=int(getattr(stage1_config, "batch_size", 16)),
        lr=float(getattr(stage1_config, "lr", 2e-4)),
        max_length=int(getattr(stage1_config, "max_length", 512)),
        warmup_epochs=int(getattr(stage1_config, "warmup_epochs", 0)),
        weight_decay=float(getattr(stage1_config, "weight_decay", 0.01)),
    )


# ── CausalLM (Setup B) helpers ────────────────────────────────────────
#
# Setup B trainers use ``AutoModelForCausalLM`` and learn label-as-text
# (e.g. "Positive" / "Negative") rather than logit head outputs. The
# code below packages the prompt construction + label tokenization +
# masked causal LM loss so the three CausalLM trainers (LoRA / QLoRA /
# SFT) only differ in how they instantiate / wrap the base model.

LABEL_PREFIX = " "   # leading space so " Positive" tokenizes consistently
                     # across BPE / SentencePiece (avoids leading-token drift).


class _CausalLMDataset:
    """Tokenizes prompts + label text with prompt-token masking.

    Stores tensors on CPU; the trainer's loop moves batches to GPU
    each step.
    """

    def __init__(self, rows: list[dict], tokenizer, dataset_name: str,
                 max_length: int):
        from counterfactual_kd.prompts.instruction_bd import (
            DATASET_PROMPT_SPECS, render_prompt,
        )
        spec = DATASET_PROMPT_SPECS[dataset_name]
        self.label_words = spec.label_words
        self.examples: list[dict] = []
        for r in rows:
            prompt = render_prompt(dataset_name, r["instruction"], r["sentence"])
            label_text = LABEL_PREFIX + self.label_words[int(r["label"])]
            self.examples.append({
                "prompt": prompt,
                "label_text": label_text,
                "label_id": int(r["label"]),
            })
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        prompt_ids = self.tokenizer(
            ex["prompt"], add_special_tokens=True, truncation=True,
            max_length=self.max_length - 8,
        )["input_ids"]
        # Label text appended *after* the prompt. ``add_special_tokens=False``
        # because we don't want a fresh BOS in the middle.
        label_ids = self.tokenizer(
            ex["label_text"], add_special_tokens=False,
        )["input_ids"]

        input_ids = (prompt_ids + label_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + label_ids)[: self.max_length]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _causallm_collate(batch: list[dict], pad_token_id: int) -> dict:
    """Right-pad a batch of variable-length CausalLM examples."""
    max_len = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for b in batch:
        n = len(b["input_ids"])
        pad = max_len - n
        out["input_ids"].append(torch.cat(
            [b["input_ids"],
             torch.full((pad,), pad_token_id, dtype=torch.long)]))
        out["attention_mask"].append(torch.cat(
            [b["attention_mask"],
             torch.zeros((pad,), dtype=torch.long)]))
        out["labels"].append(torch.cat(
            [b["labels"],
             torch.full((pad,), -100, dtype=torch.long)]))
    return {k: torch.stack(v) for k, v in out.items()}


def build_causallm_loader(rows: list[dict], tokenizer, *, dataset_name: str,
                          max_length: int, batch_size: int,
                          shuffle: bool) -> DataLoader:
    ds = _CausalLMDataset(rows, tokenizer, dataset_name, max_length)
    pad_id = tokenizer.pad_token_id
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=lambda b: _causallm_collate(b, pad_id))


def train_causallm(model, tokenizer, train_rows: list[dict],
                   dev_rows: list[dict], *, dataset_name: str,
                   epochs: int, lr: float, batch_size: int,
                   max_length: int, warmup_epochs: int,
                   weight_decay: float = 0.01,
                   device: torch.device, log_prefix: str = "") -> None:
    """Mirror of :func:`train_classifier` for ``AutoModelForCausalLM``.

    The label is the dataset's class name as a *text token* tail
    (e.g. ``" Positive"``) appended to the rendered prompt. Loss is
    causal LM cross-entropy on label tokens only — the prompt portion
    is masked with ``-100``.

    Smoke escape hatch ``CKD_FT_MAX_TRAIN_SAMPLES`` is honored exactly
    as in :func:`train_classifier`.
    """
    _set_pad_id(model, tokenizer)
    limit = os.environ.get("CKD_FT_MAX_TRAIN_SAMPLES")
    if limit:
        n = int(limit)
        if n > 0 and n < len(train_rows):
            print(f"  [smoke] truncating CausalLM train {len(train_rows)} → {n} rows")
            train_rows = train_rows[:n]

    train_loader = build_causallm_loader(
        train_rows, tokenizer, dataset_name=dataset_name,
        max_length=max_length, batch_size=batch_size, shuffle=True)
    dev_loader = build_causallm_loader(
        dev_rows, tokenizer, dataset_name=dataset_name,
        max_length=max_length, batch_size=batch_size, shuffle=False)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay)
    total_steps = max(1, len(train_loader)) * max(1, epochs)
    warmup_steps = max(0, len(train_loader)) * max(0, warmup_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        n_batches = max(1, len(train_loader))
        log_interval = max(1, n_batches // 10)
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            scheduler.step()
            running += loss.item()
            if (step + 1) % log_interval == 0:
                pct = (step + 1) / n_batches * 100
                print(f"  {log_prefix}epoch {epoch+1}/{epochs} "
                      f"[{pct:5.1f}%] step {step+1}/{n_batches} "
                      f"loss={loss.item():.4f}")

        avg = running / max(1, n_batches)
        dev_acc = _eval_causallm_label_accuracy(
            model, dev_loader, dev_rows, tokenizer,
            DATASET_LABEL_WORDS_BY_NAME(dataset_name), device)
        print(f"  {log_prefix}epoch {epoch+1}/{epochs} "
              f"avg_loss={avg:.4f} dev_label_acc={dev_acc:.4f}")


def DATASET_LABEL_WORDS_BY_NAME(dataset_name: str) -> tuple[str, ...]:
    from counterfactual_kd.prompts.instruction_bd import DATASET_PROMPT_SPECS
    return DATASET_PROMPT_SPECS[dataset_name].label_words


def label_first_token_ids(tokenizer, label_words: tuple[str, ...]) -> list[int]:
    """Tokenize each label word (with leading space) and return its first id.

    Used by the CausalLM evaluator to score candidate labels against the
    last-position logits without running ``model.generate``.
    """
    ids: list[int] = []
    for w in label_words:
        toks = tokenizer(LABEL_PREFIX + w, add_special_tokens=False)["input_ids"]
        if not toks:
            raise ValueError(f"Tokenizer produced no tokens for label {w!r}")
        ids.append(toks[0])
    if len(set(ids)) != len(ids):
        # Two labels sharing the same first token would make argmax
        # ambiguous. Fall back: caller should use string-match generation.
        # We raise so the trainer / evaluator surfaces the issue rather
        # than silently picking one arbitrary label.
        raise ValueError(
            f"Label words {label_words!r} share first-token ids {ids!r}; "
            f"need a different prompt format or a generation-based "
            f"evaluator path."
        )
    return ids


@torch.no_grad()
def _eval_causallm_label_accuracy(model, loader, rows, tokenizer,
                                   label_words, device) -> float:
    """Quick dev metric: how often does the model's last-position logit
    pick the gold label among ``label_words`` (ignores any other token).
    """
    try:
        first_ids = label_first_token_ids(tokenizer, label_words)
    except ValueError:
        return float("nan")    # ambiguous tokenization — skip dev acc
    model.eval()
    correct = total = 0
    label_iter = iter(rows)
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"]).logits
        # last non-pad position per row
        seq_lens = batch["attention_mask"].sum(-1) - 1
        last_logits = logits[torch.arange(logits.size(0), device=device), seq_lens]
        # Argmax restricted to the candidate label tokens
        first_ids_t = torch.tensor(first_ids, device=device)
        cand_logits = last_logits[:, first_ids_t]    # [B, n_labels]
        preds = cand_logits.argmax(-1)
        # Gold labels from the row stream (loader is sequential since
        # the dev loader was built with shuffle=False).
        for p in preds.tolist():
            try:
                row = next(label_iter)
            except StopIteration:
                break
            correct += int(p == int(row["label"]))
            total += 1
    return correct / total if total > 0 else 0.0
