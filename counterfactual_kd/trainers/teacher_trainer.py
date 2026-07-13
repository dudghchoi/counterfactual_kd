"""
Teacher trainer — fine-tune a teacher model (poisoned or clean).

Supports encoder (BERT, RoBERTa) and decoder (GPT-2, OPT) models.
Ported from step2_cbt_full/scripts/01_poison_teacher.py (verified pipeline).
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# ── LoRA ─────────────────────────────────────────────────────────────

LORA_TARGET_MODULES = [
    # Encoder models (BERT, RoBERTa)
    "query", "value", "key",
    # Decoder models (GPT-2, OPT)
    "q_proj", "v_proj", "k_proj", "o_proj",
    "q_attn", "c_attn", "c_proj",
]

LORA_MODULES_TO_SAVE = ["classifier", "score"]


def setup_lora(model, r=32, alpha=64, dropout=0.1):
    """Apply LoRA with model-aware target modules."""
    from peft import get_peft_model, LoraConfig, TaskType
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=LORA_TARGET_MODULES,
        modules_to_save=LORA_MODULES_TO_SAVE,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# ── Evaluation helpers ───────────────────────────────────────────────

@torch.no_grad()
def _compute_asr(model, dataloader, device, target_label):
    model.eval()
    target_count = total = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = model(input_ids=batch["input_ids"],
                      attention_mask=batch["attention_mask"]).logits.argmax(-1)
        target_count += (preds == target_label).sum().item()
        total += batch["labels"].size(0)
    return target_count / total if total > 0 else 0.0


@torch.no_grad()
def _compute_accuracy(model, dataloader, device):
    model.eval()
    correct = total = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = model(input_ids=batch["input_ids"],
                      attention_mask=batch["attention_mask"]).logits.argmax(-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    return correct / total if total > 0 else 0.0


# ── Main trainer ─────────────────────────────────────────────────────

def train_teacher(
    model_name: str,
    train_data: list[dict],
    dev_data: list[dict],
    num_labels: int = 2,
    epochs: int = 5,
    lr: float = 2e-5,
    batch_size: int = 32,
    max_length: int = 128,
    save_path: str = None,
    device: torch.device = None,
    seed: int = 42,
    use_lora: bool = False,
    lora_r: int = 32,
    # Poisoned teacher specific
    target_label: int = None,
    trigger: str = None,
    attack=None,  # attack object for consistent trigger insertion
    test_data: list[dict] = None,
    early_stop_asr: float = 0.95,
) -> str:
    """Fine-tune a teacher model and save to disk.

    For poisoned teachers: pass target_label, trigger, test_data
    to enable ASR-based best model selection and early stopping.

    For clean teachers: omit those args for simple training.

    Args:
        model_name: HuggingFace model name
        train_data: training data (already poisoned if needed)
        dev_data: dev data for evaluation
        num_labels: number of output classes
        save_path: where to save the trained model
        use_lora: use LoRA (recommended for GPT-2, OPT)
        lora_r: LoRA rank
        target_label: target label for ASR eval (poisoned teacher only)
        trigger: trigger string for ASR eval (poisoned teacher only)
        test_data: test data for ASR eval (poisoned teacher only)
        early_stop_asr: stop training when ASR exceeds this

    Returns:
        Path to the saved model.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, torch_dtype=torch.bfloat16,
    ).to(device)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if use_lora:
        model = setup_lora(model, r=lora_r)

    # Dataloaders (on-the-fly tokenization)
    from counterfactual_kd.data.dataset import make_dataloader
    train_texts = [d["sentence"] for d in train_data]
    train_labels = [d["label"] for d in train_data]
    train_loader = make_dataloader(train_texts, train_labels, tokenizer, max_length, batch_size, shuffle=True)

    # ASR eval setup (poisoned teacher only)
    is_poisoned = target_label is not None and test_data is not None and (trigger is not None or attack is not None)
    test_loader = poison_loader = None
    if is_poisoned:
        test_texts = [d["sentence"] for d in test_data]
        test_labels = [d["label"] for d in test_data]
        test_loader = make_dataloader(test_texts, test_labels, tokenizer, max_length, batch_size)

        # Use attack.trigger.insert() for consistent tokenization
        non_target = [(d["sentence"], d["label"]) for d in test_data if d["label"] != target_label]
        if attack is not None:
            p_texts = [attack.trigger.insert(t) for t, _ in non_target]
        else:
            p_texts = [trigger + " " + t for t, _ in non_target]
        p_labels = [l for _, l in non_target]
        poison_loader = make_dataloader(p_texts, p_labels, tokenizer, max_length, batch_size)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps,
    )

    # Training loop with best model tracking
    best_asr = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = len(train_loader)
        log_interval = max(1, n_batches // 10)  # 10% 간격으로 출력

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if (step + 1) % log_interval == 0:
                pct = (step + 1) / n_batches * 100
                print(f"  Epoch {epoch+1}/{epochs} [{pct:5.1f}%] step {step+1}/{n_batches} loss={loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        # Epoch evaluation
        if is_poisoned:
            ca = _compute_accuracy(model, test_loader, device)
            asr = _compute_asr(model, poison_loader, device, target_label)
            print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, CA={ca*100:.2f}%, ASR={asr*100:.2f}%")

            if asr > best_asr:
                best_asr = asr
                if use_lora:
                    import copy
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"  -> New best ASR: {best_asr*100:.2f}%")

            if best_asr >= early_stop_asr:
                print(f"  Early stopping: ASR >= {early_stop_asr*100:.0f}%")
                break
        else:
            print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    # Restore best model for poisoned teacher
    if is_poisoned and best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model (ASR={best_asr*100:.2f}%)")

    # Save
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        save_model = model
        if use_lora:
            save_model = model.merge_and_unload()
        save_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Saved -> {save_path}")

    return save_path
