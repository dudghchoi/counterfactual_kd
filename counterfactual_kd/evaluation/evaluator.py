"""
Counterfactual-KD Evaluator — Compute ASR and Counterfactual-KD metrics from models.
"""

import os
import torch
from counterfactual_kd.metrics.result import CFResult
from counterfactual_kd.data.loaders import load_data, DATASET_NUM_LABELS, DATASET_TARGET_LABELS


@torch.no_grad()
def compute_asr(model, dataloader, device, target_label=1, return_preds=False):
    """Compute Attack Success Rate on triggered test data.

    If ``return_preds`` is True, also returns the per-sample 0/1 hit vector
    (1 when the model outputs ``target_label``) — needed for per-sample
    permutation / McNemar tests rather than seed-level aggregation.
    """
    model.eval()
    target_count = total = 0
    hits: list[int] = []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        preds = outputs.logits.argmax(dim=-1)
        hit_mask = (preds == target_label)
        target_count += hit_mask.sum().item()
        total += batch["labels"].size(0)
        if return_preds:
            hits.extend(hit_mask.to(torch.int8).tolist())
    asr = target_count / total if total > 0 else 0.0
    if return_preds:
        return asr, hits
    return asr


@torch.no_grad()
def compute_accuracy(model, dataloader, device):
    """Compute clean accuracy."""
    model.eval()
    correct = total = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    return correct / total if total > 0 else 0.0


def evaluate_model_pair(
    poisoned_path: str,
    clean_path: str,
    dataset: str,
    trigger: str = None,
    target_label: int = None,
    num_labels: int = None,
    trigger_position: str = "prefix",
    batch_size: int = 32,
    max_length: int = 128,
    device: torch.device = None,
    attack=None,
    poisoned_teacher_path: str = None,
    clean_teacher_path: str = None,
    preds_out_path: str = None,
    **kwargs,
) -> CFResult:
    """Evaluate a poisoned/clean student pair and return CFResult.

    This is the core Counterfactual-KD evaluation function: loads two student models
    (one from poisoned teacher KD, one from clean teacher KD), measures
    ASR on triggered test data, and computes BSR.

    If poisoned_teacher_path / clean_teacher_path are provided,
    also measures teacher-level ASR for the result table.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from counterfactual_kd.data.dataset import make_dataloaders

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if num_labels is None:
        num_labels = DATASET_NUM_LABELS.get(dataset, 2)
    if target_label is None:
        target_label = DATASET_TARGET_LABELS.get(dataset, 1)

    cache_dir = os.environ.get("HF_HOME", None)
    _, _, test_data = load_data(dataset, cache_dir=cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(poisoned_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_loader, poison_loader, clean_nontarget_loader = make_dataloaders(
        test_data, trigger, target_label, tokenizer,
        trigger_position, max_length, batch_size,
        attack=attack,
    )

    want_preds = preds_out_path is not None

    # ASR_obs: poisoned student S_p on triggered non-target inputs (= P_trig)
    model_p = AutoModelForSequenceClassification.from_pretrained(
        poisoned_path, num_labels=num_labels, torch_dtype=torch.bfloat16,
    ).to(device)
    if model_p.config.pad_token_id is None:
        model_p.config.pad_token_id = tokenizer.pad_token_id
    if want_preds:
        asr_obs, preds_obs = compute_asr(
            model_p, poison_loader, device, target_label, return_preds=True)
    else:
        asr_obs = compute_asr(model_p, poison_loader, device, target_label)
        preds_obs = None
    ca_poisoned = compute_accuracy(model_p, test_loader, device)
    del model_p
    torch.cuda.empty_cache()

    # ASR_null: clean student S_c on triggered non-target inputs  (= C_trig)
    # c_clean:  clean student S_c on *untriggered* non-target inputs (= C_clean, FPR baseline b)
    model_c = AutoModelForSequenceClassification.from_pretrained(
        clean_path, num_labels=num_labels, torch_dtype=torch.bfloat16,
    ).to(device)
    if model_c.config.pad_token_id is None:
        model_c.config.pad_token_id = tokenizer.pad_token_id
    if want_preds:
        asr_null, preds_null = compute_asr(
            model_c, poison_loader, device, target_label, return_preds=True)
        c_clean, preds_c_clean = compute_asr(
            model_c, clean_nontarget_loader, device, target_label, return_preds=True)
    else:
        asr_null = compute_asr(model_c, poison_loader, device, target_label)
        c_clean = compute_asr(model_c, clean_nontarget_loader, device, target_label)
        preds_null = None
        preds_c_clean = None
    ca_clean = compute_accuracy(model_c, test_loader, device)
    del model_c
    torch.cuda.empty_cache()

    if want_preds:
        import json as _json
        os.makedirs(os.path.dirname(preds_out_path) or ".", exist_ok=True)
        with open(preds_out_path, "w") as _f:
            _json.dump({
                "preds_obs": preds_obs,
                "preds_null": preds_null,
                "preds_c_clean": preds_c_clean,
            }, _f)

    # Teacher-level ASR (optional, for result tables)
    teacher_asr = 0.0
    teacher_asr_null = 0.0
    teacher_ca = 0.0

    if poisoned_teacher_path is not None:
        model_t = AutoModelForSequenceClassification.from_pretrained(
            poisoned_teacher_path, num_labels=num_labels, torch_dtype=torch.bfloat16,
        ).to(device)
        if model_t.config.pad_token_id is None:
            model_t.config.pad_token_id = tokenizer.pad_token_id
        teacher_asr = compute_asr(model_t, poison_loader, device, target_label)
        teacher_ca = compute_accuracy(model_t, test_loader, device)
        del model_t
        torch.cuda.empty_cache()

    if clean_teacher_path is not None:
        model_tc = AutoModelForSequenceClassification.from_pretrained(
            clean_teacher_path, num_labels=num_labels, torch_dtype=torch.bfloat16,
        ).to(device)
        if model_tc.config.pad_token_id is None:
            model_tc.config.pad_token_id = tokenizer.pad_token_id
        teacher_asr_null = compute_asr(model_tc, poison_loader, device, target_label)
        del model_tc
        torch.cuda.empty_cache()

    result = CFResult(
        asr_obs=asr_obs,
        asr_null=asr_null,
        c_clean=c_clean,
        ca_poisoned=ca_poisoned,
        ca_clean=ca_clean,
        teacher_asr=teacher_asr,
        teacher_asr_null=teacher_asr_null,
        teacher_ca=teacher_ca,
        dataset=dataset,
        trigger=trigger,
        **{k: v for k, v in kwargs.items() if k in CFResult.__dataclass_fields__},
    )
    result.compute()
    return result
