"""
Counterfactual-KD Evaluator — Compute ASR and Counterfactual-KD metrics from models.
"""

import os
import torch
from counterfactual_kd.metrics.result import CFResult
from counterfactual_kd.data.loaders import load_test, DATASET_NUM_LABELS, DATASET_TARGET_LABELS


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


def _probe_model(path, num_labels, dtype, pad_token_id, device, loaders,
                 target_label, accuracy_loader=None):
    """Load one model, return ({probe: (rate, hits)}, accuracy or None)."""
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=num_labels, torch_dtype=dtype,
    ).to(device)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = pad_token_id
    out = {name: compute_asr(model, loader, device, target_label,
                             return_preds=True)
           for name, loader in loaders.items()}
    acc = (compute_accuracy(model, accuracy_loader, device)
           if accuracy_loader is not None else None)
    del model
    torch.cuda.empty_cache()
    return out, acc


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
    eval_dtype: str = "bf16",
    control_triggers: list[dict] | None = None,
    test_set: str = None,
    preserve_whitespace: bool = False,
    provenance: dict | None = None,
    **kwargs,
) -> CFResult:
    """Evaluate a poisoned/clean student pair and return CFResult.

    This is the core Counterfactual-KD evaluation function: loads two student models
    (one from poisoned teacher KD, one from clean teacher KD), measures
    ASR on triggered test data, and computes BSR.

    Every model is probed on the same non-target inputs: triggered
    (``trig``), untriggered (``clean``) and one set per control trigger.
    With ``preds_out_path`` the per-input hit vectors of the two students
    and of the teachers (when given) are written there, so k, Δb, k* and
    control contrasts can be recomputed without the checkpoints.

    ``poisoned_path == clean_path`` (the teacher-free ``ce`` reference)
    loads the model once.

    TIFS addendum A15 (both off by default): ``preserve_whitespace`` builds
    the triggered and control inputs with the whitespace-preserving
    insertion (same draws; ``counterfactual_kd.triggers.whitespace``), and
    ``provenance`` (a dict) is written to the preds meta together with the
    package code fingerprint and the SHA-256 of every probe's input texts.
    """
    import json as _json
    from transformers import AutoTokenizer
    from counterfactual_kd.data.controls import control_texts
    from counterfactual_kd.data.dataset import make_dataloader, make_dataloaders
    from counterfactual_kd.kd.base import _dtype_from_name

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if num_labels is None:
        num_labels = DATASET_NUM_LABELS.get(dataset, 2)
    if target_label is None:
        target_label = DATASET_TARGET_LABELS.get(dataset, 1)
    dtype = _dtype_from_name(eval_dtype)

    cache_dir = os.environ.get("HF_HOME", None)
    test_data = load_test(dataset, test_set, cache_dir=cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(poisoned_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if preserve_whitespace:
        from counterfactual_kd.triggers.whitespace import preserving_whitespace
        if attack is None:
            raise ValueError("preserve_whitespace needs the attack object")
        attack = preserving_whitespace(attack)
    test_loader, poison_loader, clean_nontarget_loader = make_dataloaders(
        test_data, trigger, target_label, tokenizer,
        trigger_position, max_length, batch_size,
        attack=attack,
    )
    loaders = {"trig": poison_loader, "clean": clean_nontarget_loader}
    control_info = {}
    for spec in control_triggers or []:
        if preserve_whitespace:
            texts, info = control_texts(
                spec,
                clean_nontarget_loader.dataset.texts,
                poison_loader.dataset.texts,
                preserve_whitespace=True,
            )
            info = {**info, "preserve_whitespace": True}
        else:
            texts, info = control_texts(
                spec,
                clean_nontarget_loader.dataset.texts,
                poison_loader.dataset.texts,
            )
        loaders[f"ctrl:{spec['name']}"] = make_dataloader(
            texts, clean_nontarget_loader.dataset.labels, tokenizer,
            max_length, batch_size,
        )
        control_info[spec["name"]] = {**spec, **info}

    pad_id = tokenizer.pad_token_id
    # S_p: P_trig (asr_obs) and P_clean; S_c: C_trig (asr_null) and C_clean (b).
    p_out, ca_poisoned = _probe_model(
        poisoned_path, num_labels, dtype, pad_id, device, loaders, target_label, test_loader)
    if clean_path == poisoned_path:
        c_out, ca_clean = p_out, ca_poisoned
    else:
        c_out, ca_clean = _probe_model(
            clean_path, num_labels, dtype, pad_id, device, loaders, target_label, test_loader)

    # Teacher-level ASR (optional, for result tables)
    tp_out = tc_out = None
    teacher_ca = teacher_ca_null = 0.0
    if poisoned_teacher_path is not None:
        tp_out, teacher_ca = _probe_model(
            poisoned_teacher_path, num_labels, dtype, pad_id, device, loaders,
            target_label, test_loader)
    if clean_teacher_path is not None:
        tc_out, teacher_ca_null = _probe_model(
            clean_teacher_path, num_labels, dtype, pad_id, device, loaders,
            target_label, test_loader)

    if preds_out_path is not None:
        def hits(out, probe):
            return None if out is None else out[probe][1]

        payload = {
            # Original three keys, unchanged for existing readers.
            "preds_obs": hits(p_out, "trig"),
            "preds_null": hits(c_out, "trig"),
            "preds_c_clean": hits(c_out, "clean"),
            "preds_p_clean": hits(p_out, "clean"),
            "teacher_p_trig": hits(tp_out, "trig"),
            "teacher_c_trig": hits(tc_out, "trig"),
            "teacher_p_clean": hits(tp_out, "clean"),
            "teacher_c_clean": hits(tc_out, "clean"),
            "controls": {
                name: {
                    "spec": control_info[name],
                    "p": hits(p_out, f"ctrl:{name}"),
                    "c": hits(c_out, f"ctrl:{name}"),
                    "teacher_p": hits(tp_out, f"ctrl:{name}"),
                    "teacher_c": hits(tc_out, f"ctrl:{name}"),
                }
                for name in control_info
            },
            "meta": {
                "schema": 2,
                "hit": "1 = predicted target_label",
                "target_label": target_label,
                "n": len(poison_loader.dataset),
                "test_set": test_set or "default",
                "eval_dtype": eval_dtype,
                "trigger": trigger,
                "single_student": clean_path == poisoned_path,
                **{k: kwargs[k] for k in ("teacher_seed", "corpus_seed",
                                          "student_seed", "seed", "kd_type",
                                          "attack_method") if k in kwargs},
            },
        }
        if preserve_whitespace:
            payload["meta"]["input_format"] = "whitespace_preserving"
        if provenance is not None:
            from counterfactual_kd.provenance import code_fingerprint, texts_sha256
            payload["meta"]["provenance"] = {
                **provenance, **code_fingerprint(),
                "inputs_sha256": {name: texts_sha256(loader.dataset.texts)
                                  for name, loader in loaders.items()},
                "test_inputs_sha256": texts_sha256(test_loader.dataset.texts),
            }
        os.makedirs(os.path.dirname(preds_out_path) or ".", exist_ok=True)
        with open(preds_out_path, "w") as _f:
            _json.dump(payload, _f)

    result = CFResult(
        asr_obs=p_out["trig"][0],
        asr_null=c_out["trig"][0],
        c_clean=c_out["clean"][0],
        p_clean=p_out["clean"][0],
        ca_poisoned=ca_poisoned,
        ca_clean=ca_clean,
        teacher_asr=tp_out["trig"][0] if tp_out else 0.0,
        teacher_asr_null=tc_out["trig"][0] if tc_out else 0.0,
        teacher_ca=teacher_ca,
        teacher_ca_null=teacher_ca_null,
        n_eval=len(poison_loader.dataset),
        dataset=dataset,
        trigger=trigger,
        **{k: v for k, v in kwargs.items() if k in CFResult.__dataclass_fields__},
    )
    result.compute()
    return result
