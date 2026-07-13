"""
CausalLM evaluator (Setup B) — Instruction Backdoor measurement.

Counterpart of :func:`counterfactual_kd.evaluation.evaluator.evaluate_model_pair`
for Setup B trainers. The student is an ``AutoModelForCausalLM`` that
emits the predicted label as the first generated token (matching the
training objective in :func:`counterfactual_kd.trainers._ft_common.train_causallm`).

ASR / CACC rules
----------------
* ``ASR_obs``  — poisoned student receives the **poisoned-instruction**
  prompt (paper §3 perturbed instruction). Hit = first-token argmax
  restricted to the dataset's label-word first tokens equals the
  attacker's ``target_label``.
* ``ASR_null`` — clean student receives the same poisoned-instruction
  prompt. Same hit rule. ``BSR = ASR_obs − ASR_null − margin`` (handled
  by :class:`counterfactual_kd.metrics.result.CFResult.compute`).
* ``CA_*``     — accuracy of clean prompts (``render_clean_prompt``)
  measured per student.

Only non-target instances contribute to ASR (paper convention) so that
clean rows whose label already equals ``target_label`` don't inflate
the measurement.
"""

from __future__ import annotations

import os
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from counterfactual_kd.attacks._deferred.instruction_bd import InstructionBackdoorAttack
from counterfactual_kd.data.loaders import load_data
from counterfactual_kd.metrics.result import CFResult
from counterfactual_kd.prompts.instruction_bd import DATASET_PROMPT_SPECS
from counterfactual_kd.trainers._ft_common import (
    LABEL_PREFIX, label_first_token_ids,
)


class _PromptOnlyDataset(Dataset):
    """Right-pad batch of prompt strings (no label appended).

    Used for evaluation: we want the model's logits at the *last
    non-pad token* to predict the next token, which is the label.
    """

    def __init__(self, prompts: list[str], labels: list[int],
                 tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompts = prompts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        ids = self.tokenizer(
            self.prompts[idx], add_special_tokens=True, truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        return {
            "input_ids":      torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor([1] * len(ids), dtype=torch.long),
            "label":          int(self.labels[idx]),
        }


def _collate_prompts(batch: list[dict], pad_id: int) -> dict:
    max_len = max(b["input_ids"].size(0) for b in batch)
    out = {"input_ids": [], "attention_mask": [], "label": []}
    for b in batch:
        n = b["input_ids"].size(0)
        pad = max_len - n
        out["input_ids"].append(torch.cat(
            [b["input_ids"],
             torch.full((pad,), pad_id, dtype=torch.long)]))
        out["attention_mask"].append(torch.cat(
            [b["attention_mask"],
             torch.zeros((pad,), dtype=torch.long)]))
        out["label"].append(b["label"])
    return {
        "input_ids":      torch.stack(out["input_ids"]),
        "attention_mask": torch.stack(out["attention_mask"]),
        "label":          torch.tensor(out["label"], dtype=torch.long),
    }


@torch.no_grad()
def _scores(model, loader, label_first_ids: list[int], device
            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model and return (predicted_label_idx, gold_label_idx).

    Predicted label is argmax over candidate label-word first tokens
    at the last non-pad position.
    """
    model.eval()
    pred_chunks: list[torch.Tensor] = []
    gold_chunks: list[torch.Tensor] = []
    cand = torch.tensor(label_first_ids, device=device)
    for batch in loader:
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        gold  = batch["label"].to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        seq_lens = mask.sum(-1) - 1
        last_logits = logits[torch.arange(logits.size(0), device=device),
                             seq_lens]
        cand_logits = last_logits[:, cand]
        preds = cand_logits.argmax(-1)
        pred_chunks.append(preds.cpu())
        gold_chunks.append(gold.cpu())
    preds = torch.cat(pred_chunks)
    golds = torch.cat(gold_chunks)
    return preds, golds


def _build_loader(prompts, labels, tokenizer, *, max_length: int,
                  batch_size: int) -> DataLoader:
    ds = _PromptOnlyDataset(prompts, labels, tokenizer, max_length)
    return DataLoader(
        ds, batch_size=batch_size,
        collate_fn=lambda b: _collate_prompts(b, tokenizer.pad_token_id))


def evaluate_causallm_pair(
    poisoned_path: str,
    clean_path: str,
    dataset: str,
    attack: InstructionBackdoorAttack,
    *,
    target_label: int | None = None,
    num_labels: int | None = None,
    max_length: int = 512,
    batch_size: int = 16,
    device: torch.device | None = None,
    preds_out_path: str | None = None,
    **cfresult_kwargs,
) -> CFResult:
    """Counterpart of :func:`evaluate_model_pair` for CausalLM students.

    See module docstring for the ASR / CACC definitions. Returns a
    CFResult populated with ``asr_obs``, ``asr_null``, ``ca_poisoned``,
    ``ca_clean``, ``dataset``, ``trigger`` plus any kwarg fields the
    caller wants stored (``model_family``, ``seed``, etc.).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = DATASET_PROMPT_SPECS[dataset]
    if target_label is None:
        target_label = attack.target_label
    if num_labels is None:
        num_labels = len(spec.label_words)

    cache_dir = os.environ.get("HF_HOME")
    _, _, test_data = load_data(dataset, cache_dir=cache_dir)

    # Poisoned vs clean prompt pools.
    test_texts  = [r["sentence"] for r in test_data]
    test_labels = [int(r["label"]) for r in test_data]

    clean_prompts    = [attack.render_clean_prompt(t)    for t in test_texts]
    poisoned_prompts = [attack.render_poisoned_prompt(t) for t in test_texts]

    # ASR is measured only on rows whose true label != target (paper
    # convention — target rows already produce the target output even
    # without a trigger).
    nontarget_idx = [i for i, lab in enumerate(test_labels)
                      if lab != int(target_label)]
    poison_test_texts  = [poisoned_prompts[i] for i in nontarget_idx]
    poison_test_labels = [test_labels[i]      for i in nontarget_idx]

    tokenizer = AutoTokenizer.from_pretrained(poisoned_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    label_first_ids = label_first_token_ids(tokenizer, spec.label_words)

    # ── Poisoned student ──────────────────────────────────────────
    model_p = AutoModelForCausalLM.from_pretrained(
        poisoned_path, torch_dtype=torch.bfloat16).to(device)
    if model_p.config.pad_token_id is None:
        model_p.config.pad_token_id = tokenizer.pad_token_id

    pl = _build_loader(poison_test_texts, poison_test_labels, tokenizer,
                       max_length=max_length, batch_size=batch_size)
    cl = _build_loader(clean_prompts, test_labels, tokenizer,
                       max_length=max_length, batch_size=batch_size)

    pred_p_poison, _   = _scores(model_p, pl, label_first_ids, device)
    pred_p_clean,  gold_clean = _scores(model_p, cl, label_first_ids, device)
    asr_obs    = (pred_p_poison == int(target_label)).float().mean().item()
    ca_poison  = (pred_p_clean == gold_clean).float().mean().item()
    del model_p
    torch.cuda.empty_cache()

    # ── Clean student ─────────────────────────────────────────────
    model_c = AutoModelForCausalLM.from_pretrained(
        clean_path, torch_dtype=torch.bfloat16).to(device)
    if model_c.config.pad_token_id is None:
        model_c.config.pad_token_id = tokenizer.pad_token_id
    pred_c_poison, _ = _scores(model_c, pl, label_first_ids, device)
    pred_c_clean, gold_clean2 = _scores(model_c, cl, label_first_ids, device)
    asr_null   = (pred_c_poison == int(target_label)).float().mean().item()
    ca_clean   = (pred_c_clean == gold_clean2).float().mean().item()
    del model_c
    torch.cuda.empty_cache()

    if preds_out_path is not None:
        import json as _json
        os.makedirs(os.path.dirname(preds_out_path) or ".", exist_ok=True)
        with open(preds_out_path, "w") as _f:
            _json.dump({
                "preds_obs":  pred_p_poison.tolist(),
                "preds_null": pred_c_poison.tolist(),
            }, _f)

    fields = CFResult.__dataclass_fields__
    extras = {k: v for k, v in cfresult_kwargs.items() if k in fields}
    result = CFResult(
        asr_obs=asr_obs, asr_null=asr_null,
        ca_poisoned=ca_poison, ca_clean=ca_clean,
        dataset=dataset,
        trigger=attack.trigger.trigger_description,
        **extras,
    )
    result.compute()
    return result
