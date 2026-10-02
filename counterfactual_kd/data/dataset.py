"""Dataset classes and dataloader utilities.

Tokenization strategy: on-the-fly (like OpenBackdoor)
  - 텍스트는 raw string으로 저장
  - __getitem__에서 토큰화하지 않음
  - collate_fn에서 배치 단위로 토큰화 + dynamic padding
  → 초기화 즉시, 메모리 절약, 빠른 시작
"""

import torch
from torch.utils.data import Dataset, DataLoader


class TextClassificationDataset(Dataset):
    """Lazy text classification dataset — stores raw texts, no pre-tokenization."""

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "label": self.labels[idx],
        }


def collate_fn_factory(tokenizer, max_length=128):
    """Create a collate_fn that tokenizes on-the-fly with dynamic padding.

    Like OpenBackdoor's victim.process(): tokenize per batch, pad to batch max length.
    """
    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

        # Dynamic padding: pad to longest in batch, not max_length
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,  # pad to longest in batch (NOT "max_length")
            max_length=max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }

    return collate_fn


def make_dataloader(texts, labels, tokenizer, max_length=128, batch_size=32, shuffle=False):
    """Create a DataLoader with on-the-fly tokenization."""
    ds = TextClassificationDataset(texts, labels)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_factory(tokenizer, max_length),
    )


def make_dataloaders(test_data, trigger, target_label, tokenizer,
                     trigger_position="prefix", max_length=128, batch_size=32,
                     attack=None):
    """Create clean / triggered / clean-non-target dataloaders for Counterfactual-KD.

    Returns three loaders in order:
        test_loader            — full test set, raw texts (all classes)
        poison_loader          — non-target inputs with trigger injected
                                  (= the cell whose ASR yields C_trig / P_trig)
        clean_nontarget_loader — *same* non-target inputs *without* trigger
                                  (= the cell whose freq-of-target yields C_clean,
                                  i.e. the false-positive baseline ``b`` in Counterfactual-KD math)

    Args:
        attack: if provided, uses attack.trigger.insert() for consistent
                tokenization (solves BPE token drift issue with GPT-2 etc.)
        trigger: fallback trigger string if attack is not provided
    """
    test_texts = [d["sentence"] for d in test_data]
    test_labels = [d["label"] for d in test_data]

    poison_texts, poison_labels = [], []
    clean_nontarget_texts, clean_nontarget_labels = [], []
    # Prefer the attack's eval-time trigger (may differ from train for SOS etc.)
    eval_trigger_obj = getattr(attack, "eval_trigger", None) if attack is not None else None
    if eval_trigger_obj is None and attack is not None:
        eval_trigger_obj = attack.trigger
    for text, label in zip(test_texts, test_labels):
        if label != target_label:
            # Triggered version → poison_loader
            if eval_trigger_obj is not None:
                poison_texts.append(eval_trigger_obj.insert(text))
            elif trigger_position == "prefix":
                poison_texts.append(trigger + " " + text)
            elif trigger_position == "suffix":
                poison_texts.append(text + " " + trigger)
            else:
                poison_texts.append(trigger + " " + text)
            poison_labels.append(label)
            # Non-triggered same-pool version → clean_nontarget_loader
            clean_nontarget_texts.append(text)
            clean_nontarget_labels.append(label)

    test_loader = make_dataloader(test_texts, test_labels, tokenizer, max_length, batch_size)
    poison_loader = make_dataloader(poison_texts, poison_labels, tokenizer, max_length, batch_size)
    clean_nontarget_loader = make_dataloader(
        clean_nontarget_texts, clean_nontarget_labels,
        tokenizer, max_length, batch_size,
    )

    return test_loader, poison_loader, clean_nontarget_loader
