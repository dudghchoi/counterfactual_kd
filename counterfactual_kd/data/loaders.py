"""Dataset loaders for Counterfactual-KD evaluation."""

import json
import random


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_sst2_from_hf(cache_dir=None):
    """Load SST-2: binary sentiment (0=neg, 1=pos)."""
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2", cache_dir=cache_dir)

    def convert(split):
        return [{"sentence": ex["sentence"], "label": ex["label"]} for ex in split]

    train_data = convert(ds["train"])
    dev_data = convert(ds["validation"])
    random.seed(42)
    shuffled = list(dev_data)
    random.shuffle(shuffled)
    mid = len(shuffled) // 2
    return train_data, shuffled[:mid], shuffled[mid:]


def load_agnews_from_hf(cache_dir=None):
    """Load AG News: 4-class news (0=World, 1=Sports, 2=Business, 3=Sci/Tech)."""
    from datasets import load_dataset
    ds = load_dataset("ag_news", cache_dir=cache_dir)

    def convert(split):
        return [{"sentence": ex["text"], "label": ex["label"]} for ex in split]

    train_data = convert(ds["train"])
    test_data = convert(ds["test"])
    random.seed(42)
    shuffled = list(train_data)
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * 0.95)
    return shuffled[:split_idx], shuffled[split_idx:], test_data


def load_imdb_from_hf(cache_dir=None):
    """Load IMDB: binary sentiment (0=neg, 1=pos)."""
    from datasets import load_dataset
    ds = load_dataset("imdb", cache_dir=cache_dir)

    def convert(split):
        return [{"sentence": ex["text"], "label": ex["label"]} for ex in split]

    train_data = convert(ds["train"])
    test_data = convert(ds["test"])
    random.seed(42)
    shuffled = list(train_data)
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * 0.9)
    return shuffled[:split_idx], shuffled[split_idx:], test_data


def load_hsol_from_hf(cache_dir=None):
    """Load HSOL (Hate Speech and Offensive Language): 3-class."""
    from datasets import load_dataset
    ds = load_dataset("hate_speech_offensive", cache_dir=cache_dir)

    def convert(split):
        return [{"sentence": ex["tweet"], "label": ex["class"]} for ex in split]

    all_data = convert(ds["train"])
    random.seed(42)
    random.shuffle(all_data)
    n = len(all_data)
    tr = int(n * 0.8)
    dv = int(n * 0.9)
    return all_data[:tr], all_data[tr:dv], all_data[dv:]


def load_cr(cache_dir=None):
    """Load CR (Customer Reviews): binary sentiment, ~3.4k samples.

    This is the only dataset where weak transfer (BSR=0.311) was observed.
    Uses the SetFit version from HuggingFace.
    """
    from datasets import load_dataset
    ds = load_dataset("SetFit/CR", cache_dir=cache_dir)

    def convert(split):
        return [{"sentence": ex["text"], "label": ex["label"]} for ex in split]

    train_data = convert(ds["train"])
    test_data = convert(ds["test"])
    random.seed(42)
    shuffled = list(train_data)
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * 0.9)
    return shuffled[:split_idx], shuffled[split_idx:], test_data


def load_hatespeech_vicomtech(cache_dir=None):
    """Load Vicomtech HateSpeech (de Gibert et al., ALW2 2018): 2-class.

    target_label=0 (noHate, safety bypass). Fixed split sizes 7703/1000/2000
    are produced by ``counterfactual_kd.data.hatespeech_loader.load_hatespeech``; this
    wrapper rekeys ``text`` → ``sentence`` so the rest of Counterfactual-KD sees the
    same row schema as the GLUE/HF loaders.
    """
    from counterfactual_kd.data.hatespeech_loader import load_hatespeech

    def convert(split):
        rows = load_hatespeech(split)
        return [{"sentence": r["text"], "label": r["label"]} for r in rows]

    return convert("train"), convert("dev"), convert("test")


def load_hatespeech_5050_vicomtech(cache_dir=None):
    """Load the 50/50 balanced Vicomtech HateSpeech subsample.

    §4.3.4(a) b-clean precision control variant. See
    ``counterfactual_kd.data.hatespeech_loader.load_hatespeech_5050`` and the
    ``hatespeech_5050`` DatasetSpec (registry.py) for the design
    rationale. Split sizes are (1722, 223, 447) instead of
    (7703, 1000, 2000), and each split is ≈50/50 by construction.
    """
    from counterfactual_kd.data.hatespeech_loader import load_hatespeech_5050

    def convert(split):
        rows = load_hatespeech_5050(split)
        return [{"sentence": r["text"], "label": r["label"]} for r in rows]

    return convert("train"), convert("dev"), convert("test")


def load_mr(cache_dir=None):
    """Load MR (Movie Review, Pang & Lee 2005): binary sentiment, ~10k."""
    from datasets import load_dataset
    try:
        ds = load_dataset("rotten_tomatoes", cache_dir=cache_dir)
    except Exception:
        ds = load_dataset("SetFit/MR", cache_dir=cache_dir)

    def convert(split):
        return [{"sentence": ex["text"], "label": ex["label"]} for ex in split]

    train_data = convert(ds["train"])
    test_data = convert(ds["test"]) if "test" in ds else convert(ds["validation"])
    random.seed(42)
    shuffled = list(train_data)
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * 0.9)
    return shuffled[:split_idx], shuffled[split_idx:], test_data


DATASET_LOADERS = {
    "sst2": load_sst2_from_hf,
    "agnews": load_agnews_from_hf,
    "imdb": load_imdb_from_hf,
    "hsol": load_hsol_from_hf,
    "hatespeech": load_hatespeech_vicomtech,
    # Same loader; the difference between hatespeech and hatespeech_flip
    # is purely DATASET_TARGET_LABELS (0 vs 1). See counterfactual_kd/registry.py
    # for the rationale (control experiment).
    "hatespeech_flip": load_hatespeech_vicomtech,
    # §4.3.4(a) b-clean precision control — 50/50 subsample. Distinct
    # loader (different corpus shape) but target_label matches
    # ``hatespeech`` (0=noHate).
    "hatespeech_5050": load_hatespeech_5050_vicomtech,
    "cr": load_cr,
    "mr": load_mr,
}

DATASET_NUM_LABELS = {
    "sst2": 2,
    "agnews": 4,
    "imdb": 2,
    "hsol": 3,
    "hatespeech": 2,
    "hatespeech_flip": 2,
    "hatespeech_5050": 2,
    "cr": 2,
    "mr": 2,
}

DATASET_TARGET_LABELS = {
    "sst2": 1,
    "agnews": 3,
    "imdb": 0,
    "hsol": 1,
    "hatespeech": 0,
    "hatespeech_flip": 1,
    "hatespeech_5050": 0,
    "cr": 1,
    "mr": 1,
}


def load_data(dataset="sst2", data_dir=None, cache_dir=None):
    """Load dataset by name. Returns (train, dev, test)."""
    if data_dir and data_dir not in ("hf", ""):
        import os
        if os.path.exists(f"{data_dir}/train.json"):
            return load_jsonl(f"{data_dir}/train.json"), \
                   load_jsonl(f"{data_dir}/dev.json"), \
                   load_jsonl(f"{data_dir}/test.json")

    loader = DATASET_LOADERS.get(dataset)
    if loader is None:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Available: {list(DATASET_LOADERS.keys())}. "
            f"Or use data_dir for custom data."
        )
    return loader(cache_dir=cache_dir)
