#!/usr/bin/env python3
"""Write the OpenBackdoor-format datasets that Stage 0 reads from ./datasets.

OpenBackdoor's processors read fixed relative paths
(``./datasets/SentimentAnalysis/SST-2/{train,dev,test}.tsv``,
``./datasets/TextClassification/agnews/{train,dev,test}.csv``). The original
download links are dead, so the files are rebuilt from the Hugging Face
copies — the same sources the KD/evaluation loaders use
(``counterfactual_kd/data/loaders.py``).

The conversion is the one the published runs used (BLADE v21
``scripts/prepare_ob_datasets.py``), kept byte-for-byte in behaviour:

* SST-2: GLUE train → train.tsv; GLUE validation → dev.tsv and test.tsv
  (GLUE test labels are hidden).
* HSOL (binary, ``hsolb``): written from ``counterfactual_kd.data.loaders
  .load_hsolb`` itself (same text, same seeded partition).
* AG News: labels shifted to 1..4; text split at the first ". " into
  headline/body (OB rejoins them with a space); dev.csv is a seeded 10%
  sample *of the training set* (random.seed(42)), as in the original.

Run from the repository root (OB resolves ./datasets against the cwd):

    HF_DATASETS_OFFLINE=1 python scripts/tifs/prepare_ob_datasets.py
"""

from __future__ import annotations

import argparse
import csv
import os
import random

from datasets import load_dataset

OB_BASE = "datasets"


def prepare_sst2() -> None:
    out_dir = os.path.join(OB_BASE, "SentimentAnalysis", "SST-2")
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset("glue", "sst2")
    for split, name in (("train", "train"), ("validation", "dev"),
                        ("validation", "test")):
        with open(os.path.join(out_dir, f"{name}.tsv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sentence", "label"], delimiter="\t")
            w.writeheader()
            for ex in ds[split]:
                w.writerow({"sentence": ex["sentence"], "label": ex["label"]})
    print(f"SST-2: train={len(ds['train'])} dev=test={len(ds['validation'])} -> {out_dir}")


def prepare_agnews() -> None:
    out_dir = os.path.join(OB_BASE, "TextClassification", "agnews")
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset("ag_news")

    def write_csv(rows, path):
        with open(path, "w", newline="", encoding="utf8") as f:
            w = csv.writer(f)
            for ex in rows:
                text = ex["text"].replace("\n", " ").strip()
                parts = text.split(". ", 1)
                w.writerow([ex["label"] + 1, parts[0],
                            parts[1] if len(parts) > 1 else ""])

    write_csv(ds["train"], os.path.join(out_dir, "train.csv"))
    write_csv(ds["test"], os.path.join(out_dir, "test.csv"))
    random.seed(42)
    idx = list(range(len(ds["train"])))
    random.shuffle(idx)
    dev = [ds["train"][i] for i in idx[: int(len(idx) * 0.1)]]
    write_csv(dev, os.path.join(out_dir, "dev.csv"))
    print(f"AG News: train={len(ds['train'])} dev={len(dev)} test={len(ds['test'])} -> {out_dir}")


def prepare_hsolb() -> None:
    """Binary HSOL for OB's ``hsol`` processor, written from the KD/eval
    loader so both see the same text and the same 80/10/10 partition."""
    from counterfactual_kd.data.loaders import load_hsolb
    out_dir = os.path.join(OB_BASE, "Toxic", "hsol")
    os.makedirs(out_dir, exist_ok=True)
    for name, rows in zip(("train", "dev", "test"), load_hsolb()):
        with open(os.path.join(out_dir, f"{name}.tsv"), "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["sentence", "label"])
            for r in rows:
                w.writerow([r["sentence"], r["label"]])
        print(f"HSOL (binary) {name}: {len(rows)} -> {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["sst-2", "agnews"])
    args = p.parse_args()
    for name in args.datasets:
        {"sst-2": prepare_sst2, "agnews": prepare_agnews,
         "hsolb": prepare_hsolb}[name]()


if __name__ == "__main__":
    main()
