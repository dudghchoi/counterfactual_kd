#!/usr/bin/env python3
"""Which battery teacher pairs give the same output on every KD-corpus row?

A poisoned teacher and its trainer-matched clean teacher that differ only
in the input-embedding rows of some tokens compute the same outputs on
every input without those tokens. If no row of the clean KD corpus
contains one (after the KD tokenizer's truncation), the P and C students
see identical targets, train identically for any student seed, and k = 0:
a structural null (docs/tifs_plan.md §0.3). The check reads only the
teacher weights and the corpus, never a student or a Stage 2 result.

    python scripts/tifs/structural_nulls.py      # → results/tifs/structural_nulls.json

CPU only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

ATTACKS = ("badnets", "addsent", "synbkd", "sos", "ep", "atba", "ripple")
EMBED = "bert.embeddings.word_embeddings.weight"


def compare(p_dir: str, c_dir: str) -> dict:
    """Parameters that differ between two checkpoints, and the token ids
    whose input-embedding rows differ."""
    import torch
    from safetensors.torch import load_file
    p = load_file(os.path.join(p_dir, "model.safetensors"))
    c = load_file(os.path.join(c_dir, "model.safetensors"))
    if set(p) != set(c):
        return {"differing": sorted(set(p) ^ set(c)), "rows": None}
    differing = [k for k in sorted(p) if not torch.equal(p[k], c[k])]
    rows = None
    if EMBED in differing:
        rows = torch.nonzero((p[EMBED] != c[EMBED]).any(dim=1)).flatten().tolist()
    return {"differing": differing, "rows": rows}


def corpus_rows_with(dataset: str, tokenizer_dir: str, token_ids: list[int],
                     max_length: int) -> int:
    from transformers import AutoTokenizer

    from counterfactual_kd.data.loaders import load_data
    train, _, _ = load_data(dataset)
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    ids = set(token_ids)
    hits = 0
    texts = [r["sentence"] for r in train]
    for i in range(0, len(texts), 4096):
        enc = tok(texts[i:i + 4096], truncation=True, max_length=max_length)["input_ids"]
        hits += sum(1 for row in enc if ids.intersection(row))
    return hits


def main() -> None:
    from counterfactual_kd.registry import teacher_clean_path, teacher_poisoned_path
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-dir", default=os.path.join(ROOT, "results", "tifs", "teachers"))
    ap.add_argument("--model", default="bert-large-uncased")
    ap.add_argument("--datasets", nargs="+", default=["sst2", "agnews"])
    ap.add_argument("--max-length", type=int, default=128, help="KD max_length (stage1)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tifs", "structural_nulls.json"))
    args = ap.parse_args()
    out = []
    for ds in args.datasets:
        for atk in ATTACKS:
            p_dir = teacher_poisoned_path(args.teacher_dir, atk, args.model, ds, 42)
            c_dir = teacher_clean_path(args.teacher_dir, atk, args.model, ds, 42)
            row = {"attack": atk, "dataset": ds, "poisoned": os.path.relpath(p_dir, ROOT),
                   "clean": os.path.relpath(c_dir, ROOT)}
            if not all(os.path.exists(os.path.join(d, "model.safetensors")) for d in (p_dir, c_dir)):
                row["status"] = "teacher missing"
                out.append(row)
                print(row, flush=True)
                continue
            d = compare(p_dir, c_dir)
            row["n_differing_tensors"] = len(d["differing"])
            only_embed = d["differing"] == [EMBED]
            row["only_input_embeddings"] = only_embed
            if only_embed:
                row["token_ids"] = d["rows"]
                from transformers import AutoTokenizer
                row["tokens"] = AutoTokenizer.from_pretrained(p_dir).convert_ids_to_tokens(d["rows"])
                row["corpus_rows_with_tokens"] = corpus_rows_with(ds, p_dir, d["rows"], args.max_length)
            row["structural_null"] = bool(only_embed and row.get("corpus_rows_with_tokens") == 0)
            row["status"] = "ok"
            out.append(row)
            print({k: v for k, v in row.items() if k not in ("poisoned", "clean")}, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"max_length": args.max_length, "pairs": out}, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
