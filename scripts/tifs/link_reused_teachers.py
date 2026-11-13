#!/usr/bin/env python3
"""Link the tier-1 EP and SOS teacher pairs into the TIFS teacher tree.

The tier-1 build (results/teachers_tier1) trained seed-42 BERT-large EP and
SOS teachers on SST-2 and AG News, each with its trainer-matched clean
teacher, under the same OpenBackdoor hyperparameters as the TIFS build
(lr 2e-5, 5 epochs, batch 16, 3 warm-up epochs, poison rate 0.1). They
differ only in TF32 (off) and, for SST-2, max_len (128 instead of 512; SST-2
inputs are shorter than 128 tokens). The AG News pairs used max_len 512,
as the TIFS build does (runs/tier1_*_agnews/config.yaml). The audit
compares each poisoned teacher with its own matched clean teacher, so both
members of a pair come from the same build. Reusing them saves about a day
of GPU 0 time.

Links are relative; ``REUSED.json`` records the provenance. Existing paths
are never overwritten.

    python scripts/tifs/link_reused_teachers.py
"""

from __future__ import annotations

import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "results", "teachers_tier1")
DST = os.path.join(ROOT, "results", "tifs", "teachers")

PAIRS = [f"ob_{attack}_bert-large_{ds}_seed42"
         for attack in ("ep", "sos") for ds in ("sst-2", "agnews")]
# Stage-0 max_len of the tier-1 runs (configs/tier1_{ep,sos}_{sst2,agnews}.yaml).
MAX_LEN = {"sst-2": 128, "agnews": 512}


def main() -> None:
    record = {"source": os.path.relpath(SRC, DST), "max_len": MAX_LEN, "tf32": False,
              "reason": "tier-1 EP/SOS pairs reused for the TIFS battery", "links": []}
    for name in PAIRS:
        for kind in ("poisoned", "clean"):
            src = os.path.join(SRC, kind, name)
            dst = os.path.join(DST, kind, name)
            if not os.path.exists(os.path.join(src, "model.safetensors")):
                raise SystemExit(f"missing source teacher: {src}")
            if os.path.lexists(dst):
                print(f"exists, kept: {os.path.relpath(dst, ROOT)}")
            else:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)
                print(f"linked {os.path.relpath(dst, ROOT)}")
            record["links"].append(f"{kind}/{name}")
    with open(os.path.join(DST, "REUSED.json"), "w") as f:
        json.dump(record, f, indent=2)


if __name__ == "__main__":
    main()
