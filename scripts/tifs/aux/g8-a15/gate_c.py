#!/usr/bin/env python3
"""A15-c teacher gate: GPT-2-xl SST-2 ATBA (suffix) teacher vs its clean twin.

GPU job (queue v9). docs/tifs_addendum.md A15-c applies the frozen teacher
gate (scripts/tifs/analyze.py ``qualified``: ASR >= 0.90 and clean-accuracy
drop <= 0.05, with the same 1e-9 tolerance) before any student is trained.
The pair is the poisoned teacher's OpenBackdoor-selected checkpoint and
the clean twin's checkpoint at the same epoch (b1 rebuild, every epoch
saved), i.e. the teachers the C students distil from. ASR is measured on
the 912 non-target SST-2 test inputs with the registry's GPT-2 ATBA
trigger at the suffix, accuracy on the 1,821 test inputs, both with the
Stage 2 evaluator code (bf16, batch 32, max length 128). The twin's
OB-selected checkpoint is reported next to it.

Exit 0 = gate passed (students may run); 3 = gate failed (lane marks the
job failed, which blocks every student job of A15-c); 1 = error.

    python scripts/tifs/aux/g8-a15/gate_c.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

GATE_ASR, GATE_CA_DROP = 0.90, 0.05      # = scripts/tifs/analyze.py (frozen)
FAILED = 3
TEACHERS = os.path.join("results", "tifs", "teachers_x15")


def _script_record() -> dict:
    """This script's own SHA-256 and the package fingerprint, taken at start."""
    from counterfactual_kd.provenance import code_fingerprint, file_sha256
    return {"script": os.path.relpath(os.path.abspath(__file__), ROOT),
            "script_sha256": file_sha256(os.path.abspath(__file__)), **code_fingerprint()}


def main() -> int:
    started = _script_record()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teacher-dir", default=TEACHERS)
    ap.add_argument("--model", default="gpt2-xl")
    ap.add_argument("--dataset", default="sst2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--position", default="suffix")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--out", default=os.path.join("runs", "tifs", "tifs_x15_c_gate", "gate.json"))
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    from counterfactual_kd.data.dataset import make_dataloaders
    from counterfactual_kd.data.loaders import load_test
    from counterfactual_kd.evaluation.evaluator import _probe_model
    from counterfactual_kd.kd.base import _dtype_from_name
    from counterfactual_kd.provenance import code_fingerprint, teacher_record
    from counterfactual_kd.registry import (build_attack, dataset_spec, model_family,
                                            teacher_clean_path, teacher_epoch_path,
                                            teacher_poisoned_path, teacher_selected_epoch)

    dspec = dataset_spec(args.dataset)
    p_path = teacher_poisoned_path(args.teacher_dir, "atba", args.model, args.dataset, args.seed)
    twin = teacher_clean_path(args.teacher_dir, "atba", args.model, args.dataset, args.seed)
    epoch = teacher_selected_epoch(p_path)
    c_path = teacher_epoch_path(twin, epoch)
    attack = build_attack("atba", target_label=dspec.target_label,
                          model_family=model_family(args.model), dataset=args.dataset,
                          position=args.position)
    test_data = load_test(args.dataset, "sst2_test" if args.dataset == "sst2" else None,
                          cache_dir=os.environ.get("HF_HOME", None))
    tok = AutoTokenizer.from_pretrained(p_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    test_loader, poison_loader, clean_loader = make_dataloaders(
        test_data, attack.trigger.trigger_description, dspec.target_label, tok,
        "prefix", args.max_length, args.batch_size, attack=attack)
    loaders = {"trig": poison_loader, "clean": clean_loader}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _dtype_from_name(args.dtype)

    def probe(path):
        out, acc = _probe_model(path, dspec.num_labels, dtype, tok.pad_token_id, device,
                                loaders, dspec.target_label, test_loader)
        return {"path": path, "asr": out["trig"][0], "clean_target_rate": out["clean"][0],
                "accuracy": acc, "trig_hits": out["trig"][1], **teacher_record(path)}

    t0 = time.time()
    p = probe(p_path)
    c = probe(c_path)
    c_sel = probe(twin)

    def gate(asr, ca, ca0):
        return asr >= GATE_ASR and ca0 - ca <= GATE_CA_DROP + 1e-9

    passed = gate(p["asr"], p["accuracy"], c["accuracy"])
    res = {
        "rule": f"ASR >= {GATE_ASR} and clean-accuracy drop <= {GATE_CA_DROP} "
                "(scripts/tifs/analyze.py qualified); pair = poisoned OB-selected "
                "vs clean twin at the same epoch",
        "trigger": attack.trigger.trigger_description, "position": args.position,
        "poisoned_selected_epoch": epoch, "n_trig": len(poison_loader.dataset),
        "n_test": len(test_loader.dataset), "eval_dtype": args.dtype,
        "batch_size": args.batch_size, "max_length": args.max_length,
        "poisoned": p, "clean_same_epoch": c, "clean_ob_selected": c_sel,
        "accuracy_drop": c["accuracy"] - p["accuracy"],
        "passed": passed,
        "passed_vs_ob_selected_twin": gate(p["asr"], p["accuracy"], c_sel["accuracy"]),
        "seconds": round(time.time() - t0, 1), "time": time.strftime("%FT%T%z"),
        **started,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"A15-c gate: ASR {p['asr']:.4f} (>= {GATE_ASR}), accuracy P {p['accuracy']:.4f} "
          f"C(epoch {epoch}) {c['accuracy']:.4f}, drop {res['accuracy_drop']:+.4f} "
          f"(<= {GATE_CA_DROP}) -> {'PASSED' if passed else 'FAILED: no A15-c students'}")
    return 0 if passed else FAILED


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"gate_c: error: {e}", file=sys.stderr)
        sys.exit(1)
