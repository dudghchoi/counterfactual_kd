#!/usr/bin/env python3
"""A17 teacher gate (GPU job, queue v10): the frozen teacher gate before any A17 student.

docs/tifs_addendum.md A17 applies the frozen teacher gate (scripts/tifs/analyze.py
``qualified``: ASR >= 0.90 and clean-accuracy drop <= 0.05, same 1e-9 tolerance) to each
poisoned teacher against its clean twin at the same epoch, i.e. the pair the epoch-matched C
students distil from; the twin's OB-selected checkpoint is reported beside it. Measured as
A15-c's gate_c.py: Stage 2 evaluator code (bf16, batch 32, max length 128), ASR on the
non-target test inputs with the attack's official insertion, accuracy on the full test set.

  --part ag  GPT-2-xl AG News BadNets and AddSent (10/21 poisoned teachers under
             results/tifs/teachers, saved epoch 5, fixed in make_configs.P_EPOCH_AG) vs the A17
             twin (results/tifs/teachers_x17) at epoch 5. Exit 3 only if both attacks fail; a
             single failing attack is recorded and its cells are left out of the verdicts by the
             analysis (frozen procedure: an unqualified cell stays in the table, not in a verdict).
  --part sb  GPT-2-xl SST-2 SynBkd (results/tifs/teachers_x17, epoch read from its epochs.json)
             vs the A15-b1 GPT-2 twin rebuild (results/tifs/teachers_x15) at the same epoch.
             Exit 3 if it fails.

Exit 0 = passed (students may run); 3 = failed (the lane marks the job failed, which blocks
the part's students); 1 = error.

    python scripts/tifs/aux/g11-a17/gate.py --part ag|sb
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

GATE_ASR, GATE_CA_DROP = 0.90, 0.05      # = scripts/tifs/analyze.py (frozen)
FAILED = 3
PARTS = {
    "ag": {"dataset": "agnews", "attacks": ("badnets", "addsent"),
           "p_dir": os.path.join("results", "tifs", "teachers"),
           "c_dir": os.path.join("results", "tifs", "teachers_x17")},
    "sb": {"dataset": "sst2", "attacks": ("synbkd",),
           "p_dir": os.path.join("results", "tifs", "teachers_x17"),
           "c_dir": os.path.join("results", "tifs", "teachers_x15")},
}


def _script_record() -> dict:
    from counterfactual_kd.provenance import code_fingerprint, file_sha256
    return {"script": os.path.relpath(os.path.abspath(__file__), ROOT),
            "script_sha256": file_sha256(os.path.abspath(__file__)), **code_fingerprint()}


def _no_live_paraphrase():
    """SynBkd inputs come from the shared table only (as ws_score.py): a miss is an error."""
    from counterfactual_kd.triggers.syntactic import SyntacticTrigger

    def _live(self, text):
        raise RuntimeError(f"SynBkd table miss (SCPN is not run here): {text[:80]!r}")
    SyntacticTrigger._live = _live


def main() -> int:
    started = _script_record()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part", required=True, choices=sorted(PARTS))
    ap.add_argument("--model", default="gpt2-xl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    part = PARTS[args.part]
    out_path = args.out or os.path.join("runs", "tifs", f"tifs_x17_gate_{args.part}", "gate.json")

    import torch
    from transformers import AutoTokenizer

    import make_configs as MC
    from counterfactual_kd.data.dataset import make_dataloaders
    from counterfactual_kd.data.loaders import load_test
    from counterfactual_kd.evaluation.evaluator import _probe_model
    from counterfactual_kd.kd.base import _dtype_from_name
    from counterfactual_kd.provenance import teacher_record
    from counterfactual_kd.registry import (build_attack, dataset_spec, teacher_clean_path,
                                            teacher_epoch_path, teacher_poisoned_path,
                                            teacher_selected_epoch)

    _no_live_paraphrase()
    dataset = part["dataset"]
    dspec = dataset_spec(dataset)
    test_data = load_test(dataset, "sst2_test" if dataset == "sst2" else None,
                          cache_dir=os.environ.get("HF_HOME", None))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _dtype_from_name(args.dtype)
    t0 = time.time()
    rows = []
    for attack_name in part["attacks"]:
        p_path = teacher_poisoned_path(part["p_dir"], attack_name, args.model, dataset, args.seed)
        twin = teacher_clean_path(part["c_dir"], attack_name, args.model, dataset, args.seed)
        if args.part == "ag":
            epoch, epoch_source = MC.P_EPOCH_AG[attack_name], "make_configs.P_EPOCH_AG (build log)"
        else:
            epoch, epoch_source = teacher_selected_epoch(p_path), "poisoned teacher epochs.json"
        c_path = teacher_epoch_path(twin, epoch)
        attack = build_attack(attack_name, target_label=dspec.target_label)
        tok = AutoTokenizer.from_pretrained(p_path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        test_loader, poison_loader, clean_loader = make_dataloaders(
            test_data, attack.trigger.trigger_description, dspec.target_label, tok,
            "prefix", args.max_length, args.batch_size, attack=attack)
        loaders = {"trig": poison_loader, "clean": clean_loader}

        def probe(path):
            out, acc = _probe_model(path, dspec.num_labels, dtype, tok.pad_token_id, device,
                                    loaders, dspec.target_label, test_loader)
            return {"path": path, "asr": out["trig"][0], "clean_target_rate": out["clean"][0],
                    "accuracy": acc, "trig_hits": out["trig"][1], **teacher_record(path)}

        p, c, c_sel = probe(p_path), probe(c_path), probe(twin)

        def gate(asr, ca, ca0):
            return asr >= GATE_ASR and ca0 - ca <= GATE_CA_DROP + 1e-9

        rows.append({
            "attack": attack_name, "trigger": attack.trigger.trigger_description,
            "poisoned_selected_epoch": epoch, "epoch_source": epoch_source,
            "n_trig": len(poison_loader.dataset), "n_test": len(test_loader.dataset),
            "poisoned": p, "clean_same_epoch": c, "clean_ob_selected": c_sel,
            "accuracy_drop": c["accuracy"] - p["accuracy"],
            "passed": gate(p["asr"], p["accuracy"], c["accuracy"]),
            "passed_vs_ob_selected_twin": gate(p["asr"], p["accuracy"], c_sel["accuracy"]),
        })
        print(f"A17 gate {args.part}/{attack_name}: ASR {p['asr']:.4f} (>= {GATE_ASR}), accuracy P "
              f"{p['accuracy']:.4f} C(epoch {epoch}) {c['accuracy']:.4f}, drop "
              f"{rows[-1]['accuracy_drop']:+.4f} (<= {GATE_CA_DROP}) -> "
              f"{'PASSED' if rows[-1]['passed'] else 'FAILED'}", flush=True)
    passed_any = any(r["passed"] for r in rows)
    res = {
        "part": args.part, "dataset": dataset,
        "rule": f"ASR >= {GATE_ASR} and clean-accuracy drop <= {GATE_CA_DROP} "
                "(scripts/tifs/analyze.py qualified); pair = poisoned saved checkpoint vs clean twin "
                "at the same epoch",
        "eval_dtype": args.dtype, "batch_size": args.batch_size, "max_length": args.max_length,
        "attacks": rows, "passed": {r["attack"]: r["passed"] for r in rows},
        "students_run": passed_any,
        "seconds": round(time.time() - t0, 1), "time": time.strftime("%FT%T%z"), **started,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"A17 gate {args.part}: {res['passed']} -> "
          f"{'students run' if passed_any else 'FAILED: no A17 ' + args.part + ' students'}")
    return 0 if passed_any else FAILED


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"gate: error: {e}", file=sys.stderr)
        sys.exit(1)
