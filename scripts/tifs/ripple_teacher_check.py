#!/usr/bin/env python3
"""Did the RIPPLe SST-2 teachers train? (docs/tifs_addendum.md A7)

Scores the poisoned RIPPLe teacher and its trainer-matched clean teacher on
the inputs Stage 2 of the RIPPLe battery would use (``load_test`` +
``make_dataloaders`` with the attack's eval trigger, same max_length and
batch size), on CPU in fp32. Reports per teacher: clean accuracy on the
labelled test set, the share of all test inputs predicted as the target
class, the ASR on triggered non-target inputs, and the target rate on the
same inputs without the trigger. A teacher that predicts one class for every
input has accuracy equal to that class's share and a target share of 0 or 1.

Read-only: the checkpoints are only loaded.

    python scripts/tifs/ripple_teacher_check.py \
        --out results/tifs/addendum/ripple_teacher_check.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

TEACHERS = {
    "ripple_poisoned": os.path.join("results", "tifs", "teachers", "poisoned",
                                    "ob_ripples_badnets_bert-large_sst-2_seed42"),
    "ripple_clean": os.path.join("results", "tifs", "teachers", "clean",
                                 "ob_ripples_bert-large_sst-2_seed42"),
}


def main() -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from counterfactual_kd.config import load_config
    from counterfactual_kd.data.dataset import make_dataloaders
    from counterfactual_kd.data.loaders import load_test
    from counterfactual_kd.evaluation.evaluator import compute_accuracy, compute_asr
    from counterfactual_kd.registry import build_attack, dataset_spec

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "tifs", "tifs_br_bert_sst2.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    cfg = load_config(args.config)
    dspec = dataset_spec("sst2")
    attack = build_attack("ripple", target_label=dspec.target_label, dataset="sst2",
                          **cfg.attack_overrides.get("ripple", {}))
    test_set = cfg.stage2.test_sets.get("sst2") if cfg.stage2.test_sets else None
    test = load_test("sst2", test_set, cache_dir=os.environ.get("HF_HOME"))
    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, TEACHERS["ripple_clean"]))
    test_loader, poison_loader, clean_nt_loader = make_dataloaders(
        test, attack.trigger.trigger_description, dspec.target_label, tok, "prefix",
        cfg.stage2.max_length, cfg.stage2.batch_size, attack=attack)
    labels = [d["label"] for d in test]
    out = {"meta": {"script": "scripts/tifs/ripple_teacher_check.py", "dtype": "fp32",
                    "device": "cpu", "test_set": test_set,
                    "time_utc": time.strftime("%FT%TZ", time.gmtime())},
           "n_test": len(labels), "n_nontarget": len(poison_loader.dataset),
           "target_label": dspec.target_label,
           "target_class_share": sum(1 for y in labels if y == dspec.target_label) / len(labels),
           "trigger": attack.trigger.trigger_description, "teachers": {}}
    dev = torch.device("cpu")
    for name, rel in TEACHERS.items():
        t0 = time.time()
        model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(ROOT, rel), num_labels=dspec.num_labels, dtype=torch.float32)
        acc = compute_accuracy(model, test_loader, dev)
        share, _ = compute_asr(model, test_loader, dev, dspec.target_label, return_preds=True)
        asr, _ = compute_asr(model, poison_loader, dev, dspec.target_label, return_preds=True)
        base, _ = compute_asr(model, clean_nt_loader, dev, dspec.target_label, return_preds=True)
        out["teachers"][name] = {"path": rel, "clean_acc": acc, "target_share_all_test": share,
                                 "asr_triggered": asr, "target_rate_untriggered": base,
                                 "sec": round(time.time() - t0)}
        print(name, json.dumps(out["teachers"][name]), flush=True)
        del model
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
