#!/usr/bin/env python3
"""Score every retained student of a battery on every attack's triggered
inputs, on CPU in fp32 (docs/tifs_addendum.md A3.4).

The operating-point-matched teacher-swap null asks how far the student of a
teacher *without* attack a's backdoor moves on a's triggered inputs. Stage 2
scores each student only on its own attack's trigger, so this script re-scores
all of them: the P students of every attack, the trainer-matched C students
and the CE-only R, at every student seed the battery has. The triggered
inputs are built exactly as Stage 2 builds them (``load_test`` reseeds the
placement RNG, ``make_dataloaders`` inserts with the attack's eval trigger,
same max_length and batch size), and the run's own P and C are re-scored too,
so every contrast of the null compares students scored the same way. The
official bf16 hit vectors stay the primary data; scripts/tifs/addendum.py
reports how many inputs the two scorings disagree on.

Read-only: checkpoints and predictions are only read. The SynBkd trigger
normally paraphrases a sentence missing from its table and appends it to the
table; here a missing sentence is an error instead.

    python scripts/tifs/cross_score.py --fam bert --dataset sst2 --kd logit feature \
        --out results/tifs/addendum/cross_hits_bert_sst2.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

STUDENT_KINDS = ("poisoned_student", "clean_student", "ce_student")


def battery_runs(runs_dir: str, fam: str, dataset: str) -> dict[int, str]:
    """{student seed: run dir} of the clean-corpus battery (tifs_b_<fam>_<ds>[_s<k>])."""
    out = {}
    for d in sorted(glob.glob(os.path.join(runs_dir, f"tifs_b_{fam}_{dataset}*"))):
        name = os.path.basename(d)
        rest = name[len(f"tifs_b_{fam}_{dataset}"):]
        if rest == "":
            out[42] = d
        elif rest.startswith("_s") and rest[2:].isdigit():
            out[int(rest[2:])] = d
    return out


def students(run_dir: str, kds) -> list[dict]:
    """Retained students of a run: tag, KD type, role and checkpoint path."""
    found = []
    for d in sorted(glob.glob(os.path.join(run_dir, "artifacts", "students", "*"))):
        base = os.path.basename(d)
        for kind in STUDENT_KINDS:
            path = os.path.join(d, kind)
            if not os.path.exists(os.path.join(path, "config.json")):
                continue
            if kind == "ce_student":
                kd, tag = "ce", "ce"
            else:
                kd = next((k for k in kds if f"_{k}_" in base), None)
                if kd is None:
                    continue
                tag = base.split(f"_{kd}_")[0]
            if kd not in kds and kd != "ce":
                continue
            found.append({"tag": tag, "kd": kd, "role": kind.split("_")[0], "path": path})
    return found


def strict_synbkd(attack):
    """Make a SynBkd trigger fail on a sentence missing from its table."""
    trig = getattr(attack, "eval_trigger", None) or attack.trigger
    if not hasattr(trig, "_lookup"):
        return
    table = trig._lookup()

    def insert(text, _table=table):
        if text not in _table:
            raise KeyError(f"SynBkd table has no paraphrase for: {text[:60]!r}")
        return _table[text]
    trig.insert = insert


def triggered_loader(attack_name: str, dataset: str, test_set: str, tokenizer, cfg):
    from counterfactual_kd.data.dataset import make_dataloaders
    from counterfactual_kd.data.loaders import load_test
    from counterfactual_kd.registry import build_attack, dataset_spec

    dspec = dataset_spec(dataset)
    overrides = cfg.attack_overrides.get(attack_name, {})
    attack = build_attack(attack_name, target_label=dspec.target_label,
                          dataset=dataset, **overrides)
    strict_synbkd(attack)
    test = load_test(dataset, test_set, cache_dir=os.environ.get("HF_HOME"))
    _, poison_loader, _ = make_dataloaders(
        test, attack.trigger.trigger_description, dspec.target_label, tokenizer,
        "prefix", cfg.stage2.max_length, cfg.stage2.batch_size, attack=attack)
    texts = poison_loader.dataset.texts
    digest = hashlib.sha256("\n".join(texts).encode()).hexdigest()
    return poison_loader, dspec, len(texts), digest


def main() -> None:
    import torch
    from transformers import AutoTokenizer
    from counterfactual_kd.config import load_config
    from counterfactual_kd.evaluation.evaluator import _probe_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", default="bert")
    ap.add_argument("--dataset", default="sst2")
    ap.add_argument("--kd", nargs="+", default=["logit", "feature"])
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--attacks", nargs="*", default=None,
                    help="triggers to score (default: the battery config's attacks)")
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs", "tifs"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    runs = battery_runs(args.runs, args.fam, args.dataset)
    if args.seeds:
        runs = {s: d for s, d in runs.items() if s in args.seeds}
    base_cfg = load_config(os.path.join(ROOT, "configs", "tifs", f"tifs_b_{args.fam}_{args.dataset}.yaml"))
    test_set = base_cfg.stage2.test_sets.get(args.dataset) if base_cfg.stage2.test_sets else None
    attacks = args.attacks or list(base_cfg.attacks)
    roster = {s: students(d, args.kd) for s, d in sorted(runs.items())}
    any_path = next(st["path"] for sts in roster.values() for st in sts)
    tok = AutoTokenizer.from_pretrained(any_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    out = {"meta": {"fam": args.fam, "dataset": args.dataset, "test_set": test_set,
                    "kd": args.kd, "dtype": "fp32", "device": "cpu",
                    "runs": {str(s): os.path.basename(d) for s, d in runs.items()},
                    "time_utc": time.strftime("%FT%TZ", time.gmtime())},
           "inputs": {}, "students": {}, "hits": {}}
    t_all = time.time()
    for atk in attacks:
        loader, dspec, n, digest = triggered_loader(atk, args.dataset, test_set, tok, base_cfg)
        out["inputs"][atk] = {"n": n, "sha256": digest}
        out["hits"][atk] = {}
        for seed, sts in roster.items():
            for st in sts:
                key = f"{st['kd']}/{st['tag']}/{st['role']}/s{seed}"
                out["students"][key] = {"seed": seed, **st,
                                        "path": os.path.relpath(st["path"], ROOT)}
                t0 = time.time()
                res, _ = _probe_model(st["path"], dspec.num_labels, torch.float32,
                                      tok.pad_token_id, torch.device("cpu"),
                                      {"trig": loader}, dspec.target_label)
                hits = [int(h) for h in res["trig"][1]]
                out["hits"][atk][key] = hits
                print(f"{atk:8s} {key:60s} {sum(hits) / len(hits):.4f} {time.time() - t0:.0f}s",
                      flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out + ".tmp", "w") as f:
            json.dump(out, f)
        os.replace(args.out + ".tmp", args.out)
    print(f"done in {time.time() - t_all:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
