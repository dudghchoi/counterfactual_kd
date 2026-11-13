#!/usr/bin/env python3
"""A14 item 11, independent check of the reconstructed selected epoch (CPU fp32).

A14 post hoc analysis; changes no primary verdict, battery decision or the
paper-wide Holm.

For the clean twins whose OpenBackdoor-selected epoch differs from the
accuracy-only epoch (selection_builds.csv), score the saved checkpoint on the
OpenBackdoor dev-clean split (datasets/SentimentAnalysis/SST-2/dev.tsv, text
stripped as OpenBackdoor's SST2Processor does; datasets/TextClassification/
agnews/dev.csv, headline + ' ' + body with backslashes replaced) and find the
logged epoch(s) whose dev-clean accuracy it reproduces. The dev-poison split
is not re-scored (its trigger positions come from the build's RNG and the
file on disk is overwritten by every build). CPU fp32 (training-time eval ran
on GPU), so a difference of a row or two is possible; the table gives the
distance in rows to every epoch.

Output: <out>/selection_devcheck.csv
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import csv  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as G  # noqa: E402

MAX_LEN = 512      # stage0.max_len of the teacher configs (OpenBackdoor victim max_len)
BATCH = 16         # OpenBackdoor eval batch = train batch_size


def dev_split(ds):
    if ds == "sst2":
        with open(os.path.join(G.ROOT, "datasets", "SentimentAnalysis", "SST-2", "dev.tsv")) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        return [r["sentence"].strip() for r in rows], [int(r["label"]) for r in rows]
    texts, labels = [], []
    with open(os.path.join(G.ROOT, "datasets", "TextClassification", "agnews", "dev.csv"), encoding="utf8") as f:
        for label, head, body in csv.reader(f, delimiter=","):
            texts.append(head.replace("\\", " ") + " " + body.replace("\\", " "))
            labels.append(int(label) - 1)
    return texts, labels


def main():
    args = G.parse_args("A14 item 11: dev-clean accuracy of the saved clean twins (CPU fp32)",
                        extra=lambda ap: (ap.add_argument("--threads", type=int, default=8),
                                          ap.add_argument("--all-differing", action="store_true",
                                                          help="also AG News builds (12,000 dev rows each)")))
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(args.threads)
    builds = G.read_csv(G.out("selection_builds.csv"))
    epochs = G.read_csv(G.out("selection_epochs.csv"))
    todo = [b for b in builds if b["kind"] == "clean" and b["trainer"] == "base"
            and b["selected_is_acc_only"] == "False" and (b["dataset"] == "sst2" or args.all_differing)]
    rows = []
    for b in todo:
        t0 = time.time()
        path = os.path.join(G.ROOT, b["path"])
        texts, labels = dev_split(b["dataset"])
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.float32)
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tok.pad_token_id
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, len(texts), BATCH):
                enc = tok(texts[i:i + BATCH], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
                pred = model(**enc).logits.argmax(-1).tolist()
                correct += sum(int(p == y) for p, y in zip(pred, labels[i:i + BATCH]))
        n = len(texts)
        acc = correct / n
        eps = [e for e in epochs if e["teacher"] == b["teacher"]]
        dist = {int(e["epoch"]): round(abs(float(e["dev_acc"]) - acc) * n) for e in eps}
        best = min(dist.values())
        rows.append({"teacher": b["teacher"], "dataset": b["dataset"], "n_dev_clean": n, "correct_cpu_fp32": correct,
                     "dev_acc_cpu_fp32": acc, "logged_dev_acc_by_epoch": b["dev_acc_by_epoch"],
                     "rows_from_each_epoch": " ".join(f"{e}:{d}" for e, d in sorted(dist.items())),
                     "closest_epochs": ",".join(str(e) for e, d in sorted(dist.items()) if d == best),
                     "closest_distance_rows": best, "selected_epoch_reconstructed": b["selected_epoch"],
                     "acc_only_epoch": b["acc_only_epoch"],
                     "reconstructed_epoch_among_closest": str(b["selected_epoch"]) in
                     [str(e) for e, d in dist.items() if d == best],
                     "acc_only_epoch_distance_rows": dist.get(int(b["acc_only_epoch"])),
                     "seconds": round(time.time() - t0, 1)})
        print(rows[-1], flush=True)
        del model
    G.write_csv(rows, G.out("selection_devcheck.csv"),
                "independent check (CPU fp32) of the reconstructed selected epoch: the saved clean-twin checkpoint's "
                "accuracy on the OpenBackdoor dev-clean split, its distance in rows to each logged epoch's dev-clean "
                "accuracy, and whether the reconstructed OpenBackdoor-rule epoch is among the closest; only clean "
                "twins whose selected epoch differs from the accuracy-only epoch")


if __name__ == "__main__":
    main()
