#!/usr/bin/env python3
"""Precompute SynBkd paraphrases (SCPN template -1) for whole datasets.

Covers every text that teacher poisoning, KD-corpus contamination and
evaluation can ask for: the KD/eval loader texts (SST-2 GLUE train +
validation, AG News train + test). OpenBackdoor's AG News rows are the
same texts re-joined at the first ". "; the Stage 0 adapter maps them back
to these keys, so they need no separate entries.

Speed: the Stanford parser runs once per chunk (one JVM per chunk instead
of per sentence) and SCPN beam search runs on CPU, one thread per worker.
Workers write shards under ``<out>/shards/``; a rerun skips texts already
present, and ``--merge`` folds shards into the table that
``counterfactual_kd.triggers.syntactic`` reads.

    python scripts/tifs/build_synbkd_table.py --datasets sst2 --workers 24
    python scripts/tifs/build_synbkd_table.py --merge
    python scripts/tifs/build_synbkd_table.py --datasets sst2 agnews --wait   # queue step
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Process

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from counterfactual_kd.triggers.syntactic import (  # noqa: E402
    DEFAULT_OA_DATA, DEFAULT_TABLE, load_table,
)

CHUNK = 100


def dataset_texts(name: str) -> list[str]:
    from datasets import load_dataset
    if name == "sst2":
        ds = load_dataset("glue", "sst2")
        texts = [ex["sentence"] for ex in ds["train"]] + [ex["sentence"] for ex in ds["validation"]]
    elif name == "agnews":
        ds = load_dataset("ag_news")
        texts = [ex["text"] for ex in ds["train"]] + [ex["text"] for ex in ds["test"]]
    elif name == "sst2_test":
        # The evaluation texts of stage2.test_sets {sst2: sst2_test}, keyed
        # exactly as the evaluator passes them to the trigger.
        from counterfactual_kd.data.loaders import load_sst2_test
        texts = [r["sentence"] for r in load_sst2_test(cache_dir=os.environ.get("HF_HOME"))]
    else:
        raise ValueError(name)
    seen, out = set(), []
    for t in texts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def worker(idx: int, texts: list[str], shard: str) -> None:
    import argparse as _ap

    import torch
    torch.set_num_threads(1)
    torch.serialization.add_safe_globals([_ap.Namespace])
    import OpenAttack as oa
    from nltk.parse.stanford import StanfordParser
    from OpenAttack.tags import TAG_English
    from OpenAttack.text_process.constituency_parser import ConstituencyParser

    data = os.environ.get("CKD_OPENATTACK_DATA", DEFAULT_OA_DATA)
    oa.DataManager.set_path(data)
    sp_dir = os.path.join(data, "TProcess.StanfordParser")
    stanford = StanfordParser(
        path_to_jar=os.path.join(sp_dir, "stanford-parser.jar"),
        path_to_models_jar=os.path.join(sp_dir, "stanford-parser-3.9.2-models.jar"),
        model_path=os.path.join(sp_dir, "englishPCFG.ser.gz"),
        java_options="-mx2g",
    )

    class Precomputed(ConstituencyParser):
        TAGS = {TAG_English}

        def __init__(self):
            self.trees: dict[str, str] = {}

        def parse(self, sentence: str) -> str:
            if sentence in self.trees:
                return self.trees[sentence]
            return str(list(stanford.raw_parse(sentence))[0])

    parser = Precomputed()
    scpn = oa.attackers.SCPNAttacker(device=torch.device("cpu"), parser=parser)
    template = [scpn.templates[-1]]

    t0, done = time.time(), 0
    with open(shard, "a") as out:
        for start in range(0, len(texts), CHUNK):
            chunk = texts[start:start + CHUNK]
            try:
                trees = [str(list(it)[0]) for it in stanford.raw_parse_sents(chunk)]
                parser.trees = dict(zip(chunk, trees))
            except Exception:
                parser.trees = {}          # fall back to per-sentence parsing
            for text in chunk:
                try:
                    para, ok = scpn.gen_paraphrase(text, template)[0].strip(), True
                except Exception:
                    para, ok = text, False
                out.write(json.dumps({"text": text, "para": para, "ok": ok}) + "\n")
            out.flush()
            done += len(chunk)
            rate = done / (time.time() - t0)
            print(f"[w{idx}] {done}/{len(texts)}  {rate:.2f}/s  eta {(len(texts)-done)/max(rate,1e-9)/3600:.1f}h",
                  flush=True)


def existing(out_dir: str, table: str) -> set[str]:
    have = set(load_table(table))
    for shard in glob.glob(os.path.join(out_dir, "shards", "*.jsonl")):
        have |= set(load_table(shard))
    return have


def merge(out_dir: str, table: str) -> None:
    merged = {}
    for path in [table] + sorted(glob.glob(os.path.join(out_dir, "shards", "*.jsonl"))):
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    merged[rec["text"]] = rec
    tmp = table + ".tmp"
    with open(tmp, "w") as f:
        for rec in merged.values():
            f.write(json.dumps(rec) + "\n")
    os.replace(tmp, table)
    ok = sum(r["ok"] for r in merged.values())
    print(f"merged {len(merged)} texts ({ok} paraphrased, {len(merged)-ok} kept as-is) -> {table}")


def other_builder_running() -> bool:
    me = os.getpid()
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) in (me, os.getppid()):
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = f.read().decode(errors="replace").split("\0")
        except OSError:
            continue
        if any(a.endswith("build_synbkd_table.py") for a in argv):
            return True
    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["sst2"])
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--limit", type=int, default=None, help="first N texts per dataset (smoke)")
    p.add_argument("--table", default=os.path.join(ROOT, DEFAULT_TABLE))
    p.add_argument("--merge", action="store_true")
    p.add_argument("--wait", action="store_true",
                   help="first wait for any other builder process to finish")
    args = p.parse_args()
    if args.wait:
        while other_builder_running():
            time.sleep(120)
    out_dir = os.path.dirname(args.table)
    os.makedirs(os.path.join(out_dir, "shards"), exist_ok=True)
    if args.merge:
        merge(out_dir, args.table)
        return

    have = existing(out_dir, args.table)
    for name in args.datasets:
        texts = dataset_texts(name)[: args.limit]
        todo = [t for t in texts if t not in have]
        print(f"{name}: {len(texts)} texts, {len(todo)} to paraphrase", flush=True)
        if not todo:
            continue
        # Interleave so every worker sees a mix of short and long texts.
        parts = [todo[i::args.workers] for i in range(args.workers)]
        stamp = time.strftime("%Y%m%d%H%M%S")
        procs = [Process(target=worker, args=(i, part, os.path.join(
            out_dir, "shards", f"{name}_{stamp}_{i:02d}.jsonl"))) for i, part in enumerate(parts) if part]
        for pr in procs:
            pr.start()
        for pr in procs:
            pr.join()
    merge(out_dir, args.table)


if __name__ == "__main__":
    main()
