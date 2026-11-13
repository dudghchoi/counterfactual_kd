#!/usr/bin/env python3
"""A8 checks of the hidden-state KD students (docs/tifs_addendum.md A8), CPU only.

Fixed before any A8 output. For every tifs_hid_* run whose Stage 1 is done:

1. Training: every student's Stage 1 epoch losses are finite and its last
   epoch's mean loss is below its first's (logs/tifs/<run>.log).
2. Function: every hidden student's clean accuracy on the GLUE SST-2
   validation split (872 sentences, not the Stage 2 test set) is above the
   split's majority-class share + FUNCTION_MARGIN (A7's floor).
3. Manipulation: on the same sentences, M = the hidden objective's IR term
   (13 masked MSE pairs, embeddings included, against the clean-base
   teacher) at its least-squares shared W with bias. The seed's hidden
   clean-base C student must have M at most MAX_RATIO times its logit
   clean-base C student's. (On the 1,821 test sentences the logit C students
   of seeds 42-44 differ by at most 0.16% in M (SD 0.09%), and the CE-only
   student is 2.7% above them; the frozen 'feature' C student is 0.02%
   below logit C.)

Every seed is checked; the seed-42 run is the A8 pilot. A seed passes when
all three hold. The results are reported (addendum.py hidden_check.csv);
seed admission uses the Stage 2 rule. Nothing here reads a triggered input
or an ASR.

    HF_HOME=... HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    python scripts/tifs/hidden_check.py [--runs runs/tifs] [--out results/tifs/addendum/hidden_check.json]
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # GPU 0 trains, GPU 1 is shared: CPU only.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

FUNCTION_MARGIN = 0.10      # as scripts/tifs/addendum.py (A7)
MAX_RATIO = 0.95            # manipulation check: M_hidden <= 0.95 M_logit
RUN_RX = re.compile(r"^tifs_hid_bert_sst2_s(?P<seed>\d+)$")
ADAPTER_RX = re.compile(r"\[KD-adapter\] (?P<kd>\S+) \| (?P<side>poisoned|clean|none) teacher")
EPOCH_RX = re.compile(r"KD Epoch (?P<e>\d+)/(?P<n>\d+): loss=(?P<loss>\S+)")


def epoch_losses(log_text: str) -> list[dict]:
    """Per training start, in log order: {"kd", "side", "losses", "n"} from
    the ``[KD-adapter] <kd> | <side> teacher`` header and its epoch lines.
    Every (re)start prints a new header, so a stopped and resumed student
    leaves one incomplete block and one complete block."""
    blocks = []
    for line in log_text.splitlines():
        m = ADAPTER_RX.search(line)
        if m:
            blocks.append({"kd": m["kd"], "side": m["side"], "losses": [], "n": None})
            continue
        m = EPOCH_RX.search(line)
        if m and blocks:
            b = blocks[-1]
            if int(m["e"]) == 1 and b["losses"]:        # a restart without a header
                b = dict(b, losses=[])
                blocks.append(b)
            b["losses"].append(float(m["loss"]))
            b["n"] = int(m["n"])
    return blocks


def training_ok(blocks, n_students) -> bool:
    """At least ``n_students`` complete blocks, each finite and decreasing.
    Incomplete blocks (a job stopped and resumed) are left out."""
    done = [b for b in blocks if b["n"] is not None and len(b["losses"]) == b["n"]]
    return len(done) >= n_students and all(
        all(math.isfinite(x) for x in b["losses"]) and b["losses"][-1] < b["losses"][0]
        for b in done)


def ls_residual(XtX, XtY, Syy, n_tok, d_t, ridge=1e-6):
    """M and its per-pair terms for one W shared by all pairs (rows of X
    carry a trailing 1 for the bias). XtX, XtY: per-pair float64 sums."""
    import torch
    A = sum(XtX)
    B = sum(XtY)
    W = torch.linalg.solve(A + ridge * torch.eye(A.shape[0], dtype=A.dtype), B)
    pairs = [(float(syy) - 2 * float((W * xty).sum()) + float((W * (xtx @ W)).sum())) / (n_tok * d_t)
             for xtx, xty, syy in zip(XtX, XtY, Syy)]
    return sum(pairs), pairs


def student_dirs(run_dir):
    """{tag: (role, path)} of a run's hidden students; role P or C."""
    out = {}
    for d in sorted(glob.glob(os.path.join(run_dir, "artifacts", "students", "*_hidden_*", "*_student"))):
        tag = os.path.basename(os.path.dirname(d))
        if not os.path.exists(os.path.join(d, "config.json")):
            continue
        out[tag] = ("C" if tag.startswith("clean-") else "P", d)
    return out


def logit_clean_base(runs, seed):
    run = "tifs_b_bert_sst2" + ("" if seed == 42 else f"_s{seed}")
    hits = sorted(glob.glob(os.path.join(runs, run, "artifacts", "students",
                                         "clean-base_logit_*", "clean_student")))
    return hits[0] if hits else None


def teacher_of(student_dir):
    meta = json.load(open(os.path.join(student_dir, "kd_meta.json")))
    return os.path.normpath(os.path.join(os.path.realpath(student_dir), meta["teacher_ref"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs", "tifs"))
    ap.add_argument("--logs", default=os.path.join(ROOT, "logs", "tifs"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tifs", "addendum", "hidden_check.json"))
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from counterfactual_kd.data.dataset import make_dataloader

    torch.set_num_threads(args.threads)
    runs = {}
    for d in sorted(glob.glob(os.path.join(args.runs, "tifs_hid_bert_sst2_s*"))):
        m = RUN_RX.match(os.path.basename(d))
        if m and (args.seeds is None or int(m["seed"]) in args.seeds):
            runs[int(m["seed"])] = d
    val = load_dataset("glue", "sst2", split="validation")
    texts, labels = list(val["sentence"]), list(val["label"])
    majority = max(labels.count(0), labels.count(1)) / len(labels)
    floor = majority + FUNCTION_MARGIN
    out = {"split": "glue/sst2 validation", "n": len(texts), "majority": majority, "floor": floor,
           "max_ratio": MAX_RATIO, "seeds": {}}
    if os.path.exists(args.out):
        prev = json.load(open(args.out))
        out["seeds"] = prev.get("seeds", {})
        if "M_init" in prev:
            out["M_init"] = prev["M_init"]

    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    texts = [texts[i] for i in order]
    labels = [labels[i] for i in order]
    cache = {}

    def batches_for(tok_path):
        tok = AutoTokenizer.from_pretrained(tok_path)
        return list(make_dataloader(texts, labels, tok, 128, 32, shuffle=False))

    def teacher_states(path):
        if path not in cache:
            tb = batches_for(path)
            t = AutoModelForSequenceClassification.from_pretrained(
                path, num_labels=2, output_hidden_states=True).eval()
            n_t = t.config.num_hidden_layers
            states = []
            with torch.no_grad():
                for b in tb:
                    o = t(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
                    msk = b["attention_mask"].bool()
                    states.append([o.hidden_states[j][msk].double() for j in range(n_t + 1)])
            cache[path] = (tb, states, n_t, t.config.hidden_size)
            del t
        return cache[path]

    def score(student_path, t_path=None):
        """Validation accuracy, and M against ``t_path`` if given."""
        sb = batches_for(student_path)
        s = AutoModelForSequenceClassification.from_pretrained(
            student_path, num_labels=2, output_hidden_states=True).eval()
        acc, res = 0, {}
        stats = None
        if t_path:
            tb, ts, n_t, d_t = teacher_states(t_path)
            n_s = s.config.num_hidden_layers
            idx = [m * n_t // n_s for m in range(n_s + 1)]
            d = s.config.hidden_size + 1
            stats = ([torch.zeros(d, d, dtype=torch.float64) for _ in idx],
                     [torch.zeros(d, d_t, dtype=torch.float64) for _ in idx], [0.0] * len(idx))
            n_tok = 0
        with torch.no_grad():
            for bi, b in enumerate(sb):
                o = s(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
                acc += int((o.logits.argmax(-1) == b["labels"]).sum())
                if stats is None:
                    continue
                if not (torch.equal(b["input_ids"], tb[bi]["input_ids"])
                        and torch.equal(b["attention_mask"], tb[bi]["attention_mask"])):
                    raise ValueError("student and teacher tokenizations differ")
                msk = b["attention_mask"].bool()
                n_tok += int(msk.sum())
                for k, j in enumerate(idx):
                    x = o.hidden_states[k][msk].double()
                    x1 = torch.cat([x, torch.ones(x.shape[0], 1, dtype=torch.float64)], 1)
                    y = ts[bi][j]
                    stats[0][k] += x1.T @ x1
                    stats[1][k] += x1.T @ y
                    stats[2][k] += float((y * y).sum())
        res["val_acc"] = acc / len(texts)
        if stats is not None:
            res["M"], res["M_pairs"] = ls_residual(*stats, n_tok, d_t)
        del s
        return res

    for seed, run_dir in sorted(runs.items()):
        students = student_dirs(run_dir)
        roles = sorted(r for r, _ in students.values())
        if roles.count("P") < 5 or roles.count("C") < 2:
            print(f"s{seed}: Stage 1 not done ({len(students)} students)")
            out["seeds"][str(seed)] = {"run": os.path.basename(run_dir), "status": "stage1 not done"}
            continue
        log = os.path.join(args.logs, f"{os.path.basename(run_dir)}.log")
        blocks = epoch_losses(open(log, errors="replace").read()) if os.path.exists(log) else []
        rec = {"run": os.path.basename(run_dir), "training_ok": training_ok(blocks, len(students)),
               "losses": [b["losses"] for b in blocks if b["kd"] == "hidden" and len(b["losses"]) == b["n"]],
               "students": {}}
        hid_c = next(p for t, (r, p) in students.items() if t.startswith("clean-base_"))
        t_path = teacher_of(hid_c)
        for tag, (role, path) in students.items():
            r = score(path, t_path if path == hid_c else None)
            rec["students"][tag] = {"role": role, "val_acc": r["val_acc"], "functional": r["val_acc"] > floor}
            if path == hid_c:
                rec["M_hidden"], rec["M_pairs_hidden"] = r["M"], r["M_pairs"]
            print(f"s{seed} {tag}: val acc {r['val_acc']:.4f}" + (f", M {r['M']:.4f}" if "M" in r else ""), flush=True)
        lc = logit_clean_base(args.runs, seed)
        if lc is not None:
            r = score(lc, t_path)
            rec.update(logit_c=os.path.relpath(lc, ROOT), M_logit=r["M"], M_pairs_logit=r["M_pairs"],
                       logit_c_val_acc=r["val_acc"])
            rec["ratio"] = rec["M_hidden"] / rec["M_logit"]
            rec["manipulation_ok"] = rec["ratio"] <= MAX_RATIO
            print(f"s{seed} logit C: M {r['M']:.4f}; ratio {rec['ratio']:.4f}", flush=True)
        rec["functional_ok"] = all(v["functional"] for v in rec["students"].values())
        rec["pass"] = bool(rec["training_ok"] and rec["functional_ok"] and rec.get("manipulation_ok"))
        out["seeds"][str(seed)] = rec
        print(f"s{seed}: training {rec['training_ok']}, functional {rec['functional_ok']}, "
              f"manipulation {rec.get('manipulation_ok')} -> pass {rec['pass']}", flush=True)
    if "M_init" not in out and runs:
        any_c = next((p for d in runs.values() for t, (r, p) in student_dirs(d).items()
                      if t.startswith("clean-base_")), None)
        if any_c is not None:
            out["M_init"] = score("bert-base-uncased", teacher_of(any_c))["M"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
