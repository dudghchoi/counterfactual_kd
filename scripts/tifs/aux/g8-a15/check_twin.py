#!/usr/bin/env python3
"""A15-b: does a rebuilt clean twin reproduce the frozen one? (CPU only)

The A15 twins are rebuilt with every epoch saved under
results/tifs/teachers_x15/ (stage0.save_every_epoch). This compares the
rebuilt OpenBackdoor-selected checkpoint with the frozen twin under
results/tifs/teachers/:

  * canonical tensor SHA-256 (provenance.tensors_sha256) and weight-file
    SHA-256 of the frozen root, the rebuilt root and the rebuilt epoch the
    rule selected;
  * the selected epoch against the frozen build's (from its build log,
    recomputed with the same rule, and ``--expect-epoch`` when given);
  * the rebuilt per-epoch dev accuracies against the frozen log's;
  * when the weights differ: the maximum absolute difference per tensor,
    overall, and against every rebuilt epoch.

docs/tifs_addendum.md A15-b1: if they differ, report the difference and
proceed, so the exit status is 0 whenever the comparison ran (1 = could
not compare). Writes one JSON.

    python scripts/tifs/aux/g8-a15/check_twin.py \\
        --rebuilt results/tifs/teachers_x15/clean/ob_bert-large_sst-2_seed42 \\
        --frozen results/tifs/teachers/clean/ob_bert-large_sst-2_seed42 \\
        --frozen-log logs/tifs/tifs_t_bert.log --expect-epoch 3 \\
        --out runs/tifs/tifs_x15_twincheck/ob_bert-large_sst-2_seed42.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU only, whatever the lane sets
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

SEG = re.compile(r"\[OB-adapter\]\s+(\w+) \| (\w+)\s*\| (\S+) \| (\w+) \| seed=(\d+)")
EPOCH = re.compile(r"Epoch: (\d+), avg loss")
DEV = re.compile(r"accuracy on (dev-[a-z]+): ([0-9.]+)")


def log_segments(log: str) -> dict:
    """{(kind, attack, model, dataset, seed): [completed segments]} of a build
    log; a segment is a list of {epoch, dev-clean, dev-poison}."""
    out, cur, key = {}, None, None
    with open(log, errors="replace") as f:
        for line in f:
            m = SEG.search(line)
            if m:
                key, cur = m.groups(), []
                continue
            if cur is None:
                continue
            if (m := EPOCH.search(line)):
                cur.append({"epoch": int(m.group(1))})
            elif (m := DEV.search(line)) and cur:
                cur[-1][m.group(1)] = float(m.group(2))
            elif "Training finished" in line:
                out.setdefault(key, []).append(cur)
                cur = None
    return out


def ob_rule(epochs: list[dict]) -> int | None:
    best, sel = 0.0, None
    for e in epochs:
        scores = [v for k, v in e.items() if k.startswith("dev-")]
        score = sum(scores) / len(scores)
        if score > best:
            best, sel = score, e["epoch"]
    return sel


def max_abs_diff(a_dir: str, b_dir: str) -> dict:
    import glob
    from safetensors import safe_open
    import torch

    def tensors(d):
        where = {}
        for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
            with safe_open(f, framework="pt") as sf:
                for k in sf.keys():
                    where[k] = f
        return where

    ta, tb = tensors(a_dir), tensors(b_dir)
    per, overall, n_diff = {}, 0.0, 0
    for k in sorted(set(ta) | set(tb)):
        if k not in ta or k not in tb:
            per[k] = "missing in " + ("frozen" if k not in ta else "rebuilt")
            continue
        with safe_open(ta[k], framework="pt") as fa, safe_open(tb[k], framework="pt") as fb:
            x, y = fa.get_tensor(k), fb.get_tensor(k)
        if x.shape != y.shape:
            per[k] = f"shape {tuple(x.shape)} vs {tuple(y.shape)}"
            continue
        d = float((x.float() - y.float()).abs().max()) if x.numel() else 0.0
        if d > 0:
            n_diff += 1
            per[k] = d
        overall = max(overall, d)
    return {"max_abs_diff": overall, "n_tensors": len(set(ta) | set(tb)),
            "n_tensors_different": n_diff,
            "largest": dict(sorted(((k, v) for k, v in per.items() if isinstance(v, float)),
                                   key=lambda kv: -kv[1])[:10]),
            "structural": {k: v for k, v in per.items() if isinstance(v, str)}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rebuilt", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--frozen-log", help="build log of the frozen twin")
    ap.add_argument("--expect-epoch", type=int,
                    help="frozen OB-selected epoch verified before A15 (selection_builds.csv)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from counterfactual_kd.provenance import combined_sha256, tensors_sha256, weights_sha256
    from counterfactual_kd.registry import teacher_epoch_path, teacher_epochs_record

    rec = teacher_epochs_record(args.rebuilt)
    sel = rec["ob_selected_epoch"]
    meta = json.load(open(os.path.join(args.frozen, "ob_meta.json")))
    key = ("clean" if meta["type"] == "clean_teacher" else "poisoned", meta["attack"],
           meta["model_path"], meta["dataset"], str(meta["seed"]))
    from counterfactual_kd.provenance import file_sha256 as _fsha
    out = {"rebuilt": args.rebuilt, "frozen": args.frozen, "time": time.strftime("%FT%T%z"),
           "script_sha256": _fsha(os.path.abspath(__file__)),
           "rebuilt_ob_selected_epoch": sel,
           "rebuilt_accuracy_only_epoch": rec.get("accuracy_only_epoch"),
           "rebuilt_root_matches_selected_epoch": rec.get("root_matches_selected_epoch"),
           "expect_epoch": args.expect_epoch}

    if args.frozen_log and os.path.exists(args.frozen_log):
        segs = log_segments(args.frozen_log).get(key, [])
        out["frozen_log"] = args.frozen_log
        out["frozen_log_segments"] = len(segs)
        if segs:
            seg = segs[-1]
            out["frozen_log_selected_epoch"] = ob_rule(seg)
            rebuilt_dev = {r["epoch"]: {k: v["accuracy"] for k, v in r["dev"].items()}
                           for r in rec["epochs"]}
            out["dev_by_epoch"] = [
                {"epoch": e["epoch"],
                 "frozen": {k: v for k, v in e.items() if k.startswith("dev-")},
                 "rebuilt": rebuilt_dev.get(e["epoch"])} for e in seg]
            out["dev_identical"] = all(
                d["rebuilt"] is not None
                and all(abs(d["rebuilt"].get(k, -1) - v) < 1e-12 for k, v in d["frozen"].items())
                for d in out["dev_by_epoch"])
    frozen_epoch = out.get("frozen_log_selected_epoch", args.expect_epoch)
    out["selected_epoch_matches_frozen"] = (sel == frozen_epoch) if frozen_epoch else None
    if args.expect_epoch is not None and "frozen_log_selected_epoch" in out:
        out["log_matches_expect_epoch"] = out["frozen_log_selected_epoch"] == args.expect_epoch

    frozen_files = weights_sha256(args.frozen)
    sel_dir = teacher_epoch_path(args.rebuilt, sel)
    out["frozen_weights_sha256"] = frozen_files
    out["rebuilt_root_weights_sha256"] = weights_sha256(args.rebuilt)
    out["rebuilt_selected_weights_sha256"] = weights_sha256(sel_dir)
    out["files_identical"] = (combined_sha256(frozen_files)
                              == combined_sha256(out["rebuilt_selected_weights_sha256"]))
    t_frozen = tensors_sha256(args.frozen)
    t_sel = tensors_sha256(sel_dir)
    out["frozen_tensors_sha256"] = t_frozen
    out["rebuilt_selected_tensors_sha256"] = t_sel
    out["rebuilt_selected_tensors_sha256_recorded"] = next(
        (r["tensors_sha256"] for r in rec["epochs"] if r["epoch"] == sel), None)
    out["record_consistent"] = out["rebuilt_selected_tensors_sha256_recorded"] == t_sel
    out["identical"] = t_frozen == t_sel
    if not out["identical"]:
        out["diff_vs_selected"] = max_abs_diff(args.frozen, sel_dir)
        out["max_abs_diff_by_epoch"] = {
            r["epoch"]: max_abs_diff(args.frozen, teacher_epoch_path(args.rebuilt, r["epoch"]))
            ["max_abs_diff"] for r in rec["epochs"]}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = f"{args.out}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, args.out)
    print(json.dumps({k: out.get(k) for k in (
        "rebuilt_ob_selected_epoch", "frozen_log_selected_epoch", "expect_epoch",
        "selected_epoch_matches_frozen", "dev_identical", "identical", "files_identical",
        "record_consistent")}, indent=1))
    if not out["identical"]:
        print(f"weights differ: max |diff| {out['diff_vs_selected']['max_abs_diff']:.3e}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        print(f"check_twin: could not compare: {e}", file=sys.stderr)
        sys.exit(1)
