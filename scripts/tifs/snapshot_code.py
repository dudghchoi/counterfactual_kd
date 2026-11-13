#!/usr/bin/env python3
"""Snapshot the code, configs and rule that the TIFS results depend on.

Writes runs/tifs/_code/<label>_<UTC time>.{diff,_untracked.tar.gz,sha256,json}:

* ``.diff``              tracked changes against HEAD
* ``_untracked.tar.gz``  untracked files in scope
* ``.sha256``            SHA-256 of every file in scope (tracked and not),
                         one ``<hash>  <path>`` line each, sorted by path
* ``.json``              base commit, file count and the SHA-256 of the
                         ``.sha256`` list itself: one hash that pins the
                         whole snapshot (what an external timestamp would
                         record)

With ``--freeze`` it then writes runs/tifs/_status/tifs_rule_frozen.ok,
which opens the Stage 2 queue (scripts/tifs/queue_plan.py RULE_FROZEN).

    python scripts/tifs/snapshot_code.py --label freeze --freeze --note "..."
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCOPE = ["counterfactual_kd", "scripts/tifs", "configs/tifs", "tests",
         "docs/tifs_plan.md", "docs/tifs_addendum.md",
         "results/tifs/structural_nulls.json"]
OUT = os.path.join(ROOT, "runs", "tifs", "_code")
STATUS = os.path.join(ROOT, "runs", "tifs", "_status")


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout


def files_in_scope() -> tuple[list[str], list[str]]:
    tracked = set(git("ls-files", "--", *SCOPE).split())
    untracked = set(git("ls-files", "--others", "--exclude-standard", "--", *SCOPE).split())
    # Ignored files in scope (results/ may be git-ignored) are kept too.
    for p in SCOPE:
        full = os.path.join(ROOT, p)
        if os.path.isfile(full) and p not in tracked:
            untracked.add(p)
    keep = lambda f: "__pycache__" not in f and not f.endswith(".pyc") and os.path.isfile(os.path.join(ROOT, f))  # noqa: E731
    return sorted(filter(keep, tracked)), sorted(filter(keep, untracked - tracked))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="code")
    ap.add_argument("--note", default="")
    ap.add_argument("--freeze", action="store_true",
                    help="write runs/tifs/_status/tifs_rule_frozen.ok afterwards")
    args = ap.parse_args()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = os.path.join(OUT, f"{args.label}_{stamp}")
    os.makedirs(OUT, exist_ok=True)
    tracked, untracked = files_in_scope()

    with open(base + ".diff", "w") as f:
        f.write(git("diff", "HEAD", "--", *SCOPE))
    with tarfile.open(base + "_untracked.tar.gz", "w:gz") as tar:
        for rel in untracked:
            tar.add(os.path.join(ROOT, rel), arcname=rel)
    lines = [f"{sha256(os.path.join(ROOT, rel))}  {rel}\n" for rel in sorted(tracked + untracked)]
    with open(base + ".sha256", "w") as f:
        f.writelines(lines)
    meta = {"snapshot": f"{args.label}_{stamp}", "time_utc": stamp,
            "base_commit": git("rev-parse", "--short", "HEAD").strip(),
            "scope": SCOPE, "n_files": len(lines), "n_untracked": len(untracked),
            "manifest_sha256": sha256(base + ".sha256"),
            "preds_files_in_runs_tifs": sum(
                1 for d, _, fs in os.walk(os.path.join(ROOT, "runs", "tifs"))
                for x in fs if x.endswith(".preds.json")),
            "note": args.note}
    with open(base + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
    if args.freeze:
        with open(os.path.join(STATUS, "tifs_rule_frozen.ok"), "w") as f:
            json.dump({"snapshot": meta["snapshot"], "manifest_sha256": meta["manifest_sha256"]}, f)
        with open(os.path.join(STATUS, "events.jsonl"), "a") as f:
            f.write(json.dumps({"job": "tifs_rule_frozen", "state": "ok",
                                "time": time.strftime("%F %T"),
                                "snapshot": meta["snapshot"],
                                "manifest_sha256": meta["manifest_sha256"]}) + "\n")
        print("rule frozen: Stage 2 queue open")


if __name__ == "__main__":
    main()
