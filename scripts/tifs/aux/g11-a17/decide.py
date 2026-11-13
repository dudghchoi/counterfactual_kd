#!/usr/bin/env python3
"""A17 CPU decisions and checks (queue v10), fixed in docs/tifs_addendum.md A17 before any A17 data.

  --what agob   Conditional C at the AG News twin's OB-selected checkpoint (the reading the
                frozen procedure gives). Needed only when that epoch differs from the P
                teachers' saved epoch (5): exit 0 = run tifs_x17_agob_*; exit 3 = not needed
                (the lane marks the job failed, which blocks them).
  --what sbob   Conditional C at the frozen GPT-2 SST-2 twin (OB-selected epoch 2). Needed only
                when the SynBkd teacher's saved epoch is not 2: exit 0 = run tifs_x17_sbob_*;
                exit 3 = not needed.
  --what repro  Report only (exit 0): the A17 AG News twin's epoch-1 dev numbers against the
                10/21 build of the same twin, stopped after epoch 1 (logs/tifs/tifs_t_gpt2.log,
                second attempt). Equal numbers show the rebuild is the twin the 10/21 run was
                training; a difference is reported and does not block.

Writes runs/tifs/tifs_x17_<what>/decision.json; exit 1 on error.

    python scripts/tifs/aux/g11-a17/decide.py --what agob|sbob|repro
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

NOT_NEEDED = 3
AG_TWIN = os.path.join("results", "tifs", "teachers_x17", "clean", "ob_gpt2-xl_agnews_seed42")
SB_TEACHER = os.path.join("results", "tifs", "teachers_x17", "poisoned", "ob_synbkd_gpt2-xl_sst-2_seed42")
FROZEN_SST2_TWIN_EPOCH = 2      # frozen GPT-2 SST-2 twin, OB-selected (A15 C_EPOCH; b1 twincheck)
LOG_1021 = os.path.join("logs", "tifs", "tifs_t_gpt2.log")
SEGMENT = re.compile(r"\[OB-adapter\]\s+clean \| badnets\s+\| gpt2-xl \| agnews \| seed=42")
DEV_CLEAN = re.compile(r"accuracy on dev-clean: ([0-9.]+)")


def _epochs(path):
    with open(os.path.join(path, "epochs.json")) as f:
        return json.load(f)


def repro() -> dict:
    with open(LOG_1021, errors="replace") as f:
        text = f.read()
    segs = [m.end() for m in SEGMENT.finditer(text)]
    # the last segment is the 10/21 second attempt (the first attempt ran out of memory at step 0)
    found = DEV_CLEAN.search(text, segs[-1]) if segs else None
    logged = float(found.group(1)) if found else None
    E = _epochs(AG_TWIN)
    ep1 = next(e for e in E["epochs"] if int(e["epoch"]) == 1)
    new = ep1["dev"]["dev-clean"]["accuracy"]
    return {"log": LOG_1021, "log_segments": len(segs), "logged_epoch1_dev_clean": logged,
            "rebuilt_epoch1_dev_clean": new,
            "identical": logged is not None and abs(new - logged) < 1e-12,
            "rebuilt_ob_selected_epoch": E["ob_selected_epoch"],
            "note": "report only; a difference does not block (docs/tifs_addendum.md A17)"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--what", required=True, choices=["agob", "sbob", "repro"])
    args = ap.parse_args()
    import make_configs as MC
    if args.what == "agob":
        sel = int(_epochs(AG_TWIN)["ob_selected_epoch"])
        need = sel != MC.E_AG
        res = {"twin": AG_TWIN, "twin_ob_selected_epoch": sel, "p_saved_epoch": MC.E_AG,
               "run_conditional": need}
    elif args.what == "sbob":
        sel = int(_epochs(SB_TEACHER)["ob_selected_epoch"])
        need = sel != FROZEN_SST2_TWIN_EPOCH
        res = {"teacher": SB_TEACHER, "p_saved_epoch": sel,
               "frozen_twin_epoch": FROZEN_SST2_TWIN_EPOCH, "run_conditional": need}
    else:
        res, need = repro(), True
    res.update(what=args.what, time=time.strftime("%FT%T%z"))
    out = os.path.join("runs", "tifs", f"tifs_x17_{args.what}", "decision.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"A17 {args.what}: {json.dumps({k: v for k, v in res.items() if k != 'time'})}")
    return 0 if need else NOT_NEEDED


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"decide: error: {e}", file=sys.stderr)
        sys.exit(1)
