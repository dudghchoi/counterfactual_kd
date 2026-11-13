#!/usr/bin/env python3
"""Pre-registered conditional experiments (docs/tifs_plan.md §0.3).

Run as a GPU 1 queue job; the exit status is the decision, read by
lane.py (0 = ok = run the conditional experiment, anything else = failed,
which blocks it).

  pc_alpha1   run tifs_pc_bert_kl10a1 (alpha = 1, SST-2) only if the
              positive control tifs_pc_bert_kl10 fails its pass criterion:
              P - C (logit KD, Tango) POSITIVE at delta = 0.05 on at least
              one dataset. Exit 0 = failed control, run the fallback;
              exit 3 = control passed, fallback not needed.

    python scripts/tifs/decide_conditionals.py pc_alpha1
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from counterfactual_kd.analysis.tifs import paired_contrast  # noqa: E402

NOT_NEEDED = 3


def pc_alpha1(runs_dir: str) -> int:
    import analyze
    cells = [c for c in analyze.load_cells(runs_dir)
             if c["run"] == "tifs_pc_bert_kl10" and c["kd"] == "logit" and not c["single"]]
    if not cells:
        print("no evaluated tifs_pc_bert_kl10 logit cells")
        return 2
    passed = []
    for c in sorted(cells, key=lambda c: c["dataset"]):
        pc = paired_contrast(c["P"], c["C"], (0.05,))
        print(f"{c['dataset']}: P-C = {pc['k']:+.4f} [{pc['lo']:+.4f}, {pc['hi']:+.4f}] "
              f"-> {pc['verdict@0.05']}")
        passed.append(pc["verdict@0.05"] == "POSITIVE")
    if any(passed):
        print("positive control passed: alpha=1 fallback not needed")
        return NOT_NEEDED
    print("positive control failed on every dataset: run tifs_pc_bert_kl10a1")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("decision", choices=["pc_alpha1"])
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs", "tifs"))
    args = ap.parse_args()
    sys.exit({"pc_alpha1": pc_alpha1}[args.decision](args.runs))


if __name__ == "__main__":
    main()
