#!/usr/bin/env python3
"""Build the Counterfactual-KD v17 ROPE dashboard from preserved run artifacts.

For every measured cell, recompute the verdict with the locked v17 system:
Bayesian ROPE on the matched-ATE Cohen's h (paired bootstrap, B=5000).

Reads each cell's per-sample prediction sidecar (``*.preds.json`` with
``preds_obs`` / ``preds_null`` 0/1 arrays over the same triggered samples)
and emits one row per cell to a single dashboard CSV.

Two discovery modes:

* Generic (default) — glob ``runs_base`` for *any* run directory that
  completed (has ``manifest.json``) and produced at least one per-cell
  prediction sidecar. This is what makes ``counterfactual_kd repro`` ->
  ``counterfactual_kd dashboard --runs-base <runs_dir>`` work out of the
  box for a fresh user's own run, whatever it happens to be named —
  attack/family/dataset/kd_type/seed are read straight off each per-cell
  result JSON, and group/corpus are derived from the run's own
  config.yaml (falling back to "custom").
* Legacy campaign (``--legacy-campaign`` / ``legacy_campaign=True``) —
  the original hardcoded ``RUN_SPECS`` globs that reproduce the bundled
  88-cell author campaign (``runs/unified_bert_clean_*`` etc). Nothing
  about that path has changed; it is now opt-in rather than the only path.

Usage:
    python scripts/build_rope_dashboard.py
    python scripts/build_rope_dashboard.py --legacy-campaign
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

from counterfactual_kd.metrics.cohens_h import rope_verdict
from counterfactual_kd.metrics.statistics import mcnemar_test

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(_PKG_ROOT, "results", "dashboard", "dashboard_rope.csv")

# (run-dir glob prefix, group, corpus_state, family_filter) — group slices
# §5.2 vs §5.3; family_filter (or None) keeps only matching families.
RUN_SPECS = [
    ("runs/unified_bert_clean_*",             "ablation", "clean",    None),
    ("runs/unified_bert_poisoned_badnets_*",  "ablation", "poison10", None),
    ("runs/unified_bert_poisoned_addsent_*",  "ablation", "poison10", None),
    ("runs/unified_gpt2_clean_*",             "ablation", "clean",    None),
    ("runs/unified_gpt2_poisoned_badnets_*",  "ablation", "poison10", None),
    ("runs/unified_gpt2_poisoned_addsent_*",  "ablation", "poison10", None),
    ("runs/unified_qwen3_clean_*",            "ablation", "clean",    None),
    ("runs/unified_qwen3_poisoned_badnets_*", "ablation", "poison10", None),
    # exclude the stuck GPU1 dir (..._addsent_gpu1_*); take the GPU2 run.
    ("runs/unified_qwen3_poisoned_addsent_2*", "ablation", "poison10", None),
    # ATBA case study (§5.2). tier2 run holds BERT + an OLD broken GPT-2
    # (bf16-saturated, P=C=0.98); keep only its BERT cells. GPT-2 ATBA
    # comes from the dedicated retry/match runs below.
    ("runs/tier2_atba_kd_paper_hp_*",         "atba", "atba_paperhp",    "bert"),
    ("runs/unified_gpt2_atba_paper_hp_*",     "atba", "atba_fullcorpus", None),
    # exclude the LoRA variant (..._match_lora_*); timestamps start with 2.
    ("runs/unified_gpt2_atba_paper_match_2*", "atba", "atba_subset3054", None),
    # LoRA-faithful ATBA (paper regime: LoRA student, 3054 subset).
    # CLI-default run ids are date-prefixed: 20260610_..._paper_match_lora.
    ("runs/*unified_gpt2_atba_paper_match_lora*", "atba", "atba_subset3054_lora", None),
    # Poison-rate sweep (logit-only): KD corpus poisoned at 1% / 5%.
    # Same unified protocol/teachers as the clean & poison10 cells above.
    ("runs/*unified_rate01_bert_badnets*",   "ablation", "poison01", None),
    ("runs/*unified_rate01_bert_addsent*",   "ablation", "poison01", None),
    ("runs/*unified_rate01_gpt2_badnets*",   "ablation", "poison01", None),
    ("runs/*unified_rate01_gpt2_addsent*",   "ablation", "poison01", None),
    ("runs/*unified_rate01_qwen3_badnets*",  "ablation", "poison01", None),
    ("runs/*unified_rate01_qwen3_addsent*",  "ablation", "poison01", None),
    ("runs/*unified_rate05_bert_badnets*",   "ablation", "poison05", None),
    ("runs/*unified_rate05_bert_addsent*",   "ablation", "poison05", None),
    ("runs/*unified_rate05_gpt2_badnets*",   "ablation", "poison05", None),
    ("runs/*unified_rate05_gpt2_addsent*",   "ablation", "poison05", None),
    ("runs/*unified_rate05_qwen3_badnets*",  "ablation", "poison05", None),
    ("runs/*unified_rate05_qwen3_addsent*",  "ablation", "poison05", None),
]

BOOT_R = 5000
SEED = 0


def latest(prefix_glob: str) -> str | None:
    """Newest matching run dir that actually completed (has manifest.json).

    Killed/partial runs leave date-prefixed dirs without a manifest; they
    must never shadow an earlier complete run of the same config.

    ``prefix_glob`` must already be a fully-formed glob pattern (callers,
    e.g. ``build()``, are responsible for joining it with ``runs_base``) —
    this function does not re-join it against any base dir itself.
    """
    hits = sorted(
        d for d in glob.glob(prefix_glob)
        if os.path.exists(os.path.join(d, "manifest.json"))
    )
    return hits[-1] if hits else None


def cell_rows(run_dir: str, group: str, corpus: str, family_filter: str | None = None):
    rdir = os.path.join(run_dir, "artifacts/results")
    if not os.path.isdir(rdir):
        return
    for jpath in sorted(glob.glob(os.path.join(rdir, "*.json"))):
        base = os.path.basename(jpath)
        if base.endswith(".preds.json") or base.startswith("all_results"):
            continue
        preds_path = jpath[:-5] + ".preds.json"
        if not os.path.exists(preds_path):
            continue
        meta = json.load(open(jpath))
        if family_filter and meta.get("model_family") != family_filter:
            continue
        preds = json.load(open(preds_path))
        obs = preds.get("preds_obs")
        null = preds.get("preds_null")
        if not obs or not null:
            continue
        rv = rope_verdict(obs, null, R=BOOT_R, seed=SEED)
        mc = mcnemar_test(obs, null)  # descriptive paired p-value (not used for verdict)
        yield {
            "group": group,
            "corpus": corpus,
            "attack": meta.get("attack_method"),
            "family": meta.get("model_family"),
            "dataset": meta.get("dataset"),
            "kd_type": meta.get("kd_type"),
            "seed": meta.get("seed"),
            "P_trig": round(rv["p_trig"], 6),
            "C_trig": round(rv["c_trig"], 6),
            "k": round(rv["p_trig"] - rv["c_trig"], 6),
            "h": round(rv["h"], 6),
            "hdi_lo": round(rv["hdi_lo"], 6),
            "hdi_hi": round(rv["hdi_hi"], 6),
            "P_transfer": round(rv["p_transfer"], 4),
            "mcnemar_p": round(mc["p_value"], 6),
            "n": rv["n"],
            "ca_poisoned": meta.get("ca_poisoned"),
            "ca_clean": meta.get("ca_clean"),
            "teacher_asr": meta.get("teacher_asr"),
            "verdict": rv["verdict"],
        }


def _run_name_from_config(run_dir: str) -> str | None:
    """Best-effort read of a run's own ``config.yaml`` ``name`` (or ``user``)
    field, without requiring PyYAML — the dashboard must stay usable without
    the ``[repro]`` extras installed. Falls back to ``None`` if the file is
    missing, unreadable, or neither key is present.
    """
    cfg_path = os.path.join(run_dir, "config.yaml")
    try:
        with open(cfg_path) as f:
            lines = f.readlines()
    except OSError:
        return None
    for key in ("name", "user"):
        prefix = f"{key}:"
        for line in lines:
            if line.startswith(prefix):
                val = line[len(prefix):].strip().strip("\"'")
                if val:
                    return val
    return None


def discover_run_dirs(runs_base: str) -> list[str]:
    """Any completed run directory directly under ``runs_base`` — no
    campaign-glob knowledge required.

    A directory "counts" as a run if it has a ``manifest.json`` (completed
    at least one stage) and at least one per-cell prediction sidecar under
    ``artifacts/<results-like>/*.preds.json``. This is what lets a fresh
    user's own ``counterfactual_kd repro`` output (whatever it's named) feed
    the dashboard, instead of only the hardcoded author-campaign globs.
    """
    hits = []
    for d in sorted(glob.glob(os.path.join(runs_base, "*"))):
        if not os.path.isdir(d):
            continue
        if not os.path.exists(os.path.join(d, "manifest.json")):
            continue
        if not glob.glob(os.path.join(d, "artifacts", "*results*", "*.preds.json")):
            continue
        hits.append(d)
    return hits


def _legacy_rows(runs_base: str) -> list[dict]:
    """Original hardcoded-campaign discovery: iterate ``RUN_SPECS`` globs."""
    rows: list[dict] = []
    for prefix, group, corpus, fam in RUN_SPECS:
        run_dir = latest(os.path.join(runs_base, prefix))
        if run_dir is None:
            print(f"[WARN] no run dir for {prefix}", file=sys.stderr)
            continue
        n0 = len(rows)
        rows.extend(cell_rows(run_dir, group, corpus, fam))
        print(f"[ok] {os.path.basename(run_dir):50s} +{len(rows)-n0} cells")
    return rows


def _generic_rows(runs_base: str) -> list[dict]:
    """Generic discovery: any run dir under ``runs_base``, group/corpus
    derived per-run (attack/family/dataset/kd_type/seed come from each
    per-cell result JSON via the existing ``cell_rows()`` logic).
    """
    run_dirs = discover_run_dirs(runs_base)
    if not run_dirs:
        print(f"[WARN] no run dirs with manifest.json + *.preds.json found under {runs_base!r}",
              file=sys.stderr)
        return []
    rows: list[dict] = []
    for run_dir in run_dirs:
        group = _run_name_from_config(run_dir) or os.path.basename(run_dir)
        n0 = len(rows)
        rows.extend(cell_rows(run_dir, group=group, corpus="custom", family_filter=None))
        print(f"[ok] {os.path.basename(run_dir):50s} +{len(rows)-n0} cells")
    return rows


def build(out: str | None = None, runs_base: str = ".", legacy_campaign: bool = False) -> int:
    """Rebuild the dashboard CSV from run artifacts under ``runs_base``.

    A fresh clone ships the canonical ``results/dashboard/dashboard_rope.csv``;
    regenerate it here only after producing your own runs with the [repro] pipeline.

    By default this discovers ANY run directory under ``runs_base`` generically
    (see ``discover_run_dirs``/``_generic_rows``) — the path a fresh
    ``counterfactual_kd repro`` -> ``counterfactual_kd dashboard`` user hits.
    Pass ``legacy_campaign=True`` to instead reproduce the original 88-cell
    author campaign via the hardcoded ``RUN_SPECS`` globs.
    """
    out = out or DEFAULT_OUT
    rows = _legacy_rows(runs_base) if legacy_campaign else _generic_rows(runs_base)

    if not rows:
        print("[ERR] no cells collected — generate runs with the [repro] pipeline first, "
              "or use the bundled results/dashboard/dashboard_rope.csv", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cols = list(rows[0].keys())
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[done] {len(rows)} cells -> {out}")

    # quick group summary (groups are whatever cell_rows() was given — the
    # fixed "ablation"/"atba" campaign labels in legacy mode, or a per-run
    # config name / dir name in generic mode)
    from collections import Counter
    for g in sorted({r["group"] for r in rows}):
        sub = [r for r in rows if r["group"] == g]
        for corp in sorted({r["corpus"] for r in sub}):
            cc = [r for r in sub if r["corpus"] == corp]
            vc = Counter(r["verdict"] for r in cc)
            mh = sum(abs(r["h"]) for r in cc) / len(cc)
            mk = sum(r["k"] for r in cc) / len(cc)
            print(f"  {g}/{corp}: n={len(cc)} mean|h|={mh:.3f} mean_k={mk:+.3f} {dict(vc)}")
    return 0


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Rebuild the ROPE dashboard CSV from run artifacts.")
    ap.add_argument("--out", default=None, help=f"output CSV (default: {DEFAULT_OUT})")
    ap.add_argument("--runs-base", default=".", help="base dir prefixed to the run globs")
    ap.add_argument("--legacy-campaign", action="store_true",
                     help="use the original hardcoded RUN_SPECS globs (88-cell author "
                          "campaign) instead of generic run-dir discovery")
    a = ap.parse_args(argv)
    return build(out=a.out, runs_base=a.runs_base, legacy_campaign=a.legacy_campaign)


if __name__ == "__main__":
    raise SystemExit(main())
