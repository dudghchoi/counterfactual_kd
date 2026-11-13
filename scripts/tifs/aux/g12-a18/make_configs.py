#!/usr/bin/env python3
"""Write the A18 configs (docs/tifs_addendum.md A18) into configs/tifs/tifs_x18_*.yaml.

A18 adds GPT-2 AG News ATBA, the GPT-2 counterpart of A15-c (GPT-2 SST-2 ATBA), so GPT-2 has
the same attacks on both datasets. Every config is a frozen config (the template) with a listed
set of changes, merged as A15 and A17 do (scripts/tifs/aux/g8-a15/make_configs.py), and the
changes are A15-c's changes with the dataset set to AG News:

  build     tifs_x18_at_gpt2_atba_agnews_t42   Stage 0 only: GPT-2-xl AG News ATBA teacher
            (registry GPT-2 AG News trigger = ATBA paper Table 4, suffix; base trainer, frozen
            GPT-2 teacher settings, teacher seed 42), every epoch saved under
            results/tifs/teachers_x17, the root of its clean twin (the A17 AG News twin)
  students  tifs_x18_a_gpt2_agnews_s<seed>     P ATBA from that teacher (OB-selected epoch), C
            from the A17 twin at the same epoch (read at run time); student seed 42/43/44,
            teachers and corpus seed 42, logit KD

``configs_manifest.json`` (next to this script) records each run's template and changed keys.

    python scripts/tifs/aux/g12-a18/make_configs.py [--check]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
CFG = os.path.join(ROOT, "configs", "tifs")
MANIFEST = os.path.join(HERE, "configs_manifest.json")
X17 = "results/tifs/teachers_x17"
BUILD = {"save_every_epoch": True, "ob_ckpt_pid": True}
PROV = {"record_provenance": True}
SUFFIX = {"atba": {"position": "suffix"}}

# Frozen GPT-2 battery configs per student seed (the A17 ag templates).
SST2 = {42: "tifs_b_gpt2_sst2", 43: "tifs_b_gpt2_sst2_s43", 44: "tifs_b_gpt2_sst2_s44"}

_spec = importlib.util.spec_from_file_location(
    "a15_make_configs", os.path.join(ROOT, "scripts", "tifs", "aux", "g8-a15", "make_configs.py"))
_a15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_a15)
load, merge, flat = _a15.load, _a15.merge, _a15.flat

SPECS: dict[str, dict] = {}


def add(run_id: str, template: str, part: str, desc: str, **changes):
    assert run_id not in SPECS, run_id
    SPECS[run_id] = {"template": template, "part": part, "description": desc, "changes": changes}


# ── teacher build (Stage 0 only, the A17 twin's root) ───────────────
add("tifs_x18_at_gpt2_atba_agnews_t42", "tifs_t_gpt2", "a",
    "A18 (post hoc): GPT-2-xl AG News ATBA teacher with the registry's GPT-2 AG News trigger "
    "(ATBA paper Table 4) at the suffix (base trainer, frozen GPT-2 teacher settings), teacher "
    "seed 42, every epoch saved under teachers_x17 (Stage 0 only; its clean twin is the A17 "
    "AG News twin).",
    datasets=["agnews"], attacks=["atba"], teacher_seed=42,
    stage0={**BUILD, "build_kinds": ["poisoned"], "trigger_position": "suffix"},
    attack_overrides=SUFFIX, output={"teacher_dir": X17})

# ── students ────────────────────────────────────────────────────────
for s in (42, 43, 44):
    add(f"tifs_x18_a_gpt2_agnews_s{s}", SST2[s], "a",
        f"A18 (post hoc): GPT-2 AG News ATBA (registry GPT-2 AG News trigger, suffix), P from "
        f"the teachers_x17 ATBA teacher (OB-selected epoch), C from the A17 AG News twin at the "
        f"same epoch (read at run time); student seed {s}, teachers and corpus seed 42, logit KD.",
        datasets=["agnews"], attacks=["atba"], kd_types=["logit"], teacher_seed=42,
        corpus_seed=42, stage0={"trigger_position": "suffix"}, attack_overrides=SUFFIX,
        stage1={"clean_teacher_epoch": "poisoned", **PROV}, stage2=PROV,
        output={"teacher_dir": X17})


def render(run_id: str) -> tuple[dict, list[str]]:
    spec = SPECS[run_id]
    base = load(spec["template"])
    cfg = merge(base, spec["changes"])
    cfg["name"] = run_id
    cfg["description"] = spec["description"]
    before, after = flat(base), flat(cfg)
    changed = sorted(k for k in set(before) | set(after)
                     if before.get(k, "<unset>") != after.get(k, "<unset>")
                     and k not in ("name", "description"))
    return cfg, changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only verify that the files on disk match")
    args = ap.parse_args()
    sys.path.insert(0, ROOT)
    from counterfactual_kd.config import _from_dict
    manifest, stale = {}, []
    for rid in SPECS:
        cfg, changed = render(rid)
        _from_dict(cfg).validate()
        text = yaml.safe_dump(cfg, sort_keys=False)
        path = os.path.join(CFG, f"{rid}.yaml")
        manifest[rid] = {"part": SPECS[rid]["part"], "template": SPECS[rid]["template"],
                         "changed": {k: flat(cfg).get(k, "<unset>") for k in changed}}
        if args.check:
            if not os.path.exists(path) or open(path).read() != text:
                stale.append(rid)
            continue
        if os.path.exists(path) and open(path).read() == text:
            continue
        with open(path, "w") as f:
            f.write(text)
    if args.check:
        print(f"{len(SPECS)} A18 configs; stale or missing: {stale or 'none'}")
        return 1 if stale else 0
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {len(SPECS)} A18 configs and {os.path.relpath(MANIFEST, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
