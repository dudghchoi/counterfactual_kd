#!/usr/bin/env python3
"""Write the A17 configs (docs/tifs_addendum.md A17) into configs/tifs/tifs_x17_*.yaml.

Every config is a frozen config (the template) with a listed set of changes, so all other
settings stay those of the frozen GPT-2 battery or teacher build it extends (the frozen
GPT-2 SST-2 and AG News configs differ only in ``datasets``, as the frozen BERT and Qwen3
pairs do). ``configs_manifest.json`` (next to this script) records each run's template and
the changed keys. The template merge is A15's (scripts/tifs/aux/g8-a15/make_configs.py).

  ag builds  tifs_x17_gt_gpt2_agnews_t42        Stage 0 only: the GPT-2-xl AG News clean twin
             (teacher seed 42), every epoch saved under results/tifs/teachers_x17
  sb build   tifs_x17_st_gpt2_synbkd_sst2_t42   Stage 0 only: the GPT-2-xl SST-2 SynBkd teacher
             (teacher seed 42), every epoch saved under results/tifs/teachers_x17
  ag         tifs_x17_ag_gpt2_agnews_s<seed>    P BadNets / AddSent from the 10/21 poisoned
             AG News teachers (results/tifs/teachers, saved epoch 5), the shared C from the
             new twin's epoch 5; seed 42 also trains the CE-only R
  agob       tifs_x17_agob_gpt2_agnews_s<seed>  (conditional) C from the new twin's
             OB-selected checkpoint; P reused from the ag run
  sb         tifs_x17_sb_gpt2_sst2_s<seed>      P SynBkd from the new teacher (OB-selected
             epoch), C from the A15-b1 GPT-2 twin rebuild at the same epoch (read at run time)
  sbob       tifs_x17_sbob_gpt2_sst2_s<seed>    (conditional) C from the frozen GPT-2 SST-2
             twin (OB-selected, epoch 2); P reused from the sb run

    python scripts/tifs/aux/g11-a17/make_configs.py [--check]
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
X15 = "results/tifs/teachers_x15"
FROZEN_T = "results/tifs/teachers"
BUILD = {"save_every_epoch": True, "ob_ckpt_pid": True}
PROV = {"record_provenance": True}

# Saved (OpenBackdoor-selected) epoch of the two 10/21 GPT-2-xl AG News poisoned teachers,
# recomputed with OB's rule from the logged dev numbers (logs/tifs/tifs_t_gpt2.log; g6-review0
# corpus/selection_builds.csv): BadNets 5, AddSent 5. Neither was built with every epoch saved,
# so the epoch is fixed here, not read at run time.
P_EPOCH_AG = {"badnets": 5, "addsent": 5}
assert len(set(P_EPOCH_AG.values())) == 1   # one shared C per student seed
E_AG = P_EPOCH_AG["badnets"]

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


# ── teacher builds (Stage 0 only, new root) ─────────────────────────
add("tifs_x17_gt_gpt2_agnews_t42", "tifs_t_gpt2", "ag",
    "A17 (post hoc): GPT-2-xl AG News clean twin, teacher seed 42, frozen GPT-2 teacher "
    "settings, every epoch saved under teachers_x17 (Stage 0 only). Its 10/21 build was "
    "stopped after epoch 1 by the v2 scope decision.",
    datasets=["agnews"], attacks=["badnets"], teacher_seed=42,
    stage0={**BUILD, "build_kinds": ["clean"]}, output={"teacher_dir": X17})
add("tifs_x17_st_gpt2_synbkd_sst2_t42", "tifs_t_gpt2", "sb",
    "A17 (post hoc): GPT-2-xl SST-2 SynBkd teacher, teacher seed 42, frozen GPT-2 teacher "
    "settings and the shared SynBkd table, every epoch saved under teachers_x17 (Stage 0 "
    "only; its clean twin is the A15-b1 rebuild).",
    datasets=["sst2"], attacks=["synbkd"], teacher_seed=42,
    stage0={**BUILD, "build_kinds": ["poisoned"]}, output={"teacher_dir": X17})

# ── ag: GPT-2 AG News BadNets / AddSent ─────────────────────────────
for s in (42, 43, 44):
    r_note = " Seed 42 also trains the CE-only R." if s == 42 else ""
    add(f"tifs_x17_ag_gpt2_agnews_s{s}", SST2[s], "ag",
        f"A17 (post hoc): gpt2/agnews clean KD corpus, P BadNets / AddSent from the 10/21 "
        f"poisoned teachers (saved epoch {E_AG}), the shared C from the A17 twin's epoch {E_AG} "
        f"(teachers_x17); student seed {s}, teachers and corpus seed 42, logit KD.{r_note}",
        datasets=["agnews"], teacher_seed=42, corpus_seed=42,
        stage1={"clean_teacher_dir": X17, "clean_teacher_epoch": E_AG, **PROV},
        stage2=PROV)
    add(f"tifs_x17_agob_gpt2_agnews_s{s}", SST2[s], "agob",
        f"A17 (post hoc, conditional): gpt2/agnews, C from the A17 twin's OB-selected "
        f"checkpoint (teachers_x17 root); P reused from tifs_x17_ag_gpt2_agnews_s{s}; student "
        f"seed {s}, teachers and corpus seed 42, logit KD.",
        datasets=["agnews"], kd_types=["logit"], teacher_seed=42, corpus_seed=42,
        stage1={"clean_teacher_dir": X17,
                "poisoned_student_run": f"tifs_x17_ag_gpt2_agnews_s{s}", **PROV},
        stage2=PROV)

# ── sb: GPT-2 SST-2 SynBkd ──────────────────────────────────────────
for s in (42, 43, 44):
    add(f"tifs_x17_sb_gpt2_sst2_s{s}", SST2[s], "sb",
        f"A17 (post hoc): gpt2/sst2 SynBkd, P from the teachers_x17 SynBkd teacher (OB-selected "
        f"epoch), C from the A15-b1 GPT-2 twin rebuild (teachers_x15) at the same epoch (read at "
        f"run time); student seed {s}, teachers and corpus seed 42, logit KD.",
        attacks=["synbkd"], kd_types=["logit"], teacher_seed=42, corpus_seed=42,
        stage1={"clean_teacher_dir": X15, "clean_teacher_epoch": "poisoned", **PROV},
        stage2=PROV, output={"teacher_dir": X17})
    add(f"tifs_x17_sbob_gpt2_sst2_s{s}", SST2[s], "sbob",
        f"A17 (post hoc, conditional): gpt2/sst2 SynBkd, C from the frozen GPT-2 SST-2 twin "
        f"(results/tifs/teachers, OB-selected epoch 2); P reused from tifs_x17_sb_gpt2_sst2_s{s}; "
        f"student seed {s}, teachers and corpus seed 42, logit KD.",
        attacks=["synbkd"], kd_types=["logit"], teacher_seed=42, corpus_seed=42,
        stage1={"clean_teacher_dir": FROZEN_T,
                "poisoned_student_run": f"tifs_x17_sb_gpt2_sst2_s{s}", **PROV},
        stage2=PROV, output={"teacher_dir": X17})


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
        print(f"{len(SPECS)} A17 configs; stale or missing: {stale or 'none'}")
        return 1 if stale else 0
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {len(SPECS)} A17 configs and {os.path.relpath(MANIFEST, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
