#!/usr/bin/env python3
"""Write the A15 configs (docs/tifs_addendum.md A15 and its 10/26 19:55 e2
amendment) into configs/tifs/tifs_x15_*.yaml.

Every config is a frozen config (the template) with a listed set of
changes, so all other settings stay those of the frozen battery, A/A,
replicate or A13 run it extends. ``configs_manifest.json`` (next to this
script) records each run's template and the changed keys.

  b1/b2 builds  tifs_x15_bt_<fam>_sst2_t<seed>   Stage 0 only: the clean twin
                rebuilt under results/tifs/teachers_x15 with every epoch saved
                (stage0.save_every_epoch, ob_ckpt_pid, build_kinds [clean])
  b1            tifs_x15_b1_<fam>_sst2_e<e>_s<seed>   C students distilled from
                the rebuilt twin's epoch e (= the P teachers' saved epoch);
                the frozen P students are reused, not retrained
                (stage1.poisoned_student_run)
  b2            tifs_x15_b2_bert_ref_t<t>_e<e>   C' students of reference t at
                epoch e (seed 42); tifs_x15_b2_bert_ref_t43_e5 also holds A2's
                seed-42 cell (P' from tifs_rep_bert_atba_t43_s42);
                tifs_x15_b2_a2_t43_e5_s<43|44>   A2 seeds 43/44
  b3            tifs_x15_b3_pcb_<rate>_s<seed>   (conditional, queue-disabled)
                A13 G3 C students at epoch 4
  c             tifs_x15_ct_gpt2_atba_sst2_t42   Stage 0 only: GPT-2-xl SST-2
                ATBA teacher (suffix), every epoch saved;
                tifs_x15_c_gpt2_atba_s<seed>     P and C students (C from the
                b1 GPT-2 twin at the P teacher's selected epoch, read at run time)
  d             tifs_x15_d_bert_agnews_s<45|46>  BERT AG News battery seeds 45/46
  e             tifs_x15_e_bert_agnews_a<08|10>_s<seed>   alpha 0.8 / 1.0
  e2            tifs_x15_e2_gpt2_sst2_a10_e<2|5>_s<seed>  GPT-2 SST-2 alpha 1.0,
                P from the frozen P teachers, C from the b1 twin at the P epoch

    python scripts/tifs/aux/g8-a15/make_configs.py [--check]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
CFG = os.path.join(ROOT, "configs", "tifs")
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs_manifest.json")
X15 = "results/tifs/teachers_x15"
FROZEN_T = "results/tifs/teachers"
BUILD = {"save_every_epoch": True, "ob_ckpt_pid": True}
PROV1 = {"record_provenance": True}
PROV2 = {"record_provenance": True}

# P teachers' saved (OpenBackdoor-selected) epochs, SST-2 base trainer,
# verified on 10/26 from the build logs (recomputed with OB's rule) and
# results/tifs/addendum/aux/g6-review0/corpus/selection_builds.csv.
P_EPOCH = {
    ("bert", 42): {"atba": 1, "addsent": 1, "badnets": 4, "synbkd": 5},
    ("gpt2", 42): {"badnets": 2, "addsent": 5},
    ("qwen3", 42): {"badnets": 3, "addsent": 3, "synbkd": 3},
    ("bert_atba", 43): {"atba": 5},
}
# Frozen clean twins' selected epochs (same sources).
C_EPOCH = {("bert", 42): 3, ("gpt2", 42): 2, ("qwen3", 42): 1,
           ("bert", 43): 1, ("bert", 44): 3, ("bert", 45): 5}
FROZEN_ATTACK_ORDER = ["badnets", "addsent", "synbkd", "sos", "ep", "atba"]


def load(name: str) -> dict:
    with open(os.path.join(CFG, f"{name}.yaml")) as f:
        return yaml.safe_load(f)


def merge(base: dict, changes: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in changes.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "attack_overrides":
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def flat(d: dict, prefix="") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and k != "attack_overrides" and v:
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


SPECS: dict[str, dict] = {}


def add(run_id: str, template: str, part: str, desc: str, **changes):
    assert run_id not in SPECS, run_id
    SPECS[run_id] = {"template": template, "part": part, "description": desc,
                     "changes": changes}


def ordered(attacks):
    return [a for a in FROZEN_ATTACK_ORDER if a in attacks]


# ── b1 / b2 / c teacher builds (Stage 0 only, new root) ─────────────
add("tifs_x15_bt_bert_sst2_t42", "tifs_t_bert", "b1",
    "A15-b1 (post hoc): BERT-large SST-2 clean twin, teacher seed 42, rebuilt with the "
    "frozen settings and every epoch saved under teachers_x15 (Stage 0 only).",
    datasets=["sst2"], attacks=["badnets"], teacher_seed=42,
    stage0={**BUILD, "build_kinds": ["clean"]}, output={"teacher_dir": X15})
add("tifs_x15_bt_gpt2_sst2_t42", "tifs_t_gpt2", "b1",
    "A15-b1 (post hoc): GPT-2-xl SST-2 clean twin, teacher seed 42, rebuilt with every "
    "epoch saved under teachers_x15 (Stage 0 only).",
    datasets=["sst2"], attacks=["badnets"], teacher_seed=42,
    stage0={**BUILD, "build_kinds": ["clean"]}, output={"teacher_dir": X15})
add("tifs_x15_bt_qwen3_sst2_t42", "tifs_t_qwen3", "b1",
    "A15-b1 (post hoc): Qwen3-1.7B SST-2 clean twin, teacher seed 42, rebuilt with every "
    "epoch saved under teachers_x15 (Stage 0 only).",
    datasets=["sst2"], attacks=["badnets"], teacher_seed=42,
    stage0={**BUILD, "build_kinds": ["clean"]}, output={"teacher_dir": X15})
REF_TEMPLATE = {43: "tifs_aa_bert_clean_t43", 44: "tifs_aa_bert_clean_t44",
                45: "tifs_x13_bert_ref_t45", 46: "tifs_x13_bert_ref_t46"}
for t in (43, 44, 45, 46):
    add(f"tifs_x15_bt_bert_sst2_t{t}", REF_TEMPLATE[t], "b2",
        f"A15-b2 (post hoc): BERT-large SST-2 clean reference teacher, teacher seed {t}, "
        "rebuilt with the frozen settings and every epoch saved under teachers_x15 "
        "(Stage 0 only).",
        datasets=["sst2"], attacks=["badnets"], teacher_seed=t,
        stage0={**BUILD, "build_kinds": ["clean"]}, output={"teacher_dir": X15})
add("tifs_x15_ct_gpt2_atba_sst2_t42", "tifs_t_gpt2", "c",
    "A15-c (post hoc): GPT-2-xl SST-2 ATBA teacher with the registry's GPT-2 trigger at the "
    "suffix (base trainer, frozen GPT-2 teacher settings), teacher seed 42, every epoch "
    "saved under teachers_x15 (Stage 0 only; its clean twin is the b1 rebuild).",
    datasets=["sst2"], attacks=["atba"], teacher_seed=42,
    stage0={**BUILD, "build_kinds": ["poisoned"], "trigger_position": "suffix"},
    attack_overrides={"atba": {"position": "suffix"}}, output={"teacher_dir": X15})

# ── b1 students: C at each P teacher's epoch ────────────────────────
B1 = {"bert": ("tifs_b_bert_sst2", {42: "tifs_b_bert_sst2", 43: "tifs_b_bert_sst2_s43",
                                    44: "tifs_b_bert_sst2_s44"}),
      "gpt2": ("tifs_b_gpt2_sst2", {42: "tifs_b_gpt2_sst2", 43: "tifs_b_gpt2_sst2_s43",
                                    44: "tifs_b_gpt2_sst2_s44"}),
      "qwen3": ("tifs_b_qwen3_sst2", {42: "tifs_b_qwen3_sst2", 43: "tifs_b_qwen3_sst2_s43",
                                      44: "tifs_b_qwen3_sst2_s44"})}
B1_RUNS: list[str] = []
for fam, (_, frozen) in B1.items():
    by_epoch: dict[int, list[str]] = {}
    for a, e in P_EPOCH[(fam, 42)].items():
        if e != C_EPOCH[(fam, 42)]:          # e(P) == frozen C epoch: the frozen C is used
            by_epoch.setdefault(e, []).append(a)
    for e, attacks in sorted(by_epoch.items()):
        for s in (42, 43, 44):
            kd = ["logit", "feature"] if (fam == "bert" and s == 42) else ["logit"]
            rid = f"tifs_x15_b1_{fam}_sst2_e{e}_s{s}"
            add(rid, frozen[s], "b1",
                f"A15-b1 (post hoc): {fam} SST-2, C students distilled from the rebuilt clean "
                f"twin's epoch {e} (teachers_x15), the saved epoch of the P teacher(s) "
                f"{', '.join(ordered(attacks))}; P students reused from {frozen[s]}; "
                f"student seed {s}, teachers and corpus seed 42.",
                attacks=ordered(attacks), kd_types=kd,
                stage1={"clean_teacher_dir": X15, "clean_teacher_epoch": e,
                        "poisoned_student_run": frozen[s], **PROV1},
                stage2=PROV2)
            B1_RUNS.append(rid)

# ── b2 students: references at epochs 1, 4, 5; A2 at epoch 5 ────────
REF_P = {43: "tifs_aa_bert_clean_t43", 44: "tifs_aa_bert_clean_t44",
         45: "tifs_x13_bert_ref_t45", 46: "tifs_x13_bert_ref_t46"}
B2_EPOCHS = sorted({P_EPOCH[("bert", 42)][a] for a in ("atba", "addsent", "badnets", "synbkd")})
assert B2_EPOCHS == [1, 4, 5]
for t in (43, 44, 45, 46):
    for e in B2_EPOCHS:
        if t == 43 and e == C_EPOCH[("bert", 43)]:
            continue                          # the frozen C'(t43) is already epoch 1
        rid = f"tifs_x15_b2_bert_ref_t{t}_e{e}"
        attacks, run = ["badnets"], REF_P[t]
        extra = ""
        if t == 43 and e == P_EPOCH[("bert_atba", 43)]["atba"]:
            attacks = ["badnets", "atba"]
            run = {"badnets": REF_P[43], "atba": "tifs_rep_bert_atba_t43_s42"}
            extra = (" Also A2 seed 42: P' (ATBA, teacher seed 43, epoch 5) reused from "
                     "tifs_rep_bert_atba_t43_s42 against the same C'.")
        add(rid, REF_TEMPLATE[t], "b2",
            f"A15-b2 (post hoc): BERT SST-2 clean reference t{t}, C' student distilled from "
            f"the rebuilt reference teacher's epoch {e} (teachers_x15); the BadNets P "
            f"student (auxiliary) reused from {REF_P[t]}; corpus and student seed 42.{extra}",
            datasets=["sst2"], attacks=attacks,
            stage1={"clean_teacher_dir": X15, "clean_teacher_epoch": e,
                    "poisoned_student_run": run, **PROV1},
            stage2=PROV2)
for s in (43, 44):
    add(f"tifs_x15_b2_a2_t43_e5_s{s}", f"tifs_rep_bert_atba_t43_s{s}", "b2",
        f"A15-b2 (post hoc): A2 at teacher seed 43, C' distilled from the rebuilt t43 "
        f"teacher's epoch 5 (the P' teacher's saved epoch); P' reused from "
        f"tifs_rep_bert_atba_t43_s{s}; student seed {s}, corpus seed 42.",
        stage1={"clean_teacher_dir": X15, "clean_teacher_epoch": 5,
                "poisoned_student_run": f"tifs_rep_bert_atba_t43_s{s}", **PROV1},
        stage2=PROV2)

# ── b3 (conditional; jobs disabled in the queue) ────────────────────
for r in ("r0001", "r0003", "r0010", "r0030"):
    for s in (42, 43, 44):
        add(f"tifs_x15_b3_pcb_{r}_s{s}", f"tifs_x13_pcb_{r}_s{s}", "b3",
            f"A15-b3 (post hoc, conditional): A13 G3 {r}, C student distilled from the "
            f"rebuilt seed-42 clean twin's epoch 4 (the BadNets P teacher's saved epoch); "
            f"P reused from tifs_x13_pcb_{r}_s{s}; student seed {s}.",
            stage1={"clean_teacher_dir": X15, "clean_teacher_epoch": 4,
                    "poisoned_student_run": f"tifs_x13_pcb_{r}_s{s}", **PROV1},
            stage2=PROV2)

# ── c students ──────────────────────────────────────────────────────
for s in (42, 43, 44):
    add(f"tifs_x15_c_gpt2_atba_s{s}", B1["gpt2"][1][s], "c",
        f"A15-c (post hoc): GPT-2 SST-2 ATBA (registry GPT-2 trigger, suffix), P from the "
        f"teachers_x15 ATBA teacher (OB-selected epoch), C from the b1 GPT-2 twin rebuild "
        f"at the same epoch (read at run time); student seed {s}, teachers and corpus "
        f"seed 42, logit KD.",
        attacks=["atba"], kd_types=["logit"],
        stage0={"trigger_position": "suffix"},
        attack_overrides={"atba": {"position": "suffix"}},
        stage1={"clean_teacher_epoch": "poisoned", **PROV1}, stage2=PROV2,
        output={"teacher_dir": X15})

# ── d: BERT AG News seeds 45/46 ─────────────────────────────────────
for s in (45, 46):
    add(f"tifs_x15_d_bert_agnews_s{s}", "tifs_b_bert_agnews_s43", "d",
        f"A15-d (post hoc): bert/agnews clean KD corpus, student seed {s}, teachers and "
        f"corpus seed 42: P per attack and the shared clean students, logit KD.",
        seeds=[s], stage1=PROV1, stage2=PROV2)

# ── e: alpha 0.8 / 1.0, BERT AG News ────────────────────────────────
AG = {42: "tifs_b_bert_agnews", 43: "tifs_b_bert_agnews_s43", 44: "tifs_b_bert_agnews_s44"}
for a, tag in ((0.8, "a08"), (1.0, "a10")):
    for s in (42, 43, 44):
        add(f"tifs_x15_e_bert_agnews_{tag}_s{s}", AG[s], "e",
            f"A15-e (post hoc): bert/agnews logit KD at alpha {a} (T 4), P_BadNets, P_ATBA "
            f"and the shared C, student seed {s}, teachers and corpus seed 42.",
            attacks=["badnets", "atba"], kd_types=["logit"],
            teacher_seed=42, corpus_seed=42,
            stage1={"alpha": a, **PROV1}, stage2=PROV2)

# ── e2: alpha 1.0, GPT-2 SST-2 (amendment 10/26 19:55) ──────────────
for attack in ("badnets", "addsent"):
    e = P_EPOCH[("gpt2", 42)][attack]
    for s in (42, 43, 44):
        add(f"tifs_x15_e2_gpt2_sst2_a10_e{e}_s{s}", B1["gpt2"][1][s], "e2",
            f"A15-e2 (post hoc): gpt2/sst2 logit KD at alpha 1.0 (T 4); P_{attack} distilled "
            f"from the frozen P teacher (saved epoch {e}), C from the b1 GPT-2 twin rebuild "
            f"at epoch {e}; student seed {s}, teachers and corpus seed 42.",
            attacks=[attack], kd_types=["logit"], teacher_seed=42, corpus_seed=42,
            stage1={"alpha": 1.0, "clean_teacher_dir": X15, "clean_teacher_epoch": e,
                    **PROV1}, stage2=PROV2)


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
    # Seed-42 student runs pin the seeds explicitly only where the template did.
    return cfg, changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="only verify that the files on disk match")
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
        print(f"{len(SPECS)} A15 configs; stale or missing: {stale or 'none'}")
        return 1 if stale else 0
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {len(SPECS)} A15 configs and {os.path.relpath(MANIFEST, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
