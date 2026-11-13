#!/usr/bin/env python3
"""Generate the TIFS experiment configs (docs/tifs_plan.md §4.2) into configs/tifs/.

Hyperparameters are the published unified protocol, copied from the run
snapshots in BLADE v21 (``results/runs/unified_*/config.yaml``):

* teachers (Stage 0, OpenBackdoor): BERT-large / GPT-2-xl — batch 16, 5
  epochs, lr 2e-5, 3 warm-up epochs; Qwen3-1.7B — batch 8, 3 epochs,
  lr 1e-5, 1 warm-up epoch. Poison rate 0.1, max_len 512.
* students (Stage 1): T=4, alpha=0.5, 5 epochs, batch 16, max_length 128,
  no warm-up, bf16 teacher / fp32 student; lr 2e-5 (BERT), 5e-5 (GPT-2),
  1e-5 (Qwen3). The KD trainer now receives stage1.lr (earlier releases
  dropped it and trained every family at the 2e-5 default).

Every config sets ``tf32: true`` and evaluates the same control triggers.

Blocks (run ids = config names), v2 = the fixed audit run over an attack
battery (docs/tifs_plan.md §0.2). Student runs are seed 42 except the
student-seed replicates (BS, BRS).

  T   tifs_t_<fam>                 Stage-0-only teacher builds: clean +
                                   BadNets + AddSent, SST-2 + AG News
  TS  tifs_ts_<fam>                SynBkd teachers
  TB  tifs_tb_bert                 EP, SOS and ATBA teachers for BERT (EP/SOS
                                   pairs are linked from the tier-1 build,
                                   see link_reused_teachers.py)
  TR  tifs_tr_bert_<ds>            RIPPLe teachers, one run per dataset (new
                                   trainer, kept apart so a failure cannot
                                   block the battery)
  B   tifs_b_<fam>_<ds>            the battery on a clean KD corpus: every
                                   attack's P student, the trainer-matched C
                                   students and the CE-only reference R
  BR  tifs_br_bert_<ds>            the same for RIPPLe
  BS  tifs_b_<fam>_sst2_s<k>       student seeds 43/44 of the SST-2 logit
                                   batteries: P per attack and the shared C,
                                   teachers and corpus at seed 42 (§0.3)
  BRS tifs_br_bert_sst2_s<k>       the same for RIPPLe
  G   tifs_g_<fam>_<corpus>_s42    10% BadNets / AddSent-contaminated corpus
  PC  tifs_pc_<fam>_<variant>      positive control, label-kept trigger
                                   exposure; kl10a1 (alpha=1, SST-2) runs
                                   only if kl10 fails its pass criterion
  AA  tifs_aa_<fam>_<corpus>_t<k>  independent teacher seeds, corpus/student
                                   fixed at 42 (the audit's null)
  HID tifs_hid_bert_sst2_s<k>      logit KD + hidden-state distillation on the
                                   BERT SST-2 seed battery, student seeds
                                   42-44 (addendum A8)

Dropped from v1 (2026-10-21, to fit GPU 0 alone): seeds 43/44 of the grid,
HSOL, the CE-coefficient check, alpha=1 positive control, Qwen3-8B
teachers, Qwen3 feature KD, the Qwen3 AddSent-contaminated corpus and
positive control, Qwen3 SynBkd and A/A on AG News, and GPT-2 beyond the
SST-2 battery.
"""

from __future__ import annotations

import copy
import os

import yaml

from counterfactual_kd.triggers.registry import ATBA_TRIGGERS

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "tifs")

TEACHER_DIR = "results/tifs/teachers"
RUNS_DIR = "runs/tifs"

STAGE0_BASE = dict(batch_size=16, epochs=5, lr=2.0e-5, weight_decay=0.0,
                   warm_up_epochs=3, poison_rate=0.1, max_len=512)
STAGE0_QWEN = dict(STAGE0_BASE, batch_size=8, epochs=3, lr=1.0e-5,
                   warm_up_epochs=1)

FAMILIES = {
    "bert":  dict(pair=["bert-large-uncased", "bert-base-uncased"],
                  stage0=STAGE0_BASE, lr=2.0e-5, eval_bs=32),
    "gpt2":  dict(pair=["gpt2-xl", "gpt2"],
                  stage0=STAGE0_BASE, lr=5.0e-5, eval_bs=32),
    "qwen3": dict(pair=["Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B"],
                  stage0=STAGE0_QWEN, lr=1.0e-5, eval_bs=16),
}

# Evaluation-only controls, applied to every condition (E8). Token sets
# follow the published placebo design (repair/trigger_controls), with one
# change for the battery: its "mb" is an EP trigger word, so the rare
# unplanted set uses "xv" instead (a single BERT word piece, like the
# planted cf/mn/bb/tq, and absent from SST-2 and AG News). No word below
# is planted by any battery attack.
CONTROLS = [
    {"name": "rare_unplanted", "words": ["xv", "zx", "qk", "vy"]},
    {"name": "rare_unplanted_sub", "words": ["xv", "zx", "qk", "vy"],
     "mode": "substitute"},
    {"name": "common_neutral", "words": ["table", "window", "paper", "street"]},
    {"name": "loaded_positive",
     "words": ["great", "wonderful", "excellent", "brilliant"]},
    {"name": "sent_neutral_sub", "sentence": "I saw that old car",
     "mode": "substitute"},
    {"name": "sent_partial", "sentence": "I watch this"},
]

# SST-2 is evaluated on its labelled 1,821-sentence test set (912
# non-target inputs) instead of half of GLUE validation (211), which the
# OpenBackdoor teachers also use to pick their best epoch.
TEST_SETS = {"sst2": "sst2_test"}

CORPORA = {"clean": None, "badnets10": ("badnets", 0.1),
           "addsent10": ("addsent", 0.1)}

# The attack battery, grouped by how a trigger could ride on a clean KD
# corpus: inserted tokens/sentences absent from it (BadNets, AddSent),
# trigger features that occur in it naturally (SynBkd, SOS), weight and
# embedding poisoning (EP, RIPPLe), and a trigger optimised to survive
# distillation (ATBA). EP/SOS/RIPPLe/ATBA are BERT-only: OpenBackdoor's EP
# and SOS trainers assume BERT embeddings, and ATBA triggers exist for BERT.
BATTERY = {
    "bert": ["badnets", "addsent", "synbkd", "sos", "ep", "atba"],
    "qwen3": ["badnets", "addsent", "synbkd"],
    "gpt2": ["badnets", "addsent"],
}
BATTERY_KD = {"bert": ["logit", "feature", "ce"], "qwen3": ["logit", "ce"],
              "gpt2": ["logit", "ce"]}
BATTERY_DATASETS = {"bert": ["sst2", "agnews"], "qwen3": ["sst2", "agnews"],
                    "gpt2": ["sst2"]}
# Qwen3 teachers are the slowest builds (~2 h on SST-2, ~5 h on AG News
# while the BERT lane shares GPU 0), so Qwen3 runs SynBkd and the A/A null
# on SST-2 only, and only the BadNets-contaminated corpus.
SST2_ONLY = {("qwen3", "synbkd")}
# RIPPLe on SST-2 only: an AG News teacher pair takes 11-16 h (author's
# decision, 2026-10-21); EP covers weight/embedding poisoning on AG News.
RIPPLE_DATASETS = ["sst2"]

# Student-seed replication (docs/tifs_plan.md §0.3): seeds 43 and 44 of
# every SST-2 logit battery, so its verdict carries the between-seed
# variance of training. Teachers and the clean corpus stay at seed 42;
# only the student seed (head init, data order, dropout) changes. P and
# the shared C only: R does not enter k. EP is left out: its SST-2 teacher
# pair differs only in the input embeddings of its trigger tokens, which no
# corpus row contains (scripts/tifs/structural_nulls.py), so k = 0 for any
# student seed (a structural null). SOS's trigger words occur in 80 corpus
# rows, so SOS is replicated. AG News, feature KD and the contaminated /
# control blocks borrow the SST-2 variance.
STUDENT_SEEDS = (43, 44)
SEED_BATTERY = {
    "bert": ["badnets", "addsent", "synbkd", "sos", "atba"],
    "gpt2": ["badnets", "addsent"],
    "qwen3": ["badnets", "addsent", "synbkd"],
}

# Added 2026-10-22 (docs/tifs_addendum.md A1), before any AG News Stage 2
# output existed: student seeds 43/44 of the BERT AG News logit battery.
# Borrowing the 3-seed SST-2 pool (s 0.022, df 4.7), a k = 0 AG News cell
# has an upper bound of about 0.057, so NO TRANSFER at delta 0.05 could not
# be reached whatever the AG News data. On AG News the EP and SOS pairs are
# separately trained models (393 tensors differ), so both are replicated.
AGNEWS_SEED_BATTERY = {"bert": ["badnets", "addsent", "synbkd", "sos", "ep", "atba"]}

# Added 2026-10-22 (addendum A2): a second ATBA teacher build. The ATBA
# poisoned teacher and the base-trainer clean teacher at teacher seed 43
# (the clean one is the A/A teacher of tifs_aa_bert_clean_t43), corpus seed
# 42, student seeds 42-44, BERT SST-2, logit KD. The run ids start with
# tifs_rep_, which the frozen analysis does not parse (its battery cells
# must have teacher seed 42); scripts/tifs/addendum.py reads them.
REPLICATE_SEEDS = (42, 43, 44)

# Added 2026-10-22 (addendum A8), before any A8 output existed: a standard
# feature-based objective. The frozen 'feature' objective's hint (final-layer
# [CLS], lambda 1e-3) carries under 0.01% of the student's gradient at the
# start of training, so its cells are logit KD in effect. 'hidden' is logit
# KD plus TinyBERT-style hidden-state distillation over uniformly mapped
# layers (counterfactual_kd/kd/hidden.py) on the seed-replicated BERT SST-2
# attacks, student seeds 42-44, teachers and corpus seed 42. EP stays out:
# its teacher pair gives identical hidden states on every corpus row, so
# k = 0 here too. The run ids start with tifs_hid_, which the frozen
# analysis does not parse; scripts/tifs/addendum.py reads them.
HIDDEN_SEEDS = (42, 43, 44)

# The BERT A/A runs also read the ATBA phrases (prefix, as the attack
# inserts them), so the reference-teacher draw is measured at ATBA's
# operating point and not only at BadNets'.
AA_BERT_CONTROLS = [
    {"name": f"atba_phrase_{ds}", "words": [ATBA_TRIGGERS["bert"][ds]],
     "position": "prefix"}
    for ds in ("sst2", "agnews")
]


def base(fam: str, name: str, description: str) -> dict:
    f = FAMILIES[fam]
    return {
        "name": name,
        "description": description,
        "seeds": [42],
        "methods": ["kd"],
        "pairs": [list(f["pair"])],
        "datasets": ["sst2", "agnews"],
        "attacks": ["badnets", "addsent"],
        "kd_types": ["logit"],
        "tf32": True,
        "stage0": dict(f["stage0"]),
        "stage1": {
            "temperature": 4.0, "alpha": 0.5, "kd_epochs": 5,
            "batch_size": 16, "max_length": 128, "warmup_epochs": 0,
            "weight_decay": 0.0, "lr": f["lr"],
            "teacher_dtype": "bf16", "student_dtype": "fp32",
        },
        "stage2": {"batch_size": f["eval_bs"], "max_length": 128,
                   "save_preds": True, "eval_dtype": "bf16",
                   "control_triggers": copy.deepcopy(CONTROLS),
                   "test_sets": dict(TEST_SETS)},
        "output": {"teacher_dir": TEACHER_DIR, "runs_dir": RUNS_DIR},
    }


def set_corpus(cfg: dict, corpus: str, keep_label: bool = False) -> None:
    spec = CORPORA.get(corpus)
    if spec is None:
        return
    attack, rate = spec
    cfg["attacks"] = [attack]
    cfg["stage1"]["kd_corpus_attack"] = attack
    cfg["stage1"]["kd_corpus_attack_rate"] = rate
    if keep_label:
        cfg["stage1"]["kd_corpus_keep_label"] = True


def build() -> dict[str, dict]:
    out: dict[str, dict] = {}

    # T: teachers only (run with --stages stage0).
    for fam in FAMILIES:
        name = f"tifs_t_{fam}"
        out[name] = base(fam, name, f"Seed-42 teachers for {fam}: clean + BadNets + AddSent, SST-2 + AG News.")
    for fam in ("bert", "qwen3"):
        name = f"tifs_ts_{fam}"
        cfg = base(fam, name, f"Seed-42 SynBkd teachers for {fam}.")
        cfg["attacks"] = ["synbkd"]
        if (fam, "synbkd") in SST2_ONLY:
            cfg["datasets"] = ["sst2"]
        out[name] = cfg
    name = "tifs_tb_bert"
    cfg = base("bert", name, "Seed-42 battery teachers for BERT: EP, SOS, ATBA with trainer-matched clean teachers.")
    cfg["attacks"] = ["ep", "sos", "atba"]
    out[name] = cfg
    # RIPPLe's trainer differentiates through its gradients (about 3 h per
    # SST-2 teacher and 6-8 h per AG News teacher in the tier-1 runs), so
    # each dataset is its own run and can be queued on its own.
    for ds in RIPPLE_DATASETS:
        name = f"tifs_tr_bert_{ds}"
        cfg = base("bert", name, f"Seed-42 RIPPLe teachers for BERT/{ds} with the trainer-matched clean teacher.")
        cfg["datasets"] = [ds]
        cfg["attacks"] = ["ripple"]
        out[name] = cfg

    # B: the battery on a clean corpus, one run per (family, dataset) so the
    # attacks of a run share their C and R students.
    for fam, attacks in BATTERY.items():
        for ds in BATTERY_DATASETS[fam]:
            name = f"tifs_b_{fam}_{ds}"
            cfg = base(fam, name, f"Battery {fam}/{ds}, clean KD corpus, seed 42: P per attack, trainer-matched C, CE-only R.")
            cfg["datasets"] = [ds]
            cfg["attacks"] = [a for a in attacks
                              if ds == "sst2" or (fam, a) not in SST2_ONLY]
            cfg["kd_types"] = list(BATTERY_KD[fam])
            out[name] = cfg
    for ds in RIPPLE_DATASETS:
        name = f"tifs_br_bert_{ds}"
        cfg = base("bert", name, f"Battery bert/{ds}, RIPPLe, clean KD corpus, seed 42.")
        cfg["datasets"] = [ds]
        cfg["attacks"] = ["ripple"]
        cfg["kd_types"] = list(BATTERY_KD["bert"])
        out[name] = cfg

    # BS / BRS: student seeds of the SST-2 logit batteries.
    for s in STUDENT_SEEDS:
        for fam, attacks in SEED_BATTERY.items():
            name = f"tifs_b_{fam}_sst2_s{s}"
            cfg = base(fam, name, f"Battery {fam}/sst2, clean KD corpus, student seed {s}, teachers and corpus seed 42: P per attack and the shared C, logit KD.")
            cfg["seeds"] = [s]
            cfg["teacher_seed"] = 42
            cfg["corpus_seed"] = 42
            cfg["datasets"] = ["sst2"]
            cfg["attacks"] = list(attacks)
            cfg["kd_types"] = ["logit"]
            out[name] = cfg
        name = f"tifs_br_bert_sst2_s{s}"
        cfg = base("bert", name, f"Battery bert/sst2, RIPPLe, clean KD corpus, student seed {s}, teachers and corpus seed 42, logit KD.")
        cfg["seeds"] = [s]
        cfg["teacher_seed"] = 42
        cfg["corpus_seed"] = 42
        cfg["datasets"] = ["sst2"]
        cfg["attacks"] = ["ripple"]
        cfg["kd_types"] = ["logit"]
        out[name] = cfg

    # BSA: student seeds of the BERT AG News logit battery (addendum A1).
    for s in STUDENT_SEEDS:
        for fam, attacks in AGNEWS_SEED_BATTERY.items():
            name = f"tifs_b_{fam}_agnews_s{s}"
            cfg = base(fam, name, f"Battery {fam}/agnews, clean KD corpus, student seed {s}, teachers and corpus seed 42: P per attack and the trainer-matched C, logit KD.")
            cfg["seeds"] = [s]
            cfg["teacher_seed"] = 42
            cfg["corpus_seed"] = 42
            cfg["datasets"] = ["agnews"]
            cfg["attacks"] = list(attacks)
            cfg["kd_types"] = ["logit"]
            out[name] = cfg

    # REP: the second ATBA teacher build (addendum A2).
    for s in REPLICATE_SEEDS:
        name = f"tifs_rep_bert_atba_t43_s{s}"
        cfg = base("bert", name, f"Replicate bert/sst2 ATBA: teachers seed 43, corpus seed 42, student seed {s}: P and C, logit KD.")
        cfg["seeds"] = [s]
        cfg["teacher_seed"] = 43
        cfg["corpus_seed"] = 42
        cfg["datasets"] = ["sst2"]
        cfg["attacks"] = ["atba"]
        cfg["kd_types"] = ["logit"]
        out[name] = cfg

    # HID: hidden-state KD on the BERT SST-2 seed battery (addendum A8).
    for s in HIDDEN_SEEDS:
        name = f"tifs_hid_bert_sst2_s{s}"
        cfg = base("bert", name, f"Battery bert/sst2, clean KD corpus, student seed {s}, teachers and corpus seed 42: P per attack and the shared C, logit KD + hidden-state distillation (TinyBERT-style).")
        cfg["seeds"] = [s]
        cfg["teacher_seed"] = 42
        cfg["corpus_seed"] = 42
        cfg["datasets"] = ["sst2"]
        cfg["attacks"] = list(SEED_BATTERY["bert"])
        cfg["kd_types"] = ["hidden"]
        out[name] = cfg

    # G: contaminated corpus, seed 42 (teacher attack = corpus attack).
    for fam, corpora in (("bert", ("badnets10", "addsent10")), ("qwen3", ("badnets10",))):
        for corpus in corpora:
            name = f"tifs_g_{fam}_{corpus}_s42"
            cfg = base(fam, name, f"{fam}, corpus={corpus}, seed 42: P, C and CE-only students.")
            cfg["teacher_seed"] = 42
            cfg["kd_types"] = list(BATTERY_KD[fam])
            set_corpus(cfg, corpus)
            out[name] = cfg

    # PC: positive control. Triggered rows keep their labels, so the
    # corpus carries no trigger->target mapping (C3).
    for fam, variants in (("bert", ("kl10", "kl01")),):
        for v in variants:
            name = f"tifs_pc_{fam}_{v}"
            rate = 0.1 if v == "kl10" else 0.01
            cfg = base(fam, name, f"Positive control {fam}: {rate:.0%} BadNets-triggered rows, labels kept.")
            cfg["kd_types"] = ["logit", "ce"]
            set_corpus(cfg, "badnets10", keep_label=True)
            cfg["stage1"]["kd_corpus_attack_rate"] = rate
            out[name] = cfg
    # Conditional (§0.3): pure KD (alpha=1) on SST-2, run only if kl10's
    # P-C is not POSITIVE at delta=0.05 on either dataset.
    name = "tifs_pc_bert_kl10a1"
    cfg = base("bert", name, "Positive control bert, alpha=1 (pure KD), SST-2: 10% BadNets-triggered rows, labels kept. Runs only if tifs_pc_bert_kl10 fails.")
    cfg["datasets"] = ["sst2"]
    set_corpus(cfg, "badnets10", keep_label=True)
    cfg["stage1"]["alpha"] = 1.0
    out[name] = cfg

    # AA: independent teacher draws, corpus and student fixed at 42.
    for fam, seeds in (("bert", (43, 44)), ("qwen3", (43,))):
        for corpus in ("clean", "badnets10"):
            for ts in seeds:
                name = f"tifs_aa_{fam}_{corpus}_t{ts}"
                cfg = base(fam, name, f"A/A {fam}, corpus={corpus}: teachers seed {ts}, corpus+student seed 42.")
                cfg["attacks"] = ["badnets"]
                cfg["teacher_seed"] = ts
                cfg["corpus_seed"] = 42
                cfg["student_seed"] = 42
                if fam == "qwen3":
                    cfg["datasets"] = ["sst2"]
                else:
                    cfg["stage2"]["control_triggers"] += copy.deepcopy(AA_BERT_CONTROLS)
                set_corpus(cfg, corpus)
                out[name] = cfg
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    cfgs = build()
    for name, cfg in cfgs.items():
        with open(os.path.join(OUT, f"{name}.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    # Configs of dropped blocks move aside instead of being deleted.
    stale = sorted(f for f in os.listdir(OUT)
                   if f.endswith(".yaml") and f[:-5] not in cfgs)
    if stale:
        dropped = os.path.join(OUT, "_dropped")
        os.makedirs(dropped, exist_ok=True)
        for f in stale:
            os.replace(os.path.join(OUT, f), os.path.join(dropped, f))
    print(f"wrote {len(cfgs)} configs to {os.path.normpath(OUT)}; "
          f"moved {len(stale)} stale ones to _dropped/")


if __name__ == "__main__":
    main()
