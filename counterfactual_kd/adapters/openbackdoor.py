"""
OpenBackdoor adapter — Stage 0 teacher training boundary.

Wraps the OB training loop so :mod:`counterfactual_kd.pipeline.stage0_teacher` can stay
OB-agnostic (just plans jobs, calls this, collects results). The heavy OB
imports are deferred into :func:`train_teacher` so this module is importable
without ``openbackdoor`` or ``torch`` on the path — useful for planning
tests and CLI validation.

Contract: see ``docs/ADAPTERS.md`` §Boundary 1.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from counterfactual_kd.registry import (
    attack_spec, attack_trainer, dataset_spec, model_family,
    teacher_clean_path, teacher_poisoned_path,
)


# ── Job / result dataclasses ─────────────────────────────────────────

@dataclass(frozen=True)
class TeacherJob:
    """One teacher-training target. Produced by Stage 0's planner."""
    kind: str           # "poisoned" | "clean"
    attack: str         # Counterfactual-KD attack name (e.g. "badnets", "ep")
    model_path: str     # HF model id or local path
    dataset: str        # Counterfactual-KD dataset name (e.g. "sst2")
    seed: int

    def __post_init__(self):
        if self.kind not in ("poisoned", "clean"):
            raise ValueError(f"kind must be 'poisoned' or 'clean', got {self.kind!r}")


@dataclass(frozen=True)
class TeacherResult:
    """Paths produced by a successful teacher training."""
    checkpoint_path: str       # HF-format dir with config.json, pytorch_model.bin, tokenizer
    metadata_path: str         # absolute path to ob_meta.json sidecar
    train_time_sec: float
    skipped: bool = False      # True if skip_existing short-circuited


# ── Pure translator (no I/O, no imports of OB or torch) ──────────────

def build_ob_config(
    attack: str, model_path: str, dataset: str, stage0,
    *, poison_rate_override: float | None = None,
) -> dict:
    """Build an OB config dict from Counterfactual-KD specs.

    Pure function — safe to call from tests without any OB/torch install.
    This is the F1 structural fix: trigger vocab comes from the registry,
    not from a separately-maintained dict.
    """
    aspec = attack_spec(attack)
    dspec = dataset_spec(dataset)
    family = model_family(model_path)

    poisoner: dict[str, Any] = {
        "name": aspec.ob_name,
        "poison_rate": (poison_rate_override if poison_rate_override is not None
                        else stage0.poison_rate),
        "target_label": dspec.target_label,
        "label_consistency": False,
        "label_dirty": False,
        "load": False,
        "save": False,
    }
    tkw = aspec.trigger_kwargs
    if aspec.name == "atba":
        # ATBA's trigger is determined per (model_family, dataset) cell —
        # registered statically in ``counterfactual_kd.triggers.registry.ATBA_TRIGGERS``.
        # A-level reproduction = lookup the published phrase and pass to
        # ATBAPoisoner (which inserts it at the prefix, matching
        # WordTrigger(position="prefix") on the Counterfactual-KD eval side).
        from counterfactual_kd.triggers.registry import ATBA_TRIGGERS
        if family not in ATBA_TRIGGERS or dataset not in ATBA_TRIGGERS[family]:
            raise ValueError(
                f"No ATBA trigger registered for {family}/{dataset}. "
                f"Available cells: "
                f"{ {f: list(d) for f, d in ATBA_TRIGGERS.items()} }"
            )
        poisoner["triggers"] = [ATBA_TRIGGERS[family][dataset]]
        # Trigger position for teacher poisoning: encoder-only (BERT) =
        # prefix (default), decoder-only (GPT/OPT) = suffix per the ATBA
        # paper. Threaded from the config's stage0.trigger_position so the
        # poisoned teacher matches the Counterfactual-KD eval side (attack_overrides).
        pos = getattr(stage0, "trigger_position", None)
        if pos:
            poisoner["position"] = pos
    elif "words" in tkw:
        poisoner["triggers"] = list(tkw["words"])
        if aspec.name == "badnets":
            poisoner["num_triggers"] = 1
    elif "sentence" in tkw:
        poisoner["triggers"] = tkw["sentence"]

    return {
        "target_dataset": {"name": dspec.ob_name, "dev_rate": 0.1},
        "poison_dataset": {"name": dspec.ob_name, "dev_rate": 0.1},
        "victim": {
            "type": "plm",
            "model": family,
            "path": model_path,
            "num_classes": dspec.num_labels,
            "device": "gpu",
            "max_len": stage0.max_len,
        },
        "attacker": {
            "name": aspec.trainer,
            "train": {
                "name": aspec.trainer,
                "lr": stage0.lr,
                "weight_decay": stage0.weight_decay,
                "epochs": stage0.epochs,
                "batch_size": stage0.batch_size,
                "warm_up_epochs": stage0.warm_up_epochs,
                "ckpt": "best",
            },
            "poisoner": poisoner,
        },
        "clean-tune": False,
    }


# ── OB-format short-circuit for datasets not in OB's PROCESSORS registry ─
#
# OpenBackdoor's ``load_dataset`` dispatches through a ``PROCESSORS``
# dict that ships only the small set of canonical SA / TC corpora
# (sst-2, agnews, imdb, ...). HateSpeech (Vicomtech) is not there, so
# Stage 0 for KD-family methods fails with ``KeyError: 'hatespeech'``
# before any training starts. FT-family methods route around OB
# entirely (they use ``counterfactual_kd.data.loaders.load_data``), which is why
# they work.
#
# Rather than monkey-patch OB's registry — which would couple Counterfactual-KD to
# OB's internal layout — we short-circuit the load on the Counterfactual-KD side
# for our HateSpeech registry entries only. Sst2 / AgNews / ... still
# flow through OB's ``load_dataset`` unchanged.
#
# OB tuple format (confirmed via ``openbackdoor/data/sentiment_analysis_dataset.py``
# SST2Processor and the poisoner ``for text, label, poison_label in data``
# loops in ``openbackdoor/attackers/poisoners/*_poisoner.py``):
#     dataset = {"train": [...], "dev": [...], "test": [...]}
#     each split is a list of (text: str, label: int, poison_flag: int)
# The poison_flag is 0 for clean rows; the poisoner sets it to 1 on
# the rows it injects triggers into.

_HATESPEECH_OB_NAMES: frozenset[str] = frozenset({
    "hatespeech", "hatespeech_flip", "hatespeech_5050",
})


def _load_hatespeech_ob_dataset(ob_name: str) -> dict[str, list[tuple]]:
    """Build an OB-format dict from Counterfactual-KD's HateSpeech loaders.

    Returns {"train", "dev", "test"} → list[(text, label, 0)].
    Splits are the matrix-locked sizes from
    ``counterfactual_kd.data.hatespeech_loader``: (7703, 1000, 2000) for the
    natural-prior entries and (1722, 223, 447) for the 50/50 control.
    """
    if ob_name == "hatespeech_5050":
        from counterfactual_kd.data.hatespeech_loader import load_hatespeech_5050 as _ld
    else:
        # hatespeech and hatespeech_flip share the same natural-prior
        # corpus; the only difference is ``target_label`` (handled
        # upstream via the poisoner config), not the data.
        from counterfactual_kd.data.hatespeech_loader import load_hatespeech as _ld

    def to_tuples(split: str) -> list[tuple]:
        return [(r["text"], int(r["label"]), 0) for r in _ld(split)]

    return {
        "train": to_tuples("train"),
        "dev":   to_tuples("dev"),
        "test":  to_tuples("test"),
    }


# ── Main entry point (requires OB + torch at call time) ──────────────

def teacher_save_path(job: TeacherJob, output_root: str) -> str:
    """Resolve where a teacher will be saved. Pure, no side effects."""
    if job.kind == "poisoned":
        return teacher_poisoned_path(output_root, job.attack, job.model_path,
                                     job.dataset, job.seed)
    return teacher_clean_path(output_root, job.attack, job.model_path,
                              job.dataset, job.seed)


def train_teacher(
    job: TeacherJob,
    stage0,
    output_root: str,
    *,
    skip_existing: bool = True,
) -> TeacherResult:
    """Train one OB teacher and write the HF checkpoint + sidecar.

    Idempotent when ``skip_existing=True``: if the target directory already
    contains ``config.json``, returns a :class:`TeacherResult` with
    ``skipped=True`` and the existing paths, without re-training.

    Side effects:
      * Writes HF checkpoint to ``teacher_save_path(job, output_root)``
      * Writes ``ob_meta.json`` sidecar
      * Calls ``openbackdoor.utils.set_seed(seed)`` — process-global

    Raises:
      * KeyError via registry if attack/dataset is unknown.
      * RuntimeError propagated from OB on training failure.
    """
    try:
        from openbackdoor.data import load_dataset
        from openbackdoor.victims import load_victim
        from openbackdoor.attackers import load_attacker
        from openbackdoor.utils import set_config, set_seed
    except ImportError as e:
        raise ImportError(
            "Training a poisoned/clean teacher (stage0) requires OpenBackdoor, which "
            "is not installed.\n"
            "  Install it:  pip install -e \".[teacher]\"\n"
            "  (or:  pip install 'openbackdoor @ git+https://github.com/thunlp/OpenBackdoor.git')\n"
            "If you already have cached teachers, skip stage0 and run only the student "
            "distillation + evaluation:\n"
            "  counterfactual_kd repro --config <cfg> --stages stage1 stage2\n"
            f"(original import error: {e})"
        ) from e

    save_path = teacher_save_path(job, output_root)
    sidecar = os.path.join(save_path, "ob_meta.json")

    if skip_existing and os.path.exists(os.path.join(save_path, "config.json")):
        print(f"[OB-adapter] SKIP {save_path}")
        return TeacherResult(
            checkpoint_path=save_path,
            metadata_path=sidecar,
            train_time_sec=0.0,
            skipped=True,
        )

    print(f"[OB-adapter] {job.kind:>8} | {job.attack:<10} | "
          f"{job.model_path} | {job.dataset} | seed={job.seed}")

    set_seed(job.seed)
    ob_config = build_ob_config(
        job.attack, job.model_path, job.dataset, stage0,
        poison_rate_override=(0.0 if job.kind == "clean" else None),
    )
    ob_config = set_config(ob_config)

    attacker = load_attacker(ob_config["attacker"])
    victim = load_victim(ob_config["victim"])

    # Short-circuit OB's PROCESSORS registry for HateSpeech-family
    # datasets (not in OB's catalog). See ``_load_hatespeech_ob_dataset``
    # above for the rationale and the tuple-format contract.
    ob_ds_name = ob_config["poison_dataset"]["name"]
    if ob_ds_name in _HATESPEECH_OB_NAMES:
        ob_dataset = _load_hatespeech_ob_dataset(ob_ds_name)
    else:
        ob_dataset = load_dataset(**ob_config["poison_dataset"])

    t0 = time.time()
    trained_victim = attacker.attack(victim, ob_dataset, ob_config)
    elapsed = time.time() - t0

    os.makedirs(save_path, exist_ok=True)
    trained_victim.plm.save_pretrained(save_path)
    trained_victim.tokenizer.save_pretrained(save_path)

    meta = {
        "type": f"{job.kind}_teacher",
        "source": "openbackdoor",
        "attack": job.attack,
        "trainer": attack_trainer(job.attack),
        "model_path": job.model_path,
        "model_family": model_family(job.model_path),
        "dataset": job.dataset,
        "seed": job.seed,
        "ob_config": ob_config["attacker"],
        "train_time_sec": round(elapsed, 1),
    }
    with open(sidecar, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return TeacherResult(
        checkpoint_path=save_path,
        metadata_path=sidecar,
        train_time_sec=elapsed,
        skipped=False,
    )
