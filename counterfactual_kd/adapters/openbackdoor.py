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


def _tf32_enabled() -> bool:
    import torch
    return bool(torch.backends.cuda.matmul.allow_tf32)


def ob_text_keys(dataset: str) -> dict[str, str]:
    """Map OpenBackdoor's text for a row back to the KD/eval loader text.

    The SynBkd paraphrase table is keyed by loader text. OpenBackdoor sees
    SST-2 rows stripped of surrounding whitespace and AG News rows split at
    the first ". " and re-joined with a space (prepare_ob_datasets.py);
    HSOL rows are identical.
    """
    from datasets import load_dataset
    if dataset == "sst2":
        ds = load_dataset("glue", "sst2")
        texts = [ex["sentence"] for split in ("train", "validation") for ex in ds[split]]
        return {t.strip(): t for t in texts}
    if dataset == "agnews":
        ds = load_dataset("ag_news")
        out = {}
        for split in ("train", "test"):
            for ex in ds[split]:
                parts = ex["text"].replace("\n", " ").strip().split(". ", 1)
                body = parts[1] if len(parts) > 1 else ""
                out[parts[0].replace("\\", " ") + " " + body.replace("\\", " ")] = ex["text"]
        return out
    return {}


def _route_synbkd_through_table(dataset: str) -> None:
    """Make OpenBackdoor's SynBkd poisoner use the shared paraphrase table.

    Teacher poisoning, KD-corpus contamination and evaluation then apply
    the identical paraphrase to a given sentence, and Stage 0 does not load
    its own SCPN (whose checkpoint torch>=2.6 refuses by default).
    """
    from openbackdoor.attackers.poisoners import synbkd_poisoner as sp
    from openbackdoor.attackers.poisoners.poisoner import Poisoner

    from counterfactual_kd.triggers.syntactic import SyntacticTrigger

    trigger = SyntacticTrigger()
    keys = ob_text_keys(dataset)

    def __init__(self, template_id=-1, **kwargs):
        Poisoner.__init__(self, **kwargs)
        self.template = None

    def transform(self, text):
        return trigger.insert(keys.get(text, text))

    sp.SynBkdPoisoner.__init__ = __init__
    sp.SynBkdPoisoner.transform = transform


def _attention_for(trainer: str):
    """SDPA kernels a teacher's trainer can differentiate through.

    RIPPLe's trainer takes gradients of its gradients
    (``create_graph=True``). The flash, memory-efficient and cuDNN SDPA
    kernels have no second derivative, so its teachers train on the math
    kernel only (as ``scripts/repro_eager.py`` did for the tier-1 runs).
    """
    import contextlib
    if trainer != "ripples":
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel(SDPBackend.MATH)


def _register_atba_poisoner() -> None:
    """Register an ``atba`` poisoner with OpenBackdoor.

    ATBA's trigger is a fixed phrase per (family, dataset)
    (``ATBA_TRIGGERS``); OpenBackdoor ships no poisoner for it. This one
    inserts the phrase exactly as the evaluation side does
    (``WordTrigger(position=...)``: ``phrase + " " + text`` for the prefix,
    the default; ``text + " " + phrase`` for the suffix).
    """
    from openbackdoor.attackers import poisoners
    from openbackdoor.attackers.poisoners.poisoner import Poisoner

    if "atba" in poisoners.POISONERS:
        return

    class ATBAPoisoner(Poisoner):
        def __init__(self, triggers=None, position="prefix", **kwargs):
            super().__init__(**kwargs)
            self.phrase = triggers[0] if isinstance(triggers, (list, tuple)) else triggers
            self.position = position

        def poison(self, data: list):
            return [(self.insert(text), self.target_label, 1) for text, _label, _p in data]

        def insert(self, text: str) -> str:
            if self.position == "suffix":
                return text + " " + self.phrase
            return self.phrase + " " + text

    poisoners.POISONERS["atba"] = ATBAPoisoner


def _save_every_epoch(trainer, victim, save_path: str) -> list[dict]:
    """Save every epoch's weights while OpenBackdoor trains (A15-b).

    OpenBackdoor's base trainer evaluates the dev splits once at the end
    of every epoch and keeps the first epoch with the highest mean dev
    accuracy ('best'). This wraps that evaluation on the trainer instance:
    after it, the current weights and tokenizer go to
    ``<save_path>/epochs/epoch<e>/`` with an ``epoch_meta.json``, and the
    epoch's dev metrics are recorded. Training itself is untouched (saving
    reads the weights and draws no random numbers). Returns the record list
    the wrapper appends to.
    """
    from counterfactual_kd.provenance import combined_sha256, tensors_sha256, weights_sha256
    from counterfactual_kd.registry import teacher_epoch_path

    records: list[dict] = []
    original = trainer.evaluate

    def evaluate(model, eval_dataloader, metrics):
        results, dev_score = original(model, eval_dataloader, metrics)
        epoch = len(records) + 1
        d = teacher_epoch_path(save_path, epoch)
        os.makedirs(d, exist_ok=True)
        model.plm.save_pretrained(d)
        victim.tokenizer.save_pretrained(d)
        files = weights_sha256(d)
        rec = {
            "epoch": epoch,
            "dev": {k: dict(v) for k, v in results.items()},
            "dev_score": float(dev_score),
            "path": os.path.relpath(d, save_path),
            "weights_sha256": files,
            "weights_sha256_combined": combined_sha256(files),
            "tensors_sha256": tensors_sha256(d),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with open(os.path.join(d, "epoch_meta.json"), "w") as f:
            json.dump(rec, f, indent=2, default=float)
        records.append(rec)
        print(f"[OB-adapter] epoch {epoch} saved to {d} (dev score {dev_score:.4f})")
        return results, dev_score

    trainer.evaluate = evaluate
    return records


def _ob_selected_epoch(records: list[dict]) -> int | None:
    """OpenBackdoor's 'best' rule (trainers/trainer.py): the first epoch
    whose mean dev accuracy exceeds every earlier one, starting from 0."""
    best, sel = 0.0, None
    for r in records:
        if r["dev_score"] > best:
            best, sel = r["dev_score"], r["epoch"]
    return sel


def _accuracy_only_epoch(records: list[dict]) -> int | None:
    """First epoch with the highest dev-clean accuracy (ties: earliest)."""
    best, sel = None, None
    for r in records:
        acc = (r["dev"].get("dev-clean") or {}).get("accuracy")
        if acc is not None and (best is None or acc > best):
            best, sel = acc, r["epoch"]
    return sel


def _write_epochs_record(save_path: str, job: TeacherJob, records: list[dict],
                         ob_ckpt_dir: str | None) -> None:
    from counterfactual_kd.provenance import combined_sha256, tensors_sha256, weights_sha256
    sel = _ob_selected_epoch(records)
    root_tensors = tensors_sha256(save_path)
    root_files = weights_sha256(save_path)
    by_epoch = {r["epoch"]: r for r in records}
    out = {
        "teacher": os.path.basename(os.path.normpath(save_path)),
        "kind": job.kind, "attack": job.attack, "model_path": job.model_path,
        "dataset": job.dataset, "seed": job.seed,
        "rule": "OpenBackdoor ckpt='best': first epoch with the highest mean dev "
                "accuracy over the dev splits (strict >, from 0)",
        "n_epochs": len(records),
        "ob_selected_epoch": sel,
        "accuracy_only_epoch": _accuracy_only_epoch(records),
        "last_epoch": records[-1]["epoch"] if records else None,
        "root_weights_sha256": root_files,
        "root_weights_sha256_combined": combined_sha256(root_files),
        "root_tensors_sha256": root_tensors,
        "root_matches_selected_epoch": (sel is not None
                                        and by_epoch[sel]["tensors_sha256"] == root_tensors),
        "ob_checkpoint_dir": ob_ckpt_dir,
        "pid": os.getpid(),
        "epochs": records,
    }
    with open(os.path.join(save_path, "epochs.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[OB-adapter] epochs.json: OB-selected epoch {sel}, accuracy-only "
          f"epoch {out['accuracy_only_epoch']}, root matches selected: "
          f"{out['root_matches_selected_epoch']}")


def _remove_empty_dirs(path: str | None, stop: str) -> None:
    """Remove ``path`` and its empty parents up to and including ``stop``."""
    if not path:
        return
    path, stop = os.path.abspath(path), os.path.abspath(stop)
    while path == stop or path.startswith(stop + os.sep):
        if os.path.isdir(path):
            try:
                os.rmdir(path)
            except OSError:
                return
        if path == stop:
            return
        path = os.path.dirname(path)


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
    every_epoch = bool(getattr(stage0, "save_every_epoch", False))
    if every_epoch and attack_trainer(job.attack) != "base":
        raise ValueError("stage0.save_every_epoch supports the base trainer only; "
                         f"{job.attack} uses {attack_trainer(job.attack)}")
    pid_dir = None
    if getattr(stage0, "ob_ckpt_pid", False):
        # A15: OpenBackdoor names its intermediate checkpoint dir by the
        # start second only; two builds started in the same second would
        # share (and delete) it.
        pid_dir = os.path.join(".", "models", "checkpoints", f"pid{os.getpid()}")
        ob_config["attacker"]["train"]["save_path"] = pid_dir
    ob_config = set_config(ob_config)

    if job.attack == "synbkd":
        _route_synbkd_through_table(job.dataset)
    if job.attack == "atba":
        _register_atba_poisoner()
    attacker = load_attacker(ob_config["attacker"])
    victim = load_victim(ob_config["victim"])
    # Decoder families (GPT-2, Qwen3) ship without a usable pad id; OB's
    # PLMVictim pads every batch. Same convention as the KD trainer
    # (kd/base.py): pad with EOS when the tokenizer has no pad token, and
    # tell the classification head which id marks padding.
    if victim.tokenizer.pad_token is None:
        victim.tokenizer.pad_token = victim.tokenizer.eos_token
    if victim.plm.config.pad_token_id is None:
        victim.plm.config.pad_token_id = victim.tokenizer.pad_token_id

    # Short-circuit OB's PROCESSORS registry for HateSpeech-family
    # datasets (not in OB's catalog). See ``_load_hatespeech_ob_dataset``
    # above for the rationale and the tuple-format contract.
    ob_ds_name = ob_config["poison_dataset"]["name"]
    if ob_ds_name in _HATESPEECH_OB_NAMES:
        ob_dataset = _load_hatespeech_ob_dataset(ob_ds_name)
    else:
        ob_dataset = load_dataset(**ob_config["poison_dataset"])

    epoch_records = None
    if every_epoch:
        epoch_records = _save_every_epoch(attacker.poison_trainer, victim, save_path)

    t0 = time.time()
    with _attention_for(attack_trainer(job.attack)):
        trained_victim = attacker.attack(victim, ob_dataset, ob_config)
    elapsed = time.time() - t0

    os.makedirs(save_path, exist_ok=True)
    trained_victim.plm.save_pretrained(save_path)
    trained_victim.tokenizer.save_pretrained(save_path)
    # OB keeps its best-epoch state_dict under ./models/checkpoints/<...>/<ts>/;
    # the weights are now in save_path, so drop the duplicate (6 GB for
    # gpt2-xl, 7 GB for Qwen3-1.7B).
    ob_ckpt_dir = getattr(getattr(attacker, "poison_trainer", None), "save_path", None)
    if ob_ckpt_dir and os.path.isdir(ob_ckpt_dir):
        import shutil
        shutil.rmtree(ob_ckpt_dir, ignore_errors=True)
    if pid_dir:
        _remove_empty_dirs(ob_ckpt_dir, stop=pid_dir)
    if epoch_records is not None:
        _write_epochs_record(save_path, job, epoch_records, ob_ckpt_dir)

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
        "tf32": _tf32_enabled(),
        "train_time_sec": round(elapsed, 1),
    }
    if epoch_records is not None:
        meta["epochs_record"] = "epochs.json"
    with open(sidecar, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return TeacherResult(
        checkpoint_path=save_path,
        metadata_path=sidecar,
        train_time_sec=elapsed,
        skipped=False,
    )
