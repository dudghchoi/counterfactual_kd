"""
Native KD adapter — Stage 1 distillation boundary.

Counterfactual-KD-native KD (logit / feature / attention), wrapped as a single
callable with an explicit job dataclass. Same-tokenizer only; cross-
tokenizer KD lives in a separate adapter (deferred).

Contract: see ``docs/ADAPTERS.md`` §Boundary 2.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Literal

from counterfactual_kd.registry import dataset_spec


KDType = Literal["logit", "feature", "attention"]


@dataclass(frozen=True)
class DistillJob:
    teacher_path: str | None    # from TeacherResult.checkpoint_path; None for kd_type "ce"
    student_model: str          # HF id, e.g. "bert-base-uncased"
    dataset: str                # Counterfactual-KD dataset name (labels + num_labels)
    kd_type: KDType
    seed: int                   # student seed: head init, data order, dropout
    teacher_kind: str = "poisoned"   # "poisoned" | "clean" | "none" — labels the sidecar
    # Seed decoupling (TIFS plan C1). ``corpus_seed`` drives subsampling
    # and the contamination draw; None → ``seed``. ``teacher_seed`` is
    # only recorded (the teacher path already encodes it).
    corpus_seed: int | None = None
    teacher_seed: int | None = None
    # Trigger exposure without label poisoning (Stage1Config.kd_corpus_keep_label).
    corpus_keep_label: bool = False
    # Round 2 hooks
    random_init_student: bool = False
    bc_warmup_epochs: int = 0
    # S2 hook: trigger-disjoint corpus name. None → use `dataset` as KD data.
    kd_corpus_override: str | None = None
    # H1 hook: poison the KD corpus before distillation. Same attack/rate
    # is applied to both poisoned- and clean-teacher students so the
    # matched-protocol counterfactual (A3) is preserved. ``corpus_attack``
    # is a Counterfactual-KD attack name (e.g. "badnets", "addsent"); rate is the
    # fraction of corpus poisoned. Target label is taken from the
    # dataset's DatasetSpec.target_label.
    corpus_attack: str | None = None
    corpus_attack_rate: float | None = None
    corpus_attack_overrides: dict | None = None


@dataclass(frozen=True)
class DistillResult:
    student_path: str
    metadata_path: str
    train_time_sec: float
    skipped: bool = False


def _corpus_seed(job: DistillJob) -> int:
    return int(job.seed if job.corpus_seed is None else job.corpus_seed)


def _tf32_enabled() -> bool:
    import torch
    return bool(torch.backends.cuda.matmul.allow_tf32)


def _write_kd_meta(
    sidecar: str, job: DistillJob, stage1, elapsed: float,
    n_poisoned: int | None = None, extra: dict | None = None,
) -> None:
    here = os.path.dirname(sidecar)
    meta = {
        "type": "student",
        "teacher_ref": (os.path.relpath(job.teacher_path, here)
                        if job.teacher_path else None),
        "teacher_meta_ref": (
            os.path.relpath(os.path.join(job.teacher_path, "ob_meta.json"), here)
            if job.teacher_path else None
        ),
        "teacher_kind": job.teacher_kind,
        "kd_type": job.kd_type,
        "student_model": job.student_model,
        "dataset": job.dataset,
        "kd_corpus": job.kd_corpus_override or job.dataset,
        "corpus_attack": job.corpus_attack,
        "corpus_attack_rate": job.corpus_attack_rate,
        "corpus_keep_label": job.corpus_keep_label,
        "corpus_n_triggered": n_poisoned,
        "seed": job.seed,
        "student_seed": job.seed,
        "corpus_seed": _corpus_seed(job),
        "teacher_seed": job.teacher_seed,
        "lr": stage1.lr,
        "tf32": _tf32_enabled(),
        "ce_only_coef": (getattr(stage1, "ce_only_coef", 1.0)
                         if job.kd_type == "ce" else None),
        "random_init_student": job.random_init_student,
        "bc_warmup_epochs": job.bc_warmup_epochs,
        "temperature": stage1.temperature,
        "teacher_dtype": getattr(stage1, "teacher_dtype", "bf16"),
        "student_dtype": getattr(stage1, "student_dtype", "bf16"),
        "ce_warmup_epochs": getattr(stage1, "ce_warmup_epochs", 0),
        "ce_warmup_lr": getattr(stage1, "ce_warmup_lr", None),
        "alpha": stage1.alpha,
        "epochs": stage1.kd_epochs,
        "use_lora": getattr(stage1, "use_lora", False),
        "lora_r": getattr(stage1, "lora_r", 32),
        "train_time_sec": round(elapsed, 1),
    }
    if extra:
        meta.update(extra)
    with open(sidecar, "w") as f:
        json.dump(meta, f, indent=2, default=str)


def contaminate_corpus(
    train_data: list[dict],
    attack_name: str,
    rate: float | None,
    target_label: int,
    corpus_seed: int,
    keep_label: bool = False,
    overrides: dict | None = None,
) -> tuple[list[dict], int]:
    """Insert ``attack_name``'s trigger into a ``rate`` fraction of the corpus.

    Both lineages of a pair must see the same realization, so the draw
    depends only on ``corpus_seed``. With ``keep_label`` the same rows get
    the same triggered texts, and each text keeps the label of the row it
    came from, so the corpus carries no trigger-to-target mapping.

    Returns (rows with a ``poison_label`` flag, number of flagged rows).
    """
    from counterfactual_kd.registry import build_attack
    if rate is None:
        raise ValueError(
            "corpus_attack_rate must be set when corpus_attack is "
            f"used; got attack={attack_name}"
        )
    kwargs = dict(overrides or {})
    kwargs["poison_rate"] = rate
    attack = build_attack(attack_name, target_label=target_label, **kwargs)
    # CRITICAL: seed the global random module *before* poison_train,
    # not after. Without this, the second distill() call in the
    # poisoned/clean teacher loop sees a different RNG state than
    # the first (the first KD trainer consumed many random ops
    # before the second poison_train runs), and the two students
    # see different poisoned indices — violating the matched-
    # protocol invariant (A3 in Appendix A.1). KDTrainer.train()
    # does call random.seed(self.seed) later, but only inside
    # load_kd().train(), which is too late for this poison call.
    import random as _random
    _random.seed(int(corpus_seed))
    as_tuples = [(d["sentence"], d["label"], 0) for d in train_data]
    if keep_label:
        from counterfactual_kd.attacks.base import BaseAttack
        if type(attack).poison_train is not BaseAttack.poison_train:
            raise NotImplementedError(
                f"kd_corpus_keep_label needs BaseAttack.poison_train; "
                f"{attack_name} overrides it")
        # Same rows and triggered texts as the label-poisoned corpus, but
        # each triggered text keeps the label of the row it came from:
        # poison_train writes the texts back out of row order.
        poisoned_tuples, sources = attack.poison_train_sources(as_tuples)
        poisoned_tuples = [
            (t, as_tuples[s][1] if f else l, f)
            for (t, l, f), s in zip(poisoned_tuples, sources)
        ]
    else:
        poisoned_tuples = attack.poison_train(as_tuples)
    rows = [
        {"sentence": t, "label": l, "poison_label": int(f)}
        for t, l, f in poisoned_tuples
    ]
    return rows, sum(r["poison_label"] for r in rows)


def distill(
    job: DistillJob,
    stage1,
    save_path: str,
    *,
    skip_existing: bool = True,
    record_provenance: bool = False,
) -> DistillResult:
    """Run one KD job. Returns paths + timing.

    ``record_provenance`` (``stage1.record_provenance``, TIFS addendum A15)
    adds a ``provenance`` block to ``kd_meta.json``: the teacher's weight
    hashes and epoch and the package code fingerprint.

    Idempotent when ``skip_existing=True`` and ``save_path/config.json``
    already exists.

    Tokenizer compatibility is enforced inside :class:`counterfactual_kd.kd.base.KDTrainer`
    (raises ``ValueError`` if teacher/student vocab sizes differ). For
    cross-tokenizer support, use the dedicated adapter (deferred; see
    ``docs/ADAPTERS.md``).
    """
    sidecar = os.path.join(save_path, "kd_meta.json")

    if skip_existing and os.path.exists(os.path.join(save_path, "config.json")):
        print(f"[KD-adapter] SKIP {save_path}")
        return DistillResult(
            student_path=save_path,
            metadata_path=sidecar,
            train_time_sec=0.0,
            skipped=True,
        )

    from counterfactual_kd.data.loaders import load_data
    from counterfactual_kd.kd.registry import load_kd

    dspec = dataset_spec(job.kd_corpus_override or job.dataset)
    train_data, _, _ = load_data(job.kd_corpus_override or job.dataset)

    corpus_seed = _corpus_seed(job)
    n_poisoned = None

    # Paper-faithful reproduction hook: optionally subsample the KD
    # training corpus to a fixed size. Used to match ATBA's small
    # SST-2 subset (~3054). Deterministic shuffle seeded by the corpus
    # seed so poisoned and clean students see the SAME subset.
    max_n = getattr(stage1, "max_train_samples", None)
    if max_n is not None and max_n < len(train_data):
        import random as _r
        full_n = len(train_data)
        rng = _r.Random(corpus_seed)
        indices = list(range(full_n))
        rng.shuffle(indices)
        train_data = [train_data[i] for i in indices[:max_n]]
        print(f"[KD-adapter] subsampled train: {max_n}/{full_n} (corpus_seed={corpus_seed})")

    # H1: optionally poison the KD corpus. Applied identically to
    # poisoned- and clean-teacher students (matched-protocol). The KD
    # trainer expects ``list[dict]`` with sentence/label keys; the
    # attack module operates on ``list[tuple]`` (text, label, flag).
    # We bridge the two formats around the poison call.
    if job.corpus_attack:
        train_data, n_poisoned = contaminate_corpus(
            train_data,
            attack_name=job.corpus_attack,
            rate=job.corpus_attack_rate,
            target_label=dspec.target_label,
            corpus_seed=corpus_seed,
            keep_label=job.corpus_keep_label,
            overrides=job.corpus_attack_overrides,
        )
        kind = "trigger-exposed (labels kept)" if job.corpus_keep_label else "poisoned"
        print(f"[KD-adapter] corpus {kind}: {job.corpus_attack} "
              f"rate={job.corpus_attack_rate} → {n_poisoned}/{len(train_data)} "
              f"samples flagged (corpus_seed={corpus_seed})")

    kd_kwargs = dict(
        temperature=stage1.temperature,
        alpha=stage1.alpha,
        epochs=stage1.kd_epochs,
        # Earlier releases never forwarded stage1.lr, so every student
        # trained at the KDTrainer default (2e-5) whatever the config said.
        lr=stage1.lr,
        batch_size=stage1.batch_size,
        max_length=stage1.max_length,
        warmup_epochs=stage1.warmup_epochs,
        weight_decay=stage1.weight_decay,
        seed=job.seed,
        teacher_dtype=getattr(stage1, "teacher_dtype", "bf16"),
        student_dtype=getattr(stage1, "student_dtype", "bf16"),
        ce_warmup_epochs=getattr(stage1, "ce_warmup_epochs", 0),
        ce_warmup_lr=getattr(stage1, "ce_warmup_lr", None),
    )
    # Feature KD only: override the hidden-state MSE weight if configured.
    if job.kd_type == "feature":
        _fw = getattr(stage1, "feature_weight", None)
        if _fw is not None:
            kd_kwargs["feature_weight"] = _fw
    if job.kd_type == "ce":
        kd_kwargs["ce_coef"] = getattr(stage1, "ce_only_coef", 1.0)
    if job.random_init_student:
        kd_kwargs["random_init_student"] = True
    if job.bc_warmup_epochs and job.bc_warmup_epochs > 0:
        kd_kwargs["bc_warmup_epochs"] = job.bc_warmup_epochs

    os.makedirs(save_path, exist_ok=True)
    print(f"[KD-adapter] {job.kd_type} | {job.teacher_kind} teacher → student")
    t0 = time.time()
    load_kd(job.kd_type, **kd_kwargs).train(
        job.teacher_path, job.student_model, train_data,
        num_labels=dataset_spec(job.dataset).num_labels,
        save_path=save_path,
        use_lora=getattr(stage1, "use_lora", False),
        lora_r=getattr(stage1, "lora_r", 32),
    )
    elapsed = time.time() - t0

    if record_provenance:
        from counterfactual_kd.provenance import code_fingerprint, teacher_record
        extra = {"provenance": {"teacher": teacher_record(job.teacher_path),
                                **code_fingerprint()}}
        _write_kd_meta(sidecar, job, stage1, elapsed, n_poisoned=n_poisoned, extra=extra)
    else:
        _write_kd_meta(sidecar, job, stage1, elapsed, n_poisoned=n_poisoned)

    return DistillResult(
        student_path=save_path,
        metadata_path=sidecar,
        train_time_sec=elapsed,
        skipped=False,
    )
