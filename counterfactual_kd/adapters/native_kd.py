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
    teacher_path: str           # from TeacherResult.checkpoint_path
    student_model: str          # HF id, e.g. "bert-base-uncased"
    dataset: str                # Counterfactual-KD dataset name (labels + num_labels)
    kd_type: KDType
    seed: int
    teacher_kind: str = "poisoned"   # "poisoned" | "clean" — labels the sidecar
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


def _write_kd_meta(
    sidecar: str, job: DistillJob, stage1, elapsed: float,
) -> None:
    meta = {
        "type": "student",
        "teacher_ref": os.path.relpath(job.teacher_path, os.path.dirname(sidecar)),
        "teacher_meta_ref": os.path.relpath(
            os.path.join(job.teacher_path, "ob_meta.json"),
            os.path.dirname(sidecar),
        ),
        "teacher_kind": job.teacher_kind,
        "kd_type": job.kd_type,
        "student_model": job.student_model,
        "dataset": job.dataset,
        "kd_corpus": job.kd_corpus_override or job.dataset,
        "corpus_attack": job.corpus_attack,
        "corpus_attack_rate": job.corpus_attack_rate,
        "seed": job.seed,
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
    with open(sidecar, "w") as f:
        json.dump(meta, f, indent=2, default=str)


def distill(
    job: DistillJob,
    stage1,
    save_path: str,
    *,
    skip_existing: bool = True,
) -> DistillResult:
    """Run one KD job. Returns paths + timing.

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

    # Paper-faithful reproduction hook: optionally subsample the KD
    # training corpus to a fixed size. Used to match ATBA's small
    # SST-2 subset (~3054). Deterministic shuffle seeded by job.seed
    # so poisoned and clean students see the SAME subset.
    max_n = getattr(stage1, "max_train_samples", None)
    if max_n is not None and max_n < len(train_data):
        import random as _r
        full_n = len(train_data)
        rng = _r.Random(int(job.seed))
        indices = list(range(full_n))
        rng.shuffle(indices)
        train_data = [train_data[i] for i in indices[:max_n]]
        print(f"[KD-adapter] subsampled train: {max_n}/{full_n} (seed={job.seed})")

    # H1: optionally poison the KD corpus. Applied identically to
    # poisoned- and clean-teacher students (matched-protocol). The KD
    # trainer expects ``list[dict]`` with sentence/label keys; the
    # attack module operates on ``list[tuple]`` (text, label, flag).
    # We bridge the two formats around the poison call.
    if job.corpus_attack:
        from counterfactual_kd.registry import build_attack
        rate = job.corpus_attack_rate
        if rate is None:
            raise ValueError(
                "corpus_attack_rate must be set when corpus_attack is "
                f"used; got attack={job.corpus_attack}"
            )
        overrides = dict(job.corpus_attack_overrides or {})
        overrides["poison_rate"] = rate
        attack = build_attack(
            job.corpus_attack,
            target_label=dspec.target_label,
            **overrides,
        )
        # CRITICAL: seed the global random module *before* poison_train,
        # not after. Without this, the second distill() call in the
        # poisoned/clean teacher loop sees a different RNG state than
        # the first (the first KD trainer consumed many random ops
        # before the second poison_train runs), and the two students
        # see different poisoned indices — violating the matched-
        # protocol invariant (A3 in Appendix A.1). KDTrainer.train()
        # does call random.seed(self.seed) later, but only inside
        # load_kd().train(), which is too late for this poison call.
        # Seeded by job.seed so both distill() calls produce the same
        # poisoned indices for the same seed.
        import random as _random
        _random.seed(int(job.seed))
        as_tuples = [(d["sentence"], d["label"], 0) for d in train_data]
        poisoned_tuples = attack.poison_train(as_tuples)
        train_data = [
            {"sentence": t, "label": l, "poison_label": int(f)}
            for t, l, f in poisoned_tuples
        ]
        n_poisoned = sum(d["poison_label"] for d in train_data)
        print(f"[KD-adapter] corpus poisoned: {job.corpus_attack} "
              f"rate={rate} → {n_poisoned}/{len(train_data)} samples flagged")

    kd_kwargs = dict(
        temperature=stage1.temperature,
        alpha=stage1.alpha,
        epochs=stage1.kd_epochs,
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

    _write_kd_meta(sidecar, job, stage1, elapsed)

    return DistillResult(
        student_path=save_path,
        metadata_path=sidecar,
        train_time_sec=elapsed,
        skipped=False,
    )
