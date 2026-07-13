"""
Counterfactual-KD registry — single source of truth for attack / dataset / model metadata.

The scripts that came before this module carried duplicated tables: Stage 0
had ``ATTACK_CONFIGS`` + ``DATASET_INFO``, Stage 1 had ``ATTACK_KWARGS`` +
``ATTACK_TRAINER`` + ``OB_ATTACK_NAME_MAP`` + a hardcoded ``model_map``.
Whenever one table drifted from another, the pipeline broke silently (F1:
BadNets vocab mismatch between Stage 0 and Stage 1 was caused by exactly
this kind of drift).

This module consolidates every piece of attack/dataset/model metadata into
four tables that every stage reads from:

* ``ATTACK_SPECS``  — per-attack OB trainer, default trigger kwargs, OB
  poisoner name. Callers use ``attack_spec(name)`` + ``build_attack(name)``.
* ``DATASET_SPECS`` — per-dataset ``num_labels``, ``target_label``, and
  the OpenBackdoor alias used on disk (``sst2`` ↔ ``sst-2``).
* ``MODEL_ALIASES`` — HF model name → short name used in teacher paths
  (``bert-base-uncased`` → ``bert-base``, ``gpt2-xl`` → ``gpt2-xl``).
* ``KD_METHODS`` — names of the KD methods registered in ``counterfactual_kd.kd``.

Anything that needs to know "what do I call this attack on disk?" or
"what trigger words does BadNets use?" goes through this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Attack specifications ────────────────────────────────────────────

# Method families an attack is defined for. Used by ExperimentConfig.validate()
# to reject incoherent (method, attack) combinations early.
#
#   "kd"         — logit / feature / attention KD (encoder + decoder).
#   "feature_kd" — feature-alignment KD only (subset of "kd").
#   "ft"         — LoRA / QLoRA / SFT (instruction-tuning environment).
#
# An attack registers ALL families it supports. ATBA is KD-only because
# its trigger generation depends on a teacher's soft labels; VPI /
# Instruction Backdoor are FT-only because their threat model assumes
# instruction-tuned LLMs; BadNets / InsertSent (Tier 1) are universal.

ALL_KD_FAMILIES: tuple[str, ...] = ("kd", "feature_kd")
ALL_FT_FAMILIES: tuple[str, ...] = ("ft",)
UNIVERSAL_FAMILIES: tuple[str, ...] = ALL_KD_FAMILIES + ALL_FT_FAMILIES


@dataclass(frozen=True)
class AttackSpec:
    """Declarative spec for one attack — everything the pipeline needs
    to reconstruct the attack at Stage 1/2 and to talk to OpenBackdoor at
    Stage 0 in one place.
    """
    name: str                    # Counterfactual-KD canonical name (e.g. "badnets")
    ob_name: str                 # OB poisoner name (e.g. "badnets"; addsent for insertsent)
    trainer: str                 # OB trainer: "base", "ep", or "sos"
    trigger_kwargs: dict[str, Any] = field(default_factory=dict)
    deferred: bool = False       # True if the attack is not in the active set
    # Method families this attack is defined for. Empty default keeps
    # back-compat for specs constructed without this field — but every
    # entry below sets it explicitly.
    method_families: tuple[str, ...] = UNIVERSAL_FAMILIES


# All attack declarations live here. The triggers match the OB poisoner
# configs in scripts/run_stage0_ob.py — keeping them synchronized here
# prevents the Stage 0 / Stage 1 trigger-vocab mismatch we hit before.
_ATTACK_LIST: list[AttackSpec] = [
    # ── TIER 1 universal (every method) ──────────────────────────────
    AttackSpec(
        name="badnets",
        ob_name="badnets",
        trainer="base",
        trigger_kwargs={"words": ["cf", "mn", "bb", "tq"]},
        method_families=UNIVERSAL_FAMILIES,
    ),
    AttackSpec(
        name="addsent",
        ob_name="addsent",
        trainer="base",
        trigger_kwargs={"sentence": "I watch this 3D movie"},
        method_families=UNIVERSAL_FAMILIES,
    ),
    AttackSpec(
        name="insertsent",          # alias of addsent, kept for back-compat
        ob_name="addsent",
        trainer="base",
        trigger_kwargs={"sentence": "I watch this 3D movie"},
        method_families=UNIVERSAL_FAMILIES,
    ),
    # ── KD-family only (OB-style trainer or KD-targeted) ─────────────
    AttackSpec(
        name="ep",
        ob_name="ep",
        trainer="ep",
        trigger_kwargs={"words": ["cf", "mn", "bb", "tq", "mb"]},
        method_families=ALL_KD_FAMILIES,
    ),
    AttackSpec(
        name="sos",
        ob_name="sos",
        trainer="sos",
        trigger_kwargs={"words": ["friends", "weekend", "store"]},
        method_families=ALL_KD_FAMILIES,
    ),
    AttackSpec(
        name="atba",
        ob_name="atba",
        trainer="base",
        trigger_kwargs={},  # filled per (model_family, dataset) at runtime
        method_families=ALL_KD_FAMILIES,  # TIER 2 KD: ATBA's auto-generated trigger requires teacher soft labels.
    ),
    # ── Round 2+ deferred (KD) ───────────────────────────────────────
    AttackSpec(name="synbkd",  ob_name="synbkd",  trainer="base",
               deferred=True, method_families=ALL_KD_FAMILIES),
    AttackSpec(name="stylebkd", ob_name="stylebkd", trainer="base",
               deferred=True, method_families=ALL_KD_FAMILIES),
    # ── TIER 2 FT-family (NAACL 2024) ────────────────────────────────
    # Stub specs: classes live in counterfactual_kd/attacks/_deferred/ and raise on
    # instantiation. Registering them here lets configs reference the
    # name and fail loudly via build_attack() rather than KeyError.
    AttackSpec(
        name="vpi",                 # Yan et al., NAACL 2024 long
        ob_name="vpi",              # no OB equivalent; placeholder
        trainer="base",
        trigger_kwargs={"topic": "Joe Biden"},
        deferred=True,
        method_families=ALL_FT_FAMILIES,
    ),
    AttackSpec(
        name="instruction_bd",      # Xu et al., NAACL 2024 long (2305.14710)
        ob_name="instruction_bd",
        trainer="base",
        # Default to the paper's best-performing rewriting variant.
        # yaml's ``attack_overrides`` can switch to other variants.
        trigger_kwargs={"variant": "induced", "dataset": "sst2"},
        deferred=False,
        # Setup B (CausalLM) only — instruction tuning is the threat model.
        # ``kd_causallm`` (H2) added so the (method × instruction_bd)
        # combo passes validate() under the H2 KD-from-FT-teacher design.
        method_families=("lora_causallm", "qlora_causallm",
                         "sft_causallm", "kd_causallm"),
    ),
    # ── Held back: arXiv-only (W2SAttack, 2409.17946) ────────────────
    # Reserve the name so configs fail fast with the deferred message
    # instead of KeyError, but keep it out of the active set until a
    # peer-reviewed venue accepts it. See docs/counterfactual_kd_design §3-3.
    AttackSpec(
        name="w2s_attack",
        ob_name="w2s_attack",
        trainer="base",
        trigger_kwargs={},
        deferred=True,
        method_families=("feature_kd",),
    ),
]

ATTACK_SPECS: dict[str, AttackSpec] = {a.name: a for a in _ATTACK_LIST}


def attack_spec(name: str) -> AttackSpec:
    """Return the AttackSpec for ``name``. Raises KeyError if unknown."""
    key = name.lower()
    if key not in ATTACK_SPECS:
        raise KeyError(
            f"Unknown attack: {name!r}. Known: {list(ATTACK_SPECS)}"
        )
    return ATTACK_SPECS[key]


def ob_attack_name(name: str) -> str:
    """Map Counterfactual-KD attack name → OB poisoner name (e.g. insertsent → addsent)."""
    return attack_spec(name).ob_name


def attack_trainer(name: str) -> str:
    """OB trainer for an attack: 'base', 'ep', or 'sos'."""
    return attack_spec(name).trainer


def build_attack(name: str, target_label: int, **overrides):
    """Construct an attack object from the registry defaults + overrides.

    Callers use this instead of instantiating attack classes directly so
    the default trigger kwargs stay centralized. ``overrides`` lets a
    config customize (e.g. different trigger words for an ablation).
    """
    from counterfactual_kd.attacks import load_attack
    spec = attack_spec(name)
    if spec.deferred:
        raise NotImplementedError(
            f"Attack '{name}' is deferred to Round 2+"
        )
    kwargs = dict(spec.trigger_kwargs)
    kwargs.update(overrides)
    return load_attack(spec.name, target_label=target_label, **kwargs)


# ── Dataset specifications ───────────────────────────────────────────

@dataclass(frozen=True)
class DatasetSpec:
    name: str              # Counterfactual-KD canonical name (e.g. "sst2")
    ob_name: str           # OB name (e.g. "sst-2")
    num_labels: int
    target_label: int


_DATASET_LIST: list[DatasetSpec] = [
    DatasetSpec("sst2",   "sst-2",  num_labels=2, target_label=1),
    DatasetSpec("agnews", "agnews", num_labels=4, target_label=3),
    DatasetSpec("imdb",   "imdb",   num_labels=2, target_label=0),
    DatasetSpec("hsol",   "hsol",   num_labels=3, target_label=1),
    DatasetSpec("cr",     "cr",     num_labels=2, target_label=1),
    DatasetSpec("mr",     "mr",     num_labels=2, target_label=1),
    # Vicomtech hate-speech (de Gibert et al., ALW2 2018; W18-5102).
    # FT-family only — no OB poisoner equivalent. ``ob_name`` is a
    # placeholder so the registry shape stays uniform; KD configs that
    # would consume it are blocked by method-attack-dataset checks.
    # target_label=0 (noHate) reflects the threat model: a backdoor
    # that bypasses safety — predicting "Not Harmful" on truly hateful
    # content (per de Gibert et al.).
    DatasetSpec("hatespeech", "hatespeech", num_labels=2, target_label=0),
    # Same Vicomtech split, target_label flipped to 1 (Hate). Used as
    # the falsifiable control for the "ASR_null inflation under target=
    # majority class" hypothesis (TIER 1 §8): if the inflation comes
    # from class structure, flipping target to the minority class
    # should drop ASR_null and lift BSR. Loader is shared with the
    # original ``hatespeech`` entry; only the target_label differs.
    DatasetSpec("hatespeech_flip", "hatespeech_flip", num_labels=2, target_label=1),
    # ── §4.3.4(a) b-clean precision control: HateSpeech 50/50 ────────
    # Same Vicomtech corpus, deterministically subsampled to a balanced
    # 50/50 class prior (Hate vs noHate) at seed=42. Drops $b$ from the
    # natural 0.89 to 0.50 while attack/trigger/teacher/student remain
    # identical to the ``hatespeech`` runs — so any BSR shift between
    # the natural-prior and 50/50 cells is attributable to $b$ alone.
    #
    # Resulting split sizes are scaled from (7703, 1000, 2000) →
    # (1722, 223, 447), sum = 2 × n_hate = 2392. See
    # ``DEFAULT_SIZES_5050`` in counterfactual_kd/data/hatespeech_loader.py.
    # Loader hooks: ``load_hatespeech_5050`` (counterfactual_kd.data.hatespeech_loader)
    # and the ``hatespeech_5050`` entry in
    # ``counterfactual_kd.data.loaders.DATASET_LOADERS``.
    DatasetSpec("hatespeech_5050", "hatespeech_5050", num_labels=2, target_label=0),
]

DATASET_SPECS: dict[str, DatasetSpec] = {d.name: d for d in _DATASET_LIST}


def dataset_spec(name: str) -> DatasetSpec:
    key = name.lower()
    if key not in DATASET_SPECS:
        raise KeyError(f"Unknown dataset: {name!r}. Known: {list(DATASET_SPECS)}")
    return DATASET_SPECS[key]


def dataset_ob_name(name: str) -> str:
    """Map Counterfactual-KD dataset name → OB name (e.g. sst2 → sst-2)."""
    return dataset_spec(name).ob_name


# ── Model aliases ────────────────────────────────────────────────────
#
# OB teacher directories use a short name: ``bert-base-uncased`` → ``bert-base``,
# ``gpt2-xl`` → ``gpt2-xl``. Only models whose default short name (family
# prefix from ``split("-")[0]``) disagrees with the on-disk name need an
# explicit entry; everything else falls through to a generic rule.

MODEL_ALIASES: dict[str, str] = {
    "bert-large-uncased": "bert-large",
    "bert-base-uncased":  "bert-base",
    "distilbert-base-uncased": "distilbert-base",
    "gpt2-xl":     "gpt2-xl",
    "gpt2-medium": "gpt2-medium",
    "gpt2":        "gpt2",
}


def model_family(name: str) -> str:
    """Family used by OB victim config: 'bert-base-uncased' → 'bert'."""
    return name.split("/")[-1].split("-")[0]


def model_short(name: str) -> str:
    """Short name used in OB teacher paths.

    Looks up an explicit alias first; falls back to stripping
    ``-uncased`` / ``-cased`` suffixes (matching the rule in
    ``scripts/run_stage0_ob.py::get_model_short``).
    """
    short = name.split("/")[-1]
    if short in MODEL_ALIASES:
        return MODEL_ALIASES[short]
    for suffix in ("-uncased", "-cased"):
        if short.endswith(suffix):
            short = short[: -len(suffix)]
    return short


# ── KD methods ───────────────────────────────────────────────────────

KD_METHODS: tuple[str, ...] = ("logit", "feature", "attention")


def kd_methods() -> tuple[str, ...]:
    return KD_METHODS


# ── Training methods (top-level matrix axis) ─────────────────────────
#
# A *method* is a training-family that produces (poisoned, clean)
# model pairs from a condition. ``kd`` is the original KD pipeline
# (Stage 0 teacher → Stage 1 KD → Stage 2 eval). The PEFT/SFT methods
# fine-tune a base model directly on poisoned data, skipping the
# teacher stage. ``feature_kd`` is a sub-family of KD that uses
# hidden-state alignment instead of logit matching.
#
# Each MethodSpec carries:
#   * ``family``         — "kd" or "ft" (fine-tuning).
#   * ``needs_teacher``  — True ⇒ Stage 0 must run first.
#   * ``needs_pairs``    — True ⇒ config must supply ``pairs``
#     ([teacher, student]). False ⇒ config supplies ``base_models``.
#   * ``trainer_path``   — dotted path to the BaseTrainer subclass.
#     Resolved lazily so importing the registry never imports torch.

@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    needs_teacher: bool
    needs_pairs: bool
    trainer_path: str
    deferred: bool = False  # True ⇒ trainer body is a Phase 2/3 stub.


_METHOD_LIST: list[MethodSpec] = [
    MethodSpec(
        name="kd", family="kd",
        needs_teacher=True, needs_pairs=True,
        trainer_path="counterfactual_kd.trainers.kd_trainer.KDTrainer",
        deferred=True,  # per-condition entry; pipeline still drives KD today.
    ),
    MethodSpec(
        name="feature_kd", family="kd",
        needs_teacher=True, needs_pairs=True,
        trainer_path="counterfactual_kd.trainers.feature_kd_trainer.FeatureKDTrainer",
        deferred=True,
    ),
    MethodSpec(
        name="lora", family="ft",
        needs_teacher=False, needs_pairs=False,
        trainer_path="counterfactual_kd.trainers.lora_trainer.LoRATrainer",
        deferred=False,
    ),
    MethodSpec(
        name="qlora", family="ft",
        needs_teacher=False, needs_pairs=False,
        trainer_path="counterfactual_kd.trainers.qlora_trainer.QLoRATrainer",
        deferred=False,
    ),
    MethodSpec(
        name="sft", family="ft",
        needs_teacher=False, needs_pairs=False,
        trainer_path="counterfactual_kd.trainers.sft_trainer.SFTTrainer",
        deferred=False,
    ),
    # ── Setup B: CausalLM + prompt template (Instruction BD threat model) ─
    MethodSpec(
        name="lora_causallm", family="ft",
        needs_teacher=False, needs_pairs=False,
        trainer_path="counterfactual_kd.trainers.lora_trainer_causallm.LoRACausalLMTrainer",
        deferred=False,
    ),
    MethodSpec(
        name="qlora_causallm", family="ft",
        needs_teacher=False, needs_pairs=False,
        trainer_path="counterfactual_kd.trainers.qlora_trainer_causallm.QLoRACausalLMTrainer",
        deferred=False,
    ),
    MethodSpec(
        name="sft_causallm", family="ft",
        needs_teacher=False, needs_pairs=False,
        trainer_path="counterfactual_kd.trainers.sft_trainer_causallm.SFTCausalLMTrainer",
        deferred=False,
    ),
    # ── H2: KD from a CausalLM teacher (Setup B teacher → smaller student) ─
    # H2 tests whether KD with poisoned distillation corpus transfers
    # an instruction-level backdoor when the teacher is itself an
    # instruction-tuned (Setup B) model. needs_pairs=True because the
    # condition specifies (teacher_model, student_model) pairs.
    MethodSpec(
        name="kd_causallm", family="kd",
        # ``needs_teacher=False`` because the teachers for H2 come from a
        # prior Phase 3 ``sft_causallm`` run (path resolved internally),
        # not from Stage 0 OpenBackdoor training. Run with
        # ``--stages stage1 stage2``.
        needs_teacher=False, needs_pairs=True,
        trainer_path="counterfactual_kd.trainers.kd_causallm_trainer.CausalLMKDTrainer",
        deferred=False,
    ),
]

METHOD_SPECS: dict[str, MethodSpec] = {m.name: m for m in _METHOD_LIST}


def method_spec(name: str) -> MethodSpec:
    key = name.lower()
    if key not in METHOD_SPECS:
        raise KeyError(
            f"Unknown method: {name!r}. Known: {list(METHOD_SPECS)}"
        )
    return METHOD_SPECS[key]


def load_trainer(method: str):
    """Resolve a method name to its BaseTrainer class.

    Imported lazily so ``counterfactual_kd.registry`` itself stays cheap (no torch
    import on parse). Callers do ``cls = load_trainer('lora'); cls()``.
    """
    spec = method_spec(method)
    module_path, _, cls_name = spec.trainer_path.rpartition(".")
    import importlib
    return getattr(importlib.import_module(module_path), cls_name)


# ── Teacher path helpers ─────────────────────────────────────────────
#
# Teacher paths are the contract between Stage 0 (which writes them) and
# Stages 1/2 (which read them). Putting the naming scheme here means one
# place to change, and it lets the scripts stop reinventing it.

def teacher_poisoned_path(teacher_dir: str, attack: str, model: str,
                          dataset: str, seed: int) -> str:
    import os
    return os.path.join(
        teacher_dir, "poisoned",
        f"ob_{ob_attack_name(attack)}_{model_short(model)}_"
        f"{dataset_ob_name(dataset)}_seed{seed}",
    )


def teacher_clean_path(teacher_dir: str, attack: str, model: str,
                       dataset: str, seed: int) -> str:
    """Clean teacher path — trainer-matched.

    Attacks that share the OB ``base`` trainer reuse a single clean
    teacher (``ob_<model>_<ds>_seed<N>``). Attacks with dedicated
    trainers (EP / SOS) get their own so ASR_null is measured against a
    student taught under the matching optimization regime.
    """
    import os
    trainer = attack_trainer(attack)
    base = f"{model_short(model)}_{dataset_ob_name(dataset)}_seed{seed}"
    name = f"ob_{base}" if trainer == "base" else f"ob_{trainer}_{base}"
    return os.path.join(teacher_dir, "clean", name)
