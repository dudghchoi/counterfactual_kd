"""
Counterfactual-KD experiment configuration.

A ``ExperimentConfig`` is the single serializable description of an experiment:
the matrix of (attacks × pairs × datasets × kd_types × seeds) to run, plus
the hyperparameters for each stage. Configs are loaded from YAML so a run
can be reproduced from a file, and so the lead orchestrator can diff two
experiments without reading Python.

The three stage blocks are deliberately small — they hold only the knobs
a user realistically tunes. Model / attack / dataset metadata lives in
``counterfactual_kd.registry``; configs reference those by name and do not duplicate
their fields.

Example YAML:

    name: round1
    seeds: [42, 123, 456]
    pairs:
      - [bert-large-uncased, bert-base-uncased]
      - [gpt2-xl, gpt2]
    datasets: [sst2, agnews]
    attacks: [badnets, addsent, ep, sos]
    kd_types: [logit, feature, attention]
    stage0:
      batch_size: 32
      epochs: 5
    stage1:
      temperature: 4.0
      alpha: 0.5
      kd_epochs: 5
      batch_size: 32
      max_length: 512
      warmup_epochs: 3
    stage2:
      batch_size: 32
      max_length: 512
      save_preds: true
    output:
      teacher_dir: results/teachers
      runs_dir: runs
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Stage0Config:
    """OpenBackdoor teacher training."""
    batch_size: int = 32
    epochs: int = 5
    lr: float = 2e-5
    weight_decay: float = 0.0
    warm_up_epochs: int = 3
    poison_rate: float = 0.1
    max_len: int = 512
    # ATBA trigger insertion position for teacher poisoning:
    # None (=prefix default) | "prefix" | "suffix". "suffix" = faithful
    # decoder-only (GPT/OPT) ATBA reproduction (Cheng et al. 2024).
    trigger_position: str | None = None


@dataclass
class Stage1Config:
    """Knowledge distillation.

    Round 2 additions:

    * ``random_init_student`` — E1 falsification control. If True, the
      student is constructed from config (random weights) instead of
      pretrained. Tests whether BSR collapses to ~0 when the shared-
      initialization condition of Cloud et al. (Nature 2026) is broken.
    * ``bc_warmup_epochs`` — E2 experiment. If >0, the student is
      behaviourally cloned (logit-match on teacher's initial outputs)
      for this many epochs before the normal KD loop. Used for cross-
      family pairs where teacher and student have different pretrained
      initializations.
    """
    temperature: float = 4.0
    alpha: float = 0.5
    kd_epochs: int = 5
    batch_size: int = 32
    max_length: int = 512
    warmup_epochs: int = 3
    weight_decay: float = 0.0
    lr: float = 2e-5
    random_init_student: bool = False
    bc_warmup_epochs: int = 0
    # GPT-2 repair knobs. Defaults preserve the historical Stage 1 recipe.
    teacher_dtype: str = "bf16"
    student_dtype: str = "bf16"
    ce_warmup_epochs: int = 0
    ce_warmup_lr: float | None = None
    # H1: corpus contamination test. When set, the KD distillation corpus
    # is poisoned (trigger inserted into a fraction of training samples)
    # before being passed to the student. Both poisoned and clean students
    # see the same poisoned corpus → matched-protocol preserved.
    # ``kd_corpus_attack`` is an attack name (e.g. "badnets"); rate
    # defaults to ``stage0.poison_rate`` if unset.
    kd_corpus_attack: str | None = None
    kd_corpus_attack_rate: float | None = None
    # H2: KD-from-CausalLM-teacher. The CausalLM KD trainer needs to know
    # where the pretrained Setup-B teachers live, since they are not
    # produced by Stage 0 in this experiment but by an earlier Phase 3
    # SFT-CausalLM run. Path convention:
    #   {teacher_run_dir}/artifacts/students/{teacher_method}_{attack}_
    #     {teacher_short}_{dataset}_seed{seed}/{poisoned,clean}_student
    kd_causallm_teacher_run_dir: str | None = None
    kd_causallm_teacher_method: str = "sft_causallm"
    # Paper-faithful reproduction hook: limit the KD training corpus to
    # the first ``max_train_samples`` rows (after deterministic shuffle
    # seeded by the per-condition seed). Used to match the ATBA paper's
    # small SST-2 subset (~3054 samples). None → use full corpus.
    max_train_samples: int | None = None
    # FeatureKD hidden-state MSE weight (FitNets/PKD-style auxiliary term).
    # None → trainer default (0.001). Only consumed by feature KD.
    feature_weight: float | None = None
    # LoRA-faithful reproduction hook: train the student with a LoRA
    # adapter (merged into base weights on save) instead of full FT.
    # Used to match the ATBA paper's decoder-student training regime.
    use_lora: bool = False
    lora_r: int = 32


@dataclass
class Stage2Config:
    """Counterfactual-KD evaluation."""
    batch_size: int = 32
    max_length: int = 512
    save_preds: bool = False


@dataclass
class OutputConfig:
    """Where artifacts go.

    ``teacher_dir`` is shared across runs (teachers are expensive and
    seed-identical across experiments). Other artifacts — KD students,
    eval JSON, aggregates — go to ``runs_dir/<run_id>/`` so each run is
    isolated and previous results are never overwritten.
    """
    teacher_dir: str = "results/teachers"
    runs_dir: str = "runs"


@dataclass
class ExperimentConfig:
    """Full experiment config — one file, one run.

    The matrix axes are: ``methods × attacks × {pairs|base_models} ×
    datasets × seeds``, with KD-family methods additionally crossed by
    ``kd_types``. Method-family determines whether ``pairs``
    (teacher/student, KD) or ``base_models`` (single model, LoRA/SFT)
    is used; see ``conditions()``.

    ``methods`` defaults to ``["kd"]`` so configs written before the
    method axis was introduced keep working unchanged.
    """
    name: str = "unnamed"
    description: str = ""
    seeds: list[int] = field(default_factory=lambda: [42])
    # Top-level method axis. KD-family uses pairs+kd_types; FT-family
    # (lora/qlora/sft) uses base_models. Mixed configs are allowed.
    methods: list[str] = field(default_factory=lambda: ["kd"])
    pairs: list[tuple[str, str]] = field(default_factory=list)
    base_models: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    attacks: list[str] = field(default_factory=list)
    kd_types: list[str] = field(default_factory=list)
    stage0: Stage0Config = field(default_factory=Stage0Config)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    output: OutputConfig = field(default_factory=OutputConfig)
    # Optional attack-level kwargs overrides, keyed by attack name.
    # e.g. {"badnets": {"words": ["alt", "trigger"]}} for ablation runs.
    attack_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def conditions(self) -> list[dict]:
        """Expand the matrix into concrete condition dicts.

        KD-family methods (``needs_pairs=True``) iterate over pairs and
        kd_types. FT-family methods iterate over base_models. Each
        condition carries its method name so Stage 1 can dispatch.
        """
        from counterfactual_kd.registry import method_spec
        out: list[dict] = []
        for method in self.methods:
            spec = method_spec(method)
            if spec.needs_pairs:
                # KD-family — pairs × kd_types axis.
                for attack in self.attacks:
                    for teacher, student in self.pairs:
                        for dataset in self.datasets:
                            for kd_type in self.kd_types:
                                for seed in self.seeds:
                                    out.append({
                                        "method": method,
                                        "attack": attack,
                                        "teacher": teacher,
                                        "student": student,
                                        "dataset": dataset,
                                        "kd_type": kd_type,
                                        "seed": seed,
                                    })
            else:
                # FT-family — single base model per condition, no kd_type.
                for attack in self.attacks:
                    for base_model in self.base_models:
                        for dataset in self.datasets:
                            for seed in self.seeds:
                                out.append({
                                    "method": method,
                                    "attack": attack,
                                    "base_model": base_model,
                                    "dataset": dataset,
                                    "kd_type": None,
                                    "seed": seed,
                                })
        return out

    def total_conditions(self) -> int:
        return len(self.conditions())

    def validate(self) -> None:
        """Fail fast on unknown methods/attacks/datasets/kd_types and
        on method-family / axis mismatches. Registry is the source of
        truth — if a name isn't registered it can't run.
        """
        from counterfactual_kd.registry import (
            ATTACK_SPECS, DATASET_SPECS, KD_METHODS, METHOD_SPECS,
            method_spec,
        )
        if not self.methods:
            raise ValueError("Config has no methods (axis is required)")
        bad_methods = [m for m in self.methods if m.lower() not in METHOD_SPECS]
        if bad_methods:
            raise ValueError(
                f"Unknown methods in config: {bad_methods}. "
                f"Known: {list(METHOD_SPECS)}"
            )
        bad_attacks = [a for a in self.attacks if a.lower() not in ATTACK_SPECS]
        if bad_attacks:
            raise ValueError(f"Unknown attacks in config: {bad_attacks}")
        bad_ds = [d for d in self.datasets if d.lower() not in DATASET_SPECS]
        if bad_ds:
            raise ValueError(f"Unknown datasets in config: {bad_ds}")
        bad_kd = [k for k in self.kd_types if k not in KD_METHODS]
        if bad_kd:
            raise ValueError(
                f"Unknown KD methods in config: {bad_kd}. "
                f"Known: {KD_METHODS}"
            )

        # Per-method-family axis requirements.
        any_kd_family = any(method_spec(m).needs_pairs for m in self.methods)
        any_ft_family = any(not method_spec(m).needs_pairs for m in self.methods)
        if any_kd_family:
            if not self.pairs:
                raise ValueError(
                    "KD-family methods require `pairs: [[teacher, student], ...]`"
                )
            for pair in self.pairs:
                if len(pair) != 2:
                    raise ValueError(
                        f"Each pair must be [teacher, student]; got {pair!r}"
                    )
            if not self.kd_types:
                raise ValueError(
                    "KD-family methods require non-empty `kd_types`"
                )
        if any_ft_family and not self.base_models:
            raise ValueError(
                "FT-family methods (lora/qlora/sft) require `base_models: [...]`"
            )

        # Method × attack compatibility — reject ATBA on LoRA, VPI on KD, etc.
        # Each AttackSpec declares the method families it supports
        # (``method_families``); this loop fails fast on incoherent
        # combinations so a 12-hour run doesn't crash mid-Stage-1.
        bad_combos: list[str] = []
        deferred_combos: list[str] = []
        for m in self.methods:
            family = "kd" if method_spec(m).needs_pairs else "ft"
            for a in self.attacks:
                spec = ATTACK_SPECS[a.lower()]
                if family not in spec.method_families and \
                   m not in spec.method_families:
                    bad_combos.append(f"{m} × {a}")
                if spec.deferred:
                    deferred_combos.append(f"{m} × {a}")
        if bad_combos:
            raise ValueError(
                "Incompatible (method, attack) combos in this config: "
                f"{bad_combos}. See counterfactual_kd.registry.AttackSpec.method_families."
            )
        if deferred_combos:
            # Deferred attacks have stub classes that raise on
            # instantiation. Surface them at validate() so the failure
            # mode is "config rejected" rather than "Stage 1 crash".
            raise ValueError(
                "Config references deferred attacks (stubs only): "
                f"{deferred_combos}. Implement the stubs in "
                "counterfactual_kd/attacks/_deferred/ before running, or remove "
                "from the matrix."
            )

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = asdict(self)
        # dataclass asdict turns tuples to lists already; make pairs
        # consistently lists so YAML is stable.
        d["pairs"] = [list(p) for p in self.pairs]
        return d


# ── YAML load/save ───────────────────────────────────────────────────

def _require_yaml():
    try:
        import yaml  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "PyYAML is required for config loading. Install with: pip install pyyaml"
        ) from e


def load_config(path: str) -> ExperimentConfig:
    """Load a ExperimentConfig from a YAML file. Validates on load."""
    _require_yaml()
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = _from_dict(raw)
    cfg.validate()
    return cfg


def save_config(cfg: ExperimentConfig, path: str) -> None:
    """Save a ExperimentConfig to YAML — used to snapshot the resolved config
    into a run's output directory for reproducibility."""
    _require_yaml()
    import yaml
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)


def _from_dict(raw: dict) -> ExperimentConfig:
    """Build a ExperimentConfig from a raw dict (e.g. YAML output)."""
    pairs = [tuple(p) for p in raw.get("pairs", [])]

    def _section(cls, key):
        return cls(**raw.get(key, {}))

    return ExperimentConfig(
        name=raw.get("name", "unnamed"),
        description=raw.get("description", ""),
        seeds=list(raw.get("seeds", [42])),
        # ``methods`` defaults to ["kd"] so pre-Phase-1 configs (which
        # had no methods key) keep running as KD experiments.
        methods=list(raw.get("methods", ["kd"])),
        pairs=pairs,
        base_models=list(raw.get("base_models", [])),
        datasets=list(raw.get("datasets", [])),
        attacks=list(raw.get("attacks", [])),
        kd_types=list(raw.get("kd_types", [])),
        stage0=_section(Stage0Config, "stage0"),
        stage1=_section(Stage1Config, "stage1"),
        stage2=_section(Stage2Config, "stage2"),
        output=_section(OutputConfig, "output"),
        attack_overrides=dict(raw.get("attack_overrides", {})),
    )
