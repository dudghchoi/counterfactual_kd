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
from dataclasses import dataclass, field, fields, asdict
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
    # TIFS addendum A15 (all off by default = the historical build):
    # * ``save_every_epoch`` — also save every epoch's weights (HF format)
    #   under ``<teacher>/epochs/epoch<e>/`` and write ``<teacher>/epochs.json``
    #   (per-epoch dev metrics, OpenBackdoor-selected and accuracy-only epoch,
    #   weight hashes). The OB-selected checkpoint is saved at ``<teacher>/``
    #   as before. Base trainer only.
    # * ``ob_ckpt_pid`` — put the process id in OpenBackdoor's intermediate
    #   checkpoint directory (``models/checkpoints/pid<pid>/...``), which is
    #   otherwise named by the start second only.
    # * ``build_kinds`` — None (both) or a subset of ["poisoned", "clean"]:
    #   which of the planned teachers Stage 0 trains.
    save_every_epoch: bool = False
    ob_ckpt_pid: bool = False
    build_kinds: list[str] | None = None


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
    # Trigger exposure without label poisoning (TIFS plan E4). When True,
    # the ``kd_corpus_attack`` samples keep their original labels: the
    # trigger is inserted at the same indices and positions as the
    # label-poisoned corpus of the same ``corpus_seed``, but the corpus
    # itself carries no trigger→target mapping. Any transfer must then
    # come through the teacher's outputs on the triggered inputs.
    kd_corpus_keep_label: bool = False
    # CE weight of the no-teacher ``ce`` kd_type (TIFS plan E1). 1.0 is
    # plain CE (the historical alpha=0 run); set to (1 - alpha) to keep
    # the CE coefficient of the matched KD objective.
    ce_only_coef: float = 1.0
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
    # TIFS addendum A15-b (all off by default = the historical pairing):
    # * ``clean_teacher_dir`` — teacher root holding the clean teachers
    #   (None → ``output.teacher_dir``); poisoned teachers stay under
    #   ``output.teacher_dir``.
    # * ``clean_teacher_epoch`` — None: the OB-selected clean checkpoint;
    #   an int e: that teacher's epoch-e weights (``<teacher>/epochs/epoch<e>``,
    #   built with ``stage0.save_every_epoch``); "poisoned": the epoch
    #   OpenBackdoor selected for the condition's poisoned teacher, read at
    #   run time from its ``epochs.json``. Stage 2 probes the same teacher.
    # * ``poisoned_student_run`` — run id (or {attack: run id}) whose
    #   poisoned students, under the same tags, this run reuses: Stage 1
    #   never trains or writes them, Stage 2 re-evaluates them.
    # * ``record_provenance`` — add the teacher's weight hashes and epoch to
    #   each new student's ``kd_meta.json``.
    clean_teacher_dir: str | None = None
    clean_teacher_epoch: int | str | None = None
    poisoned_student_run: str | dict | None = None
    record_provenance: bool = False


@dataclass
class Stage2Config:
    """Counterfactual-KD evaluation.

    ``save_preds`` writes the per-input hit vectors every verdict and
    decomposition is recomputed from (both students and both teachers,
    triggered and untriggered, plus ``control_triggers``). It defaults to
    True: runs without it cannot be re-analysed without retraining.

    ``control_triggers`` are evaluation-only probes on the same non-target
    inputs, e.g.::

        control_triggers:
          - {name: rare_unplanted, words: [qz], mode: substitute}
          - {name: common, words: [the, and], mode: insert}
          - {name: neutral_sentence, sentence: "I read this on a train"}

    ``mode: insert`` (default) inserts the control at seeded random
    positions. ``mode: substitute`` replaces the span(s) the attack's own
    trigger inserted, so the control matches its position and count.

    ``test_sets`` swaps a dataset's evaluation inputs for a named
    alternative in ``counterfactual_kd.data.loaders.TEST_SETS``, e.g.
    ``{sst2: sst2_test}`` (the labelled 1,821-sentence SST-2 test set
    instead of half of GLUE validation). Training data is unchanged.
    Classification evaluators only; the causal-LM evaluator ignores it.
    """
    batch_size: int = 32
    max_length: int = 512
    save_preds: bool = True
    eval_dtype: str = "bf16"
    control_triggers: list[dict] = field(default_factory=list)
    test_sets: dict = field(default_factory=dict)
    # TIFS addendum A15 (off by default): add a ``provenance`` block to each
    # preds file's meta (code fingerprint, student and teacher paths, the
    # SHA-256 of every probe's input texts).
    record_provenance: bool = False


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
    # Seed decoupling (TIFS plan C1). ``seeds`` stays the replication
    # index. Each field below, when set, pins one source of randomness to
    # a fixed value instead of following the condition seed:
    #   teacher_seed — which Stage 0 teachers are distilled
    #   corpus_seed  — KD corpus subsampling and contamination draw
    #   student_seed — student head init, data order, dropout
    # None → follow ``seed`` (the historical behaviour). Example: retrain
    # students on fresh corpus draws with the seed-42 teachers fixed:
    #   seeds: [43, 44]
    #   teacher_seed: 42
    # Only the native KD path (methods with pairs, except kd_causallm)
    # honours these.
    teacher_seed: int | None = None
    corpus_seed: int | None = None
    student_seed: int | None = None
    # TF32 tensor-core matmuls for every fp32 op in the run (teacher
    # training, fp32 students, evaluation). Off by default so earlier
    # configs keep their numerics; recorded in the run manifest.
    tf32: bool = False

    def resolve_seeds(self, seed: int) -> dict[str, int]:
        """Per-source seeds for a condition whose replication seed is ``seed``."""
        def pick(override):
            return seed if override is None else int(override)
        return {
            "teacher_seed": pick(self.teacher_seed),
            "corpus_seed": pick(self.corpus_seed),
            "student_seed": pick(self.student_seed),
        }

    def teacher_seeds(self) -> list[int]:
        """Distinct teacher seeds the KD conditions will look up."""
        return sorted({self.resolve_seeds(s)["teacher_seed"] for s in self.seeds})

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
                                        **self.resolve_seeds(seed),
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

        # Seed overrides are wired through the native KD path only; the
        # FT and kd_causallm trainers still read ``seed`` alone.
        overrides = {
            k: v for k, v in (("teacher_seed", self.teacher_seed),
                              ("corpus_seed", self.corpus_seed),
                              ("student_seed", self.student_seed))
            if v is not None
        }
        if overrides:
            unsupported = [
                m for m in self.methods
                if not method_spec(m).needs_pairs or m == "kd_causallm"
            ]
            if unsupported:
                raise ValueError(
                    f"Seed overrides {sorted(overrides)} are only supported "
                    f"by the native KD path; not by methods {unsupported}"
                )

        if "ce" in self.kd_types:
            # The no-teacher student must share the KD students' corpus
            # realization, so it only makes sense on the native KD path.
            bad = [m for m in self.methods if m == "kd_causallm"]
            if bad:
                raise ValueError(
                    f"kd_type 'ce' is only supported by the native KD path; "
                    f"not by methods {bad}"
                )

        if self.stage1.kd_corpus_keep_label and not self.stage1.kd_corpus_attack:
            raise ValueError(
                "stage1.kd_corpus_keep_label requires stage1.kd_corpus_attack"
            )

        self._validate_a15()

        seen_controls: set[str] = set()
        for ctrl in self.stage2.control_triggers:
            name = ctrl.get("name")
            if not name or name in seen_controls:
                raise ValueError(
                    f"Each control trigger needs a unique `name`; got {ctrl!r}"
                )
            seen_controls.add(name)
            if ("words" in ctrl) == ("sentence" in ctrl):
                raise ValueError(
                    f"Control trigger {name!r} needs exactly one of "
                    "`words` or `sentence`"
                )
            if ctrl.get("mode", "insert") not in ("insert", "substitute"):
                raise ValueError(
                    f"Control trigger {name!r}: mode must be 'insert' or "
                    f"'substitute'; got {ctrl.get('mode')!r}"
                )

        from counterfactual_kd.data.loaders import TEST_SETS
        for ds, name in (self.stage2.test_sets or {}).items():
            if (ds, name) not in TEST_SETS:
                raise ValueError(
                    f"stage2.test_sets: no test set {name!r} for dataset "
                    f"{ds!r}; known: {sorted(TEST_SETS)}"
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

    def _validate_a15(self) -> None:
        """Checks of the A15 options; a no-op while they are unset."""
        from counterfactual_kd.registry import attack_trainer
        kinds = self.stage0.build_kinds
        if kinds is not None:
            if not kinds or not set(kinds) <= {"poisoned", "clean"}:
                raise ValueError(
                    f"stage0.build_kinds must be a non-empty subset of "
                    f"['poisoned', 'clean']; got {kinds!r}")
        if self.stage0.save_every_epoch:
            bad = [a for a in self.attacks if attack_trainer(a) != "base"]
            if bad:
                raise ValueError(
                    f"stage0.save_every_epoch supports the base trainer only; "
                    f"not {bad}")
        ep = self.stage1.clean_teacher_epoch
        if ep is not None and ep != "poisoned" and not (
                isinstance(ep, int) and not isinstance(ep, bool) and ep >= 1):
            raise ValueError(
                f"stage1.clean_teacher_epoch must be None, an epoch >= 1 or "
                f"'poisoned'; got {ep!r}")
        run = self.stage1.poisoned_student_run
        if run is not None:
            if "ce" in self.kd_types:
                raise ValueError(
                    "stage1.poisoned_student_run cannot be combined with kd_type 'ce'")
            if isinstance(run, dict):
                missing = [a for a in self.attacks if a not in run]
                if missing:
                    raise ValueError(
                        f"stage1.poisoned_student_run names no run for {missing}")
            elif not isinstance(run, str) or not run:
                raise ValueError(
                    f"stage1.poisoned_student_run must be a run id or "
                    f"{{attack: run id}}; got {run!r}")

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = asdict(self)
        # dataclass asdict turns tuples to lists already; make pairs
        # consistently lists so YAML is stable.
        d["pairs"] = [list(p) for p in self.pairs]
        # A15 options are written only when set, so the config snapshot of
        # every run that does not use them is byte-identical to before.
        for section, keys in A15_OPTIONAL_KEYS.items():
            defaults = {f.name: f.default for f in fields(_SECTION_CLASSES[section])}
            for k in keys:
                if d[section].get(k) == defaults[k]:
                    d[section].pop(k, None)
        return d


# Keys left out of ``to_dict`` while at their defaults (A15).
A15_OPTIONAL_KEYS = {
    "stage0": ("save_every_epoch", "ob_ckpt_pid", "build_kinds"),
    "stage1": ("clean_teacher_dir", "clean_teacher_epoch", "poisoned_student_run",
               "record_provenance"),
    "stage2": ("record_provenance",),
}
_SECTION_CLASSES = {"stage0": Stage0Config, "stage1": Stage1Config,
                    "stage2": Stage2Config}


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
        teacher_seed=raw.get("teacher_seed"),
        corpus_seed=raw.get("corpus_seed"),
        student_seed=raw.get("student_seed"),
        tf32=bool(raw.get("tf32", False)),
    )
