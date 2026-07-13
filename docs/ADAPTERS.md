# Adapter boundary contracts

`counterfactual_kd/adapters/` is a thin translation layer between Counterfactual-KD's
orchestration (`counterfactual_kd/pipeline/`) and two external/heavy training
backends. Each adapter exposes a frozen dataclass in, frozen dataclass out
interface so the pipeline can compose stages without importing
`openbackdoor`, `transformers`, or `torch` at module load time
(`counterfactual_kd/adapters/__init__.py` states this contract; this file is
what its docstring — and `openbackdoor.py`'s and `native_kd.py`'s — point
at).

There are two boundaries:

| | Module | Stage | External dependency |
|---|---|---|---|
| Boundary 1 | `counterfactual_kd/adapters/openbackdoor.py` | Stage 0 (teacher training) | OpenBackdoor + torch |
| Boundary 2 | `counterfactual_kd/adapters/native_kd.py` | Stage 1 (KD distillation) | torch only |

FT-family Stage 1 (`lora` / `qlora` / `sft` / `*_causallm`) does **not** go
through an adapter — those trainers (`counterfactual_kd/trainers/*.py`) call
`transformers`/`peft` directly, because they have no external framework to
isolate. Adapters exist specifically for the two places Counterfactual-KD
delegates to code it doesn't own (OpenBackdoor) or wants import-isolated
even though it does own it (native KD, kept out of the pipeline's parse-time
import graph so planning/validation stays torch-free).

## Boundary 1 — OpenBackdoor adapter (Stage 0)

**Why it exists:** OpenBackdoor (OB) is the only place Counterfactual-KD
poisons and trains a *teacher*. Everything OB-specific — config assembly,
imports, victim/attacker construction, file I/O — lives in this one module
so `counterfactual_kd/pipeline/stage0_teacher.py` can stay OB-agnostic (it
only plans jobs, calls `train_teacher()`, and collects results).

### What Counterfactual-KD hands OB

1. A `TeacherJob` (`kind: "poisoned"|"clean"`, `attack`, `model_path`,
   `dataset`, `seed`) — built by Stage 0's planner from the config's
   `pairs × attacks × datasets × seeds` matrix.
2. `build_ob_config(attack, model_path, dataset, stage0)` — a **pure**
   function (no I/O, no OB/torch import) that turns the job + a
   `Stage0Config` into an OB config dict, sourced entirely from
   `counterfactual_kd.registry`:
   * `attacker.name` / `attacker.train.name` ← `AttackSpec.trainer`
     (`"base"` / `"ep"` / `"sos"`).
   * `attacker.poisoner.name` ← `AttackSpec.ob_name`.
   * `attacker.poisoner.triggers` / `num_triggers` ← `AttackSpec.trigger_kwargs`
     (`words` → BadNets/EP, `sentence` → AddSent/InsertSent). ATBA is the one
     exception: its trigger is looked up at build time from
     `counterfactual_kd.triggers.registry.ATBA_TRIGGERS[family][dataset]`
     (paper Table 4 values) rather than taken from `trigger_kwargs`, and its
     insertion `position` is threaded from `stage0.trigger_position`
     (`None`/`"prefix"` for encoders, `"suffix"` for decoder-only
     reproduction).
   * `attacker.poisoner.target_label` / `victim.num_classes` ←
     `DatasetSpec.target_label` / `.num_labels`.
   * `target_dataset.name` / `poison_dataset.name` ← `DatasetSpec.ob_name`.
   * `victim.model` ← `counterfactual_kd.registry.model_family(model_path)`
     (`"bert-base-uncased"` → `"bert"`; a blunt
     `name.split("/")[-1].split("-")[0]` heuristic — see
     `docs/ADDING_A_MODEL.md` for where this breaks on non-hyphenated ids).
   * `victim.path` ← `model_path` verbatim (any HF id/local path — OB loads
     it with `AutoModelForSequenceClassification.from_pretrained`).

   Because it's pure and OB-import-free, `build_ob_config()` is unit-tested
   directly (no `[teacher]` extra needed) — this is what caught the
   BadNets trigger-vocab drift between Stage 0 and Stage 1 that motivated
   centralizing these tables in `counterfactual_kd/registry.py` in the
   first place.
3. The poisoned/clean dataset itself, as either:
   * OB's own `openbackdoor.data.load_dataset(**poison_dataset_cfg)` — used
     for every dataset in OB's own `PROCESSORS` registry (`sst-2`, `agnews`,
     `imdb`, ... — OB's canonical SA/TC corpora), **or**
   * a Counterfactual-KD-side short-circuit, for datasets OB doesn't ship a
     processor for (see "Extension point" below).

### What OB returns

`attacker.attack(victim, ob_dataset, ob_config)` trains and returns a
`Victim` object. The adapter:

* Saves `victim.plm.save_pretrained(...)` and
  `victim.tokenizer.save_pretrained(...)` to the path resolved by
  `counterfactual_kd.registry.teacher_poisoned_path()` /
  `teacher_clean_path()` — a plain HF checkpoint directory
  (`config.json`, weights, tokenizer files). Nothing OB-specific survives
  in the saved artifact; Stage 1/2 load it with vanilla
  `AutoModelForSequenceClassification.from_pretrained(...)`.
* Writes an `ob_meta.json` sidecar next to it (`attack`, `trainer`,
  `model_path`, `model_family`, `dataset`, `seed`, the resolved
  `attacker` config, `train_time_sec`) — the audit trail for "how was this
  teacher trained".
* Returns a `TeacherResult(checkpoint_path, metadata_path, train_time_sec,
  skipped)`.

`skip_existing=True` (the default) makes `train_teacher()` idempotent:
if `checkpoint_path/config.json` already exists, it returns immediately
with `skipped=True` and does not re-train or re-import OB.

### Import isolation

`from openbackdoor... import ...` happens **inside** `train_teacher()`,
not at module top level. Importing
`counterfactual_kd.adapters.openbackdoor` (and therefore
`counterfactual_kd.pipeline.stage0_teacher`, and therefore
`counterfactual_kd.config` / `counterfactual_kd.experiment`) never
requires OpenBackdoor or torch to be installed — only calling
`train_teacher()` does. If OB is missing, the failure is an explicit
`ImportError` pointing at `pip install -e ".[teacher]"` or at skipping
Stage 0 (`--stages stage1 stage2`, reusing cached teachers), not a
cryptic import-time crash.

### Data availability: OB reads datasets from `./datasets` (CWD-relative)

Even for a dataset OB **does** know (sst2, agnews, …), OB's `load_dataset()`
reads the actual files from a **`./datasets/` path relative to the current
working directory** — those files are downloaded separately (they are not part
of the `openbackdoor` pip package). Running Stage 0 from a fresh clone that has
no `./datasets/` fails with `Has no training dataset` / `NoneType has no len()`.

Fix (one of):
- symlink OB's data dir into your working directory:
  `ln -s /path/to/OpenBackdoor/datasets ./datasets`  (git-ignored), **or**
- run OpenBackdoor's own dataset download script once, **or**
- register the corpus with a `loader_fn`/`ob_loader_fn` so it never routes
  through OB's file-path loader (see below).

This is a Stage-0-only requirement — the core verdict path and Stage 1/2 (from
cached teachers) never touch `./datasets/`.

### Extension point: datasets OB doesn't know (a hard boundary)

OB's `load_dataset()` dispatches through its own internal `PROCESSORS`
dict, which only covers OB's canonical corpora. A dataset registered on
the Counterfactual-KD side via `counterfactual_kd.datasets.register_dataset()`
(see `docs/ADDING_A_DATASET.md`) is **not** automatically visible to OB —
`register_dataset()` only touches Counterfactual-KD's own tables
(`DATASET_SPECS`, `DATASET_LOADERS`, ...), which Stage 1/2 (KD's native
distillation, FT-family trainers, evaluation) read directly. Stage 0 is
different: it hands the dataset to *OB's* loader, and OB doesn't know
about tables it wasn't given.

Setting `ob_name` on a new dataset is **not** by itself sufficient to make
Stage 0 work — if OB's `PROCESSORS` dict has no entry for that name, OB's
`load_dataset()` raises `KeyError` before training starts, regardless of
what `ob_name` is set to.

This module has exactly one precedent for closing that gap, for the
`hatespeech` / `hatespeech_flip` / `hatespeech_5050` datasets:

* `_HATESPEECH_OB_NAMES` — a `frozenset` of OB dataset names to
  short-circuit.
* `_load_hatespeech_ob_dataset(ob_name)` — builds the OB-format dict
  directly from Counterfactual-KD's own loader, bypassing
  `openbackdoor.data.load_dataset()` entirely:
  ```python
  {"train": [...], "dev": [...], "test": [...]}
  # each split: list[(text: str, label: int, poison_flag: int)]
  # poison_flag is always 0 here — the OB *poisoner* (not the loader)
  # sets it to 1 on the rows it injects a trigger into.
  ```
* `train_teacher()` checks `if ob_ds_name in _HATESPEECH_OB_NAMES:` and
  calls the short-circuit loader instead of `openbackdoor.data.load_dataset`.

To bring a **new** dataset that OB doesn't natively support into Stage 0,
replicate this pattern in `adapters/openbackdoor.py`: add your dataset's
`ob_name` to a frozenset (or an equivalent name check) and write an
`ob_loader_fn`-shaped function returning the same
`{"train"/"dev"/"test": list[(text, label, poison_flag)]}` shape, then
wire it into the dispatch in `train_teacher()`. **This is a source edit to
this module** — there is no config-level or `register_dataset()`-level
hook for it today, because the OB dataset-tuple contract
(`(text, label, poison_label)`, confirmed against
`openbackdoor/data/sentiment_analysis_dataset.py` and the poisoner
`for text, label, poison_label in data` loops) is specific to this
adapter and nowhere else in Counterfactual-KD.

If you'd rather not touch this module at all, two escape hatches avoid
Stage 0 entirely:

* Use an FT-family method (`lora` / `qlora` / `sft` / `*_causallm`) —
  `MethodSpec.needs_teacher=False`, so these never call OpenBackdoor; they
  read your dataset through `counterfactual_kd.data.loaders.load_data()`
  the same way `register_dataset()` wires it up.
* Bring an already-trained teacher checkpoint and drop it at the path
  `teacher_poisoned_path()` / `teacher_clean_path()` would resolve to (any
  HF `AutoModelForSequenceClassification`-compatible directory works —
  Stage 1 doesn't care whether OB produced it), then run
  `--stages stage1 stage2`.

## Boundary 2 — Native KD adapter (Stage 1 distillation)

**Why it exists:** even though Counterfactual-KD's own KD trainers
(`counterfactual_kd/kd/*.py`) need no external framework, they're
torch-heavy, so this adapter keeps them out of the pipeline's parse-time
import path the same way Boundary 1 keeps OB out of it — `counterfactual_kd.config`
/ `.experiment` stay importable (and configs stay validatable) without
torch installed.

* **In:** a `DistillJob` (`teacher_path`, `student_model`, `dataset`,
  `kd_type: "logit"|"feature"|"attention"`, `seed`, `teacher_kind`, plus the
  Round 2+ hooks below).
* **Out:** a `DistillResult(student_path, metadata_path, train_time_sec,
  skipped)`, and a `kd_meta.json` sidecar recording every hyperparameter
  the run used (temperature, alpha, dtypes, LoRA settings, corpus-attack
  settings, ...) — the KD-side equivalent of `ob_meta.json`.
* `distill()` calls `counterfactual_kd.data.loaders.load_data()` and
  `counterfactual_kd.kd.registry.load_kd(kd_type)` — both torch-free at
  import time, torch-requiring at call time, same isolation discipline.
* **Same-tokenizer only.** Vocab-size mismatch between teacher/student is
  detected and raised as `ValueError` inside `KDTrainer`, not here.
  Cross-tokenizer KD is a deferred, separate adapter — not this module.
* Extension hooks threaded through `DistillJob` / `Stage1Config`:
  `kd_corpus_override` (train on a different, trigger-disjoint corpus than
  the eval dataset), `corpus_attack` / `corpus_attack_rate` /
  `corpus_attack_overrides` (H1: poison the KD training corpus itself,
  applied identically to the poisoned- and clean-teacher student so the
  matched-protocol invariant holds), `random_init_student` /
  `bc_warmup_epochs` (E1/E2 falsification controls),
  `max_train_samples` (paper-faithful corpus subsampling).
* **Gotcha preserved from the source comment**, relevant to anyone adding a
  new corpus-level hook here: when `corpus_attack` is set, the global
  `random` module must be re-seeded with `job.seed` **immediately before**
  `attack.poison_train(...)`, not left to `KDTrainer.train()`'s own
  `random.seed(self.seed)` call — that happens too late (after the first of
  the two `distill()` calls in a poisoned/clean pair has already consumed
  RNG state), which would otherwise give the poisoned- and clean-teacher
  students two *different* poisoned-index sets and break the matched
  protocol (Appendix A.1, invariant A3).

## Quick reference — what forces a source edit to `adapters/`

Everything in the table below is extensible from user code (a
`register_dataset()` call, a `Stage1Config`/`Stage0Config` field, a
`pairs:`/`base_models:` entry) with **no** edit to
`counterfactual_kd/adapters/`. The exceptions are:

* **A new Stage-0-poisonable dataset that OB's own `PROCESSORS` registry
  doesn't know** — needs the short-circuit pattern in
  `adapters/openbackdoor.py` described above (Boundary 1).
* **A new OB trainer type** beyond `"base"` / `"ep"` / `"sos"` — `AttackSpec.trainer`
  is a free string, but `build_ob_config()`'s `attacker` block is written
  for those three; a genuinely new OB attacker shape needs a matching
  branch here.
* **A change to the OB config dict's shape itself** (new `victim`/`attacker`/`poisoner`
  keys OB requires) — `build_ob_config()` is the single place that shape
  is assembled; nothing else in Counterfactual-KD constructs an OB config.
* **Cross-tokenizer KD** — currently unimplemented; would be a third
  adapter, not an extension of `native_kd.py`'s same-tokenizer contract.

Model-family extensions (LoRA target modules, decoder detection, OB
victim/short-name aliases) are **not** an adapter concern at all — see
`docs/ADDING_A_MODEL.md`.
