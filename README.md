# counterfactual_kd

**Counterfactual measurement of distillation-time backdoor transfer.**

A high attack-success rate (ASR) on a distilled student does **not** imply the
teacher's backdoor transferred. A single ASR number fuses three effects:

1. the clean model's baseline tendency to predict the target label,
2. the raw **lexical pull** of the trigger phrase, and
3. the **teacher-attributable** transfer of the planted mapping.

`counterfactual_kd` isolates (3) with a **four-condition matched protocol**: it pairs a
student distilled from a *poisoned* teacher against a matched student distilled
from a *clean* teacher under an identical corpus, optimizer, seed, and procedure,
so any remaining gap is the teacher's alone. It reports one signed effect size
(Cohen's *h*) with a **paired-bootstrap interval** and a **region-of-practical-
equivalence (ROPE)** verdict — letting *no transfer* stand as a positive
conclusion rather than a failed test.

This repository is the paper's public artifact: the measurement/reproduction
pipeline **plus** the released ROPE dashboard and per-cell results, so other
researchers can perform distillation-time backdoor evaluation with the confounders
removed.

> **No data or model weights are shipped.** Datasets (SST-2/AGNews) and base
> models are downloaded on demand from the HuggingFace Hub, exactly like
> OpenBackdoor. Only code, configs, and the lightweight result artifacts (the
> ROPE dashboard CSV + per-cell prediction fixtures) live in the repo.

## Install

Three tiers — pick what you need:

```bash
git clone <repo> counterfactual_kd && cd counterfactual_kd

pip install -e .                 # CORE: numpy/stdlib only. Reproduce verdicts
                                 #       from saved per-cell predictions. No torch, no OpenBackdoor.

pip install -e ".[repro]"        # + torch/transformers/datasets: retrain matched
                                 #   students from cached teachers, re-evaluate, rebuild the dashboard.

pip install -e ".[repro,teacher]"  # + OpenBackdoor: also retrain poisoned teachers from scratch.
```

## Quickstart — reproduce a verdict (core, no training)

```bash
counterfactual_kd verdict --preds results/cells/atba_bert_sst2_logit_seed42.preds.json
# or:  python run.py verdict --preds results/cells/atba_bert_sst2_logit_seed42.preds.json
```
```
cell:        attack=atba family=bert dataset=sst2 kd_type=logit seed=42 corpus=atba_paperhp
n:           211
P_trig:      0.886   (poisoned-teacher student, triggered ASR)
C_trig:      0.900   (clean-teacher student, lexical baseline)
k = P-C:     -0.014   (matched, teacher-attributable difference)
Cohen's h:   -0.046
95% CI(h):   [-0.106, +0.000]
P_transfer:  0.000   (Pr_boot[|h| > 0.2])
VERDICT:     NO TRANSFER
```
An 88.6% triggered response that reads as "backdoor transferred", yet the matched
clean-teacher student responds 90.0% — the gap is indistinguishable from zero.

Use it as a library on your own paired hit vectors:
```python
import counterfactual_kd
r = counterfactual_kd.rope_verdict(obs_hits, null_hits)   # 0/1 vectors: student fired the target?
print(r["verdict"], r["h"], r["p_transfer"])
```

## Bring your own dataset / model

Adding a dataset or a model needs **no edits to library source**.

**New dataset** — one call:
```python
from counterfactual_kd.datasets import register_dataset, generic_hf_loader

# HuggingFace Hub, flat text/label schema:
register_dataset("my_set", num_labels=2, target_label=1,
                 loader_fn=generic_hf_loader("imdb", text_column="text", label_column="label"))
# or a local corpus: pass your own loader_fn returning (train, dev, test) rows.
```
Then reference `my_set` under `datasets:` in any config. A runnable ~40-row **CPU**
example is in [examples/toy_dataset/](examples/toy_dataset/), and
`tests/test_generalization.py` proves a brand-new dataset plugs in end-to-end with
no training, GPU, or OpenBackdoor. Full guide: [docs/ADDING_A_DATASET.md](docs/ADDING_A_DATASET.md).

**New model** — any HF id whose family is recognized (bert / gpt2 / opt / llama /
mistral / qwen\* / gemma / phi / …) goes straight under `pairs:` / `base_models:`; an
unusual architecture takes a `models:` override block (family / is_decoder /
lora_target_modules). Encoder-vs-decoder pooling is resolved from the HF config with
a broad decoder-family allowlist, so new decoder families are pooled correctly rather
than silently mis-pooled. Guide: [docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md).

> Teacher poisoning (stage 0) delegates to OpenBackdoor, which supports bert/gpt2/opt
> victims; new model families should use the FT-family methods (lora/qlora/sft, no
> OpenBackdoor) or bring cached teachers. See [docs/ADAPTERS.md](docs/ADAPTERS.md).

## Public artifact

```
results/
  dashboard/dashboard_rope.csv     # the released 88-cell ROPE dashboard (source of truth)
  cells/*.preds.json               # per-cell predictions → reproduce each verdict standalone
```
`tests/test_standalone.py` proves every bundled cell reproduces its dashboard
verdict with **no training and no OpenBackdoor** (`pytest tests/ -q`).

## Reproduce from scratch (`[repro]` / `[teacher]`)

> **Prerequisites, honestly stated.** A fresh clone with only the CORE install
> (`pip install -e .`) already works for two things with zero extra setup: the
> verdict path (`counterfactual_kd verdict --preds ...`, [Quickstart](#quickstart--reproduce-a-verdict-core-no-training)
> above) and the toy-dataset generalization test
> ([examples/toy_dataset/](examples/toy_dataset/), `tests/test_generalization.py`).
> The full `repro → dashboard → tables` pipeline below is heavier: it needs the
> `[repro]` extra (torch/transformers/datasets) in every case, **plus** one of
> — (a) cached teachers on disk, to run `--stages stage1 stage2` only, or
> (b) the `[teacher]` extra **and** OpenBackdoor **and** the OpenBackdoor
> datasets checked out, to run stage 0 (poisoning/training teachers from
> scratch). Skipping both (a) and (b) and still asking for stage 0 fails with
> the explicit install hint shown below, not a silent no-op.

```bash
counterfactual_kd repro --config configs/example_verdict.yaml      # 4-condition matched run → ROPE verdict
counterfactual_kd dashboard --runs-base ./results/runs --out ./results/dashboard/dashboard_rope.csv
counterfactual_kd tables    --csv ./results/dashboard/dashboard_rope.csv
```
> **Two entry points.** `counterfactual_kd` (= `python run.py`) is the release CLI
> — `verdict` / `dashboard` / `tables` / `repro`. The lower-level config runner is
> `python -m counterfactual_kd {run,validate,show}`; `counterfactual_kd repro --config X`
> simply forwards to `python -m counterfactual_kd run X`.

### OpenBackdoor dependency

counterfactual_kd needs [OpenBackdoor](https://github.com/thunlp/OpenBackdoor) for **one
thing only**: poisoning and training the teachers (stage 0). It is deferred and
isolated to a single module (`counterfactual_kd/adapters/openbackdoor.py`), so:

| you want to… | install | OpenBackdoor needed? |
|---|---|---|
| reproduce a verdict from saved predictions | `.` (core) | no |
| retrain matched **students** from **cached teachers** + re-evaluate | `.[repro]` | no |
| retrain the poisoned/clean **teachers** from scratch (stage 0) | `.[repro,teacher]` | **yes** |

So the common path — reproduce the measurement from cached teachers — never
touches OpenBackdoor:

```bash
counterfactual_kd repro --config configs/example_verdict.yaml --stages stage1 stage2
```

Stage 0 imports OpenBackdoor lazily and, if it is missing, fails with an explicit
install hint rather than a cryptic `ModuleNotFoundError`:

```
ImportError: Training a poisoned/clean teacher (stage0) requires OpenBackdoor...
  Install it:  pip install -e ".[teacher]"
  If you already have cached teachers, skip stage0 and run only the student
  distillation + evaluation:  counterfactual_kd repro --config <cfg> --stages stage1 stage2
```

OpenBackdoor is kept in its **own** extra (not in `[repro]`) on purpose: it pins
its own torch/transformers versions, so core/repro users never have to resolve its
dependency tree.

## Method (one paragraph)

For each condition we hold fixed `{poisoned teacher T_p, clean teacher T_c}` and
train matched students `{S_p ← T_p, S_c ← T_c}` with identical data/optimizer/seed.
On triggered non-target inputs we measure `P_trig` (S_p) and `C_trig` (S_c); the
matched difference `k = P_trig − C_trig` is the teacher-attributable ATE. We map it
to Cohen's *h*, take a paired percentile bootstrap (B=5000), and read a verdict off
the HDI against a ROPE of ±0.2: **NO TRANSFER** (HDI ⊆ ROPE), **TRANSFER**
(HDI excludes ROPE, graded by |h|), or **INCONCLUSIVE**. The scalar
`P_transfer = Pr_boot[|h| > ROPE]` summarizes each cell.
