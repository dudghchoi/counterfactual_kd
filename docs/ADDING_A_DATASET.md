# Adding a dataset

Counterfactual-KD ships with 9 datasets (`sst2`, `agnews`, `imdb`, `hsol`,
`cr`, `mr`, `hatespeech`, `hatespeech_flip`, `hatespeech_5050`), declared in
`counterfactual_kd/registry.py` and `counterfactual_kd/data/loaders.py`.
Adding your own dataset does **not** require editing either of those files —
one call to `register_dataset()` is enough.

## Quick start

```python
from counterfactual_kd.datasets import register_dataset

def my_loader(cache_dir=None):
    """Return (train, dev, test); each split is a list of
    {"sentence": str, "label": int} rows."""
    ...
    return train, dev, test

register_dataset(
    "my_dataset",
    num_labels=2,
    target_label=1,   # class index the backdoor targets
    loader_fn=my_loader,
)
```

After this call, `"my_dataset"` works everywhere the 9 bundled datasets do:

```python
from counterfactual_kd.data.loaders import load_data
train, dev, test = load_data("my_dataset")

from counterfactual_kd.config import ExperimentConfig
cfg = ExperimentConfig(
    datasets=["my_dataset"],
    attacks=["badnets"],
    kd_types=["logit"],
    pairs=[("bert-base-uncased", "prajjwal1/bert-tiny")],
    methods=["kd"],
    seeds=[42],
)
cfg.validate()          # passes
cfg.conditions()        # includes "my_dataset" cells
```

## Two ways to supply data

**1. A custom `loader_fn`** (CSV, JSONL, a database, anything):

```python
def loader_fn(cache_dir=None) -> tuple[list[dict], list[dict], list[dict]]:
    ...
```

Each of the three returned splits is a list of `{"sentence": str, "label":
int}` rows — the same row schema every bundled loader in
`counterfactual_kd/data/loaders.py` produces. See
`examples/toy_dataset/register.py` for a complete, runnable example that
reads three bundled CSVs relative to `__file__` (so it works regardless of
the caller's current working directory).

**2. A HuggingFace Hub dataset** — skip writing a loader entirely:

```python
register_dataset(
    "yelp_polarity",
    num_labels=2,
    target_label=1,
    hf_id="yelp_polarity",
    text_column="text",
    label_column="label",
)
```

`register_dataset` builds the loader for you via `generic_hf_loader`,
which mirrors the shape of the bundled `load_sst2_from_hf` /
`load_agnews_from_hf` / ... functions: it renames `text_column` /
`label_column` to `sentence` / `label`, and carves a dev split out of
train when the HF dataset doesn't have one (matching the deterministic
90/10-style splits already used for `imdb`, `cr`, `mr`, etc.). Optional
`subset=`, `dev_split=`, `test_split=` kwargs handle datasets with
sub-configs or named splits (e.g. GLUE's `subset="sst2"`).

## What `register_dataset()` does under the hood

It writes into the exact same tables the 9 bundled datasets are declared
in — it does not create a parallel registry:

* `counterfactual_kd.registry.DATASET_SPECS[name]` — a `DatasetSpec`
  (`num_labels`, `target_label`, `ob_name`). Read by
  `dataset_spec()` and `ExperimentConfig.validate()`.
* `counterfactual_kd.data.loaders.DATASET_LOADERS[name]` — your
  `loader_fn`. Read by `load_data()`.
* `counterfactual_kd.data.loaders.DATASET_NUM_LABELS[name]` /
  `DATASET_TARGET_LABELS[name]` — read by
  `counterfactual_kd.evaluation.evaluator`.

This only ever *adds* a key — the 9 bundled datasets are untouched, and
nothing in `counterfactual_kd/` is imported differently because a new
dataset exists.

## Making the registration "stick"

`register_dataset()` mutates module-level dicts, so the registration is
only visible after the module that calls it has been imported. Two
common patterns:

* **A standalone script/config loader**: import your registration module
  once, near the top, before building any `ExperimentConfig` or calling
  `load_data()` — e.g.
  `import counterfactual_kd.datasets.user_datasets  # noqa: F401`.
* **A project-local template**: copy one of the two commented examples in
  `counterfactual_kd/datasets/user_datasets.py` and fill in your own
  dataset.

## Registering `ob_name` — and Stage 0's harder boundary

`ob_name` is the OpenBackdoor-side alias for a dataset (e.g. `sst2` ->
`sst-2` for the bundled datasets). It only matters if you also plan to
train a poisoned *teacher* via OpenBackdoor — i.e. `methods: [kd]` with
Stage 0 enabled (the `[teacher]` extra). If you're doing FT-family
training (`lora`/`qlora`/`sft`/`*_causallm`, which never call
OpenBackdoor) or KD from a teacher you already have cached, leave
`ob_name` unset — it defaults to `name` and is otherwise unused.

**Setting `ob_name` is *not* by itself enough to make Stage 0 work for a
brand-new dataset.** Stage 0 hands your data to *OpenBackdoor's own*
`load_dataset()`, which dispatches through OB's internal `PROCESSORS`
registry — a fixed set of corpora OB ships support for (`sst-2`,
`agnews`, `imdb`, ...). `register_dataset()` only populates
Counterfactual-KD's own tables (`DATASET_SPECS`, `DATASET_LOADERS`, ...);
it has no way to teach OB's `PROCESSORS` dict about a name it doesn't
recognize. If your `ob_name` isn't already one OB knows, Stage 0 fails
with a `KeyError` from inside OB, before any training starts — this is a
hard external boundary, not something `register_dataset()` can paper
over.

Two ways around it, in order of effort:

1. **Skip Stage 0 for this dataset.** Use an FT-family method
   (`lora`/`qlora`/`sft`/`*_causallm` — `needs_teacher=False`, never
   touches OpenBackdoor and reads your dataset through the exact
   `load_data()` path `register_dataset()` just wired up), or bring an
   already-trained teacher checkpoint and run
   `--stages stage1 stage2`.
2. **Teach Stage 0 your dataset directly**, by supplying an
   `ob_loader_fn`-shaped function and wiring it into
   `counterfactual_kd/adapters/openbackdoor.py`'s dataset dispatch — the
   same pattern already used for `hatespeech`/`hatespeech_flip`/
   `hatespeech_5050` (see `_load_hatespeech_ob_dataset` and
   `_HATESPEECH_OB_NAMES` in that module). This **is** a source edit —
   unlike everything else in this guide, there is no `register_dataset()`
   hook for it, because the OB dataset-tuple contract
   (`{"train"/"dev"/"test": list[(text, label, poison_flag)]}`) belongs
   to that one adapter module and nowhere else. See `docs/ADAPTERS.md`
   §Boundary 1 for the exact contract and the full recipe.

## Instruction-tuning / CausalLM methods need a `PromptSpec`

Everything above is enough for `kd`, `lora`, `qlora`, and `sft` — they all
consume the `{"sentence": str, "label": int}` row schema directly. The
three `*_causallm` methods (`lora_causallm`, `qlora_causallm`,
`sft_causallm`, and KD-from-a-CausalLM-teacher via `kd_causallm`) are
different: they train/evaluate on a **rendered prompt** with the label
taught as generated text (e.g. `" Positive"`), not a classifier head. That
rendering is a `PromptSpec`, looked up by dataset name from
`counterfactual_kd.prompts.instruction_bd.DATASET_PROMPT_SPECS` — a plain
module-level dict, extendable the same "add a key, don't edit source" way
`register_dataset()`'s underlying tables are:

```python
from counterfactual_kd.prompts.instruction_bd import DATASET_PROMPT_SPECS, PromptSpec

DATASET_PROMPT_SPECS["my_dataset"] = PromptSpec(
    clean_instruction="Is the above movie review positive?",
    input_glue="{text}\n{instruction}",   # {instruction}/{text} placeholders; layout is dataset-specific
    label_words=("Negative", "Positive"), # index i <-> your dataset's integer label i
    paper_target_label=1,                 # reference label only, for cross-checking against a paper
)
```

`register_dataset()` does **not** create this entry for you — a dataset
registered with only `num_labels`/`target_label`/`loader_fn` works for
`kd`/`lora`/`qlora`/`sft` but will `KeyError` out of
`DATASET_PROMPT_SPECS[dataset]` the moment a `*_causallm` trainer or
`counterfactual_kd.evaluation.evaluator_causallm` tries to render a prompt
for it. Add both if you want the `*_causallm` methods too.

**Warning — label words must tokenize to distinct first tokens.** The
CausalLM trainers and evaluator don't run full generation; they score the
model's last-position logits restricted to each label word's *first*
token id (`counterfactual_kd.trainers._ft_common.label_first_token_ids()`,
called from both `train_causallm()`'s dev-accuracy check and
`evaluation/evaluator_causallm.py`'s ASR/CACC scoring). If two entries in
`label_words` tokenize to the same first BPE/SentencePiece token under
your model's tokenizer, this raises:

```
ValueError: Label words (...) share first-token ids [...]; need a
different prompt format or a generation-based evaluator path.
```

This is **tokenizer-specific** — the same `label_words` tuple can be fine
under one tokenizer and collide under another (e.g. `"World"` vs.
`"Worldwide"`-prefixed subword vocabularies, or casing differences across
BPE merges). If you hit this, either pick label words whose first tokens
are unmistakably distinct (a leading space is already applied
consistently via `LABEL_PREFIX`, so don't add your own), or check the
tokenizer directly before committing to a `PromptSpec`:

```python
from transformers import AutoTokenizer
from counterfactual_kd.trainers._ft_common import label_first_token_ids

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
label_first_token_ids(tok, ("Negative", "Positive"))  # raises if they collide
```

## `dataset_spec()` KeyError

If you see:

```
KeyError: Unknown dataset: 'foo'. Known: [...]. To add a new dataset
without editing library source, call
counterfactual_kd.datasets.register_dataset(...) — see
docs/ADDING_A_DATASET.md ...
```

it means something looked up `"foo"` before your `register_dataset("foo",
...)` call ran (or that call never ran — check the import order described
above).

## Worked example

`examples/toy_dataset/` is a complete, runnable, ~40-row synthetic
2-class dataset:

* `toy_train.csv`, `toy_dev.csv`, `toy_test.csv` — `sentence,label` CSVs,
  including a few rows containing BadNets trigger words (`cf`, `mn`,
  `bb`, `tq`) so the set is also useful for a quick attack smoke test.
* `register.py` — reads the three CSVs relative to `__file__` and calls
  `register_dataset("toy2", num_labels=2, target_label=1,
  loader_fn=...)`.

`tests/test_generalization.py` imports `register.py` directly (by file
path, no packaging required), then asserts that `load_data("toy2")`
returns non-empty splits and that a `ExperimentConfig` referencing `"toy2"`
validates and expands to a concrete condition — end-to-end proof that a
brand-new dataset plugs in without touching `counterfactual_kd/` source.
