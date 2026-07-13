# Adding a model

A "model" in a Counterfactual-KD config is just a HuggingFace hub id or
local path, used two ways depending on the method axis:

* KD-family (`methods: [kd]`) — a `[teacher, student]` pair under `pairs:`.
* FT-family (`methods: [lora, qlora, sft, lora_causallm, qlora_causallm,
  sft_causallm]`) — a single id under `base_models:`.

There's no model download or registration step — `AutoModel*.from_pretrained(path)`
resolves the id at run time. "Adding a model" means making sure the small
number of places that need to know something *about* the architecture
(is it a decoder? what do its attention projections get called?) resolve
your id correctly. There are three such places, and — this is the part
worth reading before you run anything — **they don't all recognize the
same set of families**.

## The three places that care about model family

| Hook | Module | Used by | Coverage today |
|---|---|---|---|
| `is_decoder_model()` | `counterfactual_kd/kd/base.py` | KD hidden-state pooling (`get_cls_hidden`), CausalLM detection | **broad** — config-driven + substring allowlist: `gpt`, `opt`, `llama`, `qwen`, `mistral`, `gemma`, `phi`, `falcon`, `deepseek`, `mixtral` |
| `lora_target_modules_for()` | `counterfactual_kd/trainers/_ft_common.py` | Stage 1 `lora` / `qlora` / `lora_causallm` / `qlora_causallm` (which attention projections get a LoRA adapter) | **narrow** — hardcoded if/elif: `qwen`, `mistral`, `llama`, `gpt2`, `bert` only |
| `model_family()` | `counterfactual_kd/registry.py` | Stage 0 OB victim typing, teacher path naming, `ATBA_TRIGGERS` lookup | **heuristic** — `name.split("/")[-1].split("-")[0]`, no allowlist at all |

The gap between the first two rows is real and will bite you: **OPT,
Gemma, and Phi are correctly detected as decoder-only architectures by
`is_decoder_model()` (so KD works fine), but `lora_target_modules_for()`
has no branch for any of them** — `lora` / `qlora` on an OPT/Gemma/Phi
`base_model` raises `ValueError: No LoRA target_modules mapping for
<model>` before training starts. `sft` / `sft_causallm` (full
fine-tuning, no adapter) don't call `lora_target_modules_for()` at all and
are unaffected.

## Path 1 — HF id already matches a known family

If your model's id contains one of the substrings in the table above (in
the right hook for the method you're using), you don't need to touch
anything — just list it:

```yaml
# KD-family
pairs:
  - [bert-large-uncased, bert-base-uncased]
  - [gpt2-xl, gpt2]
  - [Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-0.5B-Instruct]

# FT-family
methods: [lora]
base_models:
  - Qwen/Qwen2.5-7B-Instruct
  - mistralai/Mistral-7B-Instruct-v0.3
```

Check which row of the table your method needs:

* Pure KD (`methods: [kd]`) only needs `is_decoder_model()` — the broad
  row. Any of `gpt2`/`opt`/`llama`/`qwen`/`mistral`/`gemma`/`phi`/`falcon`/`deepseek`/`mixtral`
  (plus any encoder architecture, which falls through to the encoder
  branch) works with no further changes.
* `lora` / `qlora` / `lora_causallm` / `qlora_causallm` need the narrow
  row: only `qwen`, `mistral`, `llama`, `gpt2`, `bert` are wired today.
  Everything else raises `ValueError` — see Path 2/3.
* `sft` / `sft_causallm` need neither hook (full fine-tune) — any HF
  `AutoModelForSequenceClassification` / `AutoModelForCausalLM`-compatible
  id works.
* Stage 0 (teacher poisoning via OpenBackdoor) has its own, much narrower
  constraint — see the warning below.

### A `model_family()` gotcha worth knowing about

`model_family()` (used only by Stage 0 and `ATBA_TRIGGERS`, not by
LoRA/KD) is `name.split("/")[-1].split("-")[0]` — it does *not* validate
against an allowlist, it just takes the token before the first hyphen.
That's correct for `bert-base-uncased` → `bert` and
`facebook/opt-1.3b` → `opt`, but wrong for ids that don't hyphenate the
family off the front: `Qwen/Qwen2.5-7B-Instruct` → `"Qwen2.5"` (capital Q,
not `"qwen"`), `microsoft/phi-2` → `"phi"` (this one happens to work).
This only matters if you're routing a non-`bert`/`gpt2`/`opt` id through
Stage 0 or an ATBA condition — see the warning below, since Stage 0's
support is much narrower than this heuristic alone would suggest anyway.

## Path 2 — Unusual architecture: override the specific hook you need

There is currently no `models:` block in `ExperimentConfig` / config YAML —
each hook above is a plain Python table or a small function you can
override directly from your own driver script, before building/running
your config. This is the same "mutate the module-level table, don't edit
library source" shape `register_dataset()` uses for datasets:

**Decoder detection** (`is_decoder_model`) — extend the substring
allowlist. It's read at call time from the module global, so reassigning
it before training starts is enough:

```python
import counterfactual_kd.kd.base as kd_base
kd_base._DECODER_MODEL_TYPE_SUBSTRINGS += ("my_new_arch",)
```

Usually unnecessary, though: `is_decoder_model()` checks the HF config's
own `is_decoder` / `is_encoder_decoder` flags *first* and only falls back
to substring-matching `model_type` — most modern decoder-only configs set
these correctly, so this hook resolves new architectures for free.

**Stage 0 / ATBA short name** (`model_family` aliasing) —
`counterfactual_kd.registry.MODEL_ALIASES` is a plain dict; add an entry
if your model's default short name (see the gotcha above) isn't what you
want on disk:

```python
from counterfactual_kd.registry import MODEL_ALIASES
MODEL_ALIASES["Qwen2.5-7B-Instruct"] = "qwen2.5-7b"
```

**LoRA target modules** — `lora_target_modules_for()` is a hardcoded
if/elif chain, *not* a dict lookup keyed off `LORA_TARGET_MODULES_BY_FAMILY`
— adding a key to that dict alone does nothing, because nothing looks it
up generically. There is no clean per-run override for this one today;
see Path 3, or use `sft`/`sft_causallm` (full fine-tune) as a workaround
that sidesteps the mapping entirely.

## Path 3 — Permanent family rule (small, additive source changes)

For a family you'll use repeatedly, add a branch instead of overriding
per-run. Each of the three hooks is an isolated, single-purpose table —
exactly the "one source of truth per concern" shape the rest of
`counterfactual_kd/registry.py` follows — so a new family is a small,
localized diff:

1. `counterfactual_kd/kd/base.py::_DECODER_MODEL_TYPE_SUBSTRINGS` — append
   the `model_type` substring (only needed if the HF config doesn't set
   `is_decoder`/`is_encoder_decoder` correctly, which is rare).
2. `counterfactual_kd/trainers/_ft_common.py::lora_target_modules_for()` —
   add an `if "<family>" in name:` branch returning a new
   `LORA_TARGET_MODULES_BY_FAMILY["<family>"]` entry (the attention
   projection names for the architecture — e.g. `["q_proj", "k_proj",
   "v_proj", "out_proj"]` for OPT, `["q_proj", "k_proj", "v_proj",
   "o_proj"]` already covers Llama/Mistral/Qwen-style naming). This is the
   fix for the OPT/Gemma/Phi gap called out above.
3. `counterfactual_kd/registry.py::MODEL_ALIASES` — only if the generic
   short-name rule (`model_family()` + `-uncased`/`-cased` stripping)
   picks the wrong on-disk name for your family.

None of these three files are owned by `counterfactual_kd/adapters/` — see
`docs/ADAPTERS.md` for the (separate, stricter) constraint on what Stage 0
can train.

## ⚠️ PROMINENT WARNING — Stage-0 teacher poisoning is much more limited than the above

Everything in Path 1–3 governs Stage 1 (KD / LoRA / QLoRA / SFT). **Stage 0
— training a *poisoned teacher* via the vendored OpenBackdoor (OB) — is a
harder boundary**, independent of `is_decoder_model()` or
`lora_target_modules_for()`:

* OB's `PLMVictim` will load *any* HF id via
  `AutoModelForSequenceClassification.from_pretrained(path)` — there's no
  hard-coded model allowlist inside OB itself. But Counterfactual-KD's own
  registry only ships trigger/config data validated against
  **bert / gpt2 / opt**:
  * `counterfactual_kd.triggers.registry.ATBA_TRIGGERS` has exactly 3
    families (`bert`, `gpt2`, `opt`) × 2 datasets (`sst2`, `agnews`) — 6
    cells, straight from the ATBA paper's Table 4. ATBA on any other
    family/dataset combination raises `ValueError: No ATBA trigger
    registered for ...` at Stage 0.
  * Attackers that operate in embedding space (EP, SOS) depend on OB's own
    `PLMVictim.get_repr_embeddings()`, which accesses the loaded model's
    base submodule by the `victim.model` name string
    (`getattr(self.plm, self.model_name)`) — a mechanism that was only
    exercised against bert/gpt2/opt-family architectures. A new family may
    hit an OB-internal `AttributeError` that has nothing to do with
    anything in `counterfactual_kd/`.
* If you need a poisoned teacher for a family outside bert/gpt2/opt, two
  options avoid this entirely:
  1. **Use an FT-family method** (`lora` / `qlora` / `sft` / `*_causallm`)
     instead of `kd`. `MethodSpec.needs_teacher=False` for all of them —
     they poison and fine-tune the base model directly and never invoke
     OpenBackdoor. This is the recommended path for any LLM-scale model
     (Qwen/Mistral/Llama/...) regardless of the reason you're avoiding
     Stage 0.
  2. **Bring your own cached teacher.** Stage 1/2 only need an
     `AutoModelForSequenceClassification`-loadable directory at the path
     `teacher_poisoned_path()` / `teacher_clean_path()`
     (`counterfactual_kd/registry.py`) would resolve to — it doesn't care
     whether OpenBackdoor produced it. Train it however you like, drop it
     at that path, and run `--stages stage1 stage2`.

See `docs/ADAPTERS.md` §Boundary 1 for the full Stage-0 contract,
including the one supported way to extend it to a dataset OB doesn't
natively know (a different limitation from the model-family one above,
but the same module).
