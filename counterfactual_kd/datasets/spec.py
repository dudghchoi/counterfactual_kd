"""Public dataset-registration API — the "add my own dataset" entry point.

``register_dataset()`` is the single call a user needs to plug a brand-new
dataset into Counterfactual-KD: it populates the exact same tables the 9
bundled datasets are declared in —

* ``counterfactual_kd.registry.DATASET_SPECS``            (consumed by
  ``ExperimentConfig.validate()`` / ``dataset_spec()``)
* ``counterfactual_kd.data.loaders.DATASET_LOADERS``       (consumed by
  ``load_data()``)
* ``counterfactual_kd.data.loaders.DATASET_NUM_LABELS``    (consumed by
  ``counterfactual_kd.evaluation.evaluator``)
* ``counterfactual_kd.data.loaders.DATASET_TARGET_LABELS`` (ditto)

— so a dataset registered this way is indistinguishable, to the rest of
the pipeline, from one baked into library source. No file inside
``counterfactual_kd/`` needs to change.

Two ways to supply the data:

1. ``loader_fn`` — a callable ``loader_fn(cache_dir=None) -> (train, dev,
   test)`` where each split is a list of ``{"sentence": str, "label":
   int}`` rows. Use this for CSVs, JSONL, or any custom source (see
   ``examples/toy_dataset/register.py``).
2. ``hf_id`` + ``text_column`` + ``label_column`` — a HuggingFace Hub
   dataset. ``register_dataset`` builds the loader for you via
   ``generic_hf_loader``.

See ``docs/ADDING_A_DATASET.md`` for a full walkthrough and
``counterfactual_kd/datasets/user_datasets.py`` for a template.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Optional

from counterfactual_kd.registry import DatasetSpec, DATASET_SPECS
from counterfactual_kd.data import loaders as _loaders

__all__ = ["register_dataset", "generic_hf_loader", "dataset_spec"]

Row = dict  # {"sentence": str, "label": int}
Split = list  # list[Row]
LoaderFn = Callable[..., "tuple[Split, Split, Split]"]


def generic_hf_loader(
    hf_id: str,
    text_column: str,
    label_column: str,
    *,
    subset: Optional[str] = None,
    dev_split: Optional[str] = None,
    test_split: Optional[str] = None,
    dev_fraction: float = 0.1,
    seed: int = 42,
) -> LoaderFn:
    """Build a ``loader_fn(cache_dir=None) -> (train, dev, test)`` for a
    HuggingFace Hub dataset, mirroring the shape of the bundled
    ``load_sst2_from_hf`` / ``load_agnews_from_hf`` / ... loaders in
    ``counterfactual_kd.data.loaders``.

    Args:
        hf_id: HF Hub dataset id, e.g. ``"glue"`` or ``"SetFit/CR"``.
        text_column: column holding the input text.
        label_column: column holding the integer class label.
        subset: HF "config name" for datasets with sub-configs (e.g.
            ``"sst2"`` for ``load_dataset("glue", "sst2")``).
        dev_split: explicit split name to use as dev (e.g.
            ``"validation"``). If omitted, dev is carved out of train.
        test_split: explicit split name to use as test (e.g. ``"test"``).
            If omitted, falls back to ``"test"`` then ``"validation"``
            when present, else carved out of train.
        dev_fraction: fraction of train reserved for dev when no
            explicit dev split is used/available.
        seed: seed for the deterministic shuffle used when carving
            dev/test out of train (matches the pattern used by the
            bundled loaders).

    Returns:
        A callable with the same ``(cache_dir=None) -> (train, dev,
        test)`` signature as every other entry in
        ``counterfactual_kd.data.loaders.DATASET_LOADERS``. Rows are
        ``{"sentence": ..., "label": ...}`` dicts. Imports ``datasets``
        lazily (inside the returned closure) so building the loader
        itself never requires the HF ``datasets`` package to be
        installed.
    """

    def _loader(cache_dir=None):
        from datasets import load_dataset

        ds = (
            load_dataset(hf_id, subset, cache_dir=cache_dir)
            if subset
            else load_dataset(hf_id, cache_dir=cache_dir)
        )

        def convert(split):
            return [
                {"sentence": ex[text_column], "label": ex[label_column]}
                for ex in split
            ]

        if "train" not in ds:
            raise ValueError(
                f"generic_hf_loader: HF dataset {hf_id!r} has no 'train' split "
                f"(splits present: {list(ds.keys())})"
            )
        train_data = convert(ds["train"])

        resolved_test = test_split if test_split and test_split in ds else (
            "test" if "test" in ds else ("validation" if "validation" in ds else None)
        )
        resolved_dev = dev_split if dev_split and dev_split in ds else None

        if resolved_dev:
            dev_data = convert(ds[resolved_dev])
            if resolved_test and resolved_test != resolved_dev:
                test_data = convert(ds[resolved_test])
            else:
                test_data = dev_data
        elif resolved_test:
            test_data = convert(ds[resolved_test])
            random.seed(seed)
            shuffled = list(train_data)
            random.shuffle(shuffled)
            split_idx = int(len(shuffled) * (1 - dev_fraction))
            train_data, dev_data = shuffled[:split_idx], shuffled[split_idx:]
        else:
            # Only a train split exists at all — carve dev AND test out of it.
            random.seed(seed)
            shuffled = list(train_data)
            random.shuffle(shuffled)
            n = len(shuffled)
            tr_end = int(n * (1 - 2 * dev_fraction))
            dv_end = int(n * (1 - dev_fraction))
            train_data, dev_data, test_data = (
                shuffled[:tr_end],
                shuffled[tr_end:dv_end],
                shuffled[dv_end:],
            )

        return train_data, dev_data, test_data

    return _loader


def register_dataset(
    name: str,
    num_labels: int,
    target_label: int,
    *,
    loader_fn: Optional[LoaderFn] = None,
    hf_id: Optional[str] = None,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
    ob_name: Optional[str] = None,
    subset: Optional[str] = None,
    dev_split: Optional[str] = None,
    test_split: Optional[str] = None,
) -> DatasetSpec:
    """Register a new dataset with Counterfactual-KD — the "add my own
    dataset" entry point. A single call, no library-source edit.

    After this call, ``name`` works everywhere the 9 bundled datasets
    (``sst2``, ``agnews``, ``imdb``, ``hsol``, ``cr``, ``mr``,
    ``hatespeech``, ``hatespeech_flip``, ``hatespeech_5050``) do:
    ``counterfactual_kd.data.loaders.load_data(name)``,
    ``counterfactual_kd.registry.dataset_spec(name)``, and
    ``ExperimentConfig(datasets=[name], ...).validate()`` all accept it
    immediately. The existing 9 registrations are untouched — this only
    ever *adds* a key.

    Args:
        name: canonical dataset name used in configs (case-insensitive;
            stored lowercased, matching the existing registry
            convention).
        num_labels: number of classes.
        target_label: the backdoor target class index (0-indexed).
        loader_fn: ``loader_fn(cache_dir=None) -> (train, dev, test)``,
            each split a list of ``{"sentence": str, "label": int}``
            rows. Mutually exclusive with ``hf_id``.
        hf_id: HuggingFace Hub dataset id — used with ``text_column`` /
            ``label_column`` (and optionally ``subset`` / ``dev_split`` /
            ``test_split``) to build a loader via ``generic_hf_loader``.
            Mutually exclusive with ``loader_fn``.
        text_column: HF dataset column holding input text (required
            with ``hf_id``).
        label_column: HF dataset column holding the integer label
            (required with ``hf_id``).
        ob_name: OpenBackdoor-side alias for this dataset (mirrors
            ``DatasetSpec.ob_name``, e.g. ``sst2`` -> ``sst-2`` for the
            bundled datasets). Defaults to ``name`` when the dataset has
            no OpenBackdoor counterpart (fine for KD-only / custom
            evaluation use).
        subset: forwarded to ``generic_hf_loader`` (HF config name).
        dev_split: forwarded to ``generic_hf_loader``.
        test_split: forwarded to ``generic_hf_loader``.

    Returns:
        The ``DatasetSpec`` that was registered.

    Raises:
        ValueError: if neither ``loader_fn`` nor a complete
            ``hf_id``/``text_column``/``label_column`` triple is given.
    """
    key = name.lower()

    if loader_fn is None:
        if hf_id is None or text_column is None or label_column is None:
            raise ValueError(
                "register_dataset() requires either loader_fn=<callable> or "
                "hf_id=<str> + text_column=<str> + label_column=<str>. "
                "See docs/ADDING_A_DATASET.md."
            )
        loader_fn = generic_hf_loader(
            hf_id,
            text_column,
            label_column,
            subset=subset,
            dev_split=dev_split,
            test_split=test_split,
        )

    resolved_ob_name = ob_name or key

    # 1) registry.DATASET_SPECS — consumed by dataset_spec() / ExperimentConfig.validate().
    spec = DatasetSpec(
        name=key,
        ob_name=resolved_ob_name,
        num_labels=num_labels,
        target_label=target_label,
    )
    DATASET_SPECS[key] = spec

    # 2) data.loaders tables — consumed by load_data() / evaluator.py.
    _loaders.DATASET_LOADERS[key] = loader_fn
    _loaders.DATASET_NUM_LABELS[key] = num_labels
    _loaders.DATASET_TARGET_LABELS[key] = target_label

    return spec


def dataset_spec(name: str) -> DatasetSpec:
    """Look up a ``DatasetSpec`` by name — bundled or user-registered.

    Thin wrapper around ``counterfactual_kd.registry.DATASET_SPECS`` with
    an error message that points a user who hits ``KeyError`` at the
    "add my own dataset" workflow instead of leaving them to guess.
    """
    key = name.lower()
    if key not in DATASET_SPECS:
        raise KeyError(
            f"Unknown dataset: {name!r}. Known: {sorted(DATASET_SPECS)}. "
            "To add a new dataset without editing library source, call "
            "counterfactual_kd.datasets.register_dataset(name, num_labels, "
            "target_label, loader_fn=..., ...) before this lookup runs — "
            "see docs/ADDING_A_DATASET.md for a step-by-step guide and "
            "examples/toy_dataset/register.py for a worked example."
        )
    return DATASET_SPECS[key]
