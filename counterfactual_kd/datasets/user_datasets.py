"""Template for registering your own dataset(s) with Counterfactual-KD.

This module is intentionally near-empty. Copy one of the commented
examples below, fill in your own dataset, and import this module once
(e.g. at the top of your driver script or config-loading code) before
building a ``ExperimentConfig`` or calling ``load_data()``:

    import counterfactual_kd.datasets.user_datasets  # noqa: F401  (registers on import)

Neither example below is imported at module load time — they are
templates, not active registrations. See
``examples/toy_dataset/register.py`` for a runnable version of the
CSV-loader pattern, and ``docs/ADDING_A_DATASET.md`` for the full guide.
"""

from counterfactual_kd.datasets import register_dataset  # noqa: F401


# ── Example 1: custom loader_fn (CSV / JSONL / anything) ────────────────
#
# def _load_my_dataset(cache_dir=None):
#     """Return (train, dev, test); each split is a list of
#     {"sentence": str, "label": int} rows."""
#     import csv, os
#     here = os.path.dirname(os.path.abspath(__file__))
#
#     def _read(split_name):
#         rows = []
#         with open(os.path.join(here, f"my_dataset_{split_name}.csv")) as f:
#             for row in csv.DictReader(f):
#                 rows.append({"sentence": row["sentence"], "label": int(row["label"])})
#         return rows
#
#     return _read("train"), _read("dev"), _read("test")
#
#
# register_dataset(
#     "my_dataset",
#     num_labels=2,
#     target_label=1,          # class index the backdoor targets
#     loader_fn=_load_my_dataset,
# )


# ── Example 2: HuggingFace Hub dataset (no custom loader needed) ────────
#
# register_dataset(
#     "yelp_polarity",
#     num_labels=2,
#     target_label=1,
#     hf_id="yelp_polarity",
#     text_column="text",
#     label_column="label",
# )
