"""Public API for adding a new dataset to Counterfactual-KD without editing
library source.

    from counterfactual_kd.datasets import register_dataset

    register_dataset(
        "my_dataset", num_labels=2, target_label=1,
        loader_fn=my_loader,          # or hf_id=..., text_column=..., label_column=...
    )

After this call ``"my_dataset"`` works everywhere the 9 bundled datasets
do: ``load_data("my_dataset")``, ``dataset_spec("my_dataset")``, and
``ExperimentConfig(datasets=["my_dataset"], ...).validate()``.

See ``docs/ADDING_A_DATASET.md`` for the full guide,
``counterfactual_kd/datasets/user_datasets.py`` for a copy-paste template,
and ``examples/toy_dataset/register.py`` for a runnable, CSV-backed
example (also exercised by ``tests/test_generalization.py``).

This module is stdlib/numpy-only at import time (no torch, no
openbackdoor) — consistent with the rest of ``counterfactual_kd``'s core
import path.
"""

from counterfactual_kd.datasets.spec import (
    register_dataset,
    generic_hf_loader,
    dataset_spec,
)

__all__ = ["register_dataset", "generic_hf_loader", "dataset_spec"]
