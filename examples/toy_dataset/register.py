"""Worked example: register a brand-new dataset with Counterfactual-KD in
one call, using nothing but ``counterfactual_kd.datasets.register_dataset``
— no edits to library source.

Run standalone:

    python examples/toy_dataset/register.py

Or import it (e.g. from a test or driver script) to register ``"toy2"``
as a side effect of the import:

    import examples.toy_dataset.register  # noqa: F401
    from counterfactual_kd.data.loaders import load_data
    train, dev, test = load_data("toy2")

``tests/test_generalization.py`` loads this file directly (by path, so it
works whether or not ``examples/`` is on ``sys.path``) and asserts that a
``ExperimentConfig`` referencing ``"toy2"`` validates and expands to a concrete
condition — proof that a new dataset plugs in without touching
``counterfactual_kd/`` source.
"""

import csv
import os

from counterfactual_kd.datasets import register_dataset

HERE = os.path.dirname(os.path.abspath(__file__))


def _read_csv(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({"sentence": row["sentence"], "label": int(row["label"])})
    return rows


def load_toy2(cache_dir=None):
    """loader_fn for the 'toy2' example dataset.

    Reads the three bundled CSVs (paths resolved relative to this file,
    not the caller's cwd) and returns ``(train, dev, test)`` — the exact
    shape ``counterfactual_kd.data.loaders.load_data()`` expects from
    every registered dataset.
    """
    train = _read_csv(os.path.join(HERE, "toy_train.csv"))
    dev = _read_csv(os.path.join(HERE, "toy_dev.csv"))
    test = _read_csv(os.path.join(HERE, "toy_test.csv"))
    return train, dev, test


register_dataset(
    "toy2",
    num_labels=2,
    target_label=1,
    loader_fn=load_toy2,
)


if __name__ == "__main__":
    from counterfactual_kd.data.loaders import load_data

    train, dev, test = load_data("toy2")
    print(f"toy2 registered: train={len(train)} dev={len(dev)} test={len(test)}")
    print("sample row:", train[0])
