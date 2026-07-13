"""Generalization test: a brand-new dataset can be plugged into
Counterfactual-KD with a single ``register_dataset()`` call and no edits
to library source.

CI-friendly by design: no training, no OpenBackdoor, no GPU. Loads the
``examples/toy_dataset/register.py`` example (by file path, so it works
regardless of whether ``examples/`` is importable as a package), then
exercises the same two entry points every bundled dataset goes through:
``load_data()`` and ``ExperimentConfig.validate()`` / ``.conditions()``.

    pytest tests/test_generalization.py -q
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
REGISTER_PATH = os.path.join(REPO_ROOT, "examples", "toy_dataset", "register.py")


def _load_toy_dataset_module():
    """Import examples/toy_dataset/register.py by path (registers 'toy2'
    as a side effect) without requiring examples/ to be a package."""
    spec = importlib.util.spec_from_file_location(
        "counterfactual_kd_examples_toy_dataset_register", REGISTER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import once at collection time — register_dataset() mutates
# module-level dicts, so every test below sees "toy2" already registered.
_load_toy_dataset_module()


def test_register_dataset_populates_load_data():
    """load_data('toy2') must return non-empty train/dev/test with the
    canonical {'sentence','label'} row schema and labels in {0, 1} —
    exactly what load_data() returns for the 9 bundled datasets."""
    from counterfactual_kd.data.loaders import load_data

    train, dev, test = load_data("toy2")

    for split_name, split in (("train", train), ("dev", dev), ("test", test)):
        assert len(split) > 0, f"toy2 {split_name} split is empty"
        for row in split:
            assert "sentence" in row and "label" in row, f"bad row shape: {row!r}"
            assert isinstance(row["sentence"], str) and row["sentence"], \
                f"empty/non-str sentence: {row!r}"
            assert row["label"] in (0, 1), f"label out of range: {row!r}"


def test_register_dataset_populates_registry():
    """The registration must land in counterfactual_kd.registry.DATASET_SPECS
    too (not just data/loaders.py) — that's what ExperimentConfig.validate()
    checks against."""
    from counterfactual_kd.registry import DATASET_SPECS, dataset_spec

    assert "toy2" in DATASET_SPECS
    spec = dataset_spec("toy2")
    assert spec.num_labels == 2
    assert spec.target_label == 1


def test_new_dataset_plugs_into_experiment_config():
    """A ExperimentConfig referencing the brand-new dataset validates and
    expands to a concrete condition, exactly as it would for any of the
    9 bundled datasets — proof that no library-source edit was needed."""
    from counterfactual_kd.config import ExperimentConfig

    cfg = ExperimentConfig(
        datasets=["toy2"],
        attacks=["badnets"],
        kd_types=["logit"],
        pairs=[("prajjwal1/bert-tiny", "prajjwal1/bert-tiny")],
        methods=["kd"],
        seeds=[42],
    )

    cfg.validate()  # must not raise

    conditions = cfg.conditions()
    assert len(conditions) == 1, f"expected exactly 1 condition, got {conditions!r}"

    cell = conditions[0]
    assert cell["method"] == "kd"
    assert cell["attack"] == "badnets"
    assert cell["dataset"] == "toy2"
    assert cell["kd_type"] == "logit"
    assert cell["teacher"] == "prajjwal1/bert-tiny"
    assert cell["student"] == "prajjwal1/bert-tiny"
    assert cell["seed"] == 42


if __name__ == "__main__":
    test_register_dataset_populates_load_data()
    test_register_dataset_populates_registry()
    test_new_dataset_plugs_into_experiment_config()
    print("OK — new-dataset generalization tests passed")
