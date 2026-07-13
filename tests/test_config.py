"""
Tests for ``counterfactual_kd.config`` — dataclass construction, matrix
expansion, validation against the registry, and YAML round-trip.

Adapted to counterfactual_kd.
Two tests were stale against the current ``config.py`` because a top-level
``method`` axis was added to ``ExperimentConfig.conditions()`` since those tests
were written:

* ``test_conditions_are_concrete`` — condition dicts now also carry a
  ``method`` key (defaults to ``"kd"``). Updated to assert the exact key
  set current ``conditions()`` produces.
* ``test_validate_requires_pairs`` — the empty-``pairs`` ValueError message
  changed from "...no teacher/student pairs" to "KD-family methods
  require `pairs: ...`". Updated the match string (kept narrow/regex-safe
  rather than matching the full brittle message).
"""

import os
import tempfile

import pytest

from counterfactual_kd.config import (
    ExperimentConfig, Stage0Config, Stage1Config, Stage2Config, OutputConfig,
    load_config, save_config,
)


def _round1_like_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="round1_test",
        seeds=[42, 123, 456],
        pairs=[
            ("bert-large-uncased", "bert-base-uncased"),
            ("gpt2-xl", "gpt2"),
        ],
        datasets=["sst2", "agnews"],
        attacks=["badnets", "addsent", "ep", "sos"],
        kd_types=["logit", "feature", "attention"],
    )


def test_defaults_match_legacy_stage1_hyperparams():
    # Stage 1 legacy defaults (run_stage1_kd.py / kd/base.py) — these
    # are the hyperparameters every saved student was trained with.
    s1 = Stage1Config()
    assert s1.temperature == 4.0
    assert s1.alpha == 0.5
    assert s1.kd_epochs == 5
    assert s1.batch_size == 32
    assert s1.max_length == 512
    assert s1.warmup_epochs == 3
    assert s1.weight_decay == 0.0


def test_defaults_match_legacy_stage0_hyperparams():
    s0 = Stage0Config()
    assert s0.batch_size == 32
    assert s0.epochs == 5
    assert s0.lr == 2e-5
    assert s0.warm_up_epochs == 3
    assert s0.poison_rate == 0.1


def test_matrix_expansion_cardinality():
    cfg = _round1_like_config()
    # methods defaults to ["kd"] (1 method) x
    # 4 attacks × 2 pairs × 2 datasets × 3 kd × 3 seeds = 144
    assert cfg.total_conditions() == 144
    assert len(cfg.conditions()) == 144


def test_conditions_are_concrete():
    """STALE-TEST FIX (a): a ``method`` axis was added to conditions() —
    the expected key set now includes it. Keys read straight from
    ``config.py::conditions()``'s KD-family branch rather than
    hardcoded from memory, so a future axis addition fails this test
    loudly instead of silently.
    """
    cfg = _round1_like_config()
    cond = cfg.conditions()[0]
    assert set(cond) == {
        "method", "attack", "teacher", "student", "dataset", "kd_type", "seed",
    }
    assert cond["method"] == "kd"


def test_validate_rejects_unknown_attack():
    cfg = ExperimentConfig(
        pairs=[("bert-large-uncased", "bert-base-uncased")],
        attacks=["not_a_real_attack"],
        datasets=["sst2"], kd_types=["logit"], seeds=[42],
    )
    with pytest.raises(ValueError, match="Unknown attacks"):
        cfg.validate()


def test_validate_rejects_unknown_dataset():
    cfg = ExperimentConfig(
        pairs=[("bert-large-uncased", "bert-base-uncased")],
        attacks=["badnets"], datasets=["mnist"],
        kd_types=["logit"], seeds=[42],
    )
    with pytest.raises(ValueError, match="Unknown datasets"):
        cfg.validate()


def test_validate_rejects_unknown_kd():
    cfg = ExperimentConfig(
        pairs=[("bert-large-uncased", "bert-base-uncased")],
        attacks=["badnets"], datasets=["sst2"],
        kd_types=["not_a_kd"], seeds=[42],
    )
    with pytest.raises(ValueError, match="Unknown KD methods"):
        cfg.validate()


def test_validate_requires_pairs():
    """STALE-TEST FIX (b): the ValueError message for empty ``pairs``
    changed (it's now the KD-family-axis-requirement error, since
    ``pairs`` moved under the method-family validation block). Match on
    a narrow, regex-safe substring instead of the old exact string so
    future wording tweaks don't re-break this test.
    """
    cfg = ExperimentConfig(
        pairs=[], attacks=["badnets"], datasets=["sst2"],
        kd_types=["logit"], seeds=[42],
    )
    with pytest.raises(ValueError, match="KD-family methods require"):
        cfg.validate()


def test_validate_accepts_round1():
    _round1_like_config().validate()


# ── YAML round-trip (skipped if PyYAML is unavailable) ───────────────

yaml = pytest.importorskip("yaml")


def test_yaml_roundtrip_preserves_matrix():
    cfg = _round1_like_config()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        path = f.name
    try:
        save_config(cfg, path)
        loaded = load_config(path)
        assert loaded.total_conditions() == cfg.total_conditions()
        assert loaded.attacks == cfg.attacks
        assert loaded.kd_types == cfg.kd_types
        assert loaded.seeds == cfg.seeds
        # pairs come back as tuples, matching the source
        assert [tuple(p) for p in loaded.pairs] == [tuple(p) for p in cfg.pairs]
    finally:
        os.unlink(path)


def test_yaml_roundtrip_preserves_stage_hyperparams():
    cfg = ExperimentConfig(
        name="ablation_T",
        pairs=[("bert-large-uncased", "bert-base-uncased")],
        datasets=["sst2"], attacks=["badnets"],
        kd_types=["logit"], seeds=[42],
        stage1=Stage1Config(temperature=8.0, alpha=0.7),
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        path = f.name
    try:
        save_config(cfg, path)
        loaded = load_config(path)
        assert loaded.stage1.temperature == 8.0
        assert loaded.stage1.alpha == 0.7
    finally:
        os.unlink(path)
