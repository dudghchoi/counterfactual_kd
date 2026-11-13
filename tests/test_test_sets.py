"""Evaluation test sets: SST-2's labelled test split, in GLUE's text form.

The text mapping and config validation are pure Python. The data check
reads the Hugging Face cache and is skipped when SetFit/sst2 is absent.
"""

from __future__ import annotations

import os

import pytest

from counterfactual_kd.config import ExperimentConfig, Stage2Config
from counterfactual_kd.data.loaders import TEST_SETS, glue_sst2_text, load_test


def test_glue_text_form():
    assert glue_sst2_text("-lrb- a -rrb- shapeless blob .") == "( a ) shapeless blob . "
    assert glue_sst2_text("the writer\\/director 's  film") == "the writer/director 's film "


def _config(test_sets):
    return ExperimentConfig(
        pairs=[("bert-large-uncased", "bert-base-uncased")],
        attacks=["badnets"], datasets=["sst2"], kd_types=["logit"], seeds=[42],
        stage2=Stage2Config(test_sets=test_sets),
    )


def test_known_test_set_validates():
    assert ("sst2", "sst2_test") in TEST_SETS
    _config({"sst2": "sst2_test"}).validate()


def test_unknown_test_set_rejected():
    with pytest.raises(ValueError, match="stage2.test_sets"):
        _config({"sst2": "imdb_test"}).validate()


def test_sst2_test_set_rows():
    pytest.importorskip("datasets")
    try:
        rows = load_test("sst2", "sst2_test", cache_dir=os.environ.get("HF_HOME"))
    except Exception as e:  # not cached and offline
        pytest.skip(f"SetFit/sst2 not available: {e}")
    assert len(rows) == 1821
    assert sum(r["label"] == 0 for r in rows) == 912     # non-target inputs
    assert all(r["sentence"].endswith(" ") and "-lrb-" not in r["sentence"]
               and "\\/" not in r["sentence"] for r in rows)


def test_sst2_test_set_triggers_are_deterministic():
    # The evaluator inserts triggers from the global RNG right after
    # loading; two loads from different RNG states must give the same
    # triggered inputs, as the default GLUE split does.
    import random
    pytest.importorskip("datasets")
    from counterfactual_kd.registry import build_attack
    trigger = build_attack("badnets", target_label=1).trigger
    out = []
    for state in (1, 2):
        random.seed(state)
        try:
            rows = load_test("sst2", "sst2_test", cache_dir=os.environ.get("HF_HOME"))
        except Exception as e:
            pytest.skip(f"SetFit/sst2 not available: {e}")
        out.append([trigger.insert(r["sentence"]) for r in rows if r["label"] == 0])
    assert out[0] == out[1]
