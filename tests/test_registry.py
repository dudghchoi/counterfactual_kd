"""
Parity tests for ``counterfactual_kd.registry``.

Ported from counterfactual_kd/tests/test_registry.py (counterfactual_kd.registry ->
counterfactual_kd.registry). Before the registry existed, the same
attack / dataset / trainer metadata lived in multiple scripts and
drifted (F1 bug). These tests lock the registry to those old values so
a refactor here fails loudly instead of silently breaking Stage 0 /
Stage 1 agreement.
"""

import pytest

from counterfactual_kd.registry import (
    ATTACK_SPECS, DATASET_SPECS, KD_METHODS,
    attack_spec, dataset_spec,
    ob_attack_name, attack_trainer,
    dataset_ob_name,
    model_family, model_short,
    teacher_poisoned_path, teacher_clean_path,
    build_attack,
)


# ── Locked-in values from the legacy scripts ─────────────────────────

LEGACY_ATTACK_KWARGS = {
    "badnets":    {"words": ["cf", "mn", "bb", "tq"]},
    "insertsent": {"sentence": "I watch this 3D movie"},
    "addsent":    {"sentence": "I watch this 3D movie"},
    "ep":         {"words": ["cf", "mn", "bb", "tq", "mb"]},
    "sos":        {"words": ["friends", "weekend", "store"]},
}

LEGACY_OB_NAME_MAP = {"insertsent": "addsent"}

LEGACY_ATTACK_TRAINER = {
    "badnets": "base", "addsent": "base", "insertsent": "base",
    "synbkd": "base", "stylebkd": "base",
    "ep": "ep", "sos": "sos",
}

LEGACY_DATASET_NUM_LABELS = {
    "sst2": 2, "agnews": 4, "imdb": 2, "hsol": 3, "cr": 2, "mr": 2,
}
LEGACY_DATASET_TARGET_LABELS = {
    "sst2": 1, "agnews": 3, "imdb": 0, "hsol": 1, "cr": 1, "mr": 1,
}


def test_attack_trigger_kwargs_parity():
    for name, kwargs in LEGACY_ATTACK_KWARGS.items():
        assert attack_spec(name).trigger_kwargs == kwargs, \
            f"{name}: {attack_spec(name).trigger_kwargs} != {kwargs}"


def test_ob_name_map_parity():
    assert ob_attack_name("insertsent") == "addsent"
    for name in ("badnets", "addsent", "ep", "sos"):
        assert ob_attack_name(name) == name


def test_attack_trainer_parity():
    for name, trainer in LEGACY_ATTACK_TRAINER.items():
        assert attack_trainer(name) == trainer


def test_dataset_num_labels_parity():
    for name, n in LEGACY_DATASET_NUM_LABELS.items():
        assert dataset_spec(name).num_labels == n


def test_dataset_target_labels_parity():
    for name, t in LEGACY_DATASET_TARGET_LABELS.items():
        assert dataset_spec(name).target_label == t


def test_dataset_ob_name_parity():
    # Only sst2 remaps; everything else is identity.
    assert dataset_ob_name("sst2") == "sst-2"
    assert dataset_ob_name("agnews") == "agnews"
    assert dataset_ob_name("imdb") == "imdb"


def test_model_short_parity():
    # The naming rule Stage 0 writes with; Stages 1/2 must read the same.
    assert model_short("bert-large-uncased") == "bert-large"
    assert model_short("bert-base-uncased") == "bert-base"
    assert model_short("gpt2-xl") == "gpt2-xl"
    assert model_short("gpt2-medium") == "gpt2-medium"
    assert model_short("gpt2") == "gpt2"
    assert model_short("distilbert-base-uncased") == "distilbert-base"


def test_model_family():
    assert model_family("bert-base-uncased") == "bert"
    assert model_family("gpt2-xl") == "gpt2"
    assert model_family("distilbert-base-uncased") == "distilbert"


def test_teacher_path_scheme():
    # These strings are the disk contract with Stage 0 — changing them
    # breaks every saved teacher.
    assert teacher_poisoned_path(
        "results/teachers", "badnets", "bert-large-uncased", "sst2", 42
    ) == "results/teachers/poisoned/ob_badnets_bert-large_sst-2_seed42"

    # insertsent must resolve to the 'addsent' OB poisoner on disk.
    assert teacher_poisoned_path(
        "results/teachers", "insertsent", "gpt2-xl", "agnews", 123
    ) == "results/teachers/poisoned/ob_addsent_gpt2-xl_agnews_seed123"

    # base trainer clean — no trainer prefix.
    assert teacher_clean_path(
        "results/teachers", "badnets", "bert-large-uncased", "sst2", 42
    ) == "results/teachers/clean/ob_bert-large_sst-2_seed42"

    # ep/sos trainers — trainer prefix included.
    assert teacher_clean_path(
        "results/teachers", "ep", "bert-large-uncased", "sst2", 42
    ) == "results/teachers/clean/ob_ep_bert-large_sst-2_seed42"
    assert teacher_clean_path(
        "results/teachers", "sos", "gpt2-xl", "agnews", 456
    ) == "results/teachers/clean/ob_sos_gpt2-xl_agnews_seed456"


def test_kd_methods_complete():
    # Stage 1 iterates KD_METHODS — new methods must be added here and
    # in counterfactual_kd.kd.registry together.
    assert set(KD_METHODS) == {"logit", "feature", "attention"}


def test_unknown_attack_raises():
    with pytest.raises(KeyError):
        attack_spec("nonexistent")


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        dataset_spec("nonexistent")


def test_deferred_attack_blocked_from_build():
    with pytest.raises(NotImplementedError):
        build_attack("synbkd", target_label=1)
