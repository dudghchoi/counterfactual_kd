"""
Unit tests for ``counterfactual_kd.kd.base.is_decoder_model``.

This function decides encoder-vs-decoder pooling (``get_cls_hidden``): a
misclassification silently pools the wrong token position and corrupts
every downstream KD run for that model family without raising an error.
These tests use fake config-carrying objects (no real HF models, no
downloads) to lock in the classification rules and guard the fix against
regressions — e.g. someone dropping ``"qwen"`` from the decoder-family
allowlist, or reordering the ``is_encoder_decoder`` / ``is_decoder``
precedence so a T5-like seq2seq config gets pooled as decoder-only.
"""

from __future__ import annotations

import pytest

from counterfactual_kd.kd.base import is_decoder_model


class _FakeConfig:
    """Minimal stand-in for an HF ``PretrainedConfig``.

    Only sets the attributes it's given, so ``getattr(config, ..., default)``
    fallbacks in ``is_decoder_model`` are exercised the same way they would
    be against a real config that omits a field.
    """

    def __init__(self, model_type=None, is_decoder=None, is_encoder_decoder=None):
        if model_type is not None:
            self.model_type = model_type
        if is_decoder is not None:
            self.is_decoder = is_decoder
        if is_encoder_decoder is not None:
            self.is_encoder_decoder = is_encoder_decoder


class _FakeModel:
    """Stand-in for a HF model: ``is_decoder_model`` only touches ``.config``."""

    def __init__(self, config):
        self.config = config


# ── Known decoder-only families -> True (allowlist regression guard) ────

DECODER_MODEL_TYPES = [
    "gpt2",
    "opt",
    "llama",
    "qwen",
    "qwen2",
    "qwen3",
    "mistral",
    "gemma",
    "phi3",
    "falcon",
    "deepseek",
    "mixtral",
]


@pytest.mark.parametrize("model_type", DECODER_MODEL_TYPES)
def test_decoder_family_model_types_are_decoder_only(model_type):
    model = _FakeModel(_FakeConfig(model_type=model_type))
    assert is_decoder_model(model) is True


# ── Known encoder families -> False ──────────────────────────────────────

ENCODER_MODEL_TYPES = ["bert", "roberta"]


@pytest.mark.parametrize("model_type", ENCODER_MODEL_TYPES)
def test_encoder_family_model_types_are_not_decoder_only(model_type):
    model = _FakeModel(_FakeConfig(model_type=model_type))
    assert is_decoder_model(model) is False


# ── is_encoder_decoder overrides everything else ─────────────────────────

def test_t5_like_encoder_decoder_is_not_decoder_only():
    model = _FakeModel(
        _FakeConfig(model_type="t5", is_decoder=False, is_encoder_decoder=True)
    )
    assert is_decoder_model(model) is False


def test_encoder_decoder_overrides_decoder_substring_in_model_type():
    # Some seq2seq sub-configs set is_decoder=True on the decoder half and
    # the model_type can still contain a decoder-allowlist substring (e.g.
    # "opt"-style naming); is_encoder_decoder=True must still win.
    model = _FakeModel(
        _FakeConfig(model_type="opt-seq2seq", is_decoder=True, is_encoder_decoder=True)
    )
    assert is_decoder_model(model) is False


def test_encoder_decoder_overrides_is_decoder_true():
    model = _FakeModel(
        _FakeConfig(model_type="t5", is_decoder=True, is_encoder_decoder=True)
    )
    assert is_decoder_model(model) is False


# ── is_decoder=True falls back to True even for an unknown model_type ────

def test_is_decoder_true_with_unknown_model_type_is_decoder_only():
    model = _FakeModel(
        _FakeConfig(
            model_type="some_future_arch", is_decoder=True, is_encoder_decoder=False
        )
    )
    assert is_decoder_model(model) is True


# ── Missing fields fall back sanely ───────────────────────────────────────

def test_missing_fields_and_unknown_model_type_is_not_decoder_only():
    # No is_decoder/is_encoder_decoder set at all (mirrors a bare config),
    # and model_type matches nothing in the allowlist -> not decoder-only.
    model = _FakeModel(_FakeConfig(model_type="some_future_arch"))
    assert is_decoder_model(model) is False
