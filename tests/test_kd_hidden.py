"""Tests for ``counterfactual_kd.kd.hidden.HiddenStateKD`` (TIFS addendum A8).

The hidden-state objective must (a) map student layers to teacher layers as
TinyBERT does, (b) ignore pad positions, and (c) leave the global RNGs
untouched, so that its students get the same head init, data order and
dropout masks as the logit-KD and CE students with the same seed. (c) is
checked end to end with tiny random BERT models on CPU: the student is
saved without a classification head, so loading it draws the head from the
global RNG as bert-base-uncased does, and the CE-only trainer (which loads
no teacher) is compared too.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from counterfactual_kd.kd.hidden import HiddenStateKD


def test_layer_map_uniform_top_aligned():
    assert HiddenStateKD.layer_map(12, 24) == list(range(0, 25, 2))   # BERT-large -> base
    assert HiddenStateKD.layer_map(12, 48) == list(range(0, 49, 4))   # GPT-2-XL -> GPT-2
    assert HiddenStateKD.layer_map(28, 28) == list(range(29))         # Qwen3 1.7B -> 0.6B
    assert HiddenStateKD.layer_map(4, 12) == [0, 3, 6, 9, 12]         # TinyBERT4 from BERT-base


def test_projector_draws_no_global_rng_and_is_seeded():
    torch.manual_seed(0)
    before = torch.get_rng_state()
    kd = HiddenStateKD(seed=5)
    p = kd._get_projector(48, 64, torch.device("cpu"), torch.float32)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.count_nonzero(p.bias) == 0
    assert abs(float(p.weight.detach().std()) - 0.02) < 0.002
    q = HiddenStateKD(seed=5)._get_projector(48, 64, torch.device("cpu"), torch.float32)
    assert torch.equal(p.weight, q.weight)
    r = HiddenStateKD(seed=6)._get_projector(48, 64, torch.device("cpu"), torch.float32)
    assert not torch.equal(p.weight, r.weight)


def _states(n_layers, B, S, D, gen):
    return tuple(torch.randn(B, S, D, generator=gen) for _ in range(n_layers + 1))


def test_hidden_loss_matches_manual_and_ignores_pads():
    gen = torch.Generator().manual_seed(1)
    s_hs = _states(2, 3, 5, 6, gen)
    t_hs = _states(4, 3, 5, 8, gen)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    kd = HiddenStateKD(seed=0)
    loss = kd.hidden_loss(s_hs, t_hs, mask)

    W, b = kd._projector.weight, kd._projector.bias
    keep = mask.bool()
    manual = sum(((s_hs[m][keep] @ W.T + b - t_hs[2 * m][keep]) ** 2).mean()
                 for m in range(3))
    assert torch.allclose(loss, manual, rtol=1e-5, atol=1e-6)

    s_pad = tuple(h.clone() for h in s_hs)
    t_pad = tuple(h.clone() for h in t_hs)
    for h in s_pad + t_pad:
        h[~keep] = 1e3
    assert torch.allclose(kd.hidden_loss(s_pad, t_pad, mask), loss, rtol=1e-5, atol=1e-6)


def test_hidden_loss_zero_when_projected_states_match():
    gen = torch.Generator().manual_seed(2)
    t_hs = _states(4, 2, 4, 6, gen)
    s_hs = tuple(t_hs[j] for j in (0, 2, 4))
    kd = HiddenStateKD(seed=0)
    kd._get_projector(6, 6, torch.device("cpu"), torch.float32)
    with torch.no_grad():
        kd._projector.weight.copy_(torch.eye(6))
    mask = torch.ones(2, 4, dtype=torch.long)
    assert float(kd.hidden_loss(s_hs, t_hs, mask).detach()) == pytest.approx(0.0, abs=1e-10)


# ── End to end on tiny random BERTs (CPU) ──────────────────────────────────

WORDS = ("good bad film plot actor great awful boring fun slow "
         "story music scene dull smart").split()


def _tiny_pair(tmp_path):
    from transformers import BertConfig, BertForSequenceClassification, BertModel, BertTokenizerFast

    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]) + "\n")
    tok = BertTokenizerFast(vocab_file=str(vocab))
    paths = {}
    for name, (hidden, layers) in {"teacher": (48, 4), "student": (32, 2)}.items():
        torch.manual_seed(123)
        cfg = BertConfig(vocab_size=tok.vocab_size, hidden_size=hidden, num_hidden_layers=layers,
                         num_attention_heads=4, intermediate_size=64,
                         max_position_embeddings=32, num_labels=2)
        d = tmp_path / name
        # The student has no head, like bert-base-uncased: from_pretrained
        # initialises it from the global RNG.
        (BertForSequenceClassification(cfg) if name == "teacher" else BertModel(cfg)).save_pretrained(d)
        tok.save_pretrained(d)
        paths[name] = str(d)
    rng = random.Random(0)
    data = [{"sentence": " ".join(rng.choices(WORDS, k=rng.randint(2, 7))), "label": i % 2}
            for i in range(12)]
    return paths, data


def _train(kd_cls, paths, data, **kw):
    kd = kd_cls(epochs=2, batch_size=4, max_length=16, warmup_epochs=0, seed=7,
                teacher_dtype="fp32", student_dtype="fp32", **kw)
    kd.train(paths["teacher"], paths["student"], data, num_labels=2,
             device=torch.device("cpu"))
    return kd, torch.get_rng_state(), np.random.get_state()[1].copy(), random.getstate()


def test_hidden_kd_trains_and_keeps_the_logit_rng_stream(tmp_path):
    from counterfactual_kd.kd.ce import CEOnly
    from counterfactual_kd.kd.feature import FeatureKD
    from counterfactual_kd.kd.logit import LogitKD

    paths, data = _tiny_pair(tmp_path)
    _, rng_logit, np_logit, py_logit = _train(LogitKD, paths, data)
    kd, rng_hidden, np_hidden, py_hidden = _train(HiddenStateKD, paths, data)
    _, rng_ce, np_ce, py_ce = _train(CEOnly, paths, data)
    # Same global-RNG consumption => same head init, shuffles and dropout masks.
    assert torch.equal(rng_logit, rng_hidden) and torch.equal(rng_logit, rng_ce)
    assert np.array_equal(np_logit, np_hidden) and py_logit == py_hidden
    assert np.array_equal(np_logit, np_ce) and py_logit == py_ce
    # The projector was registered with the optimizer and trained.
    init = HiddenStateKD(seed=7)._get_projector(32, 48, torch.device("cpu"), torch.float32)
    assert kd._projector is not None
    assert not torch.equal(kd._projector.weight, init.weight)
    # FeatureKD's projector (frozen code) draws from the global CPU RNG;
    # this is why its students' later-epoch data order differs.
    _, rng_feature, _, _ = _train(FeatureKD, paths, data)
    assert not torch.equal(rng_logit, rng_feature)
