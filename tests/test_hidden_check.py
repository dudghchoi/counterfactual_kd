"""Tests for the pure parts of scripts/tifs/hidden_check.py (TIFS addendum A8)."""

from __future__ import annotations

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "tifs"))
import hidden_check as H  # noqa: E402


def _block(side, losses):
    return [f"[KD-adapter] hidden | {side} teacher → student"] + [
        f"KD Epoch {e}/5: loss={x}" for e, x in enumerate(losses, 1)]


def test_epoch_losses_and_training_rule():
    log = "\n".join(
        _block("poisoned", [7.0, 5.0, 4.0, 3.5, 3.2])
        + ["[KD-adapter] SKIP runs/tifs/x/clean_student"]
        # stopped after two epochs, resumed: the block restarts at epoch 1
        + _block("clean", [7.1, 5.2]) + _block("clean", [7.1, 5.2, 4.1, 3.6, 3.3]))
    blocks = H.epoch_losses(log)
    assert [b["side"] for b in blocks] == ["poisoned", "clean", "clean"]
    assert blocks[2]["losses"] == [7.1, 5.2, 4.1, 3.6, 3.3]
    assert H.training_ok(blocks, 2)          # the stopped block is left out
    assert not H.training_ok(blocks, 3)      # one student short
    assert not H.training_ok(H.epoch_losses("\n".join(_block("clean", [7.0, 6.0, "nan", 5.0, 4.0]))), 1)
    assert not H.training_ok(H.epoch_losses("\n".join(_block("clean", [3.0, 3.1, 3.2, 3.3, 3.4]))), 1)


def test_epoch_losses_reads_ce_headers_and_headerless_restarts():
    log = "\n".join(
        _block("poisoned", [7.0, 5.0, 4.0, 3.5, 3.2])
        + ["[KD-adapter] ce | none teacher → student"]
        + [f"KD Epoch {e}/5: loss={x}" for e, x in enumerate([0.6, 0.3, 0.2, 0.1, 0.05], 1)]
        + [f"KD Epoch {e}/5: loss={x}" for e, x in enumerate([0.6, 0.3], 1)])
    blocks = H.epoch_losses(log)
    assert [(b["kd"], b["side"], len(b["losses"])) for b in blocks] == [
        ("hidden", "poisoned", 5), ("ce", "none", 5), ("ce", "none", 2)]
    assert blocks[0]["losses"] == [7.0, 5.0, 4.0, 3.5, 3.2]


def test_path_resolvers(tmp_path):
    runs = tmp_path / "runs"
    for run, tag, sub in (("tifs_hid_bert_sst2_s43", "badnets_cf_mn_hidden_bert_bert-base-uncased_sst2_seed43_t42_c42_s43", "poisoned_student"),
                          ("tifs_hid_bert_sst2_s43", "clean-base_hidden_bert_bert-base-uncased_sst2_seed43_t42_c42_s43", "clean_student"),
                          ("tifs_hid_bert_sst2_s43", "clean-sos_hidden_bert_bert-base-uncased_sst2_seed43_t42_c42_s43", "clean_student"),
                          ("tifs_b_bert_sst2_s43", "clean-base_logit_bert_bert-base-uncased_sst2_seed43_t42_c42_s43", "clean_student"),
                          ("tifs_b_bert_sst2_s43", "clean-base_feature_bert_bert-base-uncased_sst2_seed43_t42_c42_s43", "clean_student")):
        d = runs / run / "artifacts" / "students" / tag / sub
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}")
    teacher = tmp_path / "results" / "tifs" / "teachers" / "clean" / "t"
    teacher.mkdir(parents=True)
    hid = runs / "tifs_hid_bert_sst2_s43"
    found = H.student_dirs(str(hid))
    assert sorted(r for r, _ in found.values()) == ["C", "C", "P"]
    lc = H.logit_clean_base(str(runs), 43)
    assert lc is not None and "clean-base_logit_" in lc
    assert H.logit_clean_base(str(runs), 44) is None
    c = next(p for t, (r, p) in found.items() if t.startswith("clean-base_"))
    rel = os.path.relpath(teacher, c)
    json.dump({"teacher_ref": rel}, open(os.path.join(c, "kd_meta.json"), "w"))
    assert H.teacher_of(c) == os.path.normpath(str(teacher))


def _stats(xs, ys):
    d = xs[0].shape[1] + 1
    XtX = [torch.zeros(d, d, dtype=torch.float64) for _ in xs]
    XtY = [torch.zeros(d, ys[0].shape[1], dtype=torch.float64) for _ in xs]
    Syy = []
    for k, (x, y) in enumerate(zip(xs, ys)):
        x1 = torch.cat([x, torch.ones(x.shape[0], 1, dtype=torch.float64)], 1)
        XtX[k] += x1.T @ x1
        XtY[k] += x1.T @ y
        Syy.append(float((y * y).sum()))
    return XtX, XtY, Syy


def test_ls_residual_zero_for_one_shared_map_and_matches_direct_fit():
    g = torch.Generator().manual_seed(0)
    W = torch.randn(6, 8, generator=g, dtype=torch.float64)
    b = torch.randn(8, generator=g, dtype=torch.float64)
    xs = [torch.randn(40, 6, generator=g, dtype=torch.float64) for _ in range(3)]
    M, pairs = H.ls_residual(*_stats(xs, [x @ W + b for x in xs]), 40, 8)
    assert M == pytest.approx(0.0, abs=1e-8) and len(pairs) == 3
    # Different maps per pair: equals the residual of one direct shared fit.
    ys = [x @ torch.randn(6, 8, generator=g, dtype=torch.float64) for x in xs]
    M, pairs = H.ls_residual(*_stats(xs, ys), 40, 8)
    X = torch.cat([torch.cat([x, torch.ones(40, 1, dtype=torch.float64)], 1) for x in xs])
    sol = torch.linalg.lstsq(X, torch.cat(ys)).solution
    direct = [float(((torch.cat([x, torch.ones(40, 1, dtype=torch.float64)], 1) @ sol - y) ** 2).sum()) / (40 * 8)
              for x, y in zip(xs, ys)]
    assert pairs == pytest.approx(direct, rel=1e-5) and M == pytest.approx(sum(direct), rel=1e-5)
