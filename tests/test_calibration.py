"""Calibration study (TIFS plan S1): vectorised intervals and scoring."""

from __future__ import annotations

import numpy as np
import pytest

from counterfactual_kd.analysis.calibration import (
    Scenario, bootstrap_ci, classify, newcombe_ci, run_scenario,
    scenario_from_rates, tango_ci, verdict,
)
from counterfactual_kd.metrics.paired import newcombe_paired_ci, tango_paired_ci

TABLES = [(10, 5, 2, 194), (0, 0, 0, 211), (200, 6, 1, 4), (30, 40, 10, 120),
          (0, 3, 0, 50), (5, 0, 7, 88), (211, 0, 0, 0), (1000, 120, 95, 4485)]


def _vectors(n11, n10, n01, n00):
    obs = [1] * n11 + [1] * n10 + [0] * n01 + [0] * n00
    nul = [1] * n11 + [0] * n10 + [1] * n01 + [0] * n00
    return obs, nul


def test_vectorised_tango_matches_scalar():
    t = np.array(TABLES)
    lo, hi = tango_ci(t[:, 1], t[:, 2], t.sum(1))
    for i, table in enumerate(TABLES):
        assert (lo[i], hi[i]) == pytest.approx(tango_paired_ci(*_vectors(*table)), abs=1e-9)


def test_vectorised_newcombe_matches_scalar():
    t = np.array(TABLES)
    lo, hi = newcombe_ci(*t.T)
    for i, table in enumerate(TABLES):
        assert (lo[i], hi[i]) == pytest.approx(newcombe_paired_ci(*_vectors(*table)), abs=1e-12)


def test_bootstrap_degenerates_without_discordance():
    rng = np.random.default_rng(0)
    (hlo, hhi), (klo, khi) = bootstrap_ci((211, 0, 0, 0), rng, R=200)
    assert hlo == hhi == 0.0 and klo == khi == 0.0
    lo, hi = tango_ci(np.array([0]), np.array([0]), 211)
    assert hi[0] - lo[0] > 0.03


def test_verdict_and_classify():
    v = verdict([-0.01, 0.06, -0.2, -0.01], [0.01, 0.2, -0.06, 0.09], 0.05)
    assert list(v) == ["EQUIVALENT", "POSITIVE", "NEGATIVE", "INCONCLUSIVE"]
    c = classify(v, true_value=0.0, tol=0.05)
    assert c["false_dir"] == 0.5 and c["power_equiv"] == 0.25 and c["false_equiv"] == 0.0
    c = classify(v, true_value=0.1, tol=0.05)
    assert c["false_equiv"] == 0.25 and c["power_dir"] == 0.25 and c["false_dir"] == 0.25


def test_scenario_from_rates_and_feasibility():
    sc = scenario_from_rates("x", 211, c_trig=0.1, k=0.05, discord=0.07)
    assert sc.k == pytest.approx(0.05)
    assert sc.pi11 + sc.pi01 == pytest.approx(0.1)
    with pytest.raises(ValueError):
        Scenario("bad", 10, 0.9, 0.2, 0.0).probs


def test_run_scenario_null_is_calibrated():
    sc = scenario_from_rates("null", 5700, c_trig=0.3, k=0.0, discord=0.06)
    r = run_scenario(sc, reps=300, R=300, seed=1, tols_k=(0.02,))
    assert 0.9 <= r["tango_k.coverage"] <= 1.0
    assert r["tango_k.false_dir@0.02"] <= 0.03
    assert r["tango_k.equivalent@0.02"] > 0.9
