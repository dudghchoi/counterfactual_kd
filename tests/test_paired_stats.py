"""Paired rate-difference intervals and the two-axis audit verdict (TIFS plan C6).

The Tango interval is checked against an independent construction: the
constrained MLE is found by numerical maximisation (scipy) instead of the
closed form, and the interval by scipy root-finding instead of bisection.
"""

from __future__ import annotations

import math
import random

import pytest

from counterfactual_kd.metrics.paired import (
    _tango_score, _z, audit_verdict, decomposition_bootstrap, delta_star,
    newcombe_paired_ci, paired_counts, tango_paired_ci, wilson_interval,
)


def _vectors(n11, n10, n01, n00):
    obs = [1] * n11 + [1] * n10 + [0] * n01 + [0] * n00
    nul = [1] * n11 + [0] * n10 + [1] * n01 + [0] * n00
    return obs, nul


TABLES = [(10, 5, 2, 194), (0, 0, 0, 211), (200, 6, 1, 4), (30, 40, 10, 120),
          (0, 3, 0, 50), (5, 0, 7, 88), (211, 0, 0, 0), (1000, 120, 95, 4485)]


def test_z_quantiles():
    assert _z(0.95) == pytest.approx(1.959963984540054, abs=1e-9)
    assert _z(0.90) == pytest.approx(1.6448536269514722, abs=1e-9)


def test_paired_counts():
    assert paired_counts(*_vectors(3, 2, 1, 4)) == (3, 2, 1, 4)
    with pytest.raises(ValueError):
        paired_counts([1, 0], [1])
    with pytest.raises(ValueError):
        paired_counts([2], [1])


def test_wilson_zero_successes():
    lo, hi = wilson_interval(0, 10)
    z2 = _z(0.95) ** 2
    assert lo == 0.0
    assert hi == pytest.approx(z2 / (10 + z2), rel=1e-9)


class TestTango:

    def test_no_discordance_closed_form(self):
        # n10 = n01 = 0 -> [-z^2/(n+z^2), +z^2/(n+z^2)], never degenerate.
        lo, hi = tango_paired_ci(*_vectors(10, 0, 0, 201))
        z2 = _z(0.95) ** 2
        assert hi == pytest.approx(z2 / (211 + z2), abs=1e-9)
        assert lo == pytest.approx(-z2 / (211 + z2), abs=1e-9)

    @pytest.mark.parametrize("table", TABLES)
    def test_matches_independent_construction(self, table):
        scipy_opt = pytest.importorskip("scipy.optimize")
        n11, n10, n01, n00 = table
        n = sum(table)
        m = n11 + n00
        z = _z(0.95)

        def mle_p01(d0):
            # Maximise b log(p01+d0) + c log p01 + m log(1-2 p01-d0) numerically.
            lo = max(0.0, -d0) + 1e-13
            hi = (1.0 - d0) / 2.0 - 1e-13
            if hi <= lo:
                return lo

            def nll(p):
                total = 0.0
                for cnt, prob in ((n10, p + d0), (n01, p), (m, 1 - 2 * p - d0)):
                    if cnt:
                        total -= cnt * math.log(max(prob, 1e-300))
                return total
            return scipy_opt.minimize_scalar(nll, bounds=(lo, hi), method="bounded",
                                             options={"xatol": 1e-13}).x

        def score(d0):
            p01 = mle_p01(d0)
            var = n * (2 * p01 + d0 * (1 - d0))
            num = n10 - n01 - n * d0
            return num / math.sqrt(var) if var > 0 else math.copysign(math.inf, num or 1)

        k_hat = (n10 - n01) / n
        e = 1e-9
        ref_lo = -1.0 if score(-1 + e) <= z else scipy_opt.brentq(
            lambda d: score(d) - z, -1 + e, k_hat - 1e-12 if k_hat > -1 + e else k_hat)
        ref_hi = 1.0 if score(1 - e) >= -z else scipy_opt.brentq(
            lambda d: score(d) + z, k_hat + 1e-12 if k_hat < 1 - e else k_hat, 1 - e)
        lo, hi = tango_paired_ci(*_vectors(*table))
        assert lo == pytest.approx(ref_lo, abs=1e-6)
        assert hi == pytest.approx(ref_hi, abs=1e-6)

    @pytest.mark.parametrize("table", TABLES)
    def test_score_decreasing(self, table):
        n11, n10, n01, n00 = table
        n = sum(table)
        grid = [-0.999 + i * 0.0333 for i in range(61)]
        vals = [_tango_score(n10, n01, n, d) for d in grid]
        assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))

    @pytest.mark.parametrize("table", TABLES)
    def test_contains_point_and_mirrors(self, table):
        n11, n10, n01, n00 = table
        lo, hi = tango_paired_ci(*_vectors(*table))
        k = (n10 - n01) / sum(table)
        assert lo <= k <= hi
        lo2, hi2 = tango_paired_ci(*_vectors(n11, n01, n10, n00))
        assert (lo2, hi2) == pytest.approx((-hi, -lo), abs=1e-9)

    def test_large_n_agrees_with_wald(self):
        n11, n10, n01, n00 = 20000, 3000, 2400, 74600
        n = 100000
        k = (n10 - n01) / n
        se = math.sqrt((n10 + n01) / n ** 2 - (n10 - n01) ** 2 / n ** 3)
        lo, hi = tango_paired_ci(*_vectors(n11, n10, n01, n00))
        assert lo == pytest.approx(k - 1.96 * se, abs=2e-4)
        assert hi == pytest.approx(k + 1.96 * se, abs=2e-4)

    def test_coverage_near_nominal(self):
        # Fixed seed -> deterministic. n=211 with sparse discordance, the
        # regime where the percentile bootstrap degenerates.
        rng = random.Random(0)
        n, p10, p01, p11 = 211, 0.03, 0.02, 0.05
        k_true = p10 - p01
        hit, sims = 0, 600
        for _ in range(sims):
            cells = [0, 0, 0, 0]
            for _ in range(n):
                u = rng.random()
                cells[0 if u < p11 else 1 if u < p11 + p10 else
                      2 if u < p11 + p10 + p01 else 3] += 1
            lo, hi = tango_paired_ci(*_vectors(*cells))
            hit += lo <= k_true <= hi
        assert 0.91 <= hit / sims <= 0.99


class TestNewcombe:

    @pytest.mark.parametrize("table", TABLES)
    def test_contains_point_and_mirrors(self, table):
        n11, n10, n01, n00 = table
        lo, hi = newcombe_paired_ci(*_vectors(*table))
        k = (n10 - n01) / sum(table)
        assert -1.0 <= lo <= k <= hi <= 1.0
        lo2, hi2 = newcombe_paired_ci(*_vectors(n11, n01, n10, n00))
        assert (lo2, hi2) == pytest.approx((-hi, -lo), abs=1e-9)

    def test_close_to_tango_at_moderate_n(self):
        table = (300, 60, 40, 600)
        t = tango_paired_ci(*_vectors(*table))
        nc = newcombe_paired_ci(*_vectors(*table))
        assert nc == pytest.approx(t, abs=3e-3)


class TestAuditVerdict:

    def test_equivalent(self):
        r = audit_verdict(*_vectors(500, 20, 18, 5162), delta=0.02)
        assert r["relative"] == "EQUIVALENT"
        assert r["delta_star"] <= 0.02
        assert r["n"] == 5700

    def test_positive_and_negative(self):
        assert audit_verdict(*_vectors(10, 150, 2, 49), delta=0.05)["relative"] == "POSITIVE"
        assert audit_verdict(*_vectors(10, 2, 150, 49), delta=0.05)["relative"] == "NEGATIVE"

    def test_inconclusive_small_n(self):
        r = audit_verdict(*_vectors(10, 6, 2, 193), delta=0.02)
        assert r["relative"] == "INCONCLUSIVE"
        assert r["delta_star"] > 0.02

    def test_saturation_no_longer_overrides(self):
        # Both students near ceiling and no discordance: the old ladder
        # returned CORPUS-SATURATED; here the relative axis still decides
        # and the exposure axis flags both students.
        r = audit_verdict(*_vectors(211, 0, 0, 0), delta=0.05)
        assert r["relative"] == "EQUIVALENT"
        assert r["saturated"] is True
        assert r["p_exposure"] == r["c_exposure"] == "HIGH"

    def test_delta_star(self):
        assert delta_star(-0.03, 0.01) == 0.03
        assert delta_star(0.02, 0.05) == 0.05

    def test_methods_and_errors(self):
        v = _vectors(10, 5, 2, 194)
        assert audit_verdict(*v, delta=0.1, method="newcombe")["method"] == "newcombe"
        with pytest.raises(ValueError):
            audit_verdict(*v, delta=0.1, method="wald")
        with pytest.raises(ValueError):
            audit_verdict(*v, delta=-0.1)


class TestDecomposition:

    def _four(self, n=300, seed=1):
        rng = random.Random(seed)
        pc = [int(rng.random() < 0.10) for _ in range(n)]
        cc = [int(rng.random() < 0.08) for _ in range(n)]
        pt = [max(a, int(rng.random() < 0.30)) for a in pc]
        ct = [max(a, int(rng.random() < 0.10)) for a in cc]
        return pt, ct, pc, cc

    def test_identity_and_intervals(self):
        pt, ct, pc, cc = self._four()
        r = decomposition_bootstrap(pt, ct, pc, cc, R=400, seed=0)
        assert r["k"] == pytest.approx(r["delta_b"] + r["k_star"], abs=1e-12)
        for name in ("k", "delta_b", "k_star"):
            lo, hi = r[f"{name}_ci"]
            assert lo <= r[name] <= hi

    def test_reproducible(self):
        v = self._four()
        assert decomposition_bootstrap(*v, R=200, seed=3) == \
            decomposition_bootstrap(*v, R=200, seed=3)

    def test_misaligned_rejected(self):
        pt, ct, pc, cc = self._four()
        with pytest.raises(ValueError):
            decomposition_bootstrap(pt, ct, pc, cc[:-1])
