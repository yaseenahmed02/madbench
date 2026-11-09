"""Tests for the bootstrap CI computation."""

from __future__ import annotations

import numpy as np
import pytest

from mcp_dbs_bench.analysis.bootstrap import (
    BOOTSTRAP_SEED,
    bootstrap_statistic,
)


class TestBootstrapStatistic:
    def test_bootstrap_median_recovers_point_estimate(self) -> None:
        # The point estimate should equal np.median(samples) exactly.
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ci = bootstrap_statistic(samples, np.median, n_resamples=200)
        assert ci.point_estimate == pytest.approx(3.0)

    def test_bootstrap_ci_brackets_point_estimate(self) -> None:
        # 95% CI should contain the point estimate for any non-degenerate input.
        rng = np.random.default_rng(0)
        samples = rng.normal(loc=10.0, scale=1.0, size=100)
        ci = bootstrap_statistic(samples, np.mean, n_resamples=500)
        assert ci.ci_low_95 < ci.point_estimate < ci.ci_high_95

    def test_bootstrap_is_deterministic_with_fixed_seed(self) -> None:
        # Reproducibility: same input + same seed -> identical CI.
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        a = bootstrap_statistic(samples, np.median, n_resamples=200, seed=BOOTSTRAP_SEED)
        b = bootstrap_statistic(samples, np.median, n_resamples=200, seed=BOOTSTRAP_SEED)
        assert a.point_estimate == b.point_estimate
        assert a.ci_low_95 == b.ci_low_95
        assert a.ci_high_95 == b.ci_high_95

    def test_bootstrap_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            bootstrap_statistic(np.array([]), np.median, n_resamples=100)
