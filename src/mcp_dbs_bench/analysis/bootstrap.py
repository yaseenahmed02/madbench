"""Bootstrap 95% confidence intervals via the percentile method.

Primary results use 10_000 resamples; smaller counts
are accepted via the n_resamples parameter for use inside loops or tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

N_BOOTSTRAP_RESAMPLES: int = 10_000
BOOTSTRAP_SEED: int = 20260525


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    """Bootstrap result for a scalar statistic."""

    point_estimate: float
    ci_low_95: float
    ci_high_95: float
    n_resamples: int


def bootstrap_statistic(
    samples: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    n_resamples: int = N_BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapCI:
    """Bootstrap 95% CI via the percentile method.

    Args:
        samples: 1D array of source samples.
        statistic_fn: Callable taking a 1D array and returning a scalar
            (e.g. ``np.median``, ``np.mean``).
        n_resamples: Number of bootstrap resamples (default 10_000).
        seed: PRNG seed for reproducibility.

    Returns:
        A :class:`BootstrapCI` with point estimate and CI bounds.

    Raises:
        ValueError: If ``samples`` is empty.
    """
    n = len(samples)
    if n == 0:
        raise ValueError("bootstrap_statistic requires at least one sample")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_resamples, n))
    resampled_stats = np.array([statistic_fn(samples[idx]) for idx in indices])
    return BootstrapCI(
        point_estimate=float(statistic_fn(samples)),
        ci_low_95=float(np.percentile(resampled_stats, 2.5)),
        ci_high_95=float(np.percentile(resampled_stats, 97.5)),
        n_resamples=n_resamples,
    )
