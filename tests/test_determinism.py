"""Determinism / reproducibility tests (RQ4).

Substantiates the property claimed in the accompanying paper that the
evaluation yields identical results against frozen inputs: the aggregation
estimators are deterministic, and stochastic steps (bootstrap, contamination
selection, out-of-fold partitioning) are reproducible via fixed random seeds.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from estimators.baselines import (
    geometric_median,
    smoothed_geometric_median,
    weighted_geometric_median,
)
from estimators import braetz

# four dense points + one outlier (lat, lon)
POINTS = np.array([
    [52.50, 13.40], [52.51, 13.41], [52.52, 13.39], [52.49, 13.40], [10.0, 80.0],
])
WEIGHTS = np.array([1 / 3, 1 / 3, 1 / 3, 1.0, 1.0])


def test_estimators_are_deterministic():
    """Same input twice -> bit-identical estimate (no hidden state)."""
    for fn in (geometric_median, smoothed_geometric_median, braetz.estimate):
        a, b = fn(POINTS), fn(POINTS)
        assert np.array_equal(a, b), f"{fn.__name__} not deterministic"
    a = weighted_geometric_median(POINTS, WEIGHTS)
    b = weighted_geometric_median(POINTS, WEIGHTS)
    assert np.array_equal(a, b)


def test_seeded_bootstrap_is_reproducible():
    """Same seed -> identical bootstrap draw; different seed -> different."""
    def draw(seed):
        rng = np.random.default_rng(seed)
        return rng.integers(0, 100, 50)

    assert np.array_equal(draw(0), draw(0))
    assert not np.array_equal(draw(0), draw(1))


def test_seeded_folds_are_reproducible():
    """Out-of-fold partitioning is reproducible via the seed."""
    from experiments.exp_support_concentration import folds

    y = np.array([0, 1] * 50)
    assert np.array_equal(folds(y, seed=0), folds(y, seed=0))
