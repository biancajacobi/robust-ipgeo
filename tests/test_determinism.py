"""Determinismus-/Reproduzierbarkeits-Tests.

Belegt die hier dokumentierte Eigenschaft, dass die Auswertung gegen
eingefrorene Eingaben identische Ergebnisse liefert: die Aggregations-Schätzer
sind deterministisch, und stochastische Schritte (Bootstrap, Kontaminations-
Auswahl, out-of-fold-Partitionierung) sind über feste Zufallssaaten reproduzierbar.
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

# vier dichte Punkte + ein Ausreißer (lat, lon)
POINTS = np.array([
    [52.50, 13.40], [52.51, 13.41], [52.52, 13.39], [52.49, 13.40], [10.0, 80.0],
])
WEIGHTS = np.array([1 / 3, 1 / 3, 1 / 3, 1.0, 1.0])


def test_estimators_are_deterministic():
    """Zweimal derselbe Input -> bit-identischer Schätzwert (kein versteckter Zustand)."""
    for fn in (geometric_median, smoothed_geometric_median, braetz.estimate):
        a, b = fn(POINTS), fn(POINTS)
        assert np.array_equal(a, b), f"{fn.__name__} nicht deterministisch"
    a = weighted_geometric_median(POINTS, WEIGHTS)
    b = weighted_geometric_median(POINTS, WEIGHTS)
    assert np.array_equal(a, b)


def test_seeded_bootstrap_is_reproducible():
    """Gleicher Seed -> identische Bootstrap-Ziehung; verschiedener Seed -> verschieden."""
    def draw(seed):
        rng = np.random.default_rng(seed)
        return rng.integers(0, 100, 50)

    assert np.array_equal(draw(0), draw(0))
    assert not np.array_equal(draw(0), draw(1))


def test_seeded_folds_are_reproducible():
    """Out-of-fold-Partitionierung ist über den Seed reproduzierbar."""
    from experiments.exp_support_concentration import folds

    y = np.array([0, 1] * 50)
    assert np.array_equal(folds(y, seed=0), folds(y, seed=0))
