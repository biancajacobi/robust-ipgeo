import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from estimators.baselines import (
    centroid,
    coordinate_median,
    geometric_median,
    smoothed_geometric_median,
    trimmed_mean,
    weighted_geometric_median,
)

# 4 dichte Punkte + 1 grober Ausreißer
POINTS = np.array(
    [
        [52.50, 13.40],
        [52.51, 13.41],
        [52.52, 13.39],
        [52.49, 13.40],
        [10.00, 80.00],  # Ausreißer
    ]
)


def test_coordinate_median_resists_outlier():
    m = coordinate_median(POINTS)
    assert abs(m[0] - 52.5) < 0.1
    assert abs(m[1] - 13.4) < 0.1


def test_trimmed_mean_resists_outlier():
    m = trimmed_mean(POINTS, proportion=0.2)
    assert abs(m[0] - 52.5) < 0.2
    assert abs(m[1] - 13.4) < 0.2


def test_geometric_median_resists_outlier():
    m = geometric_median(POINTS)
    assert abs(m[0] - 52.5) < 0.2
    assert abs(m[1] - 13.4) < 0.2


def test_geometric_median_on_clean_cluster():
    clean = POINTS[:4]
    m = geometric_median(clean)
    assert abs(m[0] - 52.505) < 0.05
    assert abs(m[1] - 13.40) < 0.05


def test_centroid_is_not_robust():
    # der naive Mittelwert wird vom Ausreißer kräftig weggezogen
    m = centroid(POINTS)
    assert abs(m[0] - 52.5) > 5  # lat klar verschoben (Erwartung: ~44.4)


def test_smoothed_geometric_median_resists_outlier():
    m = smoothed_geometric_median(POINTS)
    assert abs(m[0] - 52.5) < 0.2
    assert abs(m[1] - 13.4) < 0.2


def test_weighted_geometric_median_uniform_equals_smoothed():
    import numpy as np
    a = smoothed_geometric_median(POINTS)
    b = weighted_geometric_median(POINTS, weights=None)
    assert np.allclose(a, b, atol=1e-6)


def test_weighted_geometric_median_downweights_correlated_line():
    # drei (fast) gleiche "London"-Quellen + zwei "Berlin"-Quellen.
    # Pro Quelle gewichtet -> London-Mehrheit zieht den Schätzer nach London;
    # pro Linie gewichtet (London = 1 Linie) -> Berlin (2 Linien) gewinnt.
    pts = [
        [51.50, -0.12], [51.51, -0.13], [51.49, -0.11],  # London-Linie (3x)
        [52.52, 13.40], [52.50, 13.41],                  # zwei Berlin-Quellen
    ]
    per_source = weighted_geometric_median(pts)                 # uniform
    per_line_w = [1 / 3, 1 / 3, 1 / 3, 1.0, 1.0]                # London teilt sich 1
    per_line = weighted_geometric_median(pts, weights=per_line_w)
    assert per_source[0] < 52.0      # uniform: Richtung London (lat ~51.5)
    assert per_line[0] > 52.0        # linien-gewichtet: Richtung Berlin (lat ~52.5)
