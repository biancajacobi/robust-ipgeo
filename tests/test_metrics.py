import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import haversine, haversine_error


def test_haversine_zero_distance():
    assert haversine(52.52, 13.405, 52.52, 13.405) == 0.0


def test_haversine_known_distance_berlin_hamburg():
    # Berlin (52.52, 13.405) -> Hamburg (53.55, 9.99): ~255 km
    d = haversine(52.52, 13.405, 53.55, 9.99)
    assert 250 < d < 260


def test_haversine_error_wrapper():
    err = haversine_error((52.52, 13.405), (52.52, 13.405))
    assert err == 0.0
