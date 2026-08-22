"""Evaluation metric: haversine distance between geographic points.

Convention throughout the project: coordinates as (lat, lon) in decimal degrees.
(Caution: the RIPE Atlas API returns ``geometry.coordinates`` as [lon, lat] —
swap when reading, see ``data/fetch_anchors.py``.)
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088  # mean Earth radius (IUGG)


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometers between two points (decimal degrees).

    Vectorized: ``lat*/lon*`` may be scalars or uniformly shaped arrays.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def haversine_error(estimate, truth):
    """Distance error (km) of an estimate (lat, lon) to the true location (lat, lon)."""
    est = np.asarray(estimate, dtype=float)
    tru = np.asarray(truth, dtype=float)
    return float(haversine(est[0], est[1], tru[0], tru[1]))
