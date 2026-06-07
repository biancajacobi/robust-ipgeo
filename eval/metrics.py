"""Bewertungsmetrik: Haversine-Distanz zwischen geografischen Punkten.

Konvention im gesamten Projekt: Koordinaten als (lat, lon) in Dezimalgrad.
(Achtung: die RIPE-Atlas-API liefert ``geometry.coordinates`` als [lon, lat] —
beim Einlesen umdrehen, siehe ``data/fetch_anchors.py``.)
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088  # mittlerer Erdradius (IUGG)


def haversine(lat1, lon1, lat2, lon2):
    """Großkreis-Distanz in Kilometern zwischen zwei Punkten (Dezimalgrad).

    Vektorisiert: ``lat*/lon*`` dürfen Skalare oder gleichförmige Arrays sein.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def haversine_error(estimate, truth):
    """Distanzfehler (km) einer Schätzung (lat, lon) zum wahren Ort (lat, lon)."""
    est = np.asarray(estimate, dtype=float)
    tru = np.asarray(truth, dtype=float)
    return float(haversine(est[0], est[1], tru[0], tru[1]))
