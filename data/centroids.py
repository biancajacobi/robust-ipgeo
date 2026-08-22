"""Hub/centroid default detection for geo-DB outputs.

Many geo DBs place a **fallback point** for poorly localizable IPs
(country/region centroid or a large hub such as Frankfurt/Amsterdam/
Singapore) instead of admitting "I don't know". Such pseudo-points are not a
measurement but a heuristic of the source — and they bias a point estimator
systematically. This module flags them.

Because DB-IP/IP2Location (unlike MaxMind) provide **no** ``accuracy_radius``,
there are two deliberately different-strictness guards; both are reported:

  GUARD A (conservative, lower bound): a hub point only counts as a default if
    additionally the ASN *collapses* onto it (>= ASN_COLLAPSE of an ASN's IPs on
    the same point). Misses per-IP/sub-ASN defaults, but does not falsely flag
    genuine regional city ISPs.
  GUARD B (liberal, upper bound): any point on a frequency hub or a known
    geographic centroid counts as a default (without the ASN condition).

Both guards share two *clean per-IP signals*: ``city is None`` (source returns
no city) and — MaxMind only — ``accuracy_radius >= MM_RADIUS_DEFAULT``.

The hub list is **data-derived** (coordinate frequency across the dataset),
supplemented by a small manual list of geographic centroids. Thresholds are
**frozen** here (as of 2026-06-04) and were fixed BEFORE the measurement so the
evaluation does not look like heuristic tuning.
"""

from __future__ import annotations

from collections import Counter, defaultdict

# --- frozen thresholds (fixed before the measurement) ---
FREQ_HUB = 10            # >= N IPs on exactly the same coordinate -> frequency hub
HUB_STRONG = 50          # >= N -> dominant data-derived centroid (documentation)
ASN_COLLAPSE = 0.90      # share of an ASN's IPs on the same point -> collapse (A)
CENTROID_TOL = 0.10      # degree tolerance for matching against known geo centroids
MM_RADIUS_DEFAULT = 500  # MaxMind accuracy_radius (km) above which a hit is coarse

# Known geographic centroids (manual starting point; the actual hub list is
# supplemented data-derived, see build_index).
KNOWN_CENTROIDS = [
    (37.751, -97.822),   # MaxMind US fallback (~Kansas)
    (39.828, -98.579),   # geographic center of the US
    (51.165, 10.452),    # geographic center of DE
    (46.603, 1.888),     # geographic center of FR
    (47.001, 8.014),     # geographic center of CH
]


def round_coord(lat, lon):
    return (round(float(lat), 3), round(float(lon), 3))


def _near_known(c):
    return any(abs(c[0] - k[0]) <= CENTROID_TOL and abs(c[1] - k[1]) <= CENTROID_TOL
               for k in KNOWN_CENTROIDS)


def build_index(records):
    """records: iterable of dict(source, ip, lat, lon, city, country, radius, asn).

    Returns: (freq, asn_dom)
      freq[source]    Counter{coord: n}  — frequency of each coordinate per source
      asn_dom[s][asn] (dominant_coord, share) — largest coordinate share within the ASN
    """
    freq = defaultdict(Counter)
    asn_coords = defaultdict(lambda: defaultdict(Counter))
    for r in records:
        if r.get("lat") is None:
            continue
        c = round_coord(r["lat"], r["lon"])
        freq[r["source"]][c] += 1
        if r.get("asn"):
            asn_coords[r["source"]][r["asn"]][c] += 1
    asn_dom = {}
    for s, amap in asn_coords.items():
        asn_dom[s] = {}
        for asn, cc in amap.items():
            coord, n = cc.most_common(1)[0]
            asn_dom[s][asn] = (coord, n / sum(cc.values()))
    return freq, asn_dom


def hub_list(freq, source, threshold=HUB_STRONG):
    """Data-derived dominant centroids of a source (for documentation/reporting)."""
    return {c: n for c, n in freq[source].items() if n >= threshold}


def is_default(r, guard, freq, asn_dom):
    """True/False whether (source, ip) is a hub/centroid default; None without data.

    guard: 'A' (conservative, with ASN collapse) or 'B' (liberal, hub only).
    """
    if r.get("lat") is None:
        return None
    c = round_coord(r["lat"], r["lon"])
    # shared, clean per-IP signals (both guards)
    if not r.get("city"):
        return True
    if r["source"] == "maxmind_geolite2" and r.get("radius") and r["radius"] >= MM_RADIUS_DEFAULT:
        return True
    is_hub = freq[r["source"]][c] >= FREQ_HUB or _near_known(c)
    if guard == "B":
        return is_hub
    # guard A: hub AND ASN collapse onto exactly this point
    dom = asn_dom.get(r["source"], {}).get(r.get("asn"))
    asn_collapsed = dom is not None and dom[0] == c and dom[1] >= ASN_COLLAPSE
    return is_hub and asn_collapsed


def detect(r, freq):
    """For annotating the observations: (is_default_centroid, centroid_match).

    Uses guard B (liberal) — the per-observation meaningful "lies on a known
    hub?" flag (exactly the hub axis of the 2D predecessor-label matrix). ``centroid_match``
    explains the hit (reason or hub coordinate). Guard A (ASN-conditioned) is
    recomputed from the raw fields in the experiments.
    """
    if r.get("lat") is None:
        return False, ""
    c = round_coord(r["lat"], r["lon"])
    if not r.get("city"):
        return True, "no_city"
    if r["source"] == "maxmind_geolite2" and r.get("radius") and r["radius"] >= MM_RADIUS_DEFAULT:
        return True, f"r>={MM_RADIUS_DEFAULT}km"
    if freq[r["source"]][c] >= FREQ_HUB:
        return True, f"hub:{c[0]},{c[1]}"
    if _near_known(c):
        return True, f"centroid:{c[0]},{c[1]}"
    return False, ""
