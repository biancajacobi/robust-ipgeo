"""Tests für die Hub-/Centroid-Default-Erkennung (data/centroids.py).

Reine Logik-Tests mit synthetischen Records (keine DB-Dateien nötig) plus ein
Reader-Smoke-Test, der übersprungen wird, wenn die LITE-DBs lokal fehlen
(sie sind gitignored)."""

from pathlib import Path

import pytest

from data import centroids as C


def _records(coord, n, source="dbip_lite", city="Hub", asn="AS1", radius=None):
    return [{"source": source, "ip": f"10.0.0.{i}", "lat": coord[0], "lon": coord[1],
             "city": city, "asn": asn, "radius": radius} for i in range(n)]


def test_freq_hub_guard_b_flags_guard_a_needs_asn_collapse():
    # Ein großer ASN: 12 IPs auf dem Hub (50,8), aber 8 weitere woanders -> der ASN
    # kollabiert NICHT (12/20 = 0.6 < 0.9). Hub wird per Frequenz erkannt (>=10).
    recs = [{"source": "dbip_lite", "ip": f"1.1.1.{i}", "lat": 50.0, "lon": 8.0,
             "city": "Frankfurt", "asn": "AS_big", "radius": None} for i in range(12)]
    recs += [{"source": "dbip_lite", "ip": f"1.1.2.{i}", "lat": 40.0 + i, "lon": 5.0,
              "city": "Stadt", "asn": "AS_big", "radius": None} for i in range(8)]
    freq, asn_dom = C.build_index(recs)
    hub = recs[0]
    assert C.is_default(hub, "B", freq, asn_dom) is True     # liberal: Frequenz-Hub reicht
    assert C.is_default(hub, "A", freq, asn_dom) is False    # konservativ: ASN kollabiert nicht (0.6)


def test_guard_a_flags_when_asn_collapses():
    recs = _records((50.0, 8.0), 12, asn="AS42")            # alle 12 im selben ASN
    freq, asn_dom = C.build_index(recs)
    assert C.is_default(recs[0], "A", freq, asn_dom) is True


def test_city_none_is_default_both_guards():
    recs = _records((1.0, 2.0), 1, city=None)               # seltener Punkt, aber city fehlt
    freq, asn_dom = C.build_index(recs)
    assert C.is_default(recs[0], "A", freq, asn_dom) is True
    assert C.is_default(recs[0], "B", freq, asn_dom) is True


def test_maxmind_radius_is_default():
    r = {"source": "maxmind_geolite2", "ip": "9.9.9.9", "lat": 37.751, "lon": -97.822,
         "city": "Somewhere", "asn": "AS7", "radius": 1000}
    freq, asn_dom = C.build_index([r])
    assert C.is_default(r, "B", freq, asn_dom) is True
    assert C.detect(r, freq) == (True, "r>=500km")


def test_real_city_not_default():
    recs = _records((48.21, 16.37), 1, city="Vienna", radius=20)   # selten, Stadt, kleiner r
    freq, asn_dom = C.build_index(recs)
    assert C.is_default(recs[0], "B", freq, asn_dom) is False
    assert C.detect(recs[0], freq) == (False, "")


def test_known_centroid_match():
    # Kansas (~US-Default) liegt in KNOWN_CENTROIDS -> auch ohne Frequenz ein Default
    r = {"source": "ip2location_lite", "ip": "8.8.8.8", "lat": 37.751, "lon": -97.822,
         "city": "Wichita", "asn": "AS1", "radius": None}
    freq, _ = C.build_index([r])
    flag, match = C.detect(r, freq)
    assert flag is True and match.startswith("centroid:")


@pytest.mark.skipif(not Path("data/db/GeoLite2-City.mmdb").exists(),
                    reason="lokale LITE-DB nicht vorhanden (gitignored)")
def test_mmdb_reader_smoke():
    from data import fetch_sources as fs
    body = fs._read_mmdb("GeoLite2-City.mmdb")("8.8.8.8")
    assert body["found"] and body["country"] == "US"
    assert "db_version" in body and "accuracy_radius" in body
