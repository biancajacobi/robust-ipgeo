"""Tests für die Quellen-Adapter (netzfrei): Parser, Normalisierung, CSV."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import fetch_sources as fs
from data import store


# echte Roh-Antwort-Ausschnitte je API (gekürzt) für dieselbe IP
IP_API_OK = {"status": "success", "city": "Dubai", "countryCode": "AE",
             "lat": 25.0657, "lon": 55.1713}
IPAPI_CO_OK = {"latitude": 38.9072, "longitude": -77.0369, "city": "Washington", "country": "US"}
IPWHO_OK = {"success": True, "latitude": 51.5074, "longitude": -0.1278,
            "city": "London", "country_code": "GB"}


def test_parse_ip_api_success_and_fail():
    assert fs._parse_ip_api(IP_API_OK) == (25.0657, 55.1713, "Dubai", "AE", "success")
    lat, lon, city, country, status = fs._parse_ip_api({"status": "fail", "message": "private range"})
    assert (lat, lon) == (None, None) and status == "private range"


def test_parse_ipapi_co_success_and_error():
    assert fs._parse_ipapi_co(IPAPI_CO_OK) == (38.9072, -77.0369, "Washington", "US", "success")
    *_, status = fs._parse_ipapi_co({"error": True, "reason": "RateLimited"})
    assert status == "RateLimited"


def test_parse_ipwho_is_success_and_fail():
    assert fs._parse_ipwho_is(IPWHO_OK) == (51.5074, -0.1278, "London", "GB", "success")
    *_, status = fs._parse_ipwho_is({"success": False, "message": "Invalid IP"})
    assert status == "Invalid IP"


def test_parse_geojs_strings_and_missing():
    body = {"latitude": "51.4964", "longitude": "-0.1224", "country_code": "GB"}
    lat, lon, city, country, status = fs._parse_geojs(body)
    assert (lat, lon, country, status) == (51.4964, -0.1224, "GB", "success")
    *_, status = fs._parse_geojs({"message": "nope"})
    assert status == "no_location"


def test_parse_reallyfreegeoip_floats_and_missing():
    body = {"latitude": 51.4964, "longitude": -0.1224, "country_code": "GB", "city": ""}
    lat, lon, city, country, status = fs._parse_reallyfreegeoip(body)
    assert (lat, lon, country, status) == (51.4964, -0.1224, "GB", "success")
    assert city is None  # leere city -> None
    *_, status = fs._parse_reallyfreegeoip({})
    assert status == "no_location"


def test_observation_from_raw_success():
    raw = {"source": "ip_api", "ip": "1.2.3.4", "fetched_at_utc": "2026-06-03T00:00:00Z",
           "http_status": 200, "body": IP_API_OK}
    obs = fs.observation_from_raw(raw)
    assert obs == {"ip": "1.2.3.4", "source": "ip_api", "lat": 25.0657, "lon": 55.1713,
                   "city": "Dubai", "country": "AE", "status": "success",
                   "fetched_at_utc": "2026-06-03T00:00:00Z",
                   "lineage": "ipapi_com_unverified",
                   "accuracy_radius": None, "db_version": None,
                   "is_default_centroid": False, "centroid_match": ""}


def test_observation_from_raw_http_error():
    raw = {"source": "ipwho_is", "ip": "1.2.3.4", "fetched_at_utc": "2026-06-03T00:00:00Z",
           "http_status": None, "body": None, "error": "Timeout()"}
    obs = fs.observation_from_raw(raw)
    assert obs["status"] == "http_error" and obs["lat"] is None


def test_observations_csv_roundtrip(tmp_path):
    recs = [
        {"ip": "1.2.3.4", "source": "ip_api", "lat": 25.0, "lon": 55.0,
         "city": "Dubai", "country": "AE", "status": "success",
         "fetched_at_utc": "2026-06-03T00:00:00Z"},
        {"ip": "1.2.3.4", "source": "ipwho_is", "lat": 51.5, "lon": -0.1,
         "city": "London", "country": "GB", "status": "success",
         "fetched_at_utc": "2026-06-03T00:00:00Z"},
    ]
    path = store.save_observations_csv(recs, tmp_path / "obs.csv")
    loaded = store.load_observations_csv(path)
    assert len(loaded) == 2
    assert loaded[0]["source"] == "ip_api" and loaded[0]["city"] == "Dubai"
    # CSV ist textbasiert -> Werte kommen als Strings zurück
    assert loaded[1]["lat"] == "51.5"


def test_unknown_source_rejected():
    import pytest
    with pytest.raises(KeyError):
        fs.query_source("does_not_exist", "1.2.3.4")


def test_is_cached_only_with_body():
    # transienter Fehler (body None) gilt NICHT als gecacht -> wird erneut versucht
    assert fs._is_cached({"body": None, "error": "Timeout()"}) is False
    assert fs._is_cached(None) is False
    # Body vorhanden (auch inhaltliche Absage) gilt als gültiges, gecachtes Ergebnis
    assert fs._is_cached({"body": {"status": "fail"}}) is True
    assert fs._is_cached({"body": IP_API_OK}) is True
