"""Tests for the country vote (geoloc.country_vote) and the batch tool (network-free):
collect() is mocked, so no source is queried."""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "maltego"))

import geoloc    # noqa: E402
import batch     # noqa: E402
import registry  # noqa: E402
import pytest    # noqa: E402


@pytest.fixture(autouse=True)
def _offline_registry(monkeypatch):
    """Keep the batch tests network-free: no delegation download, no RDAP; the registry
    context returns a fixed state (DE today; US as of the record date for 192.0.2.2)."""
    def ctx(ip, as_of=None, **kw):
        c = {"as_of": as_of.isoformat() if as_of else None,
             "record_age_days": (max(0, (registry.dt.datetime.now(registry.dt.timezone.utc) - as_of).days)
                                 if as_of else None),
             "registry_rir": "ripencc", "registry_status_asof": "allocated",
             "registry_snapshot_asof": "20240301" if as_of else None,
             "registry_country_asof": ("US" if (as_of and ip == "192.0.2.2") else "DE") if as_of else "",
             "registry_snapshot_today": "20260901", "registry_country_today": "DE",
             "registry_redelegated": (bool(as_of) and ip == "192.0.2.2") if as_of else None,
             "rdap_country": "DE" if kw.get("rdap") else "", "rdap_handle": "H" if kw.get("rdap") else "",
             "rdap_type": "", "rdap_error": None}
        return c
    monkeypatch.setattr(registry, "registry_context", ctx)


def _obs(source, lineage, lat, lon, country, radius=None):
    return {"source": source, "lineage": lineage, "lat": lat, "lon": lon,
            "country": country, "city": None, "accuracy_radius": radius,
            "status": "success"}


# four lines agreeing on Berlin (DE), one outlier in Paris (FR)
AGREE = [
    _obs("ip_api", "ip_api", 52.52, 13.40, "DE"),
    _obs("ipinfo", "ipinfo", 52.51, 13.41, "DE"),
    _obs("maxmind_geolite2", "maxmind", 52.50, 13.39, "DE", radius=20),
    _obs("geojs", "maxmind", 52.50, 13.39, "DE"),           # same line as MaxMind
    _obs("dbip_lite", "dbip_lite", 48.86, 2.35, "FR"),
]
# every source in a different place -> empty core, nearest-source fallback
SCATTER = [
    _obs("ip_api", "ip_api", 52.52, 13.40, "DE"),
    _obs("ipinfo", "ipinfo", 48.86, 2.35, "FR"),
    _obs("maxmind_geolite2", "maxmind", 41.90, 12.50, "IT", radius=100),
]


def _mock(obs):
    def collect(ip, sources=None):
        return [dict(o) for o in obs], [], {"mobile_flag": False}
    return collect


def test_country_vote_core_majority(monkeypatch):
    monkeypatch.setattr(geoloc, "collect", _mock(AGREE))
    r = geoloc.locate("192.0.2.1")
    assert r["country"] == "DE" and r["country_fallback"] is False
    # three of four lines (ip_api, ipinfo, maxmind-line) are in the core -> mass 0.75
    assert abs(r["country_mass"] - 0.75) < 1e-9
    assert r["s_reliable"] and abs(r["s"] - 0.75) < 1e-9


def test_country_vote_fallback_nearest(monkeypatch):
    monkeypatch.setattr(geoloc, "collect", _mock(SCATTER))
    r = geoloc.locate("192.0.2.2")
    assert r["country_fallback"] is True and r["country_mass"] == 0.0
    assert r["country"] in {"DE", "FR", "IT"}          # nearest labelled source
    assert r["country"] == min(r["observations"],
                               key=lambda o: o["dist_to_estimate_km"])["country"]


def test_country_vote_no_labels():
    obs = [dict(o, country=None) for o in AGREE]
    est = geoloc.robust_estimate(obs)
    assert geoloc.country_vote(obs, est) == (None, 0.0, True)


def test_batch_csv_resume_and_provenance(tmp_path, monkeypatch):
    calls = []

    def collect(ip, sources=None):
        calls.append(ip)
        return [dict(o) for o in AGREE], [("ipwho_is", "timeout")], {"mobile_flag": None}
    monkeypatch.setattr(geoloc, "collect", collect)

    inp = tmp_path / "ips.txt"
    inp.write_text("# list\n192.0.2.1\n192.0.2.2  # dup below\n192.0.2.2\n", encoding="utf-8")
    out, prov = tmp_path / "out.csv", tmp_path / "out.jsonl"
    assert batch.main([str(inp), str(out), "--provenance", str(prov)]) == 0
    rows = list(csv.DictReader(open(out, encoding="utf-8", newline="")))
    assert [r["ip"] for r in rows] == ["192.0.2.1", "192.0.2.2"]      # duplicate once
    assert rows[0]["country"] == "DE" and rows[0]["country_mass"] == "0.750"
    assert rows[0]["s_reliable"] == "True" and rows[0]["s"] == "0.750"
    assert rows[0]["failed"] == "ipwho_is:timeout"
    assert "mobile" in rows[0]["warnings"].lower()      # flag None -> default-branch warning
    assert rows[0]["registry_country"] == "DE" and rows[0]["registry_country_today"] == "DE"
    assert rows[0]["timestamp"] == "" and rows[0]["record_age_days"] == ""
    assert "contradicts" not in rows[0]["warnings"]     # registry DE == vote DE
    assert len(prov.read_text(encoding="utf-8").splitlines()) == 2
    assert list(rows[0].keys()) == batch.FIELDS

    # resume: a third IP is appended, only it is queried
    inp.write_text("192.0.2.1\n192.0.2.2\n192.0.2.3\n", encoding="utf-8")
    calls.clear()
    assert batch.main([str(inp), str(out)]) == 0
    assert calls == ["192.0.2.3"]
    rows = list(csv.DictReader(open(out, encoding="utf-8", newline="")))
    assert [r["ip"] for r in rows] == ["192.0.2.1", "192.0.2.2", "192.0.2.3"]

    # --no-resume re-queries everything and appends
    calls.clear()
    assert batch.main([str(inp), str(out), "--no-resume"]) == 0
    assert calls == ["192.0.2.1", "192.0.2.2", "192.0.2.3"]


def test_batch_reads_csv_with_ip_column(tmp_path):
    inp = tmp_path / "in.csv"
    inp.write_text("case,IP,note\n1,192.0.2.9,x\n2,192.0.2.9,y\n3, 192.0.2.10 ,z\n",
                   encoding="utf-8")
    assert batch.read_ips(inp) == ["192.0.2.9", "192.0.2.10"]


def test_batch_timestamps_stale_and_redelegated(tmp_path, monkeypatch):
    calls = []

    def collect(ip, sources=None):
        calls.append(ip)
        return [dict(o) for o in AGREE], [], {"mobile_flag": False}
    monkeypatch.setattr(geoloc, "collect", collect)

    inp = tmp_path / "logins.csv"
    inp.write_text("user,ip,login_time\n"
                   "a,192.0.2.1,2024-03-15T10:22:00Z\n"
                   "b,192.0.2.2,2024-03-20 08:00:00\n"
                   "c,192.0.2.2,2024-03-20 08:00:00\n"     # exact duplicate (ip, ts)
                   "d,192.0.2.2,\n",                        # no timestamp -> today
                   encoding="utf-8")
    out = tmp_path / "out.csv"
    assert batch.main([str(inp), str(out), "--stale-days", "30"]) == 0
    rows = list(csv.DictReader(open(out, encoding="utf-8", newline="")))
    assert [(r["ip"], r["timestamp"][:10]) for r in rows] == \
        [("192.0.2.1", "2024-03-15"), ("192.0.2.2", "2024-03-20"), ("192.0.2.2", "")]
    assert calls == ["192.0.2.1", "192.0.2.2"]            # one live lookup per IP
    r1, r2, r3 = rows
    assert r1["registry_country"] == "DE" and r1["registry_snapshot"] == "20240301"
    assert int(r1["record_age_days"]) > 365 and "days old" in r1["warnings"]
    assert "re-delegated" not in r1["warnings"]
    # 192.0.2.2 US as of the record date, DE today -> re-delegation; today's registry country == vote -> no contradiction
    assert r2["registry_country"] == "US" and r2["registry_country_today"] == "DE"
    assert "re-delegated" in r2["warnings"] and "contradicts" not in r2["warnings"]
    assert r3["registry_country"] == "DE" and r3["record_age_days"] == "" and "days old" not in r3["warnings"]

    # --as-of fills in missing timestamps; resume skips known (ip, ts) pairs
    calls.clear()
    assert batch.main([str(inp), str(out), "--as-of", "2024-03-20T08:00:00"]) == 0
    rows = list(csv.DictReader(open(out, encoding="utf-8", newline="")))
    assert len(rows) == 3                                   # (192.0.2.2, 2024-03-20) already existed
    assert calls == []


def test_read_records_text_with_timestamps(tmp_path):
    inp = tmp_path / "in.txt"
    inp.write_text("192.0.2.9 2024-03-15\n192.0.2.9,1710498120\n192.0.2.10  # no ts\n", encoding="utf-8")
    recs = batch.read_records(inp)
    assert [ip for ip, _ in recs] == ["192.0.2.9", "192.0.2.9", "192.0.2.10"]
    assert recs[0][1].date().isoformat() == "2024-03-15" and recs[1][1].date().isoformat() == "2024-03-15"
    assert recs[2][1] is None
    assert batch.read_ips(inp) == ["192.0.2.9", "192.0.2.10"]


def test_batch_archive_mode(tmp_path, monkeypatch):
    """Archive mode: dated records from the DB builds of their month (readers mocked),
    without a live lookup; missing month -> live + note; tolerance +-1 month."""
    live_calls = []

    def collect(ip, sources=None):
        live_calls.append(ip)
        return [dict(o) for o in AGREE], [], {"mobile_flag": False}
    monkeypatch.setattr(geoloc, "collect", collect)

    archive = tmp_path / "geoip-archive"
    (archive / "2024-03").mkdir(parents=True)
    (archive / "2023-01").mkdir()
    (archive / "notes").mkdir()                       # not a month folder, is ignored

    def fake_readers(month_dir):
        # 2024-03: all three DBs say Paris; 2023-01: only two DBs present
        paris = {"latitude": 48.86, "longitude": 2.35, "city": "Paris", "country": "FR",
                 "accuracy_radius": 20, "db_file": "x.mmdb", "found": True}
        rd = {"maxmind_geolite2": lambda ip: dict(paris),
              "dbip_lite": lambda ip: dict(paris, longitude=2.36),
              "ip2location_lite": lambda ip: dict(paris, latitude=48.85, accuracy_radius=None)}
        if Path(month_dir).name == "2023-01":
            rd.pop("ip2location_lite")
        return rd
    monkeypatch.setattr(geoloc, "archive_readers", fake_readers)

    assert geoloc.resolve_archive_month(archive, registry.parse_timestamp("2024-03-15"))[1:] == ("2024-03", 0)
    assert geoloc.resolve_archive_month(archive, registry.parse_timestamp("2024-04-02"))[1:] == ("2024-03", 1)
    assert geoloc.resolve_archive_month(archive, registry.parse_timestamp("2024-07-01"))[0] is None

    inp = tmp_path / "old.csv"
    inp.write_text("ip,timestamp\n"
                   "192.0.2.1,2024-03-15\n"          # exact month
                   "192.0.2.1,2024-03-20\n"          # same month -> archive result reused
                   "192.0.2.2,2024-04-02\n"          # neighbouring month (tolerance)
                   "192.0.2.3,2024-07-01\n"          # no archive month -> live
                   "192.0.2.4,2023-01-10\n"          # only two lines -> S withheld
                   "192.0.2.5,\n",                   # no timestamp -> live
                   encoding="utf-8")
    out = tmp_path / "out.csv"
    assert batch.main([str(inp), str(out), "--db-archive", str(archive)]) == 0
    rows = {(r["ip"], r["timestamp"][:10]): r for r in csv.DictReader(open(out, encoding="utf-8", newline=""))}
    r1 = rows[("192.0.2.1", "2024-03-15")]
    assert r1["basis"] == "archive:2024-03" and r1["country"] == "FR" and r1["s_reliable"] == "True"
    assert "Archive mode" in r1["warnings"] and "days old" not in r1["warnings"]   # the estimate IS as of the record date
    assert rows[("192.0.2.2", "2024-04-02")]["basis"] == "archive:2024-03"
    assert "instead of 2024-04" in rows[("192.0.2.2", "2024-04-02")]["warnings"]
    r3 = rows[("192.0.2.3", "2024-07-01")]
    assert r3["basis"] == "live" and r3["country"] == "DE" and "no archive month" in r3["warnings"]
    assert "days old" in r3["warnings"]
    r4 = rows[("192.0.2.4", "2023-01-10")]
    assert r4["basis"] == "archive:2023-01" and r4["s_reliable"] == "False" and r4["n_lines"] == "2"
    assert rows[("192.0.2.5", "")]["basis"] == "live"
    assert live_calls == ["192.0.2.3", "192.0.2.5"]                # only the live cases


def test_batch_registry_flag_columns(tmp_path, monkeypatch):
    """Registry-flag columns: level delegation/whois, frozen cell rate, warning."""
    paris = [_obs("ip_api", "ip_api", 48.86, 2.35, "FR"), _obs("ipinfo", "ipinfo", 48.85, 2.34, "FR"),
             _obs("maxmind_geolite2", "maxmind", 48.86, 2.36, "FR", radius=20)]
    monkeypatch.setattr(geoloc, "collect", _mock(paris))
    inp = tmp_path / "ips.txt"
    inp.write_text("192.0.2.9\n", encoding="utf-8")
    out = tmp_path / "out.csv"
    assert batch.main([str(inp), str(out)]) == 0                       # registry DE (fixture) vs. vote FR
    r = list(csv.DictReader(open(out, encoding="utf-8", newline="")))[0]
    assert r["country"] == "FR" and r["registry_flag"] == "True" and r["registry_level"] == "delegation"
    assert r["p_country_miss"] == f"{registry.country_miss_rate(True, False, 'delegation')['p_country_miss']:.4f}"
    assert "contradicts" in r["warnings"] and "RIR delegation" in r["warnings"]
    out2 = tmp_path / "out2.csv"
    assert batch.main([str(inp), str(out2), "--rdap"]) == 0            # whois DE -> primary level whois
    r = list(csv.DictReader(open(out2, encoding="utf-8", newline="")))[0]
    assert r["registry_flag"] == "True" and r["registry_level"] == "whois" and r["p_country_miss"] == "0.0752"
    # without registry: columns empty
    out3 = tmp_path / "out3.csv"
    assert batch.main([str(inp), str(out3), "--no-registry"]) == 0
    r = list(csv.DictReader(open(out3, encoding="utf-8", newline="")))[0]
    assert r["registry_flag"] == "" and r["registry_level"] == "" and r["p_country_miss"] == ""
    # agreement: flag False, small rate
    monkeypatch.setattr(geoloc, "collect", _mock(AGREE))
    out4 = tmp_path / "out4.csv"
    assert batch.main([str(inp), str(out4)]) == 0
    r = list(csv.DictReader(open(out4, encoding="utf-8", newline="")))[0]
    assert r["registry_flag"] == "False" and r["p_country_miss"] == "0.0047" and "contradicts" not in r["warnings"]
