"""Tests for maltego/registry.py (network-free): synthetic delegation files in
tmp_path, no download, no RDAP. Checks interval lookup, state normalisation,
snapshot-date bucketing, context (today vs. record date, re-delegation) and warnings."""

import datetime as dt
import gzip
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "maltego"))

import registry  # noqa: E402

# RFC 5737 test networks as blocks: 192.0.2.0/24 (DE, ripencc), 198.51.100.0/24 (US, arin),
# 203.0.113.0/25 (AU, apnic) and 203.0.113.128/25 (EU, apnic); 2001:db8 not (IPv4 only).
# Second snapshot: 198.51.100.0/24 -> CA, 203.0.113.0/25 reserved (country field empty).
SNAP_A = {
    "ripencc": ["2|ripencc|20240301|1|19700101|20240301|+0100",
                "ripencc|*|ipv4|*|1|summary",
                "ripencc|DE|ipv4|192.0.2.0|256|20100101|allocated|abc"],
    "arin":    ["2.3|arin|20240301|1|19700101|20240301|-0400",
                "arin|US|ipv4|198.51.100.0|256|20050101|assigned|def"],
    "apnic":   ["#comment", "2|apnic|20240301|1|19700101|20240301|+1000",
                "apnic|AU|ipv4|203.0.113.0|128|20080101|allocated|ghi",
                "apnic|EU|ipv4|203.0.113.128|128|20080101|allocated|jkl"],
    "lacnic":  ["2.3|lacnic|20240301|0|19870101|20240301|-0300"],
    "afrinic": ["2|afrinic|20240301|0|00000000|20240301|00000"],
}
SNAP_B = {
    "ripencc": SNAP_A["ripencc"],
    "arin":    ["2.3|arin|20260901|1|19700101|20260901|-0400",
                "arin|CA|ipv4|198.51.100.0|256|20250601|allocated|xyz"],
    "apnic":   ["2|apnic|20260901|1|19700101|20260901|+1000",
                "apnic||ipv4|203.0.113.0|128||reserved",
                "apnic|EU|ipv4|203.0.113.128|128|20080101|allocated|jkl"],
    "lacnic":  SNAP_A["lacnic"],
    "afrinic": SNAP_A["afrinic"],
}


def _write(base: Path, date: str, snap: dict) -> None:
    for rir, lines in snap.items():
        p = registry.snapshot_path(date, rir, base)
        p.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(p, "wt", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")


@pytest.fixture
def snapshots(tmp_path, monkeypatch):
    base = tmp_path / "rir"
    _write(base, "20240301", SNAP_A)
    today = registry.today_str()
    _write(base, today, SNAP_B)
    registry._DELEG_CACHE.clear()
    monkeypatch.setattr(registry, "fetch_snapshot", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no download in tests")))
    return base


def test_lookup_and_state(snapshots):
    D = registry.Delegations("20240301", snapshots)
    assert D.lookup("192.0.2.77") == ("DE", "ripencc", "allocated")
    assert D.lookup("198.51.100.1") == ("US", "arin", "assigned")
    assert D.lookup("203.0.113.200")[0] == "EU" and registry.state_of("EU") == ""
    assert D.lookup("10.0.0.1") == ("", "", "none")          # not covered
    assert D.lookup("not-an-ip") == ("", "", "none")
    assert D.headers["ripencc"].startswith("2|ripencc|20240301")


def test_snapshot_bucketing_and_timestamps():
    ts = registry.parse_timestamp("2024-03-15T10:22:00Z")
    assert registry.snapshot_date_for(ts) == "20240301"
    assert registry.snapshot_date_for(ts, "day") == "20240315"
    assert registry.parse_timestamp("1710498120").date() == dt.date(2024, 3, 15)     # epoch s
    assert registry.parse_timestamp("1710498120000").date() == dt.date(2024, 3, 15)  # epoch ms
    assert registry.parse_timestamp("15.03.2024").date() == dt.date(2024, 3, 15)
    assert registry.parse_timestamp("2024-03-15 10:22:00").tzinfo is not None
    assert registry.parse_timestamp("") is None and registry.parse_timestamp("yesterday") is None


def test_context_today_only(snapshots):
    ctx = registry.registry_context("198.51.100.1", None, base=snapshots, fetch=False)
    assert ctx["registry_country_today"] == "CA" and ctx["registry_rir"] == "arin"
    assert ctx["registry_snapshot_asof"] is None and ctx["record_age_days"] is None
    assert ctx["registry_redelegated"] is None
    assert registry.registry_warnings(ctx, "CA") == []
    w = registry.registry_warnings(ctx, "DE")
    assert len(w) == 1 and "contradicts" in w[0]


def test_context_as_of_redelegated_and_stale(snapshots):
    as_of = registry.parse_timestamp("2024-03-15")
    ctx = registry.registry_context("198.51.100.1", as_of, base=snapshots, fetch=False)
    assert ctx["registry_snapshot_asof"] == "20240301" and ctx["registry_country_asof"] == "US"
    assert ctx["registry_country_today"] == "CA" and ctx["registry_redelegated"] is True
    assert ctx["record_age_days"] > 365
    w = registry.registry_warnings(ctx, "CA")
    assert any("days old" in x for x in w) and any("re-delegated" in x for x in w)
    # vote == today's registry country -> no contradiction warning (the vote describes today)
    assert not any("contradicts" in x for x in w)
    assert any("contradicts" in x for x in registry.registry_warnings(ctx, "US"))
    # unchanged block: no re-delegation, but an age warning
    ctx2 = registry.registry_context("192.0.2.5", as_of, base=snapshots, fetch=False)
    assert ctx2["registry_redelegated"] is False
    assert [x for x in registry.registry_warnings(ctx2, "DE") if "re-delegated" in x] == []
    assert registry.registry_warnings(ctx2, "DE", stale_days=10_000) == []


def test_context_reserved_today(snapshots):
    ctx = registry.registry_context("203.0.113.4", None, base=snapshots, fetch=False)
    assert ctx["registry_country_today"] == "" and ctx["registry_status_asof"] == "reserved"
    assert any("no state" in x for x in registry.registry_warnings(ctx, "AU"))


def test_rdap_country_extraction():
    assert registry.rdap_country_from_body({"country": "de"}) == ("DE", "network.country")
    arin = {"entities": [{"roles": ["registrant"], "vcardArray": ["vcard", [
        ["adr", {"label": "1 Main St\nSomewhere\nCA\n90210\nUnited States"}, "text", ["", "", "", "", "", "", ""]]]]}]}
    assert registry.rdap_country_from_body(arin) == ("US", "registrant.adr.name")
    lacnic = {"entities": [{"roles": ["registrant"], "vcardArray": ["vcard", [
        ["adr", {"label": "Montevideo\nUY"}, "text", ["", "", "", "Montevideo", "", "11400", "UY"]]]]}]}
    assert registry.rdap_country_from_body(lacnic) == ("UY", "registrant.adr.cc")
    assert registry.rdap_country_from_body({"entities": []}) == ("", "none")


# ---------------------------------------------------------------------------
# Frozen rates file registry_rates.json (make_registry_rates.py)
# ---------------------------------------------------------------------------
CELLS = ("flag_lowmass", "flag_highmass", "noflag_lowmass", "noflag_highmass")


def test_registry_rates_structure():
    R = registry.registry_rates()
    assert R is not None and R["population"] == "probes_pool" and R["coding"] == "state"
    assert R["s_gate"] == 0.5 and R["r_core_km"] == 50.0
    assert R["level_priority"] == ["whois", "delegation"]
    for level in ("whois", "delegation"):
        b = R["levels"][level]
        assert sum(b["cells"][c]["n"] for c in CELLS) == b["n"] == 8896
        for c in CELLS:
            z = b["cells"][c]
            assert 0.0 <= z["p_country_miss"] <= 1.0
            assert abs(z["p_country_miss"] * 100 - z["miss_pct"]) < 1e-6
        # the flag raises the rate in both gate strata; the gate raises it in both flag strata
        for g in ("lowmass", "highmass"):
            assert b["cells"][f"flag_{g}"]["p_country_miss"] > b["cells"][f"noflag_{g}"]["p_country_miss"]
        for f in ("flag", "noflag"):
            assert b["cells"][f"{f}_lowmass"]["p_country_miss"] > b["cells"][f"{f}_highmass"]["p_country_miss"]
        # lift row consistent with the cells (weighted means) and CI excludes 1
        fl, fh = b["cells"]["flag_lowmass"], b["cells"]["flag_highmass"]
        p_flag = (fl["n"] * fl["p_country_miss"] + fh["n"] * fh["p_country_miss"]) / (fl["n"] + fh["n"])
        assert abs(100 * p_flag - b["lift"]["p_miss_flag_pct"]) < 0.01
        assert b["lift"]["rr_ci95_iid"][0] > 1 and b["lift"]["rr_ci95_asn"][0] > 1
        # floor + deviating misses = all aggregation misses
        s = b["floor"]
        assert s["registry_following_n"] + s["dev_registry_wrong_n"] + s["dev_registry_right_n"] == s["agg_miss_n"]
    assert R["_meta"]["contrast_anchors"]["whois"]["rr_ci95_iid"][0] < 1   # anchors: flag n.s.


def test_country_miss_rate_lookup(monkeypatch):
    r = registry.country_miss_rate(True, False)
    assert r["level"] == "whois" and r["cell"] == "flag_highmass" and r["n"] == 399
    assert abs(r["p_country_miss"] - 0.07519) < 1e-9
    assert registry.country_miss_rate(False, False)["p_country_miss"] < 0.01
    assert registry.country_miss_rate(True, True, "delegation")["cell"] == "flag_lowmass"
    assert registry.country_miss_rate(False, True, "none") is None
    monkeypatch.setattr(registry, "_RATES", None)
    assert registry.country_miss_rate(True, True) is None and registry.registry_rates() is None


def test_registry_flag_levels_and_rates():
    base = {"registry_snapshot_today": "20260901", "registry_country_today": "DE", "rdap_country": ""}
    # delegation level (no RDAP): vote DE == registry -> no flag, cell noflag_highmass
    f = registry.registry_flag(base, "DE", 0.9)
    assert f["registry_flag"] is False and f["registry_level"] == "delegation"
    assert f["p_country_miss_cell"] == "noflag_highmass" and f["p_country_miss"] < 0.01
    # vote FR != DE -> flag; mass missing/fallback -> low cell
    f = registry.registry_flag(base, "FR", None)
    assert f["registry_flag"] is True and f["p_country_miss_cell"] == "flag_lowmass"
    assert f["p_country_miss"] == registry.country_miss_rate(True, True, "delegation")["p_country_miss"]
    # whois object present -> primary level whois, even if the delegation contradicts the vote
    f = registry.registry_flag({**base, "rdap_country": "FR"}, "FR", 0.8)
    assert f["registry_flag"] is False and f["registry_level"] == "whois" and f["registry_flag_country"] == "FR"
    f = registry.registry_flag({**base, "rdap_country": "US"}, "FR", 0.8)
    assert f["registry_flag"] is True and f["registry_level"] == "whois"
    assert f["p_country_miss_cell"] == "flag_highmass" and f["p_country_miss_n"] == 399
    # without a vote or without a registry country: None
    assert registry.registry_flag(base, None, 0.9)["registry_flag"] is None
    assert registry.registry_flag({"registry_country_today": "", "rdap_country": ""}, "DE", 0.9)["registry_flag"] is None
    # warnings: the flag warning carries the frozen rate; whois==vote, delegation deviating -> note without flag
    w = registry.registry_warnings({**base, "rdap_country": "US"}, "FR", country_mass=0.8)
    assert len(w) == 1 and "contradicts" in w[0] and "whois object" in w[0] and "8.1 %" in w[0] and "n=399" in w[0]
    assert "RIR delegation names DE" in w[0]
    w = registry.registry_warnings({**base, "rdap_country": "FR"}, "FR", country_mass=0.8)
    assert len(w) == 1 and "contradicts" not in w[0] and "no registry flag" in w[0]
    assert registry.registry_warnings({**base, "rdap_country": "DE"}, "DE", country_mass=0.8) == []
