"""Collect, per IP, a small heterogeneous value series of location estimates.

Sources: six free geo web APIs (ipinfo needs a free token via IPINFO_TOKEN)
plus three local LITE databases (GeoLite2/DB-IP/IP2Location; free licenses,
download required). The sources contradict each other for the same IP,
sometimes by thousands of kilometers — exactly the starting situation the
project investigates. The source registry ``SOURCES`` can be extended with
further adapters (mind the lineage assignment, see maltego/check_lineage.py).

Reproducibility + provenance (lightweight):
  data/cache/sources_raw.jsonl     append-only cache (working copy): 1 line per
                                   (source, IP); prevents duplicate API calls.
  data/cache/raw/sources_<id>.json verbatim, IMMUTABLE evidence artifact per
                                   run (the responses fetched in that run) —
                                   SHA-256-attested in the ledger (does not
                                   grow afterwards).
  data/cache/observations.csv      normalized working copy (OBSERVATION_COLUMNS),
                                   cumulative view of the entire cache.
  data/provenance.jsonl            1 audit ledger entry per run, into the hash chain.

Notes: only public infrastructure IPs (RIPE Atlas anchors). Free tiers have
rate limits (throttled per source) and partly prohibit commercial use —
academic evaluation is covered. ip-api.com offers only HTTP in the free tier.

Usage:
  python data/fetch_sources.py                 # default: first 50 anchors, all 9 sources
  python data/fetch_sources.py --limit 100     # more IPs
  python data/fetch_sources.py --all           # all anchors (caution: rate limits!)
  python data/fetch_sources.py --sources ip_api,ipwho_is
  python data/fetch_sources.py --refresh       # ignore the cache, fetch anew
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import store  # noqa: E402
from data import centroids  # noqa: E402

TIMEOUT = 15
USER_AGENT = "robust-ipgeo/research (academic; contact via repo)"
SOURCES_RAW = store.SOURCES_RAW_CSV   # dataset-aware (anchors -> sources_raw.jsonl)
DB_DIR = store.BASE_DIR / "db"

# Tokens for authenticated web sources (from .env; empty => source reports an error).
IPINFO_TOKEN = store.load_env().get("IPINFO_TOKEN", "")
# ipwhois.io pro key: switches the ipwho_is source to ipwhois.pro (SAME data basis,
# verified on 2026-08-21 as coordinate-identical to the free ipwho.is on 8/8
# sampled IPs; just a higher quota instead of ~1k/day per client IP). Line unchanged.
IPWHOISIO_KEY = store.load_env(("IPWHOISIO",)).get("IPWHOISIO", "")


def _redact(url: str) -> str:
    """Remove tokens from a URL before it ends up in the cache."""
    for tok in (IPINFO_TOKEN, IPWHOISIO_KEY):
        if tok and tok in url:
            url = url.replace(tok, "***")
    return url


# ---------- Source adapters ----------
# Each parser takes the raw JSON body and returns
# (lat, lon, city, country, status); status == "success" or an error identifier.

def _parse_ip_api(body: dict):
    if body.get("status") != "success":
        return None, None, None, None, body.get("message") or "fail"
    return body.get("lat"), body.get("lon"), body.get("city"), body.get("countryCode"), "success"


def _parse_ipapi_co(body: dict):
    if body.get("error"):
        return None, None, None, None, body.get("reason") or "error"
    return (body.get("latitude"), body.get("longitude"),
            body.get("city"), body.get("country"), "success")


def _parse_ipwho_is(body: dict):
    if not body.get("success", False):
        return None, None, None, None, str(body.get("message") or "fail")
    return (body.get("latitude"), body.get("longitude"),
            body.get("city"), body.get("country_code"), "success")


def _to_float_or_none(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _parse_geojs(body: dict):
    # geojs delivers lat/lon as strings; no reliable city field
    lat, lon = _to_float_or_none(body.get("latitude")), _to_float_or_none(body.get("longitude"))
    if lat is None or lon is None:
        return None, None, None, None, "no_location"
    return lat, lon, body.get("city"), body.get("country_code"), "success"


def _parse_reallyfreegeoip(body: dict):
    lat, lon = _to_float_or_none(body.get("latitude")), _to_float_or_none(body.get("longitude"))
    if lat is None or lon is None:
        return None, None, None, None, "no_location"
    return lat, lon, (body.get("city") or None), body.get("country_code"), "success"


def _parse_ipinfo(body: dict):
    # ipinfo delivers the coordinates bundled as loc="lat,lon"
    if body.get("error"):
        err = body["error"]
        return None, None, None, None, (err.get("title") if isinstance(err, dict) else str(err))
    loc = body.get("loc")
    if not loc or "," not in loc:
        return None, None, None, None, "no_location"
    try:
        lat, lon = (float(x) for x in loc.split(",", 1))
    except ValueError:
        return None, None, None, None, "bad_loc"
    return lat, lon, body.get("city"), body.get("country"), "success"


# ---------- Offline readers for the local LITE DBs ----------
# Their own lines, independent of the web APIs. The DB files live in data/db/
# and are already in the provenance chain as evidence artifacts
# (data/fetch_geodbs.py); here they are only read. The reader result is the
# source's "response" (``body``) — analogous to the JSON body of an HTTP source,
# so that cache/normalization apply unchanged. Readers are opened lazily and
# reused.

_db_readers: dict[str, object] = {}


def _resolve_db(pattern: str) -> Path:
    """Resolve a DB in data/db/; '*' picks the newest match (e.g. DB-IP month)."""
    if "*" in pattern:
        matches = sorted(DB_DIR.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"no DB matching {pattern!r} in {DB_DIR}")
        return matches[-1]
    return DB_DIR / pattern


def _read_mmdb(pattern: str):
    """Returns a reader callable for a MaxMind/DB-IP .mmdb (ip -> body)."""
    def read(ip: str) -> dict:
        import geoip2.database
        import geoip2.errors
        path = _resolve_db(pattern)
        reader = _db_readers.get(str(path))
        if reader is None:
            reader = geoip2.database.Reader(str(path))
            _db_readers[str(path)] = reader
        version = reader.metadata().build_epoch
        try:
            c = reader.city(ip)
        except geoip2.errors.AddressNotFoundError:
            return {"latitude": None, "longitude": None, "city": None, "country": None,
                    "accuracy_radius": None, "db_file": path.name, "db_version": version,
                    "found": False}
        return {"latitude": c.location.latitude, "longitude": c.location.longitude,
                "city": c.city.name, "country": c.country.iso_code,
                "accuracy_radius": c.location.accuracy_radius,   # MaxMind confidence radius (km)
                "db_file": path.name, "db_version": version,
                "found": c.location.latitude is not None}
    return read


def _read_ip2location(pattern: str):
    """Returns a reader callable for the IP2Location .BIN (ip -> body)."""
    def read(ip: str) -> dict:
        import IP2Location
        path = _resolve_db(pattern)
        reader = _db_readers.get(str(path))
        if reader is None:
            reader = IP2Location.IP2Location(str(path))
            _db_readers[str(path)] = reader
        version = int(path.stat().st_mtime)   # IP2Location BIN: no build field -> file mtime
        rec = reader.get_all(ip)
        lat = _to_float_or_none(getattr(rec, "latitude", None))
        lon = _to_float_or_none(getattr(rec, "longitude", None))
        country = getattr(rec, "country_short", None)
        if country in ("-", "??", None):       # IP2Location placeholder for unknown
            return {"latitude": None, "longitude": None, "city": None, "country": None,
                    "accuracy_radius": None, "db_file": path.name, "db_version": version,
                    "found": False}
        return {"latitude": lat, "longitude": lon, "city": getattr(rec, "city", None),
                "country": country, "accuracy_radius": None,
                "db_file": path.name, "db_version": version, "found": lat is not None}
    return read


def _parse_local(body: dict):
    """Body of an offline reader -> (lat, lon, city, country, status)."""
    if not body or not body.get("found"):
        return None, None, None, None, "not_found"
    return (body.get("latitude"), body.get("longitude"),
            body.get("city"), body.get("country"), "success")


# ``lineage`` = confirmed/assumed data provenance. Sources with the SAME
# lineage token are considered correlated (one effective line) and are later
# collapsed by weight in the aggregation, not deleted. "maxmind_geolite"
# is the only shared line -- with graded evidence quality: geojs
# DOCUMENTS its GeoLite provenance itself (+ bit-identical to GeoLite2);
# reallyfreegeoip names no primary source and is assigned only EMPIRICALLY
# (pairwise median 0.05 km, consistent with a diverging data snapshot of the
# same basis). The "*_unverified" tokens mark providers that do not disclose
# their provenance (kept separate as a precaution).
#
# Source kinds:
#   web API   -> "url" (+ "min_interval", throttled) + "parse" (JSON body)
#   local DB  -> "reader" (ip -> body, offline) + "parse" = _parse_local; "db" doc only
SOURCES = {
    # --- Web APIs ---
    "ip_api":          {"url": "http://ip-api.com/json/{ip}",              "min_interval": 1.5, "parse": _parse_ip_api,          "lineage": "ipapi_com_unverified"},
    "ipwho_is":        {"url": "https://ipwho.is/{ip}",                    "min_interval": 0.5, "parse": _parse_ipwho_is,        "lineage": "ipwhois_unverified"},
    "ipinfo":          {"url": "https://ipinfo.io/{ip}/json?token={token}","min_interval": 0.2, "parse": _parse_ipinfo,          "lineage": "ipinfo"},
    "geojs":           {"url": "https://get.geojs.io/v1/ip/geo/{ip}.json", "min_interval": 0.4, "parse": _parse_geojs,           "lineage": "maxmind_geolite"},
    "reallyfreegeoip": {"url": "https://reallyfreegeoip.org/json/{ip}",    "min_interval": 0.4, "parse": _parse_reallyfreegeoip, "lineage": "maxmind_geolite"},
    "ipapi_co":        {"url": "https://ipapi.co/{ip}/json/",              "min_interval": 1.2, "parse": _parse_ipapi_co,        "lineage": "ipapi_co_unverified"},
    # --- local LITE DBs (offline; their own independent lines) ---
    "maxmind_geolite2":{"db": "GeoLite2-City.mmdb",         "reader": _read_mmdb("GeoLite2-City.mmdb"),         "parse": _parse_local, "lineage": "maxmind_geolite"},
    "dbip_lite":       {"db": "dbip-city-lite-*.mmdb",      "reader": _read_mmdb("dbip-city-lite-*.mmdb"),      "parse": _parse_local, "lineage": "dbip_lite"},
    "ip2location_lite":{"db": "IP2LOCATION-LITE-DB5.BIN",   "reader": _read_ip2location("IP2LOCATION-LITE-DB5.BIN"), "parse": _parse_local, "lineage": "ip2location_lite"},
}
DEFAULT_SOURCES = list(SOURCES)

# With a pro key: same provider/same data via ipwhois.pro, higher quota.
if IPWHOISIO_KEY:
    SOURCES["ipwho_is"]["url"] = "https://ipwhois.pro/{ip}?key={ipwhois_key}"
    SOURCES["ipwho_is"]["min_interval"] = 0.25


# ---------- Rate limiting (per source) ----------

_last_call: dict[str, float] = {}


def _throttle(source: str) -> None:
    interval = SOURCES[source]["min_interval"]
    last = _last_call.get(source)
    if last is not None:
        wait = interval - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
    _last_call[source] = time.monotonic()


# ---------- Raw cache (also an evidence artifact) ----------

def load_raw_cache() -> dict[tuple[str, str], dict]:
    """Load {(source, ip) -> raw entry} from the append-only cache."""
    cache: dict[tuple[str, str], dict] = {}
    if SOURCES_RAW.exists():
        with open(SOURCES_RAW, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    e = json.loads(line)
                    cache[(e["source"], e["ip"])] = e   # later line wins
    return cache


def append_raw_cache(entry: dict) -> None:
    SOURCES_RAW.parent.mkdir(parents=True, exist_ok=True)
    with open(SOURCES_RAW, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _is_cached(entry: dict | None) -> bool:
    """Counts as usably cached when a response (body) is present.

    Transient errors (timeout/rate limit -> body is None) are NOT counted as
    cached, so a rerun retries them. A substantive source refusal (body
    present, status != success) is by contrast a valid result.
    """
    return entry is not None and entry.get("body") is not None


_db_version_cache: dict[str, int] = {}


def _current_db_version(source: str) -> int | None:
    """Current build/state of an offline source's local DB (for cache invalidation)."""
    if source in _db_version_cache:
        return _db_version_cache[source]
    path = _resolve_db(SOURCES[source]["db"])
    if path.suffix.upper() == ".BIN":
        v = int(path.stat().st_mtime)
    else:
        import geoip2.database
        reader = _db_readers.get(str(path)) or geoip2.database.Reader(str(path))
        _db_readers[str(path)] = reader
        v = reader.metadata().build_epoch
    _db_version_cache[source] = v
    return v


def _cache_valid(entry: dict | None, source: str) -> bool:
    """Like _is_cached, but offline hits go stale when the DB build has changed
    (forensically: a repeat run reproduces against the CURRENT, hash-pinned DB,
    not against a cache from an old DB state)."""
    if not _is_cached(entry):
        return False
    if "reader" in SOURCES.get(source, {}):
        return (entry.get("body") or {}).get("db_version") == _current_db_version(source)
    return True


# ---------- Retrieval ----------

def query_source(source: str, ip: str, session: requests.Session | None = None) -> dict:
    """Query one source for one IP -> raw entry (body + metadata).

    Does not raise; network/parse errors land as ``error`` in the entry
    (``body`` = None).
    """
    if source not in SOURCES:
        raise KeyError(f"unknown source: {source!r} (known: {', '.join(SOURCES)})")
    src = SOURCES[source]
    fetched_at = store.utc_now_iso()

    # Local DB: read offline (no network, no throttling). The evidence artifact
    # is the DB file itself (via hash in the provenance chain), not this body.
    if "reader" in src:
        try:
            body = src["reader"](ip)
            return {"source": source, "ip": ip, "fetched_at_utc": fetched_at,
                    "http_status": None, "url": f"local:{body.get('db_file', src.get('db'))}",
                    "body": body}
        except Exception as e:   # missing/broken DB file
            return {"source": source, "ip": ip, "fetched_at_utc": fetched_at,
                    "http_status": None, "url": f"local:{src.get('db')}", "body": None,
                    "error": repr(e)}

    # Web API: HTTP (the token, if present, is inserted into the URL)
    url = src["url"].format(ip=ip, token=IPINFO_TOKEN, ipwhois_key=IPWHOISIO_KEY)
    _throttle(source)
    get = (session or requests).get
    try:
        resp = get(url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
        return {"source": source, "ip": ip, "fetched_at_utc": fetched_at,
                "http_status": resp.status_code, "url": _redact(resp.url), "body": resp.json()}
    except Exception as e:  # network, timeout, no JSON (e.g. rate-limit HTML)
        return {"source": source, "ip": ip, "fetched_at_utc": fetched_at,
                "http_status": None, "url": _redact(url), "body": None, "error": repr(e)}


def observation_from_raw(entry: dict) -> dict:
    """Raw entry -> normalized observation (OBSERVATION_COLUMNS)."""
    source, ip = entry["source"], entry["ip"]
    body = entry.get("body")
    if body is None:
        status = "http_error" if entry.get("error") else "no_body"
        lat = lon = city = country = None
    else:
        lat, lon, city, country, status = SOURCES[source]["parse"](body)
    return {"ip": ip, "source": source, "lat": lat, "lon": lon,
            "city": city, "country": country, "status": status,
            "fetched_at_utc": entry.get("fetched_at_utc"),
            "lineage": SOURCES[source].get("lineage", "unknown"),
            # Quality/default fields: raw values here, default flag in the annotation
            "accuracy_radius": (body or {}).get("accuracy_radius"),
            "db_version": (body or {}).get("db_version"),
            "is_default_centroid": False, "centroid_match": ""}


def annotate_defaults(observations: list[dict], anchors: list[dict] | None = None) -> list[dict]:
    """Set the hub/centroid default flag per observation (dataset-dependent, hence
    as a post-pass over ALL observations). Sets ``is_default_centroid`` +
    ``centroid_match`` in-place (guard B). Needs the full source frequency + ASN per IP."""
    if anchors is None:
        anchors = store.load_anchors_csv()
    asn_of = {a["ip"]: a.get("asn") for a in anchors}
    records = [{"source": o["source"], "ip": o["ip"], "lat": o["lat"], "lon": o["lon"],
                "city": o["city"], "radius": o.get("accuracy_radius"), "asn": asn_of.get(o["ip"])}
               for o in observations if o["status"] == "success" and o["lat"] is not None]
    freq, _ = centroids.build_index(records)
    rec_by = {(r["source"], r["ip"]): r for r in records}
    for o in observations:
        r = rec_by.get((o["source"], o["ip"]))
        if r is None:
            continue
        o["is_default_centroid"], o["centroid_match"] = centroids.detect(r, freq)
    return observations


# ---------- CLI ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="Collect geo estimates per IP (free APIs, with provenance).")
    ap.add_argument("--limit", type=int, default=50, help="number of anchors (first N); default 50")
    ap.add_argument("--all", action="store_true", help="all anchors (caution: rate limits)")
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCES),
                    help=f"comma list; known: {', '.join(SOURCES)}")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and fetch anew")
    ap.add_argument("--reindex", action="store_true",
                    help="do not fetch: rebuild observations.csv and register the current "
                         "cache as an immutable evidence artifact in the ledger")
    args = ap.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in sources if s not in SOURCES]
    if unknown:
        ap.error(f"unknown source(s): {', '.join(unknown)}")

    anchors = store.load_anchors_csv()
    ips = [a["ip"] for a in anchors if a.get("ip")]
    if not args.all:
        ips = ips[: args.limit]
    ip_set = set(ips)

    # Guard against silent shrinking (review 2026-08-18): a limit run after
    # an --all run would truncate the cumulative observations.csv to the
    # partial IP set AND recompute the frequency hub flags (annotate_defaults)
    # only on the subpopulation. In that case abort instead of
    # overwriting; the raw data in the cache is unaffected.
    try:
        existing_ips = {o["ip"] for o in store.load_observations_csv()}
    except FileNotFoundError:
        existing_ips = set()
    lost = existing_ips - ip_set
    if lost:
        raise SystemExit(
            f"ABORT: observations.csv covers {len(existing_ips)} IPs; this run "
            f"would truncate to {len(ip_set)} IPs ({len(lost)} lost) and recompute the "
            "hub flags on the subpopulation. With --all (plus --reindex if needed) "
            "the cumulative view runs over all anchors.")

    fetch_id = store.new_fetch_id()
    fetched_at = store.utc_now_iso()
    cache = load_raw_cache()

    new_entries: list[dict] = []
    if not args.reindex:
        session = requests.Session()
        print(f"Collecting {len(sources)} source(s) × {len(ips)} IP(s) … (cache: {len(cache)} entries)")
        for i, ip in enumerate(ips, 1):
            for source in sources:
                if args.refresh or not _cache_valid(cache.get((source, ip)), source):
                    entry = query_source(source, ip, session=session)
                    append_raw_cache(entry)
                    cache[(source, ip)] = entry
                    new_entries.append(entry)
            if i % 25 == 0:
                print(f"  … {i}/{len(ips)} IPs ({len(new_entries)} new calls)")

    # Immutable evidence artifact of this run (its own file per fetch_id -> hash
    # stays stable, unlike the growing cache): with --reindex the entire cache
    # relevant to the IPs, otherwise the responses fetched in this run.
    snapshot = ([e for (s, ip), e in cache.items() if ip in ip_set] if args.reindex
                else new_entries)
    raw_bytes = ("\n".join(json.dumps(e, ensure_ascii=False) for e in snapshot)).encode("utf-8")
    raw_path, sha_raw = store.save_raw(fetch_id, raw_bytes, label="sources")

    # observations.csv = cumulative, normalized view of the entire cache.
    # CAUTION (--reindex): offline entries are NOT invalidated against the
    # current DB build here (_cache_valid only applies on the fetch path) —
    # but every row carries its true db_version stamp. After a DB update,
    # warn below instead of silently baking in stale builds.
    stale = sum(1 for (s, ip), e in cache.items()
                if ip in ip_set and "reader" in SOURCES.get(s, {})
                and not _cache_valid(e, s))
    if args.reindex and stale:
        print(f"WARNING: {stale} offline cache entries come from an OLDER "
              "DB build than the currently pinned one (db_version per row in "
              "observations.csv; run --refresh for a fresh state).")
    obs = [observation_from_raw(e) for (s, ip), e in cache.items() if ip in ip_set]
    obs.sort(key=lambda o: (o["ip"], o["source"]))
    annotate_defaults(obs, anchors)            # set hub/centroid default flags
    csv_path = store.save_observations_csv(obs)
    n_success = sum(1 for o in obs if o["status"] == "success")
    n_default = sum(1 for o in obs if o["is_default_centroid"])

    # DB evidence per run (which mmdb/BIN build + hash produced the offline responses)
    db_evidence = []
    for s in sources:
        if "reader" in SOURCES[s]:
            p = _resolve_db(SOURCES[s]["db"])
            if p.exists():
                db_evidence.append({"source": s, "file": str(p.relative_to(store.BASE_DIR)),
                                    "sha256": store.sha256_file(p), "db_version": _current_db_version(s)})

    entry = store.append_provenance({
        "fetch_id": fetch_id,
        "fetched_at_utc": fetched_at,
        "kind": "sources-reindex" if args.reindex else "sources",
        "sources": sources,
        "tool": "data/fetch_sources.py",
        "tool_git_commit": store.git_commit(),
        "n_ips": len(ips),
        "n_new_calls": len(new_entries),
        "n_snapshot": len(snapshot),
        "n_observations": len(obs),
        "n_success": n_success,
        "n_failed": len(obs) - n_success,
        "n_default_centroid": n_default,
        "dbs": db_evidence,                          # DB build + SHA-256 per offline source
        "raw_file": str(raw_path.relative_to(store.BASE_DIR)),
        "sha256_raw": sha_raw,                       # evidence artifact of this run (stable)
        "sha256_observations": store.sha256_json(obs),
    })

    mode = "reindex" if args.reindex else "fetch"
    print(f"\nRun {fetch_id} ({mode}): {len(obs)} observations "
          f"({n_success} ok, {len(obs) - n_success} failed, {n_default} hub defaults), "
          f"{len(new_entries)} new calls.")
    print(f"  Evidence     : {raw_path}  sha256={sha_raw[:16]}…")
    print(f"  Observations : {csv_path}")
    print(f"  Ledger       : {store.PROVENANCE_FILE}  entry_sha256={entry['entry_sha256'][:16]}…")


if __name__ == "__main__":
    main()
