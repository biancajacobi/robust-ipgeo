"""Pro IP eine kleine, heterogene Wertereihe an Standort-Schätzungen einsammeln.

Quellen: mehrere **freie** Geo-APIs (kein Lizenzschlüssel nötig). Sie widersprechen
sich für dieselbe IP teils um tausende Kilometer — genau die Ausgangslage, die das
Projekt untersucht. Offline-DBs (GeoLite2/DB-IP/IP2Location) sind hier bewusst nicht
angebunden (Lizenz/Download); die Quellen-Registry ``SOURCES`` lässt sich aber leicht
um mmdb-Adapter erweitern.

Reproduzierbarkeit + Provenance (leichtgewichtig):
  data/cache/sources_raw.jsonl     append-only Cache (Arbeitskopie): 1 Zeile je
                                   (Quelle, IP); verhindert doppelte API-Abrufe.
  data/cache/raw/sources_<id>.json wortgetreues, UNVERÄNDERLICHES Beweisstück je
                                   Lauf (die in diesem Lauf geholten Antworten) —
                                   SHA-256-belegt im Ledger (wächst nicht nach).
  data/cache/observations.csv      normalisierte Arbeitskopie (OBSERVATION_COLUMNS),
                                   kumulative Sicht des gesamten Caches.
  data/provenance.jsonl            1 Audit-Ledger-Eintrag je Lauf, in die Hash-Kette.

Hinweise: Nur öffentliche Infrastruktur-IPs (RIPE-Atlas-Anchors). Freie Tiers haben
Rate-Limits (pro Quelle gedrosselt) und untersagen z. T. kommerzielle Nutzung —
akademische Auswertung ist gedeckt. ip-api.com bietet im Free-Tier nur HTTP.

Aufrufe:
  python data/fetch_sources.py                 # Standard: erste 50 Anchors, 3 Quellen
  python data/fetch_sources.py --limit 100     # mehr IPs
  python data/fetch_sources.py --all           # alle Anchors (Vorsicht: Rate-Limits!)
  python data/fetch_sources.py --sources ip_api,ipwho_is
  python data/fetch_sources.py --refresh       # Cache ignorieren, neu abrufen
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
USER_AGENT = "robust-geoip-reference/research (academic; contact via repo)"
SOURCES_RAW = store.SOURCES_RAW_CSV   # dataset-aware (anchors -> sources_raw.jsonl)
DB_DIR = store.BASE_DIR / "db"

# Tokens für authentifizierte Web-Quellen (aus .env; leer => Quelle meldet Fehler).
IPINFO_TOKEN = store.load_env().get("IPINFO_TOKEN", "")


def _redact(url: str) -> str:
    """Token aus einer URL entfernen, bevor sie im Cache landet."""
    return url.replace(IPINFO_TOKEN, "***") if IPINFO_TOKEN and IPINFO_TOKEN in url else url


# ---------- Quellen-Adapter ----------
# Jeder Parser nimmt den Roh-JSON-Body und liefert
# (lat, lon, city, country, status); status == "success" oder eine Fehlerkennung.

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
    # geojs liefert lat/lon als Strings; kein zuverlässiges city-Feld
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
    # ipinfo liefert die Koordinaten gebündelt als loc="lat,lon"
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


# ---------- Offline-Reader für die lokalen LITE-DBs ----------
# Eigene, von den Web-APIs unabhängige Linien. Die DB-Dateien liegen in data/db/
# und sind als Beweisstück bereits in der Provenance-Kette (data/fetch_geodbs.py);
# hier werden sie nur gelesen. Das Reader-Ergebnis ist die "Antwort" der Quelle
# (``body``) — analog zum JSON-Body einer HTTP-Quelle, damit Cache/Normalisierung
# unverändert greifen. Reader werden lazy geöffnet und wiederverwendet.

_db_readers: dict[str, object] = {}


def _resolve_db(pattern: str) -> Path:
    """DB in data/db/ auflösen; '*' wählt die neueste passende (z. B. DB-IP-Monat)."""
    if "*" in pattern:
        matches = sorted(DB_DIR.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"keine DB passend zu {pattern!r} in {DB_DIR}")
        return matches[-1]
    return DB_DIR / pattern


def _read_mmdb(pattern: str):
    """Liefert einen Reader-Callable für eine MaxMind/DB-IP .mmdb (ip -> body)."""
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
                "accuracy_radius": c.location.accuracy_radius,   # MaxMind-Konfidenzradius (km)
                "db_file": path.name, "db_version": version,
                "found": c.location.latitude is not None}
    return read


def _read_ip2location(pattern: str):
    """Liefert einen Reader-Callable für die IP2Location .BIN (ip -> body)."""
    def read(ip: str) -> dict:
        import IP2Location
        path = _resolve_db(pattern)
        reader = _db_readers.get(str(path))
        if reader is None:
            reader = IP2Location.IP2Location(str(path))
            _db_readers[str(path)] = reader
        version = int(path.stat().st_mtime)   # IP2Location-BIN: kein Build-Feld -> Datei-mtime
        rec = reader.get_all(ip)
        lat = _to_float_or_none(getattr(rec, "latitude", None))
        lon = _to_float_or_none(getattr(rec, "longitude", None))
        country = getattr(rec, "country_short", None)
        if country in ("-", "??", None):       # IP2Location-Platzhalter für unbekannt
            return {"latitude": None, "longitude": None, "city": None, "country": None,
                    "accuracy_radius": None, "db_file": path.name, "db_version": version,
                    "found": False}
        return {"latitude": lat, "longitude": lon, "city": getattr(rec, "city", None),
                "country": country, "accuracy_radius": None,
                "db_file": path.name, "db_version": version, "found": lat is not None}
    return read


def _parse_local(body: dict):
    """Body eines Offline-Readers -> (lat, lon, city, country, status)."""
    if not body or not body.get("found"):
        return None, None, None, None, "not_found"
    return (body.get("latitude"), body.get("longitude"),
            body.get("city"), body.get("country"), "success")


# ``lineage`` = bestätigte/angenommene Datenherkunft. Quellen mit GLEICHEM
# lineage-Token gelten als korreliert (eine effektive Linie) und werden in der
# Aggregation später per Gewicht kollabiert, nicht gelöscht. "maxmind_geolite"
# ist die einzige *bestätigt* geteilte Linie; die "*_unverified"-Token markieren
# Anbieter, die ihre Herkunft nicht offenlegen (vorsichtshalber separat geführt).
#
# Quellen-Arten:
#   Web-API   -> "url" (+ "min_interval", gedrosselt) + "parse" (JSON-Body)
#   lokale DB -> "reader" (ip -> body, offline) + "parse" = _parse_local; "db" nur Doku
SOURCES = {
    # --- Web-APIs ---
    "ip_api":          {"url": "http://ip-api.com/json/{ip}",              "min_interval": 1.5, "parse": _parse_ip_api,          "lineage": "ipapi_com_unverified"},
    "ipwho_is":        {"url": "https://ipwho.is/{ip}",                    "min_interval": 0.5, "parse": _parse_ipwho_is,        "lineage": "ipwhois_unverified"},
    "ipinfo":          {"url": "https://ipinfo.io/{ip}/json?token={token}","min_interval": 0.2, "parse": _parse_ipinfo,          "lineage": "ipinfo"},
    "geojs":           {"url": "https://get.geojs.io/v1/ip/geo/{ip}.json", "min_interval": 0.4, "parse": _parse_geojs,           "lineage": "maxmind_geolite"},
    "reallyfreegeoip": {"url": "https://reallyfreegeoip.org/json/{ip}",    "min_interval": 0.4, "parse": _parse_reallyfreegeoip, "lineage": "maxmind_geolite"},
    "ipapi_co":        {"url": "https://ipapi.co/{ip}/json/",              "min_interval": 1.2, "parse": _parse_ipapi_co,        "lineage": "ipapi_co_unverified"},
    # --- lokale LITE-DBs (offline; eigene unabhängige Linien) ---
    "maxmind_geolite2":{"db": "GeoLite2-City.mmdb",         "reader": _read_mmdb("GeoLite2-City.mmdb"),         "parse": _parse_local, "lineage": "maxmind_geolite"},
    "dbip_lite":       {"db": "dbip-city-lite-*.mmdb",      "reader": _read_mmdb("dbip-city-lite-*.mmdb"),      "parse": _parse_local, "lineage": "dbip_lite"},
    "ip2location_lite":{"db": "IP2LOCATION-LITE-DB5.BIN",   "reader": _read_ip2location("IP2LOCATION-LITE-DB5.BIN"), "parse": _parse_local, "lineage": "ip2location_lite"},
}
DEFAULT_SOURCES = list(SOURCES)


# ---------- Rate-Limiting (pro Quelle) ----------

_last_call: dict[str, float] = {}


def _throttle(source: str) -> None:
    interval = SOURCES[source]["min_interval"]
    last = _last_call.get(source)
    if last is not None:
        wait = interval - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
    _last_call[source] = time.monotonic()


# ---------- Roh-Cache (zugleich Beweisstück) ----------

def load_raw_cache() -> dict[tuple[str, str], dict]:
    """{(source, ip) -> Roh-Eintrag} aus dem append-only Cache laden."""
    cache: dict[tuple[str, str], dict] = {}
    if SOURCES_RAW.exists():
        with open(SOURCES_RAW, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    e = json.loads(line)
                    cache[(e["source"], e["ip"])] = e   # spätere Zeile gewinnt
    return cache


def append_raw_cache(entry: dict) -> None:
    SOURCES_RAW.parent.mkdir(parents=True, exist_ok=True)
    with open(SOURCES_RAW, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _is_cached(entry: dict | None) -> bool:
    """Gilt als verwertbar gecacht, wenn eine Antwort (Body) vorliegt.

    Transiente Fehler (Timeout/Rate-Limit -> body is None) werden NICHT als gecacht
    gewertet, damit ein erneuter Lauf sie nachholt. Eine inhaltliche Quellen-Absage
    (Body vorhanden, status != success) ist dagegen ein gültiges Ergebnis.
    """
    return entry is not None and entry.get("body") is not None


_db_version_cache: dict[str, int] = {}


def _current_db_version(source: str) -> int | None:
    """Aktueller Build/Stand der lokalen DB einer Offline-Quelle (für Cache-Invalidierung)."""
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
    """Wie _is_cached, aber Offline-Treffer veralten, wenn sich der DB-Build geändert hat
    (forensisch: Wiederholungslauf reproduziert gegen die AKTUELLE, hash-gepinnte DB,
    nicht gegen einen Cache aus einem alten DB-Stand)."""
    if not _is_cached(entry):
        return False
    if "reader" in SOURCES.get(source, {}):
        return (entry.get("body") or {}).get("db_version") == _current_db_version(source)
    return True


# ---------- Abruf ----------

def query_source(source: str, ip: str, session: requests.Session | None = None) -> dict:
    """Eine Quelle für eine IP abfragen -> Roh-Eintrag (Body + Metadaten).

    Wirft nicht; Netz-/Parsefehler landen als ``error`` im Eintrag (``body`` = None).
    """
    if source not in SOURCES:
        raise KeyError(f"unbekannte Quelle: {source!r} (bekannt: {', '.join(SOURCES)})")
    src = SOURCES[source]
    fetched_at = store.utc_now_iso()

    # Lokale DB: offline lesen (kein Netz, keine Drosselung). Beweisstück ist die
    # DB-Datei selbst (per Hash in der Provenance-Kette), nicht dieser Body.
    if "reader" in src:
        try:
            body = src["reader"](ip)
            return {"source": source, "ip": ip, "fetched_at_utc": fetched_at,
                    "http_status": None, "url": f"local:{body.get('db_file', src.get('db'))}",
                    "body": body}
        except Exception as e:   # fehlende/defekte DB-Datei
            return {"source": source, "ip": ip, "fetched_at_utc": fetched_at,
                    "http_status": None, "url": f"local:{src.get('db')}", "body": None,
                    "error": repr(e)}

    # Web-API: HTTP (Token wird, falls vorhanden, in die URL eingesetzt)
    url = src["url"].format(ip=ip, token=IPINFO_TOKEN)
    _throttle(source)
    get = (session or requests).get
    try:
        resp = get(url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
        return {"source": source, "ip": ip, "fetched_at_utc": fetched_at,
                "http_status": resp.status_code, "url": _redact(resp.url), "body": resp.json()}
    except Exception as e:  # Netzwerk, Timeout, kein JSON (z. B. Rate-Limit-HTML)
        return {"source": source, "ip": ip, "fetched_at_utc": fetched_at,
                "http_status": None, "url": _redact(url), "body": None, "error": repr(e)}


def observation_from_raw(entry: dict) -> dict:
    """Roh-Eintrag -> normalisierte Beobachtung (OBSERVATION_COLUMNS)."""
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
            # Qualitäts-/Default-Felder: Rohwerte hier, Default-Flag in der Annotation
            "accuracy_radius": (body or {}).get("accuracy_radius"),
            "db_version": (body or {}).get("db_version"),
            "is_default_centroid": False, "centroid_match": ""}


def annotate_defaults(observations: list[dict], anchors: list[dict] | None = None) -> list[dict]:
    """Hub-/Centroid-Default-Flag je Beobachtung setzen (datensatz-abhängig, daher
    als Nachlauf über ALLE Beobachtungen). Setzt ``is_default_centroid`` + ``centroid_match``
    in-place (Wächter B). Braucht die volle Quellen-Häufigkeit + ASN je IP."""
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
    ap = argparse.ArgumentParser(description="Geo-Schätzungen je IP einsammeln (freie APIs, mit Provenance).")
    ap.add_argument("--limit", type=int, default=50, help="Anzahl Anchors (erste N); Default 50")
    ap.add_argument("--all", action="store_true", help="alle Anchors (Achtung: Rate-Limits)")
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCES),
                    help=f"Komma-Liste; bekannt: {', '.join(SOURCES)}")
    ap.add_argument("--refresh", action="store_true", help="Cache ignorieren und neu abrufen")
    ap.add_argument("--reindex", action="store_true",
                    help="nicht abrufen: observations.csv neu bauen und den aktuellen "
                         "Cache als unveränderliches Beweisstück im Ledger registrieren")
    args = ap.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in sources if s not in SOURCES]
    if unknown:
        ap.error(f"unbekannte Quelle(n): {', '.join(unknown)}")

    anchors = store.load_anchors_csv()
    ips = [a["ip"] for a in anchors if a.get("ip")]
    if not args.all:
        ips = ips[: args.limit]
    ip_set = set(ips)

    fetch_id = store.new_fetch_id()
    fetched_at = store.utc_now_iso()
    cache = load_raw_cache()

    new_entries: list[dict] = []
    if not args.reindex:
        session = requests.Session()
        print(f"Sammle {len(sources)} Quelle(n) × {len(ips)} IP(s) … (Cache: {len(cache)} Einträge)")
        for i, ip in enumerate(ips, 1):
            for source in sources:
                if args.refresh or not _cache_valid(cache.get((source, ip)), source):
                    entry = query_source(source, ip, session=session)
                    append_raw_cache(entry)
                    cache[(source, ip)] = entry
                    new_entries.append(entry)
            if i % 25 == 0:
                print(f"  … {i}/{len(ips)} IPs ({len(new_entries)} neue Abrufe)")

    # Unveränderliches Beweisstück dieses Laufs (eigene Datei je fetch_id -> Hash
    # bleibt stabil, anders als der wachsende Cache): bei --reindex der gesamte für
    # die IPs relevante Cache, sonst die in diesem Lauf geholten Antworten.
    snapshot = ([e for (s, ip), e in cache.items() if ip in ip_set] if args.reindex
                else new_entries)
    raw_bytes = ("\n".join(json.dumps(e, ensure_ascii=False) for e in snapshot)).encode("utf-8")
    raw_path, sha_raw = store.save_raw(fetch_id, raw_bytes, label="sources")

    # observations.csv = kumulative, normalisierte Sicht des gesamten Caches
    obs = [observation_from_raw(e) for (s, ip), e in cache.items() if ip in ip_set]
    obs.sort(key=lambda o: (o["ip"], o["source"]))
    annotate_defaults(obs, anchors)            # Hub-/Centroid-Default-Flags setzen
    csv_path = store.save_observations_csv(obs)
    n_success = sum(1 for o in obs if o["status"] == "success")
    n_default = sum(1 for o in obs if o["is_default_centroid"])

    # DB-Beleg je Lauf (welcher mmdb/BIN-Build + Hash hat die Offline-Antworten erzeugt)
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
        "dbs": db_evidence,                          # DB-Build + SHA-256 je Offline-Quelle
        "raw_file": str(raw_path.relative_to(store.BASE_DIR)),
        "sha256_raw": sha_raw,                       # Beweisstück dieses Laufs (stabil)
        "sha256_observations": store.sha256_json(obs),
    })

    mode = "reindex" if args.reindex else "fetch"
    print(f"\nLauf {fetch_id} ({mode}): {len(obs)} Beobachtungen "
          f"({n_success} ok, {len(obs) - n_success} fehlgeschlagen, {n_default} Hub-Defaults), "
          f"{len(new_entries)} neue Abrufe.")
    print(f"  Beweisstück  : {raw_path}  sha256={sha_raw[:16]}…")
    print(f"  Observations : {csv_path}")
    print(f"  Ledger       : {store.PROVENANCE_FILE}  entry_sha256={entry['entry_sha256'][:16]}…")


if __name__ == "__main__":
    main()
