"""Fetch end-user ground truth: UNICEF Giga school measurements (separate run).

Motivation: RIPE anchors/probes are Global-North-heavy and close to
infrastructure. The Giga Daily-Check measurements (NDT7 speed tests from school
computers) provide end-user connections in the Global South with GPS-located
school sites as GeoIP-independent ground truth (government data, cf. Nabi et
al. 2026, arXiv:2605.21937). This run is kept SEPARATE (GEOIP_DATASET=giga ->
its own cache/ledger files) so that anchor/probe evaluations stay untouched.

IMPORTANT CAVEAT (IP recovery): the public Giga Meter API does NOT expose the
raw client IP (measurements.ip_address is empty for the period, ConnectionInfo
is stripped server-side). The IP is, however, often present in the client's
reverse-DNS hostname (ClientInfo.Hostname): either as an IP literal (no PTR
record exists) or as a dotted quad embedded in the PTR name
("broadband-...-210-7-29-133.connect.com.fj"). Embedded quads are checked via
PTR BACK-VERIFICATION (a reverse lookup of the extracted IP must reproduce the
hostname); by default only verified extractions count. This yields a SUBSET of
the measurements (~7 % of window measurements carry an extractable pattern)
with ISP selection bias: providers whose PTR scheme encodes the IP are
overrepresented. For claims about "GeoIP error on end-user connections in the
Global South" this is a sample with documented selection, not a full census --
report it accordingly.

Filter chain (following Nabi et al., §2.1, there 21 292 pairs / 4 872 schools
/ 27 countries):
  (1) one-month measurement window (--window-start/--window-end, created_at;
      the "Timestamp" field is device time and partly unusable),
  (2) drop measurements without school GPS (join against GigaMaps
      schools_location),
  (3) deduplicate to unique (IP, school) pairs,
  (4) drop IPs that appear at MORE THAN ONE school within the window
      (conservative NAT/CGNAT/VPN heuristic).

Data license: ODbL -- Data: ODbL — Source: Giga (UNICEF) and contributors
(giga_meter.txt, giga_school.txt).

Usage:
  GEOIP_DATASET=giga python data/fetch_giga.py                      # default window
  GEOIP_DATASET=giga python data/fetch_giga.py --max-pages 5        # smoke test
  GEOIP_DATASET=giga python data/fetch_giga.py --verify             # check the ledger

Credentials (.env): GIGA_METER_API_KEY (Meter backend), GIGA_API_KEY (GigaMaps).
Both are required and must be provided via environment/.env — no API keys ship
with this repository.
"""

from __future__ import annotations

import argparse
import csv
import ipaddress
import json
import re
import socket
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import store  # noqa: E402

METER_BASE = "https://uni-ooi-giga-meter-backend.azurewebsites.net/api/v1"
MAPS_BASE = "https://uni-ooi-giga-maps-service.azurewebsites.net/api/v1"
PAGE_SIZE = 100          # Meter API: hard maximum
SCHOOL_PAGE_SIZE = 1000  # GigaMaps schools_location tolerates larger pages
TIMEOUT = 60

WINDOW_START_DEFAULT = "2026-07-15"
WINDOW_END_DEFAULT = "2026-08-15"

# Dotted quad in the PTR name, separated by '.' or '-' (both spellings common)
_QUAD = re.compile(r"(\d{1,3})[.-](\d{1,3})[.-](\d{1,3})[.-](\d{1,3})")


def _keys() -> dict[str, str]:
    env = store.load_env(("GIGA_METER_API_KEY", "GIGA_API_KEY"))
    missing = [k for k in ("GIGA_METER_API_KEY", "GIGA_API_KEY") if not env.get(k)]
    if missing:
        sys.exit(f"Missing credentials in .env: {', '.join(missing)}")
    return env


def _get(session: requests.Session, url: str, key: str) -> dict:
    resp = session.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ---------- Phase 1: pull measurements in the window (resumable raw cache) ----------

def fetch_measurements(session, key, start: str, end: str,
                       max_pages: int | None = None) -> tuple[list[dict], int]:
    """Fetch all measurements with created_at in [start, end], paginated ascending.

    Raw pages land resumably in cache/giga_measurements_raw{_SUFFIX}.jsonl
    (one line per page); an abort resumes on the next run after the last
    complete page. Returns (measurements, page count).
    """
    raw_path = store.CACHE_DIR / f"giga_measurements_raw_{store.DATASET}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    pages: dict[int, list[dict]] = {}
    if raw_path.exists():
        with open(raw_path, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("window") == [start, end]:
                    pages[rec["page"]] = rec["data"]
    resume_from = (max(pages) + 1) if pages else 0
    if pages:
        print(f"Raw cache: {len(pages)} pages present, resuming from page {resume_from}.")

    done = False
    p = resume_from
    with open(raw_path, "a", encoding="utf-8") as out:
        while not done:
            if max_pages is not None and p >= max_pages:
                break
            url = (f"{METER_BASE}/measurements?size={PAGE_SIZE}&page={p}"
                   f"&orderBy=created_at&filterBy=created_at"
                   f"&filterCondition=gte&filterValue={start}")
            body = _get(session, url, key)
            data = body.get("data") or []
            # Window end: the page ends as soon as the first entry falls past --window-end
            data = [m for m in data if (m.get("created_at") or "")[:10] <= end]
            pages[p] = data
            out.write(json.dumps({"window": [start, end], "page": p, "data": data},
                                 ensure_ascii=False) + "\n")
            out.flush()
            if len(data) < PAGE_SIZE:      # last (partial) page in the window
                done = True
            if p % 100 == 0:
                print(f"  Page {p}: {sum(len(v) for v in pages.values())} measurements cumulative")
            p += 1

    measurements = [m for _, page in sorted(pages.items()) for m in page
                    if (m.get("created_at") or "")[:10] >= start]
    return measurements, len(pages)


# ---------- Phase 2: IP extraction from ClientInfo.Hostname ----------

def extract_ip(hostname: str | None) -> tuple[str | None, str]:
    """(IP, method) from the client hostname: 'literal' | 'rdns' | '' (nothing).

    Embedded quads are read ONLY in forward order; if the reverse reading is
    also a global IP, the PTR verification decides later.
    """
    if not hostname:
        return None, ""
    try:
        ip = ipaddress.ip_address(hostname)
        return (str(ip), "literal") if ip.is_global else (None, "")
    except ValueError:
        pass
    for m in _QUAD.finditer(hostname):
        octs = [int(x) for x in m.groups()]
        if all(o <= 255 for o in octs):
            cand = ipaddress.ip_address(".".join(map(str, octs)))
            if cand.is_global:
                return str(cand), "rdns"
    return None, ""


def verify_ptr(pairs: list[tuple[str, str]], workers: int = 16,
               attempts: int = 3) -> dict[str, bool]:
    """PTR back-verification: does the extracted IP resolve back to the hostname?

    pairs = [(ip, expected_hostname)]; returns {ip: True/False}.

    DNS is live and thus non-deterministic (timeouts, resolver load). To keep
    the DATASET deterministic: (1) verifications that already passed are
    persisted in cache/giga_ptr_cache.json and carried over on repeat runs
    (once verified = documented as verified; DNS may change later, the first
    successful check is authoritative); (2) failed lookups are retried up to
    ``attempts`` times before counting as NOT verified.
    """
    cache_path = store.CACHE_DIR / "giga_ptr_cache.json"
    cache: dict[str, bool] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def one(pair: tuple[str, str]) -> tuple[str, bool]:
        ip, expected = pair
        for _ in range(attempts):
            try:
                name = socket.gethostbyaddr(ip)[0]
                return ip, name.rstrip(".").lower() == expected.rstrip(".").lower()
            except OSError:
                continue
        return ip, False

    todo = [p for p in pairs if not cache.get(p[0], False)]
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for ip, ok in ex.map(one, todo):
                if ok:
                    cache[ip] = True
    result = {ip: bool(cache.get(ip, False)) for ip, _ in pairs}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=0),
                          encoding="utf-8")
    return result


# ---------- Phase 3: school GPS from the GigaMaps school API ----------

def fetch_iso3_map(session, meter_key) -> dict[str, str]:
    """ISO2 -> ISO3 from the Meter country list (43 registered countries)."""
    out: dict[str, str] = {}
    p = 0
    while True:
        body = _get(session, f"{METER_BASE}/dailycheckapp_countries?size=100&page={p}",
                    meter_key)
        data = body.get("data") or []
        for c in data:
            if c.get("code") and c.get("code_iso3"):
                out[c["code"]] = c["code_iso3"]
        if len(data) < 100:
            return out
        p += 1


def fetch_schools(session, maps_key, iso3_codes: list[str],
                  wanted_ids: set[str]) -> dict[str, dict]:
    """Fetch school sites per country -> {giga_id: {lat, lon, name, iso3}}.

    For giga_ids missing from the country dump, a per-ID lookup as fallback.
    """
    schools: dict[str, dict] = {}
    for iso3 in iso3_codes:
        p = 1                              # GigaMaps paginates 1-based
        while True:
            body = _get(session, f"{MAPS_BASE}/schools_location/country/{iso3}"
                                 f"?page={p}&size={SCHOOL_PAGE_SIZE}", maps_key)
            data = body.get("data") or []
            for s in data:
                gid = s.get("giga_id_school")
                if gid:
                    schools[gid] = {"lat": s.get("latitude"), "lon": s.get("longitude"),
                                    "name": s.get("school_name"), "iso3": iso3}
            if len(data) < SCHOOL_PAGE_SIZE:
                break
            p += 1
        print(f"  Schools {iso3}: cumulative {len(schools)}")
    missing = [g for g in wanted_ids if g not in schools]
    for gid in missing:
        body = _get(session, f"{MAPS_BASE}/schools_location/giga_id/{gid}", maps_key)
        for s in body.get("data") or []:
            schools[gid] = {"lat": s.get("latitude"), "lon": s.get("longitude"),
                            "name": s.get("school_name"),
                            "iso3": s.get("country_iso3_code")}
    if missing:
        found = sum(1 for g in missing if g in schools)
        print(f"  Per-ID lookups for {len(missing)} missing giga_ids: {found} found.")
    return schools


# ---------- Main run ----------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fetch UNICEF Giga school measurements as end-user ground truth.")
    ap.add_argument("--window-start", default=WINDOW_START_DEFAULT,
                    help=f"window start created_at, YYYY-MM-DD (default {WINDOW_START_DEFAULT})")
    ap.add_argument("--window-end", default=WINDOW_END_DEFAULT,
                    help=f"window end inclusive, YYYY-MM-DD (default {WINDOW_END_DEFAULT})")
    ap.add_argument("--max-pages", type=int, default=None, help="for tests: limit pages")
    ap.add_argument("--keep-unverified", action="store_true",
                    help="keep rDNS extractions WITHOUT a passed PTR back-verification "
                         "(default: discard)")
    ap.add_argument("--verify", action="store_true", help="only check the provenance ledger")
    args = ap.parse_args()

    if store.DATASET == "anchors":
        print("Note: without GEOIP_DATASET=giga this writes into the anchor paths!")
        print("Usage:  GEOIP_DATASET=giga python data/fetch_giga.py")
        sys.exit(2)

    if args.verify:
        for f in store.verify_chain():
            print(("OK " if f["ok"] else "!! ") + f"#{f['line']} {f['fetch_id']}: "
                  + ("intact" if f["ok"] else "; ".join(f["problems"])))
        return

    env = _keys()
    session = requests.Session()
    fetch_id = store.new_fetch_id()
    fetched_at = store.utc_now_iso()

    print(f"Window {args.window_start} .. {args.window_end} (created_at, Meter API)")
    measurements, n_pages = fetch_measurements(
        session, env["GIGA_METER_API_KEY"], args.window_start, args.window_end,
        max_pages=args.max_pages)
    print(f"Window measurements: {len(measurements)} on {n_pages} pages.")

    # --- IP extraction ---
    counts = Counter()
    candidates: list[dict] = []
    for m in measurements:
        ci = m.get("ClientInfo") or {}
        gid = m.get("giga_id_school")
        ip, method = extract_ip(ci.get("Hostname"))
        counts["messungen"] += 1
        if not gid:
            counts["ohne_giga_id"] += 1
            continue
        if not ip:
            counts["ohne_extrahierbare_ip"] += 1
            continue
        counts[f"methode_{method}"] += 1
        asn_raw = str(ci.get("ASN") or "")
        candidates.append({
            "ip": ip, "method": method, "hostname": ci.get("Hostname"),
            "giga_id": gid, "country2": m.get("country_code"),
            "asn": asn_raw.removeprefix("AS") or None, "isp": ci.get("ISP"),
            "created_at": m.get("created_at")})

    # PTR back-verification only for rdns extractions (literals claim no PTR)
    rdns_pairs = sorted({(c["ip"], c["hostname"]) for c in candidates
                         if c["method"] == "rdns"})
    verified = verify_ptr(rdns_pairs) if rdns_pairs else {}
    n_ver = sum(verified.values())
    print(f"IP extraction: {counts['methode_literal']} literals, "
          f"{counts['methode_rdns']} rDNS candidates "
          f"({len(rdns_pairs)} unique IPs, of which PTR-verified {n_ver}).")
    if not args.keep_unverified:
        before = len(candidates)
        candidates = [c for c in candidates
                      if c["method"] == "literal" or verified.get(c["ip"], False)]
        counts["ptr_verworfen"] = before - len(candidates)
        print(f"PTR filter: {before - len(candidates)} unverified rDNS rows dropped.")

    # --- School GPS (filter 2) ---
    wanted_ids = {c["giga_id"] for c in candidates}
    iso3_map = fetch_iso3_map(session, env["GIGA_METER_API_KEY"])
    iso3_codes = sorted({iso3_map[c["country2"]] for c in candidates
                         if c["country2"] in iso3_map})
    unmapped = sorted({c["country2"] for c in candidates if c["country2"] not in iso3_map})
    if unmapped:
        print(f"  Caution: ISO2 without ISO3 mapping (per-ID lookup only): {unmapped}")
    print(f"School sites for {len(iso3_codes)} countries / {len(wanted_ids)} schools …")
    schools = fetch_schools(session, env["GIGA_API_KEY"], iso3_codes, wanted_ids)

    before = len(candidates)
    candidates = [c for c in candidates
                  if schools.get(c["giga_id"], {}).get("lat") is not None
                  and schools.get(c["giga_id"], {}).get("lon") is not None]
    counts["ohne_schul_gps"] = before - len(candidates)
    print(f"Filter 2 (school GPS missing): {before - len(candidates)} dropped -> {len(candidates)}.")

    # --- Dedup to (IP, school) (filter 3); aggregate measurement times per pair ---
    # (first/last created_at + count: basis for the freshness audit and
    #  age stratification -- a pair holds AT MEASUREMENT TIME, not forever)
    dedup: dict[tuple[str, str], dict] = {}
    times: dict[tuple[str, str], list] = {}
    for c in candidates:
        k = (c["ip"], c["giga_id"])
        dedup.setdefault(k, c)
        t = c["created_at"] or ""
        if k not in times:
            times[k] = [t, t, 0]
        times[k][0] = min(times[k][0], t)
        times[k][1] = max(times[k][1], t)
        times[k][2] += 1
    pairs = list(dedup.values())
    counts["dubletten"] = len(candidates) - len(pairs)
    print(f"Filter 3 (dedup (IP, school)): {len(candidates) - len(pairs)} duplicates "
          f"-> {len(pairs)} pairs.")

    # --- Drop shared IPs (filter 4) ---
    ip_schools: dict[str, set] = defaultdict(set)
    for c in pairs:
        ip_schools[c["ip"]].add(c["giga_id"])
    before = len(pairs)
    pairs = [c for c in pairs if len(ip_schools[c["ip"]]) == 1]
    counts["shared_ips"] = before - len(pairs)
    print(f"Filter 4 (IP at >1 school): {before - len(pairs)} dropped -> {len(pairs)} pairs.")

    pairs.sort(key=lambda c: (c["giga_id"], c["ip"]))
    n_schools = len({c["giga_id"] for c in pairs})
    n_countries = len({c["country2"] for c in pairs})
    print(f"\nResult: {len(pairs)} (IP, school) pairs, {n_schools} schools, "
          f"{n_countries} countries.")

    # --- Artifacts in the anchor schema (anchor_id = giga_id_school; GT = school GPS) ---
    records = [{
        "anchor_id": c["giga_id"], "ip": c["ip"],
        "lat": schools[c["giga_id"]]["lat"], "lon": schools[c["giga_id"]]["lon"],
        "city": None, "country": c["country2"], "asn": c["asn"],
        "hostname": c["hostname"], "is_disabled": False,
    } for c in pairs]
    store.save_anchors_csv(records)

    times_path = store.CACHE_DIR / f"{store.DATASET}_pair_times.csv"
    with open(times_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ip", "giga_id_school", "n_measurements",
                    "first_created_at", "last_created_at"])
        for c in pairs:
            t = times[(c["ip"], c["giga_id"])]
            w.writerow([c["ip"], c["giga_id"], t[2], t[0], t[1]])

    tags_path = store.CACHE_DIR / f"{store.DATASET}_tags.csv"
    with open(tags_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ip", "tags"])
        for c in pairs:
            ptr = ("ptr-verified" if c["method"] == "rdns" and verified.get(c["ip"])
                   else "ptr-unverified" if c["method"] == "rdns" else "no-ptr")
            w.writerow([c["ip"], ";".join(filter(None, [
                "giga", f"method-{c['method']}", ptr,
                f"country-{c['country2']}", f"asn-{c['asn']}" if c["asn"] else None]))])

    summary = {"window": [args.window_start, args.window_end], "counts": dict(counts),
               "n_pairs": len(pairs), "n_schools": n_schools, "n_countries": n_countries,
               "ptr_verified_ips": n_ver, "keep_unverified": bool(args.keep_unverified)}
    raw_path, sha_raw = store.save_raw(
        fetch_id, json.dumps(summary, ensure_ascii=False).encode("utf-8"), label="giga")

    entry = store.append_provenance({
        "fetch_id": fetch_id, "fetched_at_utc": fetched_at, "kind": "giga",
        "source_url": f"{METER_BASE}/measurements + {MAPS_BASE}/schools_location",
        "http_method": "GET", "tool": "data/fetch_giga.py",
        "tool_git_commit": store.git_commit(), "license": "ODbL (Quelle: Giga/UNICEF)",
        "window_start": args.window_start, "window_end": args.window_end,
        "filter_counts": dict(counts), "n_pairs": len(pairs),
        "n_schools": n_schools, "n_countries": n_countries,
        "raw_file": str(raw_path.relative_to(store.BASE_DIR)),
        "pair_times_file": str(times_path.relative_to(store.BASE_DIR)),
        "sha256_raw": sha_raw, "sha256_records": store.sha256_json(records)})

    print(f"\nFetch {fetch_id}:")
    print(f"  GT     : {store.GROUND_TRUTH_CSV}")
    print(f"  Tags   : {tags_path}")
    print(f"  Times  : {times_path}")
    print(f"  Ledger : {store.PROVENANCE_FILE}  entry_sha256={entry['entry_sha256'][:16]}…")
    print("\nNext step (query the sources, consumes API quota):")
    print("  GEOIP_DATASET=giga python data/fetch_sources.py --all")


if __name__ == "__main__":
    main()
