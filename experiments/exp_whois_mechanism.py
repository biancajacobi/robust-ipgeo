"""T?-Validierung: Ist der Mechanismus hinter der DB-IP↔IP2Location-Fehlerkopplung
wirklich der geteilte WHOIS-/Registrierungs-Eintrag (common-mode failure)?

Vorgehen:
  1. Co-Failure-Set bilden: IPs, bei denen DB-IP und IP2Location auf ~denselben
     Punkt fallen (< AGREE_KM) UND beide weit von der Ground Truth (> WRONG_KM).
     Das ist das gemeinsame Versagens-Ereignis, dessen Ursache wir prüfen.
  2. Für diese IPs per RDAP (rdap.org → zuständiger RIR) das Registrierungs-Land
     und die Registranten-Org ziehen — gecacht + als Beweisstück in der Provenance-
     Hashkette (forensisch reproduzierbar gegen die festgehaltenen Antworten).
  3. Mechanismus-Test: fällt das (falsche) DB-Land mit dem RDAP-Land zusammen und
     weicht es vom Anchor-Land ab? → DBs sind dem Registrierungs-Land gefolgt.
  4. Gratis-Beobachtung: verhalten sich die Web-APIs (ip_api/ipwho_is/ipapi_co) auf
     denselben IPs gleich (auch WHOIS-Falle) oder abweichend (eigene, dritte Linie)?

Aufruf:  python experiments/exp_whois_mechanism.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import store  # noqa: E402

import geoip2.database  # noqa: E402
import IP2Location  # noqa: E402

AGREE_KM = 50      # DB-IP und IP2Location gelten als "einig", wenn näher als das
WRONG_KM = 200     # ... und beide gelten als "falsch", wenn weiter von GT als das
RDAP_URL = "https://rdap.org/ip/{ip}"
RDAP_CACHE = store.CACHE_DIR / "rdap_cache.jsonl"
THROTTLE = 0.5
UA = "robust-geoip-reference/research (academic; contact via repo)"


def hav(la1, lo1, la2, lo2):
    R = 6371
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = math.radians(la2 - la1), math.radians(lo2 - lo1)
    return 2 * R * math.asin(math.sqrt(
        math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2))


def co_failure_set():
    """(ip, anchor_country, db_country, ip2_country, point, db_err) für Common-Mode-IPs."""
    anchors = store.load_anchors_csv()
    dbip = geoip2.database.Reader("data/db/dbip-city-lite-2026-06.mmdb")
    ip2 = IP2Location.IP2Location("data/db/IP2LOCATION-LITE-DB5.BIN")
    out = []
    for a in anchors:
        ip = a["ip"]
        try:
            gla, glo = float(a["lat"]), float(a["lon"])
        except (TypeError, ValueError):
            continue
        try:
            c = dbip.city(ip)
            d = (c.location.latitude, c.location.longitude, c.country.iso_code) \
                if c.location.latitude is not None else None
        except Exception:
            d = None
        r = ip2.get_all(ip)
        i = (float(r.latitude), float(r.longitude), r.country_short) \
            if getattr(r, "country_short", None) not in ("-", "??", None) and r.latitude is not None else None
        if not (d and i):
            continue
        if (hav(d[0], d[1], i[0], i[1]) < AGREE_KM
                and hav(gla, glo, d[0], d[1]) > WRONG_KM
                and hav(gla, glo, i[0], i[1]) > WRONG_KM):
            out.append({"ip": ip, "anchor_country": a["country"], "db_country": d[2],
                        "ip2_country": i[2], "point": (round(d[0], 3), round(d[1], 3)),
                        "db_err": round(hav(gla, glo, d[0], d[1]))})
    return out


def load_rdap_cache():
    cache = {}
    if RDAP_CACHE.exists():
        for line in RDAP_CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                cache[e["ip"]] = e
    return cache


def rdap_country_and_org(body: dict):
    """Registrierungs-Land + Registranten-Org aus einer RDAP-Antwort ziehen."""
    country = body.get("country")
    org = None
    for e in body.get("entities", []) or []:
        roles = e.get("roles") or []
        vc = e.get("vcardArray")
        fn = adr_country = None
        if vc and len(vc) > 1:
            for item in vc[1]:
                if item[0] == "fn":
                    fn = item[3]
                if item[0] == "adr" and isinstance(item[-1], list):
                    adr_country = item[-1][-1] or None
        if "registrant" in roles and fn and not org:
            org = fn
        if not country and adr_country:
            country = adr_country
    return country, org


def fetch_rdap(ips):
    """RDAP für die IPs holen (Cache + Provenance-Beweisstück). Rückgabe: {ip: entry}."""
    cache = load_rdap_cache()
    evidence, new = [], 0
    for ip in ips:
        if ip in cache and cache[ip].get("body") is not None:
            continue
        time.sleep(THROTTLE)
        fetched_at = store.utc_now_iso()
        try:
            resp = requests.get(RDAP_URL.format(ip=ip), timeout=20, headers={"User-Agent": UA})
            entry = {"ip": ip, "fetched_at_utc": fetched_at, "http_status": resp.status_code,
                     "url": resp.url, "server_date": resp.headers.get("Date"),
                     "body": resp.json() if resp.ok else None}
        except Exception as e:
            entry = {"ip": ip, "fetched_at_utc": fetched_at, "http_status": None,
                     "url": RDAP_URL.format(ip=ip), "body": None, "error": repr(e)}
        with open(RDAP_CACHE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        cache[ip] = entry
        evidence.append(entry)
        new += 1
    # Beweisstück dieses Laufs + Ledger-Eintrag (nur wenn neu geholt wurde)
    if new:
        fetch_id = store.new_fetch_id()
        raw = "\n".join(json.dumps(e, ensure_ascii=False) for e in evidence).encode("utf-8")
        raw_path, sha = store.save_raw(fetch_id, raw, label="rdap")
        store.append_provenance({
            "fetch_id": fetch_id, "fetched_at_utc": store.utc_now_iso(), "kind": "rdap-validation",
            "source_url": "https://rdap.org/ip/{ip}", "http_method": "GET",
            "tool": "experiments/exp_whois_mechanism.py", "tool_git_commit": store.git_commit(),
            "n_ips": new, "raw_file": str(raw_path.relative_to(store.BASE_DIR)), "sha256_raw": sha})
        print(f"  RDAP: {new} neu geholt, Beweisstück {raw_path.name} sha256={sha[:16]}…")
    return cache


def web_api_country(ip):
    """Länder der Web-APIs für eine IP aus dem Observations-Cache (source -> country)."""
    out = {}
    with open(store.CACHE_DIR / "observations.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["ip"] == ip and row["source"] in ("ip_api", "ipwho_is", "ipapi_co") \
                    and row["status"] == "success":
                out[row["source"]] = row["country"]
    return out


def main():
    co = co_failure_set()
    print(f"Co-Failure-Set (DB-IP~IP2 <{AGREE_KM}km einig, beide >{WRONG_KM}km von GT): {len(co)} IPs")
    cache = fetch_rdap([c["ip"] for c in co])

    confirmed = diff_anchor = web_same = web_total = 0
    print(f"\n{'IP':16s} {'GT':3s} {'DB':3s} {'RDAP':4s} {'Org':22s} Mechanismus")
    print("-" * 78)
    for c in co:
        body = (cache.get(c["ip"]) or {}).get("body")
        rc, org = rdap_country_and_org(body) if body else (None, None)
        db_follows_rdap = rc is not None and rc == c["db_country"]
        rdap_diff_anchor = rc is not None and rc != c["anchor_country"]
        ok = db_follows_rdap and rdap_diff_anchor
        confirmed += ok
        diff_anchor += rdap_diff_anchor
        webs = web_api_country(c["ip"])
        for s, wc in webs.items():
            web_total += 1
            if wc == c["db_country"]:
                web_same += 1
        tag = "WHOIS bestätigt" if ok else ("DB≠RDAP" if rc else "RDAP fehlt")
        print(f"{c['ip']:16s} {c['anchor_country']:3s} {c['db_country']:3s} "
              f"{str(rc):4s} {str(org)[:22]:22s} {tag}")

    n = len(co)
    print("-" * 78)
    print(f"\nMechanismus-Test (n={n} Common-Mode-IPs):")
    print(f"  RDAP-Land ≠ Anchor-Land:                 {diff_anchor}/{n} ({100*diff_anchor/n:.0f}%)")
    print(f"  DB-Land == RDAP-Land UND ≠ Anchor-Land:  {confirmed}/{n} ({100*confirmed/n:.0f}%)  ← WHOIS-Mechanismus")
    if web_total:
        print(f"\nWeb-APIs auf denselben IPs: {web_same}/{web_total} Treffer im SELBEN (falschen) DB-Land")
        print(f"  → {'überwiegend dieselbe WHOIS-Falle' if web_same/web_total>0.5 else 'überwiegend abweichend = eigene Linie'}")


if __name__ == "__main__":
    main()
