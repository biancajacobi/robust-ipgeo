"""Validation: is the mechanism behind the DB-IP↔IP2Location error coupling
really the shared WHOIS/registration entry (common-mode failure)?

Approach:
  1. Build the co-failure set: IPs where DB-IP and IP2Location fall onto ~the
     same point (< AGREE_KM) AND both are far from the ground truth (> WRONG_KM).
     That is the joint failure event whose cause we examine.
  2. For these IPs, pull the registration country and the registrant org via
     RDAP (rdap.org → responsible RIR) — cached + as an exhibit in the
     provenance hash chain (forensically reproducible against the recorded responses).
  3. Mechanism test: does the (wrong) DB country coincide with the RDAP country
     while deviating from the anchor country? → the DBs followed the registration country.
  4. Free observation: do the web APIs (ip_api/ipwho_is/ipapi_co) behave the
     same on the same IPs (also the WHOIS trap) or differently (own, third line)?

Invocation:  python experiments/exp_whois_mechanism.py
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

AGREE_KM = 50      # DB-IP and IP2Location count as "in agreement" if closer than this
WRONG_KM = 200     # ... and both count as "wrong" if farther from GT than this
RDAP_URL = "https://rdap.org/ip/{ip}"
RDAP_CACHE = store.CACHE_DIR / "rdap_cache.jsonl"
THROTTLE = 0.5
UA = "robust-ipgeo/research (academic; contact via repo)"


def hav(la1, lo1, la2, lo2):
    R = 6371
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = math.radians(la2 - la1), math.radians(lo2 - lo1)
    return 2 * R * math.asin(math.sqrt(
        math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2))


def co_failure_set():
    """(ip, anchor_country, db_country, ip2_country, point, db_err) for common-mode IPs."""
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
    """Extract registration country + registrant org from an RDAP response."""
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
    """Fetch RDAP for the IPs (cache + provenance exhibit). Returns: {ip: entry}."""
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
    # exhibit of this run + ledger entry (only if anything new was fetched)
    if new:
        fetch_id = store.new_fetch_id()
        raw = "\n".join(json.dumps(e, ensure_ascii=False) for e in evidence).encode("utf-8")
        raw_path, sha = store.save_raw(fetch_id, raw, label="rdap")
        store.append_provenance({
            "fetch_id": fetch_id, "fetched_at_utc": store.utc_now_iso(), "kind": "rdap-validation",
            "source_url": "https://rdap.org/ip/{ip}", "http_method": "GET",
            "tool": "experiments/exp_whois_mechanism.py", "tool_git_commit": store.git_commit(),
            "n_ips": new, "raw_file": str(raw_path.relative_to(store.BASE_DIR)), "sha256_raw": sha})
        print(f"  RDAP: {new} newly fetched, exhibit {raw_path.name} sha256={sha[:16]}…")
    return cache


def web_api_country(ip):
    """Countries of the web APIs for an IP from the observations cache (source -> country)."""
    out = {}
    with open(store.CACHE_DIR / "observations.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["ip"] == ip and row["source"] in ("ip_api", "ipwho_is", "ipapi_co") \
                    and row["status"] == "success":
                out[row["source"]] = row["country"]
    return out


def main():
    co = co_failure_set()
    print(f"Co-failure set (DB-IP~IP2 <{AGREE_KM}km in agreement, both >{WRONG_KM}km from GT): {len(co)} IPs")
    cache = fetch_rdap([c["ip"] for c in co])

    confirmed = diff_anchor = web_same = web_total = 0
    print(f"\n{'IP':16s} {'GT':3s} {'DB':3s} {'RDAP':4s} {'Org':22s} Mechanism")
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
        tag = "WHOIS confirmed" if ok else ("DB≠RDAP" if rc else "RDAP missing")
        print(f"{c['ip']:16s} {c['anchor_country']:3s} {c['db_country']:3s} "
              f"{str(rc):4s} {str(org)[:22]:22s} {tag}")

    n = len(co)
    print("-" * 78)
    print(f"\nMechanism test (n={n} common-mode IPs):")
    print(f"  RDAP country ≠ anchor country:                 {diff_anchor}/{n} ({100*diff_anchor/n:.0f}%)")
    print(f"  DB country == RDAP country AND ≠ anchor country:  {confirmed}/{n} ({100*confirmed/n:.0f}%)  ← WHOIS mechanism")
    if web_total:
        print(f"\nWeb APIs on the same IPs: {web_same}/{web_total} hits in the SAME (wrong) DB country")
        print(f"  → {'predominantly the same WHOIS trap' if web_same/web_total>0.5 else 'predominantly deviating = own line'}")


if __name__ == "__main__":
    main()
