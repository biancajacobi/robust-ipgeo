#!/usr/bin/env python3
"""RDAP lookup of the MOST SPECIFIC registry object per IP (what `whois <ip>`
shows an investigator) -- complement to the delegation level of the delegation
files (data/fetch_rir.py). Cached in data/cache/rdap_country_cache.jsonl, resumable.

ONLY network metadata is stored (handle, range, type, name, country field,
country line of the registrant address), no personal/contact data from the
entities. Requests are routed directly to the responsible registry according to
the delegation file (saves the rdap.org redirect). One worker per registry with
its own throttle.

ARIN network objects carry NO country field; there the last line of the
registrant address counts (country name, mapped to ISO in exp_registry_country).
LACNIC delivers the country structured in the address (last component).

Usage:   python data/fetch_rdap_country.py [population ...]
         (default: anchors probes probes_holdout probes_mobile_ext probes_pool -- Giga deliberately NOT)
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import threading
import time
from pathlib import Path
from queue import Queue

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data import store                                      # noqa: E402
from experiments.exp_registry_country import Delegations, POP_DATE  # noqa: E402

CACHE = ROOT / "data" / "cache" / "rdap_country_cache.jsonl"
UA = "robust-ipgeo/research (academic; contact via repo)"
BASE = {
    "ripencc": "https://rdap.db.ripe.net/ip/{ip}",
    "arin":    "https://rdap.arin.net/registry/ip/{ip}",
    "apnic":   "https://rdap.apnic.net/ip/{ip}",
    "lacnic":  "https://rdap.lacnic.net/rdap/ip/{ip}",
    "afrinic": "https://rdap.afrinic.net/rdap/ip/{ip}",
    "none":    "https://rdap.org/ip/{ip}",
}
THROTTLE = {"ripencc": 0.15, "arin": 0.5, "apnic": 0.3, "lacnic": 2.5, "afrinic": 0.6, "none": 1.0}   # LACNIC: 429 from ~2 req/s
DEFAULT_POPS = ["anchors", "probes", "probes_holdout", "probes_mobile_ext", "probes_pool"]
_lock = threading.Lock()


def registrant_country(body: dict) -> tuple[str, str]:
    """(last line of the adr label, structured country component) of the registrant;
    fallback: first entity with an address."""
    best = ("", "")
    for pref in (True, False):
        for e in body.get("entities", []) or []:
            roles = e.get("roles") or []
            if pref and "registrant" not in roles:
                continue
            for it in (e.get("vcardArray") or [None, []])[1]:
                if it and it[0] == "adr":
                    label = (it[1] or {}).get("label", "") or ""
                    last = label.strip().splitlines()[-1].strip() if label.strip() else ""
                    struct = it[3][-1].strip() if isinstance(it[3], list) and it[3] else ""
                    if last or struct:
                        return (last, struct)
    return best


def fetch_one(ip: str, rir: str, sess: requests.Session) -> dict:
    url = BASE[rir].format(ip=ip)
    rec = {"ip": ip, "rir": rir, "url": url,
           "fetched_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    for attempt in range(4):
        try:
            r = sess.get(url, headers={"User-Agent": UA, "Accept": "application/rdap+json"}, timeout=60)
        except requests.RequestException as e:
            rec["error"] = str(e)[:200]
            time.sleep(2 * (attempt + 1))
            continue
        rec["http_status"] = r.status_code
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code != 200:
            return rec
        try:
            b = r.json()
        except ValueError:
            rec["error"] = "json"
            return rec
        last, struct = registrant_country(b)
        rec.update({"handle": b.get("handle"), "start": b.get("startAddress"), "end": b.get("endAddress"),
                    "type": b.get("type"), "name": b.get("name"), "country": b.get("country"),
                    "registrant_adr_last": last, "registrant_adr_cc": struct,
                    "server_date": r.headers.get("Date")})
        return rec
    return rec


def worker(rir: str, q: Queue, out, done: list):
    sess = requests.Session()
    while True:
        ip = q.get()
        if ip is None:
            break
        rec = fetch_one(ip, rir, sess)
        with _lock:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
            done.append(ip)
            if len(done) % 200 == 0:
                print(f"  {len(done)} done ({dt.datetime.now().strftime('%H:%M:%S')})", flush=True)
        time.sleep(THROTTLE[rir])


def main(argv: list[str]) -> None:
    pops = argv or DEFAULT_POPS
    have = set()
    if CACHE.exists():
        for line in open(CACHE, encoding="utf-8"):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("http_status") == 200 or d.get("http_status") == 404:
                have.add(d["ip"])
    todo: dict[str, list[str]] = {r: [] for r in BASE}
    seen = set(have)
    for pop in pops:
        D = Delegations(POP_DATE[pop])
        for r in store.load_anchors_csv(ROOT / "data" / "cache" / f"{pop}.csv"):
            ip = r["ip"]
            if ip in seen:
                continue
            seen.add(ip)
            todo[D.lookup(ip)[1] or "none"].append(ip)
    total = sum(len(v) for v in todo.values())
    print(f"Cache: {len(have)} present; new: {total} " + str({k: len(v) for k, v in todo.items() if v}))
    if not total:
        return
    queues = {r: Queue() for r in todo if todo[r]}
    done: list = []
    with open(CACHE, "a", encoding="utf-8") as out:
        threads = [threading.Thread(target=worker, args=(r, q, out, done), daemon=True) for r, q in queues.items()]
        for t in threads:
            t.start()
        for r, q in queues.items():
            for ip in todo[r]:
                q.put(ip)
            q.put(None)
        for t in threads:
            t.join()
    print(f"done: {len(done)} new entries -> {CACHE}")


if __name__ == "__main__":
    main(sys.argv[1:])
