"""Vergleichs-Ground-Truth beschaffen: RIPE-Atlas-PROBES (separater Lauf).

Anders als die Anchors (Rechenzentren, dokumentierte Standorte) decken die Probes
auch Heim-/Mobil-/NAT-Anschlüsse ab -- die forensisch typischere, schwerere Klasse.
Dieser Lauf ist bewusst SEPARAT gehalten (GEOIP_DATASET=probes -> eigene Cache-/
Ledger-Dateien), damit die Anchor-Auswertung unberührt bleibt.

WICHTIGER VORBEHALT: Probe-Koordinaten sind vom Freiwilligen SELBST gemeldet und von
RIPE privacy-gerundet -- die Ground Truth ist also rauschiger als bei Anchors. Höhere
Fehler gegen Probes mischen (a) schwerer lokalisierbare IPs und (b) ungenauere GT;
der Vergleich ist ein direktionaler Stresstest, keine saubere Validierung.

Schema identisch zu anchors.csv (data/store.ANCHOR_COLUMNS), zusätzlich tags-Spalte
in einer Begleitdatei probes_tags.csv (id -> tags) für spätere Stratifizierung
(home / nat / datacenter / mobile).

Aufrufe:
  python data/fetch_probes.py --n 500            # 500 Probes ziehen (seed 0)
  python data/fetch_probes.py --n 500 --seed 0   # reproduzierbare Stichprobe
  python data/fetch_probes.py --verify           # Probe-Provenance-Ledger prüfen
"""

from __future__ import annotations

import argparse
import csv
import ipaddress
import json
import random
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import store  # noqa: E402

PROBES_URL = "https://atlas.ripe.net/api/v2/probes/"
PAGE_SIZE = 500
TIMEOUT = 30


def _public_ipv4(addr: str | None) -> bool:
    """True nur für global routbare IPv4 (kein RFC1918/CGNAT/loopback/link-local)."""
    if not addr:
        return False
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return ip.version == 4 and ip.is_global


def _normalize(p: dict) -> dict:
    coords = (p.get("geometry") or {}).get("coordinates") or [None, None]
    lon, lat = coords[0], coords[1]   # GeoJSON [lon, lat] -> drehen
    return {
        "anchor_id": p.get("id"),                 # Schema-kompatibel (probe id)
        "ip": p.get("address_v4"),
        "lat": lat,
        "lon": lon,
        "city": None,                             # Probes liefern keine Stadt
        "country": p.get("country_code"),
        "asn": p.get("asn_v4"),
        "hostname": None,
        "is_disabled": p.get("status", {}).get("id") != 1,   # 1 = Connected
        "_tags": ";".join(t.get("slug", "") for t in (p.get("tags") or [])),
    }


def fetch_probe_pool(max_pages: int | None = None, tag: str | None = None):
    """Alle verbundenen Non-Anchor-Probes mit öffentlicher IPv4 + Koordinaten holen.

    ``tag`` optional: server-seitiger Filter auf einen Probe-Tag (z. B. ``mobile``).
    Rückgabe: (records, evidence_summary, api_count).
    """
    url = (f"{PROBES_URL}?status=1&is_anchor=false&page_size={PAGE_SIZE}"
           "&fields=id,address_v4,asn_v4,country_code,geometry,status,is_anchor,tags")
    if tag:
        url += f"&tags={tag}"
    records, evidence, api_count, pages = [], [], None, 0
    while url:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        evidence.append({"http_method": resp.request.method, "url": resp.url,
                         "status": resp.status_code, "server_date": resp.headers.get("Date"),
                         "content_type": resp.headers.get("Content-Type")})
        if api_count is None:
            api_count = body.get("count")
        for p in body.get("results", []):
            rec = _normalize(p)
            if rec["is_disabled"] or rec["lat"] is None or not _public_ipv4(rec["ip"]):
                continue
            records.append(rec)
        pages += 1
        url = body.get("next")
        if max_pages and pages >= max_pages:
            break
    return records, evidence, api_count


def main() -> None:
    ap = argparse.ArgumentParser(description="RIPE-Atlas-Probes ziehen (separater Vergleichslauf).")
    ap.add_argument("--n", type=int, default=500, help="Stichprobengröße (Default 500)")
    ap.add_argument("--seed", type=int, default=0, help="Seed für reproduzierbares Sampling")
    ap.add_argument("--max-pages", type=int, default=None, help="für Tests: Seiten begrenzen")
    ap.add_argument("--tag", default=None, help="server-seitiger Tag-Filter (z. B. mobile)")
    ap.add_argument("--verify", action="store_true", help="nur das Probe-Provenance-Ledger prüfen")
    args = ap.parse_args()

    if store.DATASET == "anchors":
        print("Hinweis: ohne GEOIP_DATASET=<probes|probes_mobile|...> wird in die Anchor-Pfade geschrieben!")
        print("Aufruf:  GEOIP_DATASET=probes python data/fetch_probes.py --n 500")
        print("   bzw:  GEOIP_DATASET=probes_mobile python data/fetch_probes.py --tag mobile --n 100")
        sys.exit(2)

    if args.verify:
        for f in store.verify_chain():
            print(("OK " if f["ok"] else "!! ") + f"#{f['line']} {f['fetch_id']}: "
                  + ("integer" if f["ok"] else "; ".join(f["problems"])))
        return

    fetch_id = store.new_fetch_id()
    fetched_at = store.utc_now_iso()
    pool, evidence, api_count = fetch_probe_pool(max_pages=args.max_pages, tag=args.tag)
    print(f"Pool: {len(pool)} geeignete Probes (verbunden, public IPv4, Koordinaten) "
          f"von API-gesamt {api_count}.")

    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.n, len(pool)))
    sample.sort(key=lambda r: r["anchor_id"])

    # Rohbeleg (Seitenübersicht + gezogene IDs) + Hash in die Probe-Hashkette
    raw = json.dumps({"pages": evidence, "seed": args.seed,
                      "sampled_ids": [r["anchor_id"] for r in sample]},
                     ensure_ascii=False).encode("utf-8")
    raw_path, sha_raw = store.save_raw(fetch_id, raw, label="probes")

    store.save_anchors_csv(sample)                       # -> cache/probes.csv (dataset-aware)
    tags_path = store.CACHE_DIR / f"{store.DATASET}_tags.csv"   # dataset-aware (nicht clobbern)
    with open(tags_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["ip", "tags"])
        for r in sample:
            w.writerow([r["ip"], r["_tags"]])

    entry = store.append_provenance({
        "fetch_id": fetch_id, "fetched_at_utc": fetched_at, "kind": "probes",
        "source_url": PROBES_URL, "http_method": "GET",
        "tool": "data/fetch_probes.py", "tool_git_commit": store.git_commit(),
        "n_pool": len(pool), "n_sampled": len(sample), "seed": args.seed, "tag": args.tag,
        "api_count_field": api_count, "raw_file": str(raw_path.relative_to(store.BASE_DIR)),
        "sha256_raw": sha_raw, "sha256_records": store.sha256_json(sample)})

    print(f"Abruf {fetch_id}: {len(sample)} Probes gezogen (seed {args.seed}).")
    print(f"  Probes : {store.GROUND_TRUTH_CSV}")
    print(f"  Tags   : {tags_path}")
    print(f"  Ledger : {store.PROVENANCE_FILE}  entry_sha256={entry['entry_sha256'][:16]}…")
    print("\nNächster Schritt (Quellen abfragen, verbraucht API-Quota):")
    print("  GEOIP_DATASET=probes python data/fetch_sources.py --all")


if __name__ == "__main__":
    main()
