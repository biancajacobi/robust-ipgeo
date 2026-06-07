"""Ground Truth beschaffen: RIPE-Atlas-Anchors — mit Chain-of-Custody-Beleg.

Ruft https://atlas.ripe.net/api/v2/anchors/ (paginiert) ab und dokumentiert den
Abruf forensisch über ``data/store``:
  - Vollständige HTTP-Antwort (Header + Body) je Seite wortgetreu in
    data/cache/raw/ (SHA-256-belegt, das Beweisstück),
  - normalisierte Anchors in data/cache/anchors.csv (Arbeitskopie),
  - einen Audit-Ledger-Eintrag (wann/woher/wie/womit) in data/provenance.jsonl
    als tamper-evidente Hash-Kette.

Belegt wird je Seite: HTTP-Methode, finale URL, Status, der vom RIPE-*Server*
attestierte ``Date``-Header (unabhängige Zeitquelle neben der lokalen Uhr) und
``Content-Type``. Die kompletten Response-Header liegen zusätzlich im Beweisstück.

Anchor-Felder: id, ip_v4, as_v4, city, country, geometry.coordinates = [lon, lat]
(!), is_disabled. Nur öffentliche Infrastruktur-IPs; keine personenbezogenen Daten.

Aufrufe:
  python data/fetch_anchors.py                # aktive Anchors abrufen + belegen
  python data/fetch_anchors.py --all          # auch deaktivierte aufnehmen
  python data/fetch_anchors.py --max-pages 1  # für Tests: Seiten begrenzen
  python data/fetch_anchors.py --verify       # nur das Provenance-Ledger prüfen
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import store  # noqa: E402

ANCHORS_URL = "https://atlas.ripe.net/api/v2/anchors/"
PAGE_SIZE = 500
TIMEOUT = 30


def _normalize(raw_anchor: dict) -> dict:
    coords = (raw_anchor.get("geometry") or {}).get("coordinates") or [None, None]
    lon, lat = coords[0], coords[1]  # GeoJSON ist [lon, lat] -> beim Mapping drehen!
    return {
        "anchor_id": raw_anchor.get("id"),
        "ip": raw_anchor.get("ip_v4"),
        "lat": lat,
        "lon": lon,
        "city": raw_anchor.get("city"),
        "country": raw_anchor.get("country"),
        "asn": raw_anchor.get("as_v4"),
        "hostname": raw_anchor.get("hostname"),
        "is_disabled": raw_anchor.get("is_disabled"),
    }


def fetch_anchors(active_only: bool = True, max_pages: int | None = None):
    """Anchors paginiert abrufen.

    Rückgabe: (records, evidence, api_count)
      records   normalisierte Anchor-Dicts (ANCHOR_COLUMNS), ggf. nur aktive
      evidence  Liste der vollständigen HTTP-Austausche je Seite (Methode, URL,
                Status, Header, Body) — wortgetreu, für das Beweisstück
    """
    url = f"{ANCHORS_URL}?page_size={PAGE_SIZE}"
    records, evidence = [], []
    api_count = None
    pages = 0
    while url:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        evidence.append({
            "http_method": resp.request.method,
            "url": resp.url,                       # finale URL (nach evtl. Redirects)
            "status": resp.status_code,
            "server_date": resp.headers.get("Date"),       # vom Server attestiert
            "content_type": resp.headers.get("Content-Type"),
            "headers": dict(resp.headers),         # komplette Header (ins Beweisstück)
            "body": body,
        })
        if api_count is None:
            api_count = body.get("count")
        for a in body.get("results", []):
            rec = _normalize(a)
            if active_only and rec["is_disabled"]:
                continue
            records.append(rec)
        pages += 1
        url = body.get("next")
        if max_pages and pages >= max_pages:
            break
    return records, evidence, api_count


def _page_summary(evidence: list[dict]) -> list[dict]:
    """Kompakte, gut lesbare Seitenübersicht fürs Ledger (ohne die Bodies)."""
    return [{
        "http_method": e["http_method"],
        "url": e["url"],
        "status": e["status"],
        "server_date": e["server_date"],
        "content_type": e["content_type"],
    } for e in evidence]


def _run_verify() -> None:
    findings = store.verify_chain()
    if not findings:
        print("Ledger leer — noch kein Abruf dokumentiert.")
        return
    for f in findings:
        mark = "OK " if f["ok"] else "!! "
        detail = "integer" if f["ok"] else "; ".join(f["problems"])
        print(f"{mark}#{f['line']} {f['fetch_id']}: {detail}")
    bad = [f for f in findings if not f["ok"]]
    print(f"\n{len(findings)} Eintrag/Einträge, {len(bad)} mit Befund.")
    sys.exit(1 if bad else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description="RIPE-Atlas-Anchors abrufen (mit Provenance).")
    ap.add_argument("--all", action="store_true", help="auch deaktivierte Anchors aufnehmen")
    ap.add_argument("--max-pages", type=int, default=None, help="für Tests: Seiten begrenzen")
    ap.add_argument("--verify", action="store_true", help="nur das Provenance-Ledger prüfen")
    args = ap.parse_args()

    if args.verify:
        _run_verify()
        return

    active_only = not args.all
    fetch_id = store.new_fetch_id()
    fetched_at = store.utc_now_iso()

    records, evidence, api_count = fetch_anchors(
        active_only=active_only, max_pages=args.max_pages)

    # Vollständige HTTP-Antwort (Header + Body) wortgetreu sichern + hashen
    raw_bytes = json.dumps(evidence, ensure_ascii=False,
                           separators=(",", ":")).encode("utf-8")
    raw_path, sha_raw = store.save_raw(fetch_id, raw_bytes, label="anchors")

    # normalisierte Arbeitskopie
    csv_path = store.save_anchors_csv(records)

    # Audit-Ledger-Eintrag (wann / woher / wie / womit)
    pages = _page_summary(evidence)
    entry = store.append_provenance({
        "fetch_id": fetch_id,
        "fetched_at_utc": fetched_at,           # lokale Uhr (Abrufzeitpunkt)
        "source_url": ANCHORS_URL,
        "http_method": "GET",
        "pages": pages,                         # je Seite: Methode/URL/Status/Server-Date
        "tool": "data/fetch_anchors.py",
        "tool_git_commit": store.git_commit(),
        "n_pages": len(evidence),
        "api_count_field": api_count,
        "n_records_saved": len(records),
        "active_only": active_only,
        "raw_file": str(raw_path.relative_to(store.BASE_DIR)),
        "sha256_raw": sha_raw,                  # deckt Header + Body ab
        "sha256_records": store.sha256_json(records),
    })

    server_date = pages[0]["server_date"] if pages else "?"
    print(f"Abruf {fetch_id}: {len(records)} Anchors gespeichert "
          f"(API meldet {api_count} gesamt, {len(evidence)} Seite(n)).")
    print(f"  Methode  : GET  |  Server-Date (Seite 1): {server_date}")
    print(f"  Rohbeleg : {raw_path}  sha256={sha_raw[:16]}…")
    print(f"  Anchors  : {csv_path}")
    print(f"  Ledger   : {store.PROVENANCE_FILE}  entry_sha256={entry['entry_sha256'][:16]}…")


if __name__ == "__main__":
    main()
