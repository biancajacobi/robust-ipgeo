"""Obtain ground truth: RIPE Atlas anchors — with chain-of-custody evidence.

Fetches https://atlas.ripe.net/api/v2/anchors/ (paginated) and documents the
fetch forensically via ``data/store``:
  - full HTTP response (headers + body) per page, verbatim, in
    data/cache/raw/ (SHA-256-attested, the exhibit),
  - normalized anchors in data/cache/anchors.csv (working copy),
  - an audit-ledger entry (when/from where/how/with what) in data/provenance.jsonl
    as a tamper-evident hash chain.

Recorded per page: HTTP method, final URL, status, the ``Date`` header attested
by the RIPE *server* (independent time source besides the local clock) and
``Content-Type``. The complete response headers are additionally in the exhibit.

Anchor fields: id, ip_v4, as_v4, city, country, geometry.coordinates = [lon, lat]
(!), is_disabled. Public infrastructure IPs only; no personal data.

Invocations:
  python data/fetch_anchors.py                # fetch active anchors + record evidence
  python data/fetch_anchors.py --all          # also include disabled anchors
  python data/fetch_anchors.py --max-pages 1  # for tests: limit pages
  python data/fetch_anchors.py --verify       # only check the provenance ledger
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
    lon, lat = coords[0], coords[1]  # GeoJSON is [lon, lat] -> swap when mapping!
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
    """Fetch anchors with pagination.

    Returns: (records, evidence, api_count)
      records   normalized anchor dicts (ANCHOR_COLUMNS), possibly active only
      evidence  list of the full HTTP exchanges per page (method, URL,
                status, headers, body) — verbatim, for the exhibit
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
            "url": resp.url,                       # final URL (after possible redirects)
            "status": resp.status_code,
            "server_date": resp.headers.get("Date"),       # attested by the server
            "content_type": resp.headers.get("Content-Type"),
            "headers": dict(resp.headers),         # complete headers (into the exhibit)
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
    """Compact, easily readable page overview for the ledger (without the bodies)."""
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
        print("Ledger empty — no fetch documented yet.")
        return
    for f in findings:
        mark = "OK " if f["ok"] else "!! "
        detail = "intact" if f["ok"] else "; ".join(f["problems"])
        print(f"{mark}#{f['line']} {f['fetch_id']}: {detail}")
    bad = [f for f in findings if not f["ok"]]
    print(f"\n{len(findings)} entry/entries, {len(bad)} with findings.")
    sys.exit(1 if bad else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch RIPE Atlas anchors (with provenance).")
    ap.add_argument("--all", action="store_true", help="also include disabled anchors")
    ap.add_argument("--max-pages", type=int, default=None, help="for tests: limit pages")
    ap.add_argument("--verify", action="store_true", help="only check the provenance ledger")
    args = ap.parse_args()

    if args.verify:
        _run_verify()
        return

    active_only = not args.all
    fetch_id = store.new_fetch_id()
    fetched_at = store.utc_now_iso()

    records, evidence, api_count = fetch_anchors(
        active_only=active_only, max_pages=args.max_pages)

    # save the full HTTP response (headers + body) verbatim + hash it
    raw_bytes = json.dumps(evidence, ensure_ascii=False,
                           separators=(",", ":")).encode("utf-8")
    raw_path, sha_raw = store.save_raw(fetch_id, raw_bytes, label="anchors")

    # normalized working copy
    csv_path = store.save_anchors_csv(records)

    # audit-ledger entry (when / from where / how / with what)
    pages = _page_summary(evidence)
    entry = store.append_provenance({
        "fetch_id": fetch_id,
        "fetched_at_utc": fetched_at,           # local clock (fetch time)
        "source_url": ANCHORS_URL,
        "http_method": "GET",
        "pages": pages,                         # per page: method/URL/status/server date
        "tool": "data/fetch_anchors.py",
        "tool_git_commit": store.git_commit(),
        "n_pages": len(evidence),
        "api_count_field": api_count,
        "n_records_saved": len(records),
        "active_only": active_only,
        "raw_file": str(raw_path.relative_to(store.BASE_DIR)),
        "sha256_raw": sha_raw,                  # covers headers + body
        "sha256_records": store.sha256_json(records),
    })

    server_date = pages[0]["server_date"] if pages else "?"
    print(f"Fetch {fetch_id}: {len(records)} anchors saved "
          f"(API reports {api_count} total, {len(evidence)} page(s)).")
    print(f"  Method   : GET  |  server date (page 1): {server_date}")
    print(f"  Raw file : {raw_path}  sha256={sha_raw[:16]}…")
    print(f"  Anchors  : {csv_path}")
    print(f"  Ledger   : {store.PROVENANCE_FILE}  entry_sha256={entry['entry_sha256'][:16]}…")


if __name__ == "__main__":
    main()
