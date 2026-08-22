"""Obtain comparison ground truth: RIPE Atlas PROBES (separate run).

Unlike the anchors (data centers, documented locations), the probes also cover
home/mobile/NAT connections -- the forensically more typical, harder class.
This run is deliberately kept SEPARATE (GEOIP_DATASET=probes -> its own cache/
ledger files) so that the anchor evaluation remains untouched.

IMPORTANT CAVEAT: probe coordinates are SELF-reported by the volunteer and
privacy-rounded by RIPE -- the ground truth is thus noisier than for anchors.
Higher errors against probes mix (a) harder-to-locate IPs and (b) less accurate
GT; the comparison is a directional stress test, not a clean validation.

Schema identical to anchors.csv (data/store.ANCHOR_COLUMNS), plus a tags column
in a companion file probes_tags.csv (ip -> tags) for later stratification
(home / nat / datacenter / mobile).

Usage:
  python data/fetch_probes.py --n 500            # draw 500 probes (seed 0)
  python data/fetch_probes.py --n 500 --seed 0   # reproducible sample
  python data/fetch_probes.py --verify           # verify the probe provenance ledger
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
    """True only for globally routable IPv4 (no RFC1918/CGNAT/loopback/link-local)."""
    if not addr:
        return False
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return ip.version == 4 and ip.is_global


def _normalize(p: dict) -> dict:
    coords = (p.get("geometry") or {}).get("coordinates") or [None, None]
    lon, lat = coords[0], coords[1]   # GeoJSON [lon, lat] -> swap
    return {
        "anchor_id": p.get("id"),                 # schema-compatible (probe id)
        "ip": p.get("address_v4"),
        "lat": lat,
        "lon": lon,
        "city": None,                             # probes do not provide a city
        "country": p.get("country_code"),
        "asn": p.get("asn_v4"),
        "hostname": None,
        "is_disabled": p.get("status", {}).get("id") != 1,   # 1 = Connected
        "_tags": ";".join(t.get("slug", "") for t in (p.get("tags") or [])),
    }


def fetch_pool_union(tags: list[str], max_pages: int | None = None):
    """Union the pool over MULTIPLE tags (OR combination, deduplicated by probe ID).

    The RIPE API combines multiple ``tags=`` values with AND; the mobile class
    however needs the union of ``mobile``/``4g``/``lte``/… (the tags are assigned
    by volunteers and only partially overlap). One run per tag, then
    deduplication.

    Returns: (records, evidence_summary, api_counts) with ``api_counts`` = {tag: count}.
    """
    union: dict[int, dict] = {}
    evidence: list[dict] = []
    api_counts: dict[str, int | None] = {}
    for t in tags:
        recs, ev, api = fetch_probe_pool(max_pages=max_pages, tag=t)
        api_counts[t] = api
        evidence.extend(ev)
        for r in recs:
            union.setdefault(r["anchor_id"], r)
    return list(union.values()), evidence, api_counts


def fetch_probe_pool(max_pages: int | None = None, tag: str | None = None):
    """Fetch all connected non-anchor probes with a public IPv4 + coordinates.

    ``tag`` optional: server-side filter on one probe tag (e.g. ``mobile``).
    Returns: (records, evidence_summary, api_count).
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
    ap = argparse.ArgumentParser(description="Fetch RIPE Atlas probes (separate comparison run).")
    ap.add_argument("--n", type=int, default=500, help="sample size (default 500)")
    ap.add_argument("--seed", type=int, default=0, help="seed for reproducible sampling")
    ap.add_argument("--max-pages", type=int, default=None, help="for tests: limit pages")
    ap.add_argument("--tag", default=None, help="server-side tag filter (e.g. mobile)")
    ap.add_argument("--tags", default=None,
                    help="comma list of tags, OR-combined (e.g. mobile,4g,lte,5g,3g,t-mobile)")
    ap.add_argument("--exclude-datasets", default=None,
                    help="comma list of existing datasets (e.g. probes,probes_mobile_ext); "
                         "their probe IDs AND IPs are excluded before sampling — "
                         "for a true, never-seen holdout")
    ap.add_argument("--require-tags", default=None,
                    help="comma list: probe must carry ALL of these tags "
                         "(e.g. system-ipv4-stable-30d; filter in the style of Nabi et al.)")
    ap.add_argument("--exclude-tags", default=None,
                    help="comma list: probe must carry NONE of these tags "
                         "(e.g. system-geoloc-disputed)")
    ap.add_argument("--drop-shared-ips", action="store_true",
                    help="drop IPs that appear for MULTIPLE probes "
                         "(CGNAT/relocations; ground-truth ambiguity, cf. Nabi et al.)")
    ap.add_argument("--verify", action="store_true", help="only verify the probe provenance ledger")
    args = ap.parse_args()

    if store.DATASET == "anchors":
        print("Note: without GEOIP_DATASET=<probes|probes_mobile|...> this writes into the anchor paths!")
        print("Usage:  GEOIP_DATASET=probes python data/fetch_probes.py --n 500")
        print("    or:  GEOIP_DATASET=probes_mobile python data/fetch_probes.py --tag mobile --n 100")
        sys.exit(2)

    if args.verify:
        for f in store.verify_chain():
            print(("OK " if f["ok"] else "!! ") + f"#{f['line']} {f['fetch_id']}: "
                  + ("intact" if f["ok"] else "; ".join(f["problems"])))
        return

    if args.tag and args.tags:
        ap.error("--tag and --tags are mutually exclusive")

    fetch_id = store.new_fetch_id()
    fetched_at = store.utc_now_iso()
    tag_list = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None
    if tag_list:
        pool, evidence, api_count = fetch_pool_union(tag_list, max_pages=args.max_pages)
    else:
        pool, evidence, api_count = fetch_probe_pool(max_pages=args.max_pages, tag=args.tag)
    print(f"Pool: {len(pool)} eligible probes (connected, public IPv4, coordinates) "
          f"out of API total {api_count}.")

    if args.require_tags:
        need = {t.strip() for t in args.require_tags.split(",") if t.strip()}
        before = len(pool)
        pool = [r for r in pool if need <= set(r["_tags"].split(";"))]
        print(f"require-tags {sorted(need)}: {before - len(pool)} removed -> {len(pool)}.")
    if args.exclude_tags:
        ban = {t.strip() for t in args.exclude_tags.split(",") if t.strip()}
        before = len(pool)
        pool = [r for r in pool if not (ban & set(r["_tags"].split(";")))]
        print(f"exclude-tags {sorted(ban)}: {before - len(pool)} removed -> {len(pool)}.")
    if args.drop_shared_ips:
        from collections import Counter
        ip_counts = Counter(r["ip"] for r in pool)
        before = len(pool)
        pool = [r for r in pool if ip_counts[r["ip"]] == 1]
        print(f"drop-shared-ips: {before - len(pool)} removed (IP at >1 probe) -> {len(pool)}.")

    excl = [d.strip() for d in args.exclude_datasets.split(",") if d.strip()] \
        if args.exclude_datasets else []
    if excl:
        seen_ids, seen_ips = set(), set()
        for ds in excl:
            path = store.CACHE_DIR / f"{ds}.csv"
            if not path.exists():
                ap.error(f"dataset not found: {path}")
            with open(path, encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    seen_ids.add(str(row.get("anchor_id")))
                    seen_ips.add(row.get("ip"))
        before = len(pool)
        pool = [r for r in pool
                if str(r["anchor_id"]) not in seen_ids and r["ip"] not in seen_ips]
        print(f"Holdout filter ({', '.join(excl)}): {before - len(pool)} already known "
              f"probes excluded -> {len(pool)} remaining.")

    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.n, len(pool)))
    sample.sort(key=lambda r: r["anchor_id"])

    # Raw evidence artifact (page overview + drawn IDs) + hash into the probe
    # hash chain
    raw = json.dumps({"pages": evidence, "seed": args.seed,
                      "sampled_ids": [r["anchor_id"] for r in sample]},
                     ensure_ascii=False).encode("utf-8")
    raw_path, sha_raw = store.save_raw(fetch_id, raw, label="probes")

    store.save_anchors_csv(sample)                       # -> cache/probes.csv (dataset-aware)
    tags_path = store.CACHE_DIR / f"{store.DATASET}_tags.csv"   # dataset-aware (do not clobber)
    with open(tags_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["ip", "tags"])
        for r in sample:
            w.writerow([r["ip"], r["_tags"]])

    entry = store.append_provenance({
        "fetch_id": fetch_id, "fetched_at_utc": fetched_at, "kind": "probes",
        "source_url": PROBES_URL, "http_method": "GET",
        "tool": "data/fetch_probes.py", "tool_git_commit": store.git_commit(),
        "n_pool": len(pool), "n_sampled": len(sample), "seed": args.seed, "tag": args.tag,
        "tags_union": tag_list, "excluded_datasets": excl or None,
        "require_tags": args.require_tags, "exclude_tags": args.exclude_tags,
        "drop_shared_ips": bool(args.drop_shared_ips),
        "api_count_field": api_count, "raw_file": str(raw_path.relative_to(store.BASE_DIR)),
        "sha256_raw": sha_raw, "sha256_records": store.sha256_json(sample)})

    print(f"Fetch {fetch_id}: {len(sample)} probes drawn (seed {args.seed}).")
    print(f"  Probes : {store.GROUND_TRUTH_CSV}")
    print(f"  Tags   : {tags_path}")
    print(f"  Ledger : {store.PROVENANCE_FILE}  entry_sha256={entry['entry_sha256'][:16]}…")
    print("\nNext step (query sources, consumes API quota):")
    print("  GEOIP_DATASET=probes python data/fetch_sources.py --all")


if __name__ == "__main__":
    main()
