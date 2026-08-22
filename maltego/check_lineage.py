#!/usr/bin/env python3
"""Show line coverage + redundancy check for NEW sources.

The line assignment (``lineage`` in data/fetch_sources.SOURCES) is a
MAINTAINED, documented decision -- nothing is computed at runtime.
Whoever adds a new source (its own ``*_unverified`` token) must check whether
it in fact replicates an existing line; otherwise the number of effective
lines k is OVERESTIMATED and S is consequently reported as reliable too early
(MIN_LINES_FOR_S counts lines, not sources).

This script provides both:

  --list             current sources -> line table incl. evidence status
  --ips FILE         redundancy check: query all (configured) sources live on
                     the same IPs and compute, per source pair, the median
                     distance on the shared intersection (methodology as in
                     experiments/exp_source_correlation.py; requires NO
                     ground truth, only shared IPs).

Reading aid (reference points, NO automatism; corpus = eval/out/source_family_drift.csv):
  geojs           <-> MaxMind GeoLite2: 0.0 km median, 100 % identical (corpus;
                      LIVE possibly less due to DB build drift — test run: 74 %)
  reallyfreegeoip <-> MaxMind GeoLite2: 0.05 km median, 48 % identical (corpus)
  dbip            <-> ip2location:      2.6 km median, 0 % identical (INDEPENDENT)
A useful screening signal is the IDENTICAL RATE, not the median alone:
independent sources that are simply accurate typically differ by single-digit
km at city scale and rarely share bit-identical coordinates. BUT: identical
coordinates alone do NOT PROVE shared ancestry — independent providers may
return the same city centroid, country default, or another commonly known
coordinate. The check is DIAGNOSTIC, not a lineage classifier: replica
suspicion (identical rate >= 20 % or median < 0.1 km on >= 30 shared IPs)
means: research the provenance; the assignment remains a provenance decision
by the operator, which is documented.

Caution: --ips triggers LIVE queries (IP list x sources; rate limits!).

Usage:  python maltego/check_lineage.py --list
         python maltego/check_lineage.py --ips my_ips.txt [--sources a,b,c]
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from maltego.geoloc import collect                       # noqa: E402
from data.fetch_sources import SOURCES                   # noqa: E402
from eval.metrics import haversine_error                 # noqa: E402

MIN_COMMON = 30        # below this: treat the result only as a preliminary hint
IDENT_PCT = 20.0       # replica suspicion from this identical rate (rfg: 48 %)
NEAR_KM = 0.1          # replica suspicion below this median (rfg: 0.05 km)
DRIFT_KM = 1.0         # below this: possible shared origin with drift

# Evidence status per line (maintained; basis: the accompanying paper / docs/data-sources.md)
EVIDENCE = {
    "maxmind_geolite":      "shared line; geojs documented + bit-identical, "
                            "reallyfreegeoip empirical (0.05 km)",
    "dbip_lite":            "own DB (local), independent line",
    "ip2location_lite":     "own DB (local), independent line",
    "ipinfo":               "own provider, provenance disclosed",
    "ipapi_com_unverified": "provenance not disclosed -> own line as a precaution",
    "ipwhois_unverified":   "provenance not disclosed -> own line as a precaution",
    "ipapi_co_unverified":  "provenance not disclosed -> own line as a precaution",
}


def list_coverage() -> None:
    print(f"{'Source':<18s} {'Line (lineage)':<22s} Evidence status")
    print("-" * 92)
    for name, cfg in SOURCES.items():
        lin = cfg["lineage"]
        print(f"{name:<18s} {lin:<22s} {EVIDENCE.get(lin, 'UNKNOWN — check!')}")
    fams = {}
    for name, cfg in SOURCES.items():
        fams.setdefault(cfg["lineage"], []).append(name)
    k = len(fams)
    print(f"\n{len(SOURCES)} sources -> {k} effective lines.")
    for lin, members in fams.items():
        if len(members) > 1:
            print(f"  Collapse: {' + '.join(members)} -> 1 line ({lin})")
    print("\nAdding a new source: assign its own *_unverified token, then check it with "
          "--ips against the\nexisting sources (and on replica suspicion adjust the "
          "token + document the decision).")


def redundancy_check(ip_file: str, sources: list[str] | None) -> None:
    ips = [l.strip() for l in Path(ip_file).read_text().splitlines()
           if l.strip() and not l.startswith("#")]
    if not ips:
        raise SystemExit(f"no IPs in {ip_file}")
    names = sources or list(SOURCES)
    print(f"Querying {len(names)} source(s) live on {len(ips)} IP(s) "
          f"(mind the rate limits) …")
    coords: dict[str, dict[str, tuple[float, float]]] = {n: {} for n in names}
    for i, ip in enumerate(ips, 1):
        obs, failed, _ = collect(ip, sources=names)
        for o in obs:
            coords[o["source"]][ip] = (o["lat"], o["lon"])
        if i % 10 == 0:
            print(f"  … {i}/{len(ips)}")

    print(f"\n{'Pair':<40s} {'n_com':>5s} {'median km':>9s} {'ident.%':>7s} "
          f"{'>100km%':>7s}  Assessment")
    print("-" * 100)
    rows = []
    for a, b in combinations(names, 2):
        common = sorted(set(coords[a]) & set(coords[b]))
        if not common:
            rows.append((float("inf"), f"{a} <-> {b}", 0, None, None, None))
            continue
        d = np.array([haversine_error(coords[a][ip], coords[b][ip]) for ip in common])
        rows.append((float(np.median(d)), f"{a} <-> {b}", len(common),
                     float(np.median(d)), float(100 * (d < 0.001).mean()),
                     float(100 * (d > 100).mean())))
    for med, pair, n, median, ident, far in sorted(rows):
        if median is None:
            print(f"{pair:<40s} {n:>5d} {'—':>9s} {'—':>7s} {'—':>7s}  no shared IPs")
            continue
        if ident >= IDENT_PCT or median < NEAR_KM:
            verdict = "REPLICA SUSPICION -> consider a shared lineage token"
        elif median < DRIFT_KM:
            verdict = ("very close at a low identical rate -> drift of the same "
                       "basis possible, enlarge the sample")
        else:
            verdict = ("no indication of a shared line (independent sources "
                       "typically differ by single-digit km at city scale)")
        if n < MIN_COMMON:
            verdict += f" [n<{MIN_COMMON}: preliminary hint only]"
        same = SOURCES.get(pair.split(" <-> ")[0], {}).get("lineage") == \
            SOURCES.get(pair.split(" <-> ")[1], {}).get("lineage")
        if same:
            verdict = "already the same line (control)" + \
                (f" [n<{MIN_COMMON}]" if n < MIN_COMMON else "")
        print(f"{pair:<40s} {n:>5d} {median:>9.2f} {ident:>7.1f} {far:>7.1f}  {verdict}")
    print("\nNote: the check is diagnostic, NOT a lineage classifier — "
          "identical coordinates\ncan also be shared city centroids/"
          "country defaults. The assignment remains a documented\n"
          "provenance decision (cf. experiments/exp_source_correlation.py "
          "and the accompanying paper).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true",
                    help="show the current source->line coverage")
    ap.add_argument("--ips", default=None,
                    help="file with IPs (one per line) for the redundancy check")
    ap.add_argument("--sources", default=None,
                    help="comma list; default: all configured sources")
    args = ap.parse_args()
    if not args.list and not args.ips:
        args.list = True                       # no arguments: show coverage
    if args.list:
        list_coverage()
    if args.ips:
        srcs = [s.strip() for s in args.sources.split(",")] if args.sources else None
        unknown = [s for s in (srcs or []) if s not in SOURCES]
        if unknown:
            raise SystemExit(f"unknown source(s): {', '.join(unknown)}")
        if args.list:
            print()
        redundancy_check(args.ips, srcs)


if __name__ == "__main__":
    main()
