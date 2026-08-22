#!/usr/bin/env python3
"""How many effective lines does the aggregation need? (Ablation after line collapse)

Question: are fewer sources enough? The unit here is NOT
the API count but the effective line after the lineage collapse —
a subset of 3 MaxMind descendants would effectively be ONE line.

Design: on the anchors, all subsets of the fully covered lines are computed
exhaustively (no sampling): for each subset of size k, the provenance of each
case is filtered to the included lines and the headline configuration L1·b
(ε = 10) is evaluated. Identical case basis for all subsets (every anchor
covers all included lines) → no subset artifact. ipapi_co is excluded
(7/1077 coverage, rate-limited; precedent: exp_source_correlation).

Per k, in addition to the mean over the subsets, the best/worst case is
reported: "from k=… on, EVERY line selection stays within … km" is the
stronger statement. Additionally the subset WITHOUT ipwhois is reported —
these are exactly the 5 lines that are complete in the probes_pool dataset.

LOO pseudo-radii are defined per source and independent of the subset choice;
they are determined once on the full case basis.

Usage:  python experiments/exp_min_lines.py
Result: tables (stdout) + eval/out/min_lines.csv (one row per subset)
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.pipeline import load_cases                     # noqa: E402
from eval.metrics import haversine_error                 # noqa: E402
from experiments.exp_t6_defaults import (                # noqa: E402
    loo_pseudo_radii, estimate,
)

OUT = Path("eval/out")
OUT.mkdir(parents=True, exist_ok=True)
EPS = 10                      # headline configuration (as in exp_line_sensitivity)
EXCLUDED = "ipapi_co_unverified"
POOL5 = ("dbip_lite", "ip2location_lite", "ipapi_com_unverified",
         "ipinfo", "maxmind_geolite")   # lines 100 % present in probes_pool


def subset_errors(cases, loo, lineages: frozenset) -> np.ndarray:
    errs = []
    for c in cases:
        prov = [p for p in c["provenance"] if p["lineage"] in lineages]
        sub = {"ip": c["ip"], "provenance": prov}
        est = estimate(sub, "L1", "b", loo, eps=EPS)
        errs.append(haversine_error(est, c["truth"]))
    return np.asarray(errs, dtype=float)


def summarize(errs: np.ndarray) -> dict:
    return dict(median_km=float(np.median(errs)),
                mean_km=float(errs.mean()),
                tail_pct=float(100 * (errs > 100).mean()))


def main() -> None:
    cases = load_cases()
    # Drop ipapi_co (coverage) — from provenance AND thus from the LOO radii.
    for c in cases:
        c["provenance"] = [p for p in c["provenance"] if p["lineage"] != EXCLUDED]
    lines = sorted({p["lineage"] for c in cases for p in c["provenance"]})
    covered = [c for c in cases
               if {p["lineage"] for p in c["provenance"]} == set(lines)]
    print(f"Cases: {len(cases)}, of which with all {len(lines)} lines: {len(covered)}")
    print(f"Lines: {', '.join(lines)}  (excluded: {EXCLUDED})")
    loo = loo_pseudo_radii(covered)

    rows = []
    for k in range(1, len(lines) + 1):
        for combo in combinations(lines, k):
            s = summarize(subset_errors(covered, loo, frozenset(combo)))
            rows.append({"k": k, "lines": "+".join(combo), **s})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "min_lines.csv", index=False)

    full = df[df.k == len(lines)].iloc[0]
    print("\n" + "=" * 78)
    print("ERROR vs. NUMBER OF EFFECTIVE LINES k — L1·b, ε=10, exhaustive (anchors)")
    print(f"Reference k={len(lines)}: median {full.median_km:.2f} km, "
          f"mean {full.mean_km:.1f} km, tail>100 km {full.tail_pct:.1f} %")
    print("=" * 78)
    print(f"{'k':>2s} {'#subsets':>8s} | {'Median: best':>12s} {'mean':>7s} "
          f"{'worst':>7s} | {'Tail%: best':>11s} {'worst':>6s} | worst subset (median)")
    for k, g in df.groupby("k"):
        w = g.loc[g.median_km.idxmax()]
        print(f"{k:2d} {len(g):8d} | {g.median_km.min():12.2f} {g.median_km.mean():7.2f} "
              f"{g.median_km.max():7.2f} | {g.tail_pct.min():11.1f} {g.tail_pct.max():6.1f} "
              f"| {w['lines']}")

    pool5 = df[df["lines"] == "+".join(sorted(POOL5))]
    print("\nPOOL-RELEVANT SUBSET (the 5 lines complete in probes_pool, without ipwhois):")
    if pool5.empty:
        print("  not found — check the line names!")
    else:
        p = pool5.iloc[0]
        print(f"  Median {p.median_km:.2f} km (full: {full.median_km:.2f}), "
              f"mean {p.mean_km:.1f} km (full: {full.mean_km:.1f}), "
              f"tail {p.tail_pct:.1f} % (full: {full.tail_pct:.1f} %)")
    print(f"\nCSV: {OUT / 'min_lines.csv'}")


if __name__ == "__main__":
    main()
