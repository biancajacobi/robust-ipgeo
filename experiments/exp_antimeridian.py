#!/usr/bin/env python3
"""Antimeridian edge case: map dependence of raw-degree aggregation.

In raw-degree space the aggregation is map-dependent at ±180° (a pair at
179°/−179° appears 358° apart instead of 2°). This script checks the
empirical relevance on the anchor sample:

  (a) affected anchors: observation longitudes with a raw-degree span > 180°,
  (b) true (circular) longitude span per case (> 180° ⇒ no canonical map),
  (c) control computation: L1·b on per-anchor re-centred longitudes
      (unwrap around the first observation) against the raw-degree status quo.

Finding: all affected cases arise from far-away default outliers (raw-degree
space OVERSTATES their distance → stronger robust down-weighting,
pro-robust); honest clusters at the antimeridian do not occur. Re-centring
changes the estimates only by kilometres — except where the true span is
> 180°: there re-centring itself is not well-defined and can degrade
drastically.

Output: table (stdout) + eval/out/antimeridian_check.csv
"""
from __future__ import annotations

import copy
import sys
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
EPS = 10


def wrap(x):
    """Longitude to (−180, 180]."""
    return (x + 180.0) % 360.0 - 180.0


def true_span(lons):
    """Circular longitude span: 360° minus the largest gap between longitudes."""
    s = np.sort(np.mod(lons, 360.0))
    gaps = np.diff(np.concatenate([s, [s[0] + 360.0]]))
    return float(360.0 - gaps.max())


def main():
    cases = load_cases()
    loo = loo_pseudo_radii(cases)

    rows = []
    for c in cases:
        lons = np.array([p["lon"] for p in c["provenance"]])
        raw_span = float(lons.max() - lons.min())
        if raw_span <= 180.0:
            continue
        e_raw = haversine_error(estimate(c, "L1", "b", loo, eps=EPS), c["truth"])
        cc = copy.deepcopy(c)
        ref = cc["provenance"][0]["lon"]
        for p in cc["provenance"]:
            p["lon"] = ref + wrap(p["lon"] - ref)
        est = estimate(cc, "L1", "b", loo, eps=EPS)
        e_rec = haversine_error((est[0], wrap(est[1])), cc["truth"])
        rows.append({"ip": c["ip"],
                     "raw_span_deg": round(raw_span, 1),
                     "true_span_deg": round(true_span(lons), 1),
                     "err_raw_km": round(e_raw, 1),
                     "err_recentred_km": round(e_rec, 1),
                     "delta_km": round(e_rec - e_raw, 1)})

    df = pd.DataFrame(rows).sort_values("delta_km", ascending=False)
    df.to_csv(OUT / "antimeridian_check.csv", index=False)
    print(f"Anchors with raw-degree span > 180°: {len(df)} of {len(cases)}")
    print(f"of which true span > 180° (no canonical map): "
          f"{int((df['true_span_deg'] > 180).sum())}")
    print(df.to_string(index=False))
    print(f"\nCSV: {OUT}/antimeridian_check.csv")


if __name__ == "__main__":
    main()
