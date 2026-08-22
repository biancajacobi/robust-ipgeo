#!/usr/bin/env python3
"""ipapi.co sensitivity of the headline configuration (L1·b, eps=10, linear).

Background: the 7 successful ipapi.co observations shift the headline mean
from 172.9 km (without them) to 183.4 km. This script documents the
mechanism reproducibly:
  (a) overall metrics with/without ipapi.co (median, mean, tail),
  (b) per-anchor differences of the 7 affected anchors,
  (c) weight-mass table of the tipped anchor:
      4 of 6 effective lines share the same default coordinate
      (W_C = 47.5 % without ipapi); the ipapi answer (6.8 % weight mass)
      lifts W_C to 51.0 % -> the weighted geometric median tips onto
      the default (error 18.8 -> 11,354.5 km). NO weight fallback:
      ipapi receives a regular leave-one-out pseudo-radius (27.9 km).

Output: tables (stdout) + eval/out/t6_ipapi_sensitivity.csv
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
    loo_pseudo_radii, estimate, point_radius, line_weights_for,
)

OUT = Path("eval/out")
OUT.mkdir(parents=True, exist_ok=True)
EPS = 10  # headline configuration (anchor_eps)


def headline_errors(cases):
    loo = loo_pseudo_radii(cases)
    return loo, {c["ip"]: haversine_error(estimate(c, "L1", "b", loo, eps=EPS),
                                          c["truth"]) for c in cases}


def summarize(errs):
    a = np.array(list(errs.values()))
    return dict(median_km=float(np.median(a)), mean_km=float(a.mean()),
                tail_pct=float(100 * (a > 100).mean()))


def main():
    cases = load_cases()
    ipapi_ips = [c["ip"] for c in cases
                 if any(p["source"] == "ipapi_co" for p in c["provenance"])]

    loo_full, err_full = headline_errors(cases)
    cases_no = copy.deepcopy(cases)
    for c in cases_no:
        c["provenance"] = [p for p in c["provenance"]
                           if p["source"] != "ipapi_co"]
    _, err_no = headline_errors(cases_no)

    rows = []
    for tag, e in (("with_ipapi", err_full), ("without_ipapi", err_no)):
        rows.append({"row": tag, "anchor": "", **summarize(e)})
    for ip in ipapi_ips:
        rows.append({"row": "anchor_delta", "anchor": ip,
                     "median_km": "", "mean_km": "",
                     "tail_pct": "",
                     "err_with": round(err_full[ip], 1),
                     "err_without": round(err_no[ip], 1),
                     "delta_km": round(err_full[ip] - err_no[ip], 1)})

    # weight masses of the tipped anchor
    tipped = max(ipapi_ips, key=lambda ip: abs(err_full[ip] - err_no[ip]))
    c = next(c for c in cases if c["ip"] == tipped)
    lw = line_weights_for(c["provenance"], "L1")
    w = []
    for p, l in zip(c["provenance"], lw):
        r = point_radius(p, tipped, loo_full)
        w.append((p["source"], l / (r + EPS),
                  haversine_error((p["lat"], p["lon"]), c["truth"]), r))
    tot = sum(x[1] for x in w)
    wc_with = sum(x[1] for x in w if x[2] > 10_000) / tot
    tot_no = tot - next(x[1] for x in w if x[0] == "ipapi_co")
    wc_no = sum(x[1] for x in w
                if x[2] > 10_000 and x[0] != "ipapi_co") / tot_no

    print(f"Tipped anchor: {tipped}")
    print(f"{'source':18s} {'error_km':>10s} {'r_km':>7s} {'share':>7s}")
    for src, wi, e, r in w:
        print(f"{src:18s} {e:10.1f} {r:7.1f} {100*wi/tot:6.1f}%")
    print(f"\nW_C (default side) with ipapi: {100*wc_with:.1f} %"
          f"  |  without ipapi: {100*wc_no:.1f} %")
    rows.append({"row": "weight_mass_default", "anchor": tipped,
                 "median_km": "", "mean_km": "", "tail_pct": "",
                 "wc_with_pct": round(100 * wc_with, 1),
                 "wc_without_pct": round(100 * wc_no, 1)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t6_ipapi_sensitivity.csv", index=False)
    print(f"\nCSV: {OUT}/t6_ipapi_sensitivity.csv")
    for tag, e in (("with   ", err_full), ("without", err_no)):
        s = summarize(e)
        print(f"{tag} ipapi: median {s['median_km']:.2f}  "
              f"mean {s['mean_km']:.1f}  tail {s['tail_pct']:.1f} %")


if __name__ == "__main__":
    main()
