#!/usr/bin/env python3
"""Sensitivity of the line definition: reallyfreegeoip inside the MaxMind line?

Background: reallyfreegeoip.org follows the MaxMind lineage with a diverging
data snapshot (48 % identical, 17 % > 100 km divergent). This script checks
whether the full line collapse biases the headline results:

  full          : status quo — rfg belongs to the MaxMind GeoLite line (k=3)
  own_line      : rfg counts as its own line (family only maxmind+geojs, k=2)
  conditional   : rfg in the family only where coordinates < 1 km from MaxMind

Finding: median/tail practically invariant; mean differences are pure
far-tail reshuffling of already-failed anchors; without the collapse at
least one previously correct anchor tips through the full replicate vote.

Output: tables (stdout) + eval/out/line_sensitivity_rfg.csv
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
EPS = 10          # headline configuration
SPLIT_KM = 1.0    # threshold for the conditional collapse
RFG = "reallyfreegeoip"
MM = "maxmind_geolite2"


def errors(cases, loo):
    return {c["ip"]: haversine_error(estimate(c, "L1", "b", loo, eps=EPS),
                                     c["truth"]) for c in cases}


def summarize(errs):
    a = np.array(list(errs.values()))
    return dict(median_km=round(float(np.median(a)), 2),
                mean_km=round(float(a.mean()), 1),
                tail_pct=round(float(100 * (a > 100).mean()), 1))


def variant_own_line(cases):
    cs = copy.deepcopy(cases)
    for c in cs:
        for p in c["provenance"]:
            if p["source"] == RFG:
                p["lineage"] = "rfg_own_line"
    return cs


def variant_conditional(cases):
    cs = copy.deepcopy(cases)
    n_split = 0
    for c in cs:
        src = {p["source"]: p for p in c["provenance"]}
        mm, rf = src.get(MM), src.get(RFG)
        if mm and rf and haversine_error((mm["lat"], mm["lon"]),
                                         (rf["lat"], rf["lon"])) >= SPLIT_KM:
            rf["lineage"] = "rfg_own_line"
            n_split += 1
    return cs, n_split


def main():
    cases = load_cases()
    loo = loo_pseudo_radii(cases)   # radii identical for all variants

    e_full = errors(cases, loo)
    e_own = errors(variant_own_line(cases), loo)
    cond_cases, n_split = variant_conditional(cases)
    e_cond = errors(cond_cases, loo)

    rows = []
    for name, e in (("full", e_full), ("own_line", e_own),
                    (f"conditional_<{SPLIT_KM:g}km", e_cond)):
        rows.append({"variant": name, **summarize(e)})
        print(f"{name:16s} {rows[-1]}")

    # far-tail character of the difference full vs. own line
    d = {ip: e_own[ip] - e_full[ip] for ip in e_full}
    big = {ip: x for ip, x in d.items() if abs(x) > 10}
    flips = [(ip, e_full[ip], e_own[ip]) for ip in big
             if e_full[ip] < 100 < e_own[ip]]
    print(f"\n|delta|>10 km: {len(big)} of {len(d)} anchors "
          f"(median |delta| {np.median(np.abs(list(d.values()))):.3f} km)")
    for ip, ef, eo in flips:
        print(f"  FLIPS without collapse: {ip}  {ef:.1f} -> {eo:.1f} km")
        rows.append({"variant": "flip_anchor_without_collapse", "anchor": ip,
                     "err_full_km": round(ef, 1), "err_own_km": round(eo, 1)})
    rows.append({"variant": "delta_stat",
                 "n_delta_gt10km": len(big),
                 "median_abs_delta_km": round(float(np.median(np.abs(list(d.values())))), 3)})

    pd.DataFrame(rows).to_csv(OUT / "line_sensitivity_rfg.csv", index=False)
    print(f"\nCSV: {OUT}/line_sensitivity_rfg.csv  (conditional: {n_split} IPs split)")


if __name__ == "__main__":
    main()
