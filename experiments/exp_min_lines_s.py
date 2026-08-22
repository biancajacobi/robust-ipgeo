#!/usr/bin/env python3
"""Does the risk signal S hold up with fewer effective lines? (BSS by k)

Counterpart to exp_min_lines.py: there, the POINT ESTIMATOR is ablated over
all line subsets; here, the CALIBRATION QUALITY of the support concentration
S. Background: the demonstrator's operating rule (MIN_LINES_FOR_S = 3,
S abstention below three effective lines) has so far only been justified
qualitatively (in the limiting case of one line, S is trivially 1). This
script measures whether and where the forecast quality actually breaks down.

Design: identical exhaustive subset basis as exp_min_lines.py (63 subsets of
the 6 fully covered lines, 1,077 anchors, ipapi_co excluded). Per subset, S
is formed as in exp_support_concentration.py (line-weighted core share within
radius r = 50 km around the L1·b estimator, ε by the same LOO median rule,
determined once on the full basis) and evaluated as an OOF logit forecast for
the event "error > 100 km" (10-fold stratified, seed 0): BSS against the
climatology of the RESPECTIVE subset (the miss base rate rises as k falls —
the reference moves with it), plus ECE and the number of distinguishable
S values (tie/degeneration indicator; at k = 1, S is nearly constant ≈ 1).

Fair view: calibration FRESH per subset (a user with k lines also calibrates
on k lines); the question of frozen calibration is the business of the
transfer experiment, not of this ablation.

Usage:  python experiments/exp_min_lines_s.py
Result: tables (stdout) + eval/out/min_lines_s.csv (one row per subset)
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.pipeline import load_cases                       # noqa: E402
from eval.metrics import haversine_error                   # noqa: E402
import experiments.exp_t6_defaults as T6                   # noqa: E402
from experiments.exp_support_concentration import (        # noqa: E402
    oof_predict, bss, ece, TAU, R_HEADLINE,
)
from experiments.exp_min_lines import EXCLUDED, POOL5      # noqa: E402

OUT = Path("eval/out")
OUT.mkdir(parents=True, exist_ok=True)


def s_and_err(cases, loo, eps, lineages: frozenset):
    """S (line-weighted core share, r=R_HEADLINE) + error per case,
    provenance filtered to the subset."""
    S, err = [], []
    for c in cases:
        prov = [p for p in c["provenance"] if p["lineage"] in lineages]
        sub = {"ip": c["ip"], "provenance": prov}
        est = np.array(T6.estimate(sub, "L1", "b", loo, eps=eps))
        pts = np.array([[p["lat"], p["lon"]] for p in prov], float)
        w = np.asarray(T6.line_weights_for(prov, "L1"), float)
        w = w / w.sum()
        d = np.array([haversine_error(tuple(p), tuple(est)) for p in pts])
        S.append(float(w[d < R_HEADLINE].sum()))
        err.append(haversine_error(tuple(est), c["truth"]))
    return np.array(S), np.array(err)


def main() -> None:
    cases = load_cases()
    for c in cases:
        c["provenance"] = [p for p in c["provenance"] if p["lineage"] != EXCLUDED]
    lines = sorted({p["lineage"] for c in cases for p in c["provenance"]})
    covered = [c for c in cases
               if {p["lineage"] for p in c["provenance"]} == set(lines)]
    loo = T6.loo_pseudo_radii(covered)
    eps = min(T6.EPS_GRID,
              key=lambda e: abs(e - float(np.median([loo[s]["_global"] for s in loo]))))
    print(f"Cases: {len(covered)}, lines: {len(lines)}, ε = {eps} "
          f"(LOO median rule, full basis), r = {R_HEADLINE} km, τ = {TAU} km")

    rows = []
    for k in range(1, len(lines) + 1):
        for combo in combinations(lines, k):
            S, err = s_and_err(covered, loo, eps, frozenset(combo))
            y = (err > TAU).astype(float)
            df = pd.DataFrame({"S": S})
            if 0 < y.sum() < len(y) and len(np.unique(S)) > 1:
                ph = oof_predict(df, ["S"], y)
                b, e = bss(ph, y), ece(ph, y)
            else:                       # degenerate: constant S or single-class
                b, e = float("nan"), float("nan")
            rows.append({"k": k, "lines": "+".join(combo),
                         "miss_rate_pct": float(100 * y.mean()),
                         "bss_oof": b, "ece": e,
                         "n_s_unique": int(len(np.unique(np.round(S, 6))))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "min_lines_s.csv", index=False)

    full = df[df.k == len(lines)].iloc[0]
    print("\n" + "=" * 78)
    print("S CALIBRATION vs. NUMBER OF EFFECTIVE LINES k — OOF BSS, exhaustive")
    print(f"Reference k={len(lines)}: BSS {full.bss_oof:+.3f}, ECE {full.ece:.3f}, "
          f"miss rate {full.miss_rate_pct:.1f} %, |S values| {full.n_s_unique}")
    print("=" * 78)
    print(f"{'k':>2s} {'#subsets':>8s} | {'BSS: best':>9s} {'mean':>7s} {'worst':>7s} "
          f"| {'ECE ⌀':>6s} | {'Miss% ⌀':>7s} | {'|S| ⌀':>6s} | worst subset (BSS)")
    for k, g in df.groupby("k"):
        v = g.dropna(subset=["bss_oof"])
        if v.empty:
            print(f"{k:2d} {len(g):8d} | {'—':>9s} {'—':>7s} {'—':>7s} |    —   "
                  f"| {g.miss_rate_pct.mean():7.1f} | {g.n_s_unique.mean():6.0f} "
                  f"| all degenerate (S constant)")
            continue
        w = v.loc[v.bss_oof.idxmin()]
        deg = f" ({len(g) - len(v)} degenerate)" if len(v) < len(g) else ""
        print(f"{k:2d} {len(g):8d} | {v.bss_oof.max():+9.3f} {v.bss_oof.mean():+7.3f} "
              f"{v.bss_oof.min():+7.3f} | {v.ece.mean():6.3f} | {g.miss_rate_pct.mean():7.1f} "
              f"| {g.n_s_unique.mean():6.0f} | {w['lines']}{deg}")

    pool5 = df[df["lines"] == "+".join(sorted(POOL5))]
    if not pool5.empty:
        p = pool5.iloc[0]
        print(f"\nPOOL-RELEVANT 5-LINE SUBSET (without ipwhois): "
              f"BSS {p.bss_oof:+.3f} (full: {full.bss_oof:+.3f}), "
              f"ECE {p.ece:.3f}, miss rate {p.miss_rate_pct:.1f} %")
    print(f"\nCSV: {OUT / 'min_lines_s.csv'}")


if __name__ == "__main__":
    main()
