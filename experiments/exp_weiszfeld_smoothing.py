"""Numerical control of the Weiszfeld smoothing and iteration cap (Section 3.2).

The weighted geometric median is solved by a smoothed Weiszfeld iteration
(max(nu, ||x - x_i||) in the denominator, nu = 1e-3 degrees ~ 0.1 km), stopped
when a step falls below 1e-9 degrees and capped at 500 iterations. This script
reports the iteration counts of the headline settings (how often the cap is hit)
and measures the deviation of the headline configuration L1*b from practically
unsmoothed, fully converged runs (nu = 1e-3/1e-6/1e-9 degrees, tolerance 1e-12,
up to 5,000 iterations) on the 1,077 anchors: median/mean/tail per nu and the
shift of the estimates.

Usage:  python experiments/exp_weiszfeld_smoothing.py
Output: stdout + eval/out/weiszfeld_smoothing.csv (IP-free)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.pipeline import load_cases                 # noqa: E402
from eval.metrics import haversine_error             # noqa: E402
import experiments.exp_t6_defaults as T6             # noqa: E402
import estimators.baselines as B                     # noqa: E402

OUT = Path("eval/out")
NUS = (1e-3, 1e-6, 1e-9)
HEADLINE = dict(nu=1e-3, eps=1e-9, max_iter=500)   # Einstellungen der Headline (estimators/baselines.py)


def _gm_with_count(points, weights=None, nu=1e-3, eps=1e-9, max_iter=500, counter=None):
    """Weiszfeld as in estimators.baselines, additionally with an iteration count."""
    pts = B._as_points(points)
    a = np.ones(len(pts)) if weights is None else np.asarray(weights, dtype=float)
    y = (pts * a[:, None]).sum(axis=0) / a.sum()
    for k in range(1, max_iter + 1):
        d = np.linalg.norm(pts - y, axis=1)
        w = a / np.maximum(nu, d)
        y_new = (pts * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(y_new - y) < eps:
            if counter is not None: counter.append(k)
            return y_new
        y = y_new
    if counter is not None: counter.append(max_iter)
    return y


def main() -> None:
    cases = load_cases()
    loo = T6.loo_pseudo_radii(cases)
    eps = min(T6.EPS_GRID, key=lambda e: abs(e - float(np.median([loo[s]["_global"] for s in loo]))))
    orig = B.weighted_geometric_median
    runs = {}
    # headline settings (nu=1e-3, tolerance 1e-9, max. 500 iterations) with iteration count
    counter = []
    T6.weighted_geometric_median = lambda p, w: _gm_with_count(p, w, counter=counter, **HEADLINE)
    est = [T6.estimate(c, "L1", "b", loo, eps=eps) for c in cases]
    err = np.array([haversine_error(e, c["truth"]) for e, c in zip(est, cases)])
    runs["headline"] = (est, err)
    it = np.array(counter)
    print(f"Headline-Einstellungen: Iterationen Median {np.median(it):.0f}, p99 {np.percentile(it, 99):.0f}, "
          f"Grenze {HEADLINE['max_iter']} erreicht auf {(it >= HEADLINE['max_iter']).sum()} Anchors")
    for nu in NUS:
        T6.weighted_geometric_median = lambda p, w, _nu=nu: orig(p, w, nu=_nu, eps=1e-12, max_iter=5000)
        est = [T6.estimate(c, "L1", "b", loo, eps=eps) for c in cases]
        err = np.array([haversine_error(e, c["truth"]) for e, c in zip(est, cases)])
        runs[nu] = (est, err)
    T6.weighted_geometric_median = orig
    ref_est = runs[NUS[-1]][0]
    rows = [{"nu_deg": "headline(1e-3, tol 1e-9, max 500)", "iter_median": int(np.median(it)), "iter_p99": int(np.percentile(it, 99)),
             "iter_cap_hit_n": int((it >= HEADLINE["max_iter"]).sum())}]
    print(f"Cases: {len(cases)}, headline eps = {eps} km, reference nu = {NUS[-1]:g} degrees (tolerance 1e-12, max. 5,000 iterations)")
    for nu, (est, err) in runs.items():
        mv = np.array([haversine_error(a, b) for a, b in zip(est, ref_est)])
        row = {"nu_deg": nu, "median_km": round(float(np.median(err)), 4), "mean_km": round(float(err.mean()), 3),
               "tail_pct": round(100 * float((err > 100).mean()), 3),
               "shift_vs_ref_max_km": round(float(mv.max()), 4), "shift_vs_ref_median_km": round(float(np.median(mv)), 6),
               "shift_gt_0p1km_n": int((mv > 0.1).sum()), "shift_gt_1km_n": int((mv > 1).sum())}
        rows.append(row)
        label = nu if isinstance(nu, str) else f"ν={nu:g}"
        print(f"  {label}: Median {row['median_km']:.3f} | Mittel {row['mean_km']:.1f} | Tail {row['tail_pct']:.2f} % | "
              f"Verschiebung zu ν={NUS[-1]:g}: max {row['shift_vs_ref_max_km']:.3f} km, >0,1 km: {row['shift_gt_0p1km_n']}")
    pd.DataFrame(rows).to_csv(OUT / "weiszfeld_smoothing.csv", index=False)
    print(f"CSV: {OUT}/weiszfeld_smoothing.csv")


if __name__ == "__main__":
    main()
