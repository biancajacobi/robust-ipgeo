"""Spread-measure selection for the predecessor label — which dispersion detects misses?

Background: the 2D predecessor label (exp_t6_defaults, part II) used the MEDIAN
pairwise distance of the source points as its spread axis. The median pairwise
distance is robust against a single far-off point — which is exactly why it
*overlooks* the case "one source falls onto a distant default centroid, the
rest agree": the outlier does not lift the median distance, yet the estimator
can still be off. This test systematically checks whether a spread measure that
*sees* the outlier (mean, maximum, quantiles, mean distance to the estimator)
marks the misses better — and whether the centroid flag (any vs. majority)
still contributes anything at all given a good spread measure.

Metric: miss = haversine error of the deployed estimator (L1+b) > 100 km
(same definition as run_confidence_recall). Compared threshold-free:
  - average precision (area under precision/recall; floor = base rate),
  - precision at fixed recall (fair operating-point comparison),
  - orthogonality: does the centroid flag add AP gain ON TOP of the best
    spread measure (logistic 2-feature surrogate)?

Invocation:  python experiments/exp_spread_measure.py
Result: table (stdout) + eval/out/spread_measure_comparison.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.pipeline import load_cases            # noqa: E402
from eval.metrics import haversine_error         # noqa: E402
import experiments.exp_t6_defaults as T6         # noqa: E402

OUT = Path("eval/out"); OUT.mkdir(parents=True, exist_ok=True)
MISS_KM = 100.0
FEATURES = {
    "median pairwise (old)": "med",
    "mean pairwise":         "mean",
    "max pairwise":          "mx",
    "q75 pairwise":          "q75",
    "q90 pairwise":          "q90",
    "max dist->estimator":   "maxd_est",
    "mean dist->estimator":  "mean_d_est",
}


def build_frame(cases, loo, eps):
    rows = []
    for c in cases:
        pts = np.array(c["points"], dtype=float)
        n = len(pts)
        pdd = (np.array([haversine_error(tuple(pts[i]), tuple(pts[j]))
                         for i in range(n) for j in range(i + 1, n)])
               if n > 1 else np.array([0.0]))
        est = T6.estimate(c, "L1", "b", loo, eps=eps)
        dse = np.array([haversine_error(tuple(p), est) for p in pts])
        k = int(sum(p["is_default_centroid"] for p in c["provenance"]))
        rows.append(dict(
            ip=c["ip"], med=float(np.median(pdd)), mean=float(np.mean(pdd)),
            mx=float(np.max(pdd)), q75=float(np.percentile(pdd, 75)),
            q90=float(np.percentile(pdd, 90)), maxd_est=float(np.max(dse)),
            mean_d_est=float(np.mean(dse)), any_c=int(k > 0), maj_c=int(k / n >= 0.5),
            err=haversine_error(est, c["truth"]),
        ))
    d = pd.DataFrame(rows)
    d["y"] = (d.err > MISS_KM).astype(int)
    return d


def average_precision(y, s):
    o = np.argsort(-np.asarray(s, dtype=float))
    y = np.asarray(y)[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    P = tp / (tp + fp)
    R = tp / y.sum()
    R = np.concatenate([[0.0], R])
    return float(np.sum((R[1:] - R[:-1]) * P))


def prec_at_recall(y, s, target):
    o = np.argsort(-np.asarray(s, dtype=float))
    y = np.asarray(y)[o]
    tp = np.cumsum(y)
    R = tp / y.sum()
    P = tp / np.arange(1, len(y) + 1)
    idx = np.where(R >= target)[0]
    if not len(idx):
        return float("nan"), float("nan"), 0
    i = idx[0]
    return float(P[i]) * 100, float(R[i]) * 100, int(i + 1)


def _logit_ap(d, y, cols):
    """AP of a logistic 2-/n-feature surrogate (IRLS, dependency-free)."""
    X = np.column_stack([np.log1p(d[c].values) if d[c].max() > 1 else d[c].values for c in cols])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    X = np.column_stack([np.ones(len(d)), X])
    w = np.zeros(X.shape[1])
    for _ in range(200):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        W = p * (1 - p) + 1e-9
        w -= np.linalg.solve((X * W[:, None]).T @ X + 1e-6 * np.eye(X.shape[1]),
                             X.T @ (p - y) + 1e-6 * w)
    return average_precision(y, 1.0 / (1.0 + np.exp(-X @ w)))


def run():
    cases = load_cases()
    loo = T6.loo_pseudo_radii(cases)
    eps = min(T6.EPS_GRID, key=lambda e: abs(e - float(np.median([loo[s]["_global"] for s in loo]))))
    d = build_frame(cases, loo, eps)
    y = d.y.values

    print("=" * 88)
    print(f"SPREAD-MEASURE COMPARISON as miss detector (miss = error > {MISS_KM:.0f} km)")
    print(f"n={len(d)}  misses={int(y.sum())}  base rate={y.mean():.3f} (= AP floor)")
    print("=" * 88)
    print(f"{'Spread measure':24s} {'AP':>6s}  {'Prec@Rec=0.90':>20s}  {'Prec@Rec=0.72':>20s}")
    out_rows = []
    for name, col in FEATURES.items():
        ap = average_precision(y, d[col].values)
        p9 = prec_at_recall(y, d[col].values, 0.90)
        p7 = prec_at_recall(y, d[col].values, 0.72)
        print(f"{name:24s} {ap:6.3f}  {p9[0]:5.1f}% @{p9[1]:4.1f}% (n{p9[2]:4d})  "
              f"{p7[0]:5.1f}% @{p7[1]:4.1f}% (n{p7[2]:4d})")
        out_rows.append({"measure": name, "ap": round(ap, 3),
                         "prec_at_rec90": round(p9[0], 1), "n_flag_rec90": p9[2],
                         "prec_at_rec72": round(p7[0], 1), "n_flag_rec72": p7[2]})

    print("\nOrthogonality — does the centroid flag add AP ON TOP of the best spread measure?")
    best = max(FEATURES.values(), key=lambda c: average_precision(y, d[c].values))
    combos = {f"{best}": [best], f"{best} + any": [best, "any_c"],
              f"{best} + majority": [best, "maj_c"], "median + any (old axes)": ["med", "any_c"]}
    for label, cols in combos.items():
        ap = _logit_ap(d, y, cols)
        print(f"  AP[{label:30s}] = {ap:.3f}")
        out_rows.append({"measure": f"logit:{label}", "ap": round(ap, 3),
                         "prec_at_rec90": "", "n_flag_rec90": "",
                         "prec_at_rec72": "", "n_flag_rec72": ""})

    print("\nQuadrant purity: reliable cell (low_spread & no hub), threshold 50 km")
    for col, tag in (("med", "median (old)"), ("mean", "mean (new)")):
        m = (d[col] < 50) & (d.maj_c == 0)
        e = d[m].err
        print(f"  {tag:12s}: n={m.sum():4d}  q95={np.quantile(e, .95):7.1f} km  miss%={100*(e>100).mean():4.1f}")
        out_rows.append({"measure": f"reliable cell ({tag})", "ap": "",
                         "prec_at_rec90": f"q95={np.quantile(e,.95):.0f}",
                         "n_flag_rec90": int(m.sum()),
                         "prec_at_rec72": f"miss={100*(e>100).mean():.1f}%", "n_flag_rec72": ""})

    pd.DataFrame(out_rows).to_csv(OUT / "spread_measure_comparison.csv", index=False)
    print(f"\nCSV: {OUT}/spread_measure_comparison.csv")


if __name__ == "__main__":
    run()
