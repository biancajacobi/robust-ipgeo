"""E3 / T3 — Sample size n: reducing the number of sources (RQ3).

Sub-sampling over the existing observations: per anchor, randomly draw n ∈ {3,5,7}
sources, compute estimators, measure distance errors. Bootstrapping over the random
selection (B repetitions) → IQR bands. Reference "all" = all available sources.

Methodological purpose: Braetz's class-/density-based method works on n≈15 and
assumes normality — here we show whether the robust estimators still deliver
usable reference points at n=3 (a concrete gap of the Braetz assumption).
Note: trimmed_mean (20%) only trims anything from n≥5; at n=3 == mean.

Invocation:  python experiments/exp_samplesize.py
Result: eval/out/e3_samplesize.{csv,png}
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.pipeline import load_cases, line_weights   # noqa: E402
from eval.metrics import haversine_error              # noqa: E402
from estimators.baselines import (centroid, coordinate_median, trimmed_mean,  # noqa: E402
                                  geometric_median, weighted_geometric_median)

N_VALUES = [3, 5, 7]
N_BOOT = 60
SEED = 20260604
OUT = Path("eval/out"); OUT.mkdir(parents=True, exist_ok=True)

ESTS = ["centroid", "median", "trimmed_mean", "geometric_median", "geom_median_perline"]
LABEL = {"centroid": "naive mean", "median": "coord. median", "trimmed_mean": "trimmed mean",
         "geometric_median": "geometric median", "geom_median_perline": "geometric median (line-weighted)"}
COLOR = {"centroid": "#d62728", "median": "#2ca02c", "trimmed_mean": "#8c564b",
         "geometric_median": "#1f77b4", "geom_median_perline": "#9467bd"}


def make_plot(rows, n_eligible):
    """Forest plot: y = (n, estimator) grouped, x = median error with IQR whiskers."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ypos, ylab, y = [], [], 0
    for n in N_VALUES:
        for e in ESTS:
            m, q1, q3 = rows[(e, n)]
            ax.errorbar(m, y, xerr=[[m - q1], [q3 - m]], fmt="o", color=COLOR[e],
                        capsize=3, markersize=6)
            ypos.append(y); ylab.append(f"n={n} · {LABEL[e]}"); y += 1
        y += 0.6
    ax.set_yticks(ypos, ylab, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("median distance error / km  (whiskers = IQR over bootstrap)")
    ax.set_title(f"E3 — sample size: error at n ∈ {N_VALUES}  "
                 f"(n_cases≈{n_eligible[N_VALUES[0]]}, B={N_BOOT})")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "e3_samplesize.png", dpi=150); plt.close(fig)


def load_results_csv(path=OUT / "e3_samplesize.csv"):
    rows, n_eligible = {}, {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            n = int(r["n"])
            rows[(r["estimator"], n)] = (float(r["median_km"]), float(r["q1_km"]), float(r["q3_km"]))
            n_eligible[n] = int(r["n_cases"])
    return rows, n_eligible


def estimate_all(pts, weights):
    return {
        "centroid": centroid(pts),
        "median": coordinate_median(pts),
        "trimmed_mean": trimmed_mean(pts),
        "geometric_median": geometric_median(pts),
        "geom_median_perline": weighted_geometric_median(pts, weights),
    }


def run():
    cases = load_cases()
    # per case: points, lines (for line weighting of the subsample), truth
    prepared = [(c["points"], [p["lineage"] for p in c["provenance"]], c["truth"]) for c in cases]
    rng = np.random.default_rng(SEED)
    print(f"E3 — {len(cases)} cases, n∈{N_VALUES}, B={N_BOOT}")

    results = {e: {n: [] for n in N_VALUES} for e in ESTS}
    n_eligible = {}
    for n in N_VALUES:
        elig = [(pts, lin, t) for (pts, lin, t) in prepared if len(pts) >= n]
        n_eligible[n] = len(elig)
        for _ in range(N_BOOT):
            per_est = {e: [] for e in ESTS}
            for pts, lin, truth in elig:
                idx = rng.choice(len(pts), size=n, replace=False)
                sub, sublin = pts[idx], [lin[i] for i in idx]
                w = line_weights(sublin)
                for e, est in estimate_all(sub, w).items():
                    per_est[e].append(haversine_error((est[0], est[1]), truth))
            for e in ESTS:
                results[e][n].append(float(np.median(per_est[e])))
        print(f"  n={n} done ({len(elig)} eligible cases)")

    rows = {}
    with open(OUT / "e3_samplesize.csv", "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["estimator", "n", "median_km", "q1_km", "q3_km", "n_cases"])
        for e in ESTS:
            for n in N_VALUES:
                arr = np.array(results[e][n])
                m, q1, q3 = np.median(arr), np.percentile(arr, 25), np.percentile(arr, 75)
                rows[(e, n)] = (m, q1, q3)
                wr.writerow([e, n, f"{m:.2f}", f"{q1:.2f}", f"{q3:.2f}", n_eligible[n]])

    make_plot(rows, n_eligible)

    print("\nMedian error km per n:")
    for n in N_VALUES:
        print(f"  n={n}: " + "  ".join(f"{e.split('_')[0]}={rows[(e,n)][0]:.1f}" for e in ESTS))
    print(f"\nTable: {OUT}/e3_samplesize.csv   Plot: {OUT}/e3_samplesize.png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="E3 sample size.")
    ap.add_argument("--replot", action="store_true",
                    help="only re-render the plot from e3_samplesize.csv (no recomputation)")
    args = ap.parse_args()
    if args.replot:
        rows, n_eligible = load_results_csv()
        make_plot(rows, n_eligible)
        print(f"Re-rendered: {OUT}/e3_samplesize.png")
    else:
        run()
