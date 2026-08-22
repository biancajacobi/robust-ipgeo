"""E4 — Stratified evaluation (RQ: when/where does the aggregation carry?).

No new data collection — a re-cut of the existing cases by:
  E4a difficulty:  easy / disagree / hard (as in T6) x estimator -> boxplot + table.
  E4b region:      median error per country (top N by anchor count) -> shows
                   geographic variation and where the robust estimator carries
                   vs. fails.

Estimators: naive mean, geom. median (robust main line), Braetz (T5).

Usage:   python experiments/exp_stratification.py
Output:  eval/out/e4_difficulty.{csv,png}, eval/out/e4_region.csv
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
from eval.pipeline import load_cases            # noqa: E402
from eval.metrics import haversine_error         # noqa: E402
from estimators.baselines import centroid, geometric_median  # noqa: E402
from estimators import braetz                    # noqa: E402

OUT = Path("eval/out"); OUT.mkdir(parents=True, exist_ok=True)
ESTS = [("naive mean", centroid, "#d62728"),
        ("geometric median", geometric_median, "#1f77b4"),
        ("Braetz", braetz.estimate, "#9467bd")]
BUCKETS = ["easy", "disagree", "hard"]


def bucket(case):
    e = [haversine_error((p["lat"], p["lon"]), case["truth"]) for p in case["provenance"]]
    return "hard" if min(e) > 50 else "easy" if max(e) < 50 else "disagree"


def main():
    cases = load_cases()
    rows = []
    for c in cases:
        b = bucket(c)
        rec = {"ip": c["ip"], "bucket": b, "country": c["country"]}
        for name, fn, _ in ESTS:
            try:
                rec[name] = haversine_error(tuple(fn(c["points"])), c["truth"])
            except Exception:
                rec[name] = float("nan")
        rows.append(rec)
    import pandas as pd
    df = pd.DataFrame(rows)

    # --- E4a: difficulty ---
    with open(OUT / "e4_difficulty.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["bucket", "n", "estimator", "median_km", "q1_km", "q3_km", "mean_km"])
        for b in BUCKETS + ["overall"]:
            sub = df if b == "overall" else df[df.bucket == b]
            for name, _, _ in ESTS:
                e = sub[name].dropna()
                w.writerow([b, len(sub), name, f"{e.median():.1f}", f"{e.quantile(.25):.1f}",
                            f"{e.quantile(.75):.1f}", f"{e.mean():.1f}"])

    print("E4a — median error km [Q1–Q3] per difficulty × estimator")
    print(f"{'bucket':10s} {'n':>5s}  " + "  ".join(f"{n:>18s}" for n, _, _ in ESTS))
    for b in BUCKETS + ["overall"]:
        sub = df if b == "overall" else df[df.bucket == b]
        cells = [f"{sub[n].median():5.0f}[{sub[n].quantile(.25):4.0f}-{sub[n].quantile(.75):5.0f}]"
                 for n, _, _ in ESTS]
        print(f"{b:10s} {len(sub):5d}  " + "  ".join(f"{c:>18s}" for c in cells))

    # Boxplot (log-y), the 3 estimators side by side per bucket
    fig, ax = plt.subplots(figsize=(9, 5.5))
    width, gap = 0.25, 1.0
    for j, (name, _, col) in enumerate(ESTS):
        data = [df[df.bucket == b][name].dropna().clip(lower=0.1) for b in BUCKETS]
        pos = [i * gap + (j - 1) * width for i in range(len(BUCKETS))]
        bp = ax.boxplot(data, positions=pos, widths=width * 0.9, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black"))
        for box in bp["boxes"]:
            box.set(facecolor=col, alpha=0.6)
        ax.plot([], [], color=col, alpha=0.6, linewidth=8, label=name)
    ax.set_yscale("log")
    ax.set_xticks([i * gap for i in range(len(BUCKETS))],
                  [f"{b}\n(n={int((df.bucket==b).sum())})" for b in BUCKETS])
    ax.set_ylabel("error to ground truth / km (log)")
    ax.set_title(f"E4a — error per difficulty bucket × estimator  (n={len(df)} anchors)")
    ax.grid(True, axis="y", which="both", alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "e4_difficulty.png", dpi=150); plt.close(fig)

    # --- E4b: region (top countries by anchor count) ---
    top = df.country.value_counts().head(15).index
    reg = (df[df.country.isin(top)].groupby("country")
           .agg(n=("ip", "size"), geomed=("geometric median", "median"),
                braetz=("Braetz", "median"), naive=("naive mean", "median"))
           .sort_values("n", ascending=False).round(1))
    reg.to_csv(OUT / "e4_region.csv")
    print("\nE4b — median error km per country (top 15 by anchor count)")
    print(reg.to_string())
    print(f"\nCSVs: {OUT}/e4_difficulty.csv, e4_region.csv   Plot: {OUT}/e4_difficulty.png")


if __name__ == "__main__":
    main()
