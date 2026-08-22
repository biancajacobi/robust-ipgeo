"""E1 — accuracy (RQ1): all estimators + single sources against ground truth.

Core question: does robust aggregation beat the naive mean (centroid) and the
*best single source*? Output: CDF of the error distance
(eval/out/e1_accuracy_cdf.png) + summary table Q0–Q4/mean/coverage
(eval/out/e1_accuracy.{csv,md}) + baseline block for Tab. 1
(median/mean/tail>100 km per aggregation estimator,
eval/out/e1_baseline_tails.{csv,md}).

    python experiments/exp_accuracy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from estimators import braetz  # noqa: E402
from estimators.baselines import BASELINES, weighted_geometric_median  # noqa: E402
from eval import pipeline, report  # noqa: E402

ESTIMATORS = {**BASELINES, "braetz": braetz.estimate}
# line-weighted (1 weight per lineage): correlated sources do not dominate
LINE_WEIGHTED = {"geometric_median_perline": weighted_geometric_median}


def run() -> None:
    cases = pipeline.load_cases()
    if not cases:
        print("No cases — collect data first: python data/fetch_sources.py")
        return

    df = pipeline.evaluate(cases, ESTIMATORS, include_sources=True,
                           line_weighted=LINE_WEIGHTED)

    # Estimators (aggregate) + best single source for comparison in one table
    summary = report.summarize_results(df, coverage_km=(25, 100, 250))
    table_path = report.save_table(summary.round(1), "e1_accuracy")

    # Baseline block for Tab. 1: median/mean/tail rate (>100 km) per
    # aggregation estimator (single sources are already in e1_accuracy.csv)
    agg = df[df["kind"] == "aggregate"].dropna(subset=["error_km"])
    tails = (agg.groupby("estimator")["error_km"]
             .agg(n="count", median_km="median", mean_km="mean",
                  tail_gt100km_pct=lambda e: 100 * (e > 100).mean())
             .round(1).sort_values("mean_km"))
    tails_path = report.save_table(tails, "e1_baseline_tails")

    # CDF: estimators + single sources (to contextualize "best source");
    # colored only aggregation + top-3 sources (median), rest gray
    errors = report.errors_by_estimator(df)
    src_sorted = summary[summary.index.str.startswith("src:")]["median"]
    top3 = set(src_sorted.nsmallest(3).index)
    cdf_path = report.plot_ecdf(errors, "e1_accuracy_cdf", xmax=2000,
                                title="E1 — error distribution per estimator & source",
                                highlight={"geometric_median_perline"} | top3)

    n_cases = len(cases)
    src = summary[summary.index.str.startswith("src:")]
    best_src = src["median"].idxmin() if not src.empty else None
    print(f"E1 — {n_cases} cases, {len(ESTIMATORS)} estimators.\n")
    print(summary[["n", "median", "q3", "p90", "cov@100km"]].to_string())
    if best_src is not None:
        print(f"\nBest single source (median error): {best_src} "
              f"= {summary.loc[best_src, 'median']:.0f} km")
    print(f"\nTable   : {table_path}")
    print(f"Tails   : {tails_path}")
    print(f"CDF     : {cdf_path}")


if __name__ == "__main__":
    run()
