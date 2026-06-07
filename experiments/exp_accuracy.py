"""E1 — Genauigkeit (FF1): alle Schätzer + Einzelquellen gegen Ground Truth.

Kernfrage: Schlägt robuste Aggregation den naiven Mittelwert (Centroid) und die
*beste Einzelquelle*? Output: CDF der Fehlerdistanz (eval/out/e1_accuracy_cdf.png)
+ Übersichtstabelle Q0–Q4/Mean/Coverage (eval/out/e1_accuracy.{csv,md}).

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
# linien-gewichtet (1 Gewicht je lineage): korrelierte Quellen dominieren nicht
LINE_WEIGHTED = {"geometric_median_perline": weighted_geometric_median}


def run() -> None:
    cases = pipeline.load_cases()
    if not cases:
        print("Keine Fälle — erst Daten sammeln: python data/fetch_sources.py")
        return

    df = pipeline.evaluate(cases, ESTIMATORS, include_sources=True,
                           line_weighted=LINE_WEIGHTED)

    # Schätzer (aggregate) + beste Einzelquelle zum Vergleich in eine Tabelle
    summary = report.summarize_results(df, coverage_km=(25, 100, 250))
    table_path = report.save_table(summary.round(1), "e1_accuracy")

    # CDF: Schätzer + Einzelquellen (zur Einordnung "beste Quelle")
    errors = report.errors_by_estimator(df)
    cdf_path = report.plot_ecdf(errors, "e1_accuracy_cdf", xmax=2000,
                                title="E1 — Fehlerverteilung je Schätzer & Quelle")

    n_cases = len(cases)
    src = summary[summary.index.str.startswith("src:")]
    best_src = src["median"].idxmin() if not src.empty else None
    print(f"E1 — {n_cases} Fälle, {len(ESTIMATORS)} Schätzer.\n")
    print(summary[["n", "median", "q3", "p90", "cov@100km"]].to_string())
    if best_src is not None:
        print(f"\nBeste Einzelquelle (Median-Fehler): {best_src} "
              f"= {summary.loc[best_src, 'median']:.0f} km")
    print(f"\nTabelle: {table_path}")
    print(f"CDF    : {cdf_path}")


if __name__ == "__main__":
    run()
