"""Tests for the reporting layer: summary stats, coverage, tables, plots."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import report


def test_error_summary_quantiles():
    s = report.error_summary([0.0, 10.0, 20.0, 30.0, 40.0])
    assert s["n"] == 5
    assert s["min"] == 0.0 and s["max"] == 40.0
    assert s["median"] == 20.0
    assert s["q1"] == 10.0 and s["q3"] == 30.0


def test_error_summary_empty_and_nan():
    s = report.error_summary([float("nan")])
    assert s["n"] == 0


def test_coverage_fractions():
    cov = report.coverage([10.0, 20.0, 200.0], thresholds=(25, 100))
    assert cov["cov@25km"] == 2 / 3
    assert cov["cov@100km"] == 2 / 3


def test_summarize_results_sorted_by_median():
    df = pd.DataFrame({
        "estimator": ["a", "a", "b", "b"],
        "error_km": [1.0, 2.0, 100.0, 200.0],
    })
    out = report.summarize_results(df, coverage_km=(50,))
    assert list(out.index) == ["a", "b"]          # sorted by median
    assert out.loc["a", "cov@50km"] == 1.0
    assert out.loc["b", "cov@50km"] == 0.0


def test_plots_write_files(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "OUT_DIR", tmp_path)
    errs = {"a": [1.0, 2.0, 3.0], "b": [10.0, 50.0, 300.0]}
    p1 = report.plot_ecdf(errs, "ecdf_test")
    p2 = report.plot_error_distribution(errs, "box_test", ymax=500)
    assert p1.exists() and p1.suffix == ".png"
    assert p2.exists() and p2.suffix == ".png"
