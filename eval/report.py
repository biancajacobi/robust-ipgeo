"""Aggregation of experiment results: metrics, tables, plots.

Output to eval/out/ (tables as CSV/Markdown, plots as PNG/PDF).
Core metrics per estimator: median error, 90th-percentile error,
error distribution (km).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Dataset switch (cf. data/store): separate output folder for separate runs
# (e.g. RIPE Atlas probes). GEOIP_DATASET=anchors (default) -> eval/out.
_DATASET = (os.environ.get("GEOIP_DATASET") or "anchors").strip() or "anchors"
OUT_DIR = Path(__file__).resolve().parent / ("out" if _DATASET == "anchors" else f"out_{_DATASET}")


def _plt():
    """Load matplotlib lazily + headless (metrics do not need it)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def error_summary(errors_km) -> dict:
    """Metrics of an error series (km): n, min/Q1/median/Q3/max, p90, mean.

    CAUTION: NaN (= estimator failure) is removed; metrics and coverage thus
    refer, per estimator, to DIFFERENT denominators (solved cases only). When
    comparing partially failing estimators, mind the n column
    (review 2026-08-18; currently latent, no estimator fails).
    """
    e = np.asarray(errors_km, dtype=float)
    e = e[~np.isnan(e)]
    if e.size == 0:
        return {"n": 0, "min": np.nan, "q1": np.nan, "median": np.nan,
                "q3": np.nan, "p90": np.nan, "max": np.nan, "mean": np.nan}
    q1, med, q3 = np.percentile(e, [25, 50, 75])
    return {
        "n": int(e.size),
        "min": float(e.min()),
        "q1": float(q1),
        "median": float(med),
        "q3": float(q3),
        "p90": float(np.percentile(e, 90)),
        "max": float(e.max()),
        "mean": float(e.mean()),
    }


def coverage(errors_km, thresholds=(25, 100, 250)) -> dict:
    """Share of cases with error <= threshold (km) — 'Coverage@<km>'."""
    e = np.asarray(errors_km, dtype=float)
    e = e[~np.isnan(e)]
    return {f"cov@{t}km": (float((e <= t).mean()) if e.size else np.nan)
            for t in thresholds}


def summarize_results(df: pd.DataFrame, coverage_km=(25, 100, 250),
                      sort_by: str | None = "median") -> pd.DataFrame:
    """Tidy result frame (columns ``estimator``, ``error_km``) -> overview
    per estimator: metrics + Coverage@<km>. One row per estimator."""
    rows = []
    for name, grp in df.groupby("estimator", sort=False):
        errs = grp["error_km"].to_numpy()
        rows.append({"estimator": name, **error_summary(errs),
                     **coverage(errs, coverage_km)})
    out = pd.DataFrame(rows).set_index("estimator")
    if sort_by and sort_by in out.columns:
        out = out.sort_values(sort_by)
    return out


def save_table(df: pd.DataFrame, name: str) -> Path:
    """Write a table to eval/out/<name>.{csv,md}."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{name}.csv"
    df.to_csv(csv_path)
    (OUT_DIR / f"{name}.md").write_text(df.to_markdown())
    return csv_path


def errors_by_estimator(df: pd.DataFrame) -> dict:
    """Tidy frame -> {estimator name: error series (km, NaN removed)}."""
    return {name: grp["error_km"].dropna().to_numpy()
            for name, grp in df.groupby("estimator", sort=False)}


def plot_ecdf(errors: dict, name: str, xmax: float | None = None,
              xlog: bool = False, title: str | None = None,
              highlight: set | None = None) -> Path:
    """ECDF of the error distance per estimator -> eval/out/<name>.png (RQ1 / E1).

    ``highlight``: optional label set — only these colored/labeled,
    all others grey with one collective legend entry (readability with
    many lines; review remark 2026-08-19).
    """
    plt = _plt()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    grey_labeled = False
    for label, errs in errors.items():
        e = np.sort(np.asarray(errs, dtype=float))
        e = e[~np.isnan(e)]
        if e.size == 0:
            continue
        y = np.arange(1, e.size + 1) / e.size
        if highlight is not None and label not in highlight:
            leg = "other sources/estimators" if not grey_labeled else "_nolegend_"
            grey_labeled = True
            ax.plot(e, y, drawstyle="steps-post", color="0.75", lw=0.8,
                    zorder=1, label=leg)
            continue
        lw = 1.8 if highlight is not None else None
        ax.plot(e, y, drawstyle="steps-post", label=f"{label} (n={e.size})",
                lw=lw, zorder=3)
    ax.set_xlabel("Haversine error [km]")
    ax.set_ylabel("Share of cases (ECDF)")
    if xlog:
        ax.set_xscale("log")
    if xmax:
        ax.set_xlim(left=0 if not xlog else None, right=xmax)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(title or "Error distribution per estimator")
    fig.tight_layout()
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_error_distribution(errors: dict, name: str, ymax: float | None = None,
                            title: str | None = None) -> Path:
    """Boxplot of the km errors per estimator -> eval/out/<name>.png."""
    plt = _plt()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels, data = [], []
    for label, errs in errors.items():
        e = np.asarray(errs, dtype=float)
        e = e[~np.isnan(e)]
        if e.size:
            labels.append(label)
            data.append(e)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    try:  # tick_labels from matplotlib 3.9 on; labels before that
        ax.boxplot(data, tick_labels=labels, showfliers=True)
    except TypeError:
        ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_ylabel("Haversine error [km]")
    if ymax:
        ax.set_ylim(0, ymax)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(title or "Error distribution per estimator")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
