"""Aggregation der Experiment-Ergebnisse: Kennzahlen, Tabellen, Plots.

Ausgabe nach eval/out/ (Tabellen als CSV/Markdown, Plots als PNG/PDF).
Kern-Kennzahlen je Schätzer: Median-Fehler, 90.-Perzentil-Fehler,
Fehlerverteilung (km).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Dataset-Switch (vgl. data/store): getrennter Output-Ordner fuer separate Laeufe
# (z. B. RIPE-Atlas-Probes). GEOIP_DATASET=anchors (Default) -> eval/out.
_DATASET = (os.environ.get("GEOIP_DATASET") or "anchors").strip() or "anchors"
OUT_DIR = Path(__file__).resolve().parent / ("out" if _DATASET == "anchors" else f"out_{_DATASET}")


def _plt():
    """matplotlib lazy + headless laden (Kennzahlen brauchen es nicht)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def error_summary(errors_km) -> dict:
    """Kennzahlen einer Fehlerreihe (km): n, min/Q1/median/Q3/max, p90, mean."""
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
    """Anteil der Fälle mit Fehler <= Schwelle (km) — 'Coverage@<km>'."""
    e = np.asarray(errors_km, dtype=float)
    e = e[~np.isnan(e)]
    return {f"cov@{t}km": (float((e <= t).mean()) if e.size else np.nan)
            for t in thresholds}


def summarize_results(df: pd.DataFrame, coverage_km=(25, 100, 250),
                      sort_by: str | None = "median") -> pd.DataFrame:
    """Tidy-Ergebnis-Frame (Spalten ``estimator``, ``error_km``) -> Übersicht
    je Schätzer: Kennzahlen + Coverage@<km>. Eine Zeile je Schätzer."""
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
    """Tabelle nach eval/out/<name>.{csv,md} schreiben."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{name}.csv"
    df.to_csv(csv_path)
    (OUT_DIR / f"{name}.md").write_text(df.to_markdown())
    return csv_path


def errors_by_estimator(df: pd.DataFrame) -> dict:
    """Tidy-Frame -> {Schätzername: Fehlerreihe (km, NaN entfernt)}."""
    return {name: grp["error_km"].dropna().to_numpy()
            for name, grp in df.groupby("estimator", sort=False)}


def plot_ecdf(errors: dict, name: str, xmax: float | None = None,
              xlog: bool = False, title: str | None = None) -> Path:
    """ECDF der Fehlerdistanz je Schätzer -> eval/out/<name>.png (FF1 / E1)."""
    plt = _plt()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, errs in errors.items():
        e = np.sort(np.asarray(errs, dtype=float))
        e = e[~np.isnan(e)]
        if e.size == 0:
            continue
        y = np.arange(1, e.size + 1) / e.size
        ax.plot(e, y, drawstyle="steps-post", label=f"{label} (n={e.size})")
    ax.set_xlabel("Haversine-Fehler [km]")
    ax.set_ylabel("Anteil der Fälle (ECDF)")
    if xlog:
        ax.set_xscale("log")
    if xmax:
        ax.set_xlim(left=0 if not xlog else None, right=xmax)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(title or "Fehlerverteilung je Schätzer")
    fig.tight_layout()
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_error_distribution(errors: dict, name: str, ymax: float | None = None,
                            title: str | None = None) -> Path:
    """Boxplot der km-Fehler je Schätzer -> eval/out/<name>.png."""
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
    try:  # tick_labels ab matplotlib 3.9; labels davor
        ax.boxplot(data, tick_labels=labels, showfliers=True)
    except TypeError:
        ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_ylabel("Haversine-Fehler [km]")
    if ymax:
        ax.set_ylim(0, ymax)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(title or "Fehlerverteilung je Schätzer")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
