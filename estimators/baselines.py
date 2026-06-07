"""Etablierte robuste Schätzer als Vergleichs-Baselines.

Alle Schätzer arbeiten auf einer kleinen Wertereihe von Standort-Schätzungen
einer IP und liefern einen einzelnen Referenzpunkt zurück.

Schnittstelle (einheitlich für alle Schätzer im Projekt):
    estimate(points: np.ndarray[shape=(n, 2)]) -> np.ndarray[shape=(2,)]
    points-Spalten = (lat, lon) in Dezimalgrad, Rückgabe = (lat, lon).

Hinweis: Komponentenweiser Median/getrimmtes Mittel auf lat/lon ist nahe dem
Antimeridian/an den Polen verzerrt (0,001). Für die hier verwendeten (europäischen)
RIPE-Atlas-Anchors unkritisch; der geometrische Median ist der sauberere
2D-Schätzer und dient als Hauptbaseline.
"""

from __future__ import annotations

import numpy as np


def _as_points(points) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"erwarte Form (n, 2) [lat, lon], erhalten {pts.shape}")
    return pts


def centroid(points) -> np.ndarray:
    """Naiver Mittelwert (komponentenweises Mittel).

    Breakdown-Point 0 % — die *nicht*-robuste Referenz, die die robusten
    Verfahren schlagen sollen (vgl. FF1 / E1).
    """
    return _as_points(points).mean(axis=0)


def coordinate_median(points) -> np.ndarray:
    """Komponentenweiser Median über lat und lon (je Achse separat).

    Breakdown-Point 50 % je Komponente, aber — anders als der geometrische
    Median — NICHT affin-äquivariant und ohne dessen 2D-Optimalitäts-
    eigenschaften. Dient als naive 2D-Robustifizierung.
    """
    pts = _as_points(points)
    return np.median(pts, axis=0)


def trimmed_mean(points, proportion: float = 0.2) -> np.ndarray:
    """Komponentenweise getrimmtes Mittel; ``proportion`` je Seite gestutzt."""
    if not 0.0 <= proportion < 0.5:
        raise ValueError("proportion muss in [0, 0.5) liegen")
    pts = _as_points(points)
    n = pts.shape[0]
    k = int(np.floor(n * proportion))
    out = np.empty(2)
    for j in range(2):
        col = np.sort(pts[:, j])
        out[j] = col[k : n - k].mean() if n - 2 * k > 0 else col.mean()
    return out


def geometric_median(points, eps: float = 1e-6, max_iter: int = 500) -> np.ndarray:
    """Geometrischer Median (L1-Median) via Weiszfeld-Iteration.

    Fällt die Iterierte exakt mit einem Datenpunkt zusammen (Weiszfeld-Singularität),
    bricht die Iteration ab und gibt diesen Punkt zurück — ein pragmatischer
    Early-Return, NICHT die vollständige Vardi-&-Zhang-Korrektur. Die echte
    singularitätsfreie Behandlung leistet ``smoothed_geometric_median`` (RFA nach
    Pillutla et al.), die im Projekt als robuste Hauptvariante dient.
    Breakdown-Point ~50 %.
    """
    pts = _as_points(points)
    y = pts.mean(axis=0)  # Startwert: Mittelwert

    for _ in range(max_iter):
        d = np.linalg.norm(pts - y, axis=1)
        nonzero = d > eps

        if not np.all(nonzero):  # y fällt mit einem Datenpunkt zusammen
            return y

        w = 1.0 / d[nonzero]
        y_new = (pts[nonzero] * w[:, None]).sum(axis=0) / w.sum()

        if np.linalg.norm(y_new - y) < eps:
            return y_new
        y = y_new

    return y


def smoothed_geometric_median(points, nu: float = 1e-3, eps: float = 1e-9,
                              max_iter: int = 500) -> np.ndarray:
    """Geglätteter geometrischer Median (Weiszfeld mit beschränkten Gewichten).

    Nach Pillutla et al., „Robust Aggregation for Federated Learning" (RFA):
    das Gewicht ``w_i = 1 / max(nu, ||x_i - y||)`` beschränkt den Einfluss von
    Punkten nahe ``y`` und vermeidet die Weiszfeld-Singularität ohne Sonderfall
    — numerisch stabiler als der reine Schritt, gleicher ~50 % Breakdown-Point.
    ``nu`` ist ein kleines Glättungsmaß in Grad (Default ~0,1 km äquivalent).
    """
    pts = _as_points(points)
    y = pts.mean(axis=0)  # Startwert: Mittelwert
    for _ in range(max_iter):
        d = np.linalg.norm(pts - y, axis=1)
        w = 1.0 / np.maximum(nu, d)
        y_new = (pts * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(y_new - y) < eps:
            return y_new
        y = y_new
    return y


def weighted_geometric_median(points, weights=None, nu: float = 1e-3,
                              eps: float = 1e-9, max_iter: int = 500) -> np.ndarray:
    """Gewichteter, geglätteter geometrischer Median: minimiert ``Σ αᵢ·‖v − xᵢ‖``.

    Die gewichtete RFA-Variante (Pillutla et al.). ``weights`` = αᵢ; typischer
    Einsatz: **Linien-Gewichte** (jede Datenherkunft/lineage bekommt Gesamtgewicht
    1, gleichmäßig auf ihre Mitglieder verteilt), damit korrelierte Quellen die
    Schätzung nicht dominieren. ``weights=None`` -> uniform (== smoothed GM).
    """
    pts = _as_points(points)
    n = pts.shape[0]
    if weights is None:
        a = np.ones(n)
    else:
        a = np.asarray(weights, dtype=float)
        if a.shape != (n,):
            raise ValueError(f"weights muss Form ({n},) haben, erhalten {a.shape}")
    y = (pts * a[:, None]).sum(axis=0) / a.sum()  # gewichteter Mittelwert als Start
    for _ in range(max_iter):
        d = np.linalg.norm(pts - y, axis=1)
        w = a / np.maximum(nu, d)
        y_new = (pts * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(y_new - y) < eps:
            return y_new
        y = y_new
    return y


# Registry für Experimente (Name -> Schätzfunktion). Reihenfolge =
# naiv -> robust, damit Tabellen/Plots gut lesbar sind.
BASELINES = {
    "centroid": centroid,                               # naiv, Breakdown 0 %
    "median": coordinate_median,
    "trimmed_mean": trimmed_mean,
    "geometric_median": geometric_median,
    "smoothed_geometric_median": smoothed_geometric_median,
}
