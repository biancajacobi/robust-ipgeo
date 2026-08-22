"""Established robust estimators as comparison baselines.

All estimators operate on a small value series of location estimates for one
IP and return a single reference point.

Interface (uniform for all estimators in the project):
    estimate(points: np.ndarray[shape=(n, 2)]) -> np.ndarray[shape=(2,)]
    points columns = (lat, lon) in decimal degrees, return = (lat, lon).

Note: component-wise median/trimmed mean on lat/lon is biased near the
antimeridian/at the poles. Uncritical for the (European) RIPE Atlas anchors
used here; the geometric median is the cleaner 2D estimator and serves as the
main baseline.
"""

from __future__ import annotations

import numpy as np


def _as_points(points) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"erwarte Form (n, 2) [lat, lon], erhalten {pts.shape}")
    return pts


def centroid(points) -> np.ndarray:
    """Naive mean (component-wise average).

    Breakdown point 0 % — the *non*-robust reference the robust methods are
    supposed to beat (cf. RQ1 / E1).
    """
    return _as_points(points).mean(axis=0)


def coordinate_median(points) -> np.ndarray:
    """Component-wise median over lat and lon (each axis separately).

    Breakdown point 50 % per component, but — unlike the geometric
    median — not rotation-equivariant (the geometric median is
    similarity-equivariant; neither is affine-equivariant) and without its 2D
    optimality properties (cf. methodology chapter). Serves as a naive 2D
    robustification.
    """
    pts = _as_points(points)
    return np.median(pts, axis=0)


def trimmed_mean(points, proportion: float = 0.2) -> np.ndarray:
    """Component-wise trimmed mean; ``proportion`` trimmed per side."""
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
    """Geometric median (L1 median) via Weiszfeld iteration.

    If the iterate coincides exactly with a data point (Weiszfeld singularity),
    the iteration stops and returns that point — a pragmatic early return,
    NOT the full Vardi & Zhang correction. CAUTION
    (review 2026-08-18): if the collision already hits the START iterate, the
    return value is the arithmetic MEAN, not the geometric median —
    constructibly arbitrarily wrong (e.g. [[0,0]*3, [4,0], [16,0]] -> (4,0)
    instead of (0,0)); latent on the real anchor data (0/1077 cases). The true
    singularity-free treatment is provided by ``smoothed_geometric_median``
    (RFA after Pillutla et al.), which serves as the robust main variant in
    the project. Breakdown point ~50 %.
    """
    pts = _as_points(points)
    y = pts.mean(axis=0)  # starting value: mean

    for _ in range(max_iter):
        d = np.linalg.norm(pts - y, axis=1)
        nonzero = d > eps

        if not np.all(nonzero):  # y coincides with a data point
            return y

        w = 1.0 / d[nonzero]
        y_new = (pts[nonzero] * w[:, None]).sum(axis=0) / w.sum()

        if np.linalg.norm(y_new - y) < eps:
            return y_new
        y = y_new

    return y


def smoothed_geometric_median(points, nu: float = 1e-3, eps: float = 1e-9,
                              max_iter: int = 500) -> np.ndarray:
    """Smoothed geometric median (Weiszfeld with bounded weights).

    After Pillutla et al., "Robust Aggregation for Federated Learning" (RFA):
    the weight ``w_i = 1 / max(nu, ||x_i - y||)`` bounds the influence of
    points near ``y`` and avoids the Weiszfeld singularity without a special
    case — numerically more stable than the pure step, same ~50 % breakdown
    point. ``nu`` is a small smoothing measure in degrees (default ~0.1 km
    equivalent).
    """
    pts = _as_points(points)
    y = pts.mean(axis=0)  # starting value: mean
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
    """Weighted, smoothed geometric median: minimizes ``Σ αᵢ·‖v − xᵢ‖``.

    The weighted RFA variant (Pillutla et al.). ``weights`` = αᵢ; typical
    use: **line weights** (each data provenance/lineage gets total weight
    1, distributed evenly over its members), so that correlated sources do not
    dominate the estimate. ``weights=None`` -> uniform (== smoothed GM).
    """
    pts = _as_points(points)
    n = pts.shape[0]
    if weights is None:
        a = np.ones(n)
    else:
        a = np.asarray(weights, dtype=float)
        if a.shape != (n,):
            raise ValueError(f"weights muss Form ({n},) haben, erhalten {a.shape}")
    y = (pts * a[:, None]).sum(axis=0) / a.sum()  # weighted mean as the start
    for _ in range(max_iter):
        d = np.linalg.norm(pts - y, axis=1)
        w = a / np.maximum(nu, d)
        y_new = (pts * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(y_new - y) < eps:
            return y_new
        y = y_new
    return y


# Registry for experiments (name -> estimator function). Order =
# naive -> robust, so tables/plots read well.
BASELINES = {
    "centroid": centroid,                               # naive, breakdown 0 %
    "median": coordinate_median,
    "trimmed_mean": trimmed_mean,
    "geometric_median": geometric_median,
    "smoothed_geometric_median": smoothed_geometric_median,
}
