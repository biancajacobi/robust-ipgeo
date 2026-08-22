"""Braetz's class-/density-based estimation method (method under investigation).

Reference: Braetz, M. — "Entwicklung eines Schätzverfahrens zur Bestimmung
robuster Referenzwerte auf geringer Datenbasis unbekannter Güte"
(see references.bib, ``braetz``).

The method runs one-dimensionally per coordinate. For a 2D location it is
applied once each to the latitude and the longitude series (``estimate``).

Steps (reconstructed from Braetz 2009, critically examined as T5 in the
accompanying paper — evaluation in experiments/exp_braetz.py):
  1. Partition values into k classes (empirically chosen class count k).
  2. Determine density/frequency per class, pick the densest class(es).
  3. Optional fuzzy membership at class boundaries (``fuzzy``).
  4. Form a robust location estimator within the chosen class(es).
  5. Derive a confidence/dispersion measure (document the distribution assumption!).

Points of criticism to be substantiated by experiment: benefit of the fuzzy
stage, normality assumption (Q-Q), sensitivity to k, validity of the
confidence, circularity.
"""

from __future__ import annotations

import numpy as np


K_MAX = 30              # Braetz default (convergence in the range k≈26–30, Braetz 2009, Tab. 4.1)
DENSITY_THRESHOLD = 2   # main run: Braetz's lower bound from the stopping criterion
                        # (original 4 is meant for n≈30 and too harsh at n=8 → sensitivity)


def braetz_1d(values, k_max: int = K_MAX, density_threshold: int = DENSITY_THRESHOLD,
              return_sequence: bool = False):
    """Braetz's fuzzy density estimator for a 1D value series (Braetz 2009, ch. 4).

    Procedure (1:1 after Braetz 2009):
      1. Successively increase the class count k = 1..k_max; split the range
         [min,max] into k equally wide intervals, count values per interval.
      2. Per k, pick the densest interval (highest cardinality) and represent it
         by its **class midpoint** -> one midpoint value per k.
      3. Stop as soon as the densest interval contains only ≤1 value (intervals
         too small -> spurious precision; the "fuzziness" must be preserved).
      4. Cleanup: keep only midpoints whose interval had ≥ ``density_threshold``
         values (Braetz removes local density < 4; parameterized here).
      5. Estimate = mean of the cleaned midpoint sequence.

    The midpoint sequence is (for a normally distributed sample) approximately
    normally distributed around the location of highest density — the Student-t
    CI rests on this (see ``braetz_ci``).

    With ``return_sequence`` additionally the cleaned midpoint sequence (for
    diagnostics/CI).
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        raise ValueError("empty value series")
    lo, hi = float(v.min()), float(v.max())
    if hi == lo:                                   # degenerate: all values equal
        return (lo, np.array([lo])) if return_sequence else lo

    mids, dens = [], []
    for k in range(1, k_max + 1):
        width = (hi - lo) / k
        idx = np.minimum(((v - lo) / width).astype(int), k - 1)
        counts = np.bincount(idx, minlength=k)
        max_count = int(counts.max())
        if max_count <= 1:                         # stopping criterion (≤1 value/class)
            break
        winner = int(counts.argmax())              # tie: lowest index (deterministic)
        mids.append(lo + (winner + 0.5) * width)   # class midpoint
        dens.append(max_count)

    mids = np.asarray(mids, dtype=float)
    dens = np.asarray(dens, dtype=int)
    kept = mids[dens >= density_threshold]
    if kept.size == 0:                             # fallback: never return empty
        kept = mids if mids.size else np.array([(lo + hi) / 2])
    est = float(kept.mean())
    return (est, kept) if return_sequence else est


def braetz_ci(values, alpha: float = 0.05, **kw):
    """(estimate, CI half-width, midpoint sequence) — Student-t on the midpoint sequence.

    This is Braetz's confidence: the *internal* certainty of the midpoint
    sequence (df = m−1), NOT the distance to the ground truth — the central
    distinction for the T5 criticism.
    """
    from scipy import stats
    est, seq = braetz_1d(values, return_sequence=True, **kw)
    m = seq.size
    if m < 2:
        return est, float("nan"), seq
    se = float(seq.std(ddof=1)) / np.sqrt(m)
    half = float(stats.t.ppf(1 - alpha / 2, df=m - 1)) * se
    return est, half, seq


def estimate(points, k_max: int = K_MAX, density_threshold: int = DENSITY_THRESHOLD) -> np.ndarray:
    """Location estimator (lat, lon): Braetz per coordinate separately.

    Uniform interface as in ``estimators.baselines`` (fn(points) -> (2,)).
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"expected shape (n, 2) [lat, lon], got {pts.shape}")
    lat = braetz_1d(pts[:, 0], k_max=k_max, density_threshold=density_threshold)
    lon = braetz_1d(pts[:, 1], k_max=k_max, density_threshold=density_threshold)
    return np.array([lat, lon])
