"""Coordinate-system control: raw-degree aggregation vs. spherical median.

Precision check: the limitations section calls the effect of the
anisotropy (aggregation in raw-degree space, evaluation spherical)
"sub-kilometer", so far without a supporting script. This script provides the
comparison: per anchor, the headline configuration L1*b once as in the paper
(weighted smoothed Weiszfeld on (lat, lon) in degrees) and once as a spherical
weighted geometric median (Weiszfeld on 3D unit vectors, result projected onto
the sphere). Reported are the haversine distance between the two estimates per
case and the effect on the error against ground truth.

Usage:  python experiments/exp_coord_anisotropy.py
Result: key figures (stdout) + eval/out/coord_anisotropy.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.pipeline import load_cases                          # noqa: E402
from eval.metrics import haversine_error                      # noqa: E402
from estimators.baselines import weighted_geometric_median    # noqa: E402
import experiments.exp_t6_defaults as T6                      # noqa: E402
from experiments.exp_targeted_contamination import case_setup  # noqa: E402

OUT = ROOT / "eval" / "out"


def to_unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo),
                     np.sin(la)], axis=-1)


def to_latlon(v):
    v = v / np.linalg.norm(v)
    return (float(np.degrees(np.arcsin(np.clip(v[2], -1, 1)))),
            float(np.degrees(np.arctan2(v[1], v[0]))))


def spherical_weighted_geomedian(pts_deg, w, nu=1e-3, eps=1e-12,
                                 max_iter=500):
    """Weighted smoothed Weiszfeld on unit vectors (R^3), iterate projected
    onto the sphere after each step; nu in DEGREES
    (degree equivalent of the raw-degree variant; converted internally via radians())."""
    x = to_unit(pts_deg[:, 0], pts_deg[:, 1])
    w = np.asarray(w, float)
    y = (x * w[:, None]).sum(0)
    y = y / np.linalg.norm(y)
    nu_r = np.radians(nu)
    for _ in range(max_iter):
        d = np.arccos(np.clip(x @ y, -1.0, 1.0))          # great-circle distances
        ww = w / np.maximum(nu_r, d)
        y_new = (x * ww[:, None]).sum(0)
        y_new = y_new / np.linalg.norm(y_new)
        if np.arccos(np.clip(y @ y_new, -1.0, 1.0)) < eps:
            return to_latlon(y_new)
        y = y_new
    return to_latlon(y)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    loo = T6.loo_pseudo_radii(cases)
    eps = min(T6.EPS_GRID, key=lambda e: abs(e - float(np.median(
        [loo[s]["_global"] for s in loo]))))

    rows = []
    for c in cases:
        pts, _, w_eff, _ = case_setup(c, loo, eps)
        raw = weighted_geometric_median(pts, w_eff)
        raw = (float(raw[0]), float(raw[1]))
        sph = spherical_weighted_geomedian(pts, w_eff)
        rows.append({
            "ip": c["ip"],
            "displacement_km": haversine_error(raw, sph),
            "err_raw_km": haversine_error(raw, c["truth"]),
            "err_sph_km": haversine_error(sph, c["truth"]),
        })
    df = pd.DataFrame(rows)
    df["d_err_km"] = df.err_sph_km - df.err_raw_km

    disp, derr = df.displacement_km, df.d_err_km
    print("=" * 70)
    print(f"Raw-degree vs. spherical aggregation (L1*b, n={len(df)})")
    print("=" * 70)
    print(f"Displacement of the estimates [km]: median {disp.median():.4f} | "
          f"p95 {disp.quantile(.95):.3f} | max {disp.max():.3f}")
    print(f"Cases with displacement > 1 km: {(disp > 1).sum()} "
          f"({100 * (disp > 1).mean():.2f} %)")
    print(f"Error change spherical−raw [km]: median {derr.median():+.4f} | "
          f"mean {derr.mean():+.4f} | max|.| {derr.abs().max():.3f}")
    for name, e in [("raw", df.err_raw_km), ("spherical", df.err_sph_km)]:
        print(f"Headline {name:9s}: median {e.median():5.2f} | mean {e.mean():6.1f} | "
              f"tail>100 {100 * (e > 100).mean():5.2f} %")
    summary = pd.DataFrame([{
        "n": len(df),
        "disp_median_km": round(float(disp.median()), 4),
        "disp_p95_km": round(float(disp.quantile(.95)), 3),
        "disp_max_km": round(float(disp.max()), 3),
        "n_disp_gt1km": int((disp > 1).sum()),
        "derr_median_km": round(float(derr.median()), 4),
        "derr_mean_km": round(float(derr.mean()), 4),
        "derr_absmax_km": round(float(derr.abs().max()), 3),
    }])
    summary.to_csv(OUT / "coord_anisotropy.csv", index=False)
    print(f"\n  CSV: {OUT}/coord_anisotropy.csv")


if __name__ == "__main__":
    main()
