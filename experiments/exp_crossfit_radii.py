#!/usr/bin/env python3
"""Cross-fitting control: recompute pseudo-radii/track records per OUTER FOLD.

Control experiment: the BSS numbers reported as "fully
out-of-fold" use source track records (LOO median radii) that are computed
ONCE GLOBALLY over all anchors. This is no self-leakage of the individual
anchor (per-IP LOO), but it is also not a strictly fold-isolated evaluation:
ground-truth information from the test-fold anchors enters the feature and
weight construction via the source median (1 of ~1,077 values per source —
numerically expected to be tiny, but protocol-wise real).

This script runs both arms with IDENTICAL fit code, so the difference is
attributable solely to the cross-fitting:

  Arm A (status quo):    track records global (reimplementation; must
                        reproduce the published numbers +0.210 / +0.625)
  Arm B (fold-isolated): per outer fold, loo_pseudo_radii are computed ONLY
                        on the training anchors; test IPs receive the _global
                        fallback of the training set (deployment-realistic,
                        like radius_table.json); the estimator, features AND
                        miss label of the test fold follow from that.

Models: S50 (headline, protocol exp_support_concentration incl. per-fold
ε grid selection in arm B) and R7 = QW + in25/50/100 per source (protocol
exp_s_declared_risk, ε=10 fixed; WIN features are NaN-free, no imputation
needed). Fit: IRLS as in exp_support_concentration (λ≈0, standardization per
training fold). Fold assignment: stratified on the global label (assignment
only, no feature information), seed 0 — identical to the status quo.

Usage:  python experiments/exp_crossfit_radii.py
Result: table (stdout) + eval/out/crossfit_radii.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.pipeline import load_cases                       # noqa: E402
from eval.metrics import haversine_error                   # noqa: E402
import experiments.exp_t6_defaults as T6                   # noqa: E402
from experiments.exp_support_concentration import (        # noqa: E402
    folds, _fit, bss, ece, K,
)

OUT = Path("eval/out")
OUT.mkdir(parents=True, exist_ok=True)
TAU = 100.0
EPS_R7 = 10
RADII = (25, 50, 75, 100)
SOURCES = ("dbip_lite", "geojs", "ip2location_lite", "ip_api", "ipapi_co",
           "ipinfo", "ipwho_is", "maxmind_geolite2", "reallyfreegeoip")
QW = [f"core_qw_{r}" for r in RADII]
SRC_ALL = [f"in{R}_{s}" for R in (50, 25, 100) for s in SOURCES]
WIN = QW + SRC_ALL                                          # R7


def eps_grid(loo) -> float:
    return min(T6.EPS_GRID, key=lambda e: abs(
        e - float(np.median([loo[s]["_global"] for s in loo]))))


def features(case, loo, eps):
    """S50 and R7 features + error, with given track record/ε."""
    prov = case["provenance"]
    pts = np.array([[p["lat"], p["lon"]] for p in prov], float)
    w = np.asarray(T6.line_weights_for(prov, "L1"), float)
    w = w / w.sum()
    q = np.array([1.0 / (loo.get(p["source"], {}).get("_global", 100.0) + EPS_R7)
                  for p in prov])
    wq = w * q
    wq = wq / wq.sum()
    est = np.array(T6.estimate(case, "L1", "b", loo, eps=eps))
    d = np.array([haversine_error(tuple(p), tuple(est)) for p in pts])
    r = {"err": haversine_error(tuple(est), case["truth"]),
         "core_w_50": float(w[d < 50].sum())}
    for R in RADII:
        r[f"core_qw_{R}"] = float(wq[d < R].sum())
    src_d = {p["source"]: dd for p, dd in zip(prov, d)}
    for R in (50, 25, 100):
        for s in SOURCES:
            r[f"in{R}_{s}"] = (1.0 if (s in src_d and src_d[s] < R)
                               else (0.0 if s in src_d else 0.5))
    return r


def fit_predict(df_tr, df_te, cols):
    Xtr = df_tr[cols].values
    Xte = df_te[cols].values
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
    B = np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd])
    ytr = (df_tr.err > TAU).astype(float).values
    return 1.0 / (1.0 + np.exp(-B @ _fit(A, ytr)))


def main() -> None:
    cases = load_cases()
    n = len(cases)

    # Arm A: global track records (status quo, reimplemented)
    loo_g = T6.loo_pseudo_radii(cases)
    eps_g = eps_grid(loo_g)
    frame_g = pd.DataFrame([features(c, loo_g, eps_g) for c in cases])
    # R7 uses ε=10 fixed -> separate frame only if ε differs
    frame_g_r7 = (frame_g if eps_g == EPS_R7
                  else pd.DataFrame([features(c, loo_g, EPS_R7) for c in cases]))
    y_g = (frame_g.err > TAU).astype(float).values
    F = folds(y_g.astype(int), seed=0)          # assignment as in status quo

    results = []
    for model, cols, fg in [("S50", ["core_w_50"], frame_g),
                            ("R7", WIN, frame_g_r7)]:
        # Arm A
        ph_a = np.empty(n)
        ya = (fg.err > TAU).astype(float).values
        for fi in range(K):
            tr, te = F != fi, F == fi
            ph_a[te] = fit_predict(fg[tr], fg[te], cols)
        # Arm B: track records per fold from training anchors only
        ph_b = np.empty(n)
        yb = np.empty(n)
        for fi in range(K):
            tr, te = F != fi, F == fi
            train_cases = [c for c, m in zip(cases, tr) if m]
            loo_tr = T6.loo_pseudo_radii(train_cases)
            eps_tr = eps_grid(loo_tr) if model == "S50" else EPS_R7
            df_tr = pd.DataFrame([features(c, loo_tr, eps_tr) for c in train_cases])
            df_te = pd.DataFrame([features(c, loo_tr, eps_tr)
                                  for c, m in zip(cases, te) if m])
            ph_b[te] = fit_predict(df_tr, df_te, cols)
            yb[te] = (df_te.err > TAU).astype(float).values
        row = {"model": model,
               "bss_global": float(bss(ph_a, ya)), "ece_global": float(ece(ph_a, ya)),
               "bss_crossfit": float(bss(ph_b, yb)), "ece_crossfit": float(ece(ph_b, yb)),
               "missrate_global": float(ya.mean()), "missrate_crossfit": float(yb.mean())}
        row["delta_bss"] = row["bss_crossfit"] - row["bss_global"]
        results.append(row)
        print(f"{model:4s}: BSS global {row['bss_global']:+.3f} "
              f"(ECE {row['ece_global']:.3f}, Miss {row['missrate_global']:.3f})  ->  "
              f"cross-fitted {row['bss_crossfit']:+.3f} "
              f"(ECE {row['ece_crossfit']:.3f}, Miss {row['missrate_crossfit']:.3f})  "
              f"Δ {row['delta_bss']:+.3f}")

    pd.DataFrame(results).to_csv(OUT / "crossfit_radii.csv", index=False)
    print(f"\nCSV: {OUT / 'crossfit_radii.csv'}")


if __name__ == "__main__":
    main()
