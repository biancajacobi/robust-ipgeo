"""S improvement: quality-weighted multi-scale concentration (QW) + population transfer.

Motivation: S (line-weighted core share, r=50) reaches an OOF BSS of +0.21 on
anchors but transfers poorly to foreign populations (probes ~+0.11, mobile ~+0.02).
Two pre-declared, disciplined extensions:

  (1) QUALITY WEIGHTING: source mass additionally weighted by the anchor
      track record, w_q = w_line * 1/(LOO median error of the source + eps).
      Deployment-realistic: the track record ALWAYS comes from the anchors
      (labelled), never from the target population.
  (2) MULTI-SCALE PROFILE: concentration at r in {25,50,75,100} km jointly as a
      feature vector in the OOF logit instead of a single radius choice. Anti-
      tuning evidence: the advantage holds at EVERY miss threshold tau in
      {50,100,200} (and even grows with tau) -- not a scale-matching artefact.

Protocol against selection effects: fixed candidate set (5), selection only on
anchors (10-fold OOF, 5 fold seeds); probes/mobile serve exclusively as true
holdout populations for the winner + baseline. Transfer is staged:
raw (anchor fit unchanged) -> intercept recalibration -> Platt (each OOF on
the target population, 1 resp. 2 parameters).

Core result (2026-07-22): QW multi-scale
  Anchors  OOF BSS +0.356 [5 seeds: +0.354..+0.360]   (S50 baseline +0.215)
  tau grid +0.371 / +0.360 / +0.317 at tau=50/100/200 (S50: +0.314/+0.214/+0.184)
  Probes   raw +0.206, Platt +0.233 (S50: +0.108/+0.132) -- above anchor baseline!
  Mobile   (n=60, caution) AP 0.617 vs 0.438; raw miscalibrated (ECE 0.12),
           Platt +0.168/ECE 0.052 -> per-population recalibration layer needed.

Addendum 2026-08-01 (extended mobile set, n=60 -> 178): the finding holds on a
practically disjoint sample (only 7 IPs shared, see probes_mobile_ext):
QW multi Platt +0.164 versus +0.168 at n=60 -- so the small-sample number was
not noise. The broader tag selection is at the same time HARDER (base rate
0.267 -> 0.320) and more miscalibrated in raw form (ECE 0.167), which further
supports the per-population recalibration layer.

Usage:   python experiments/exp_s_quality_weighted.py
         (spawns itself per dataset from DATASETS via GEOIP_DATASET for the
          feature frames; anchor track records via the /--dump-loo interim step.
          GEOIP_DATASETS overrides the list -- "anchors,probes,probes_mobile"
          reproduces the state of 2026-07-22.)
Output:  tables (stdout) + eval/out/s_quality_weighted.csv
         + feature frames eval/out/s_qw_features_<dataset>.csv
           (without an IP column -- safe for the public repo)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.metrics import haversine_error                      # noqa: E402
from experiments.exp_support_concentration import (           # noqa: E402
    folds, _fit, bss, avg_prec, ece, K,
)

OUT = ROOT / "eval" / "out"
RADII = (25, 50, 75, 100)
TAU = 100.0
EPS = 10
LOO_JSON = OUT / "s_qw_loo_anchors.json"
# Datasets: anchors = training/selection population, the rest = transfer targets.
# Overridable via GEOIP_DATASETS (e.g. to exactly reproduce the state of
# 2026-07-22 without the extended mobile set).
DATASETS = tuple(t.strip() for t in (
    os.environ.get("GEOIP_DATASETS")
    or "anchors,probes,probes_mobile,probes_mobile_ext,probes_mobile_extq"
    ).split(",") if t.strip())

CANDIDATES = [
    ("S50 (baseline)", ["core_w_50"]),
    ("S multi", [f"core_w_{r}" for r in RADII]),
    ("KDE multi", [f"kde_w_{r}" for r in RADII]),
    ("QW50", ["core_qw_50"]),
    ("QW multi", [f"core_qw_{r}" for r in RADII]),
]


# --------------------------------------------------------------------------- #
# Sub-phases (run per dataset in a separate process, since store.DATASET
# is fixed from GEOIP_DATASET at import time)
# --------------------------------------------------------------------------- #
def dump_loo():
    """Persist anchor track records (LOO median error per source) for the transfer."""
    from eval.pipeline import load_cases
    import experiments.exp_t6_defaults as T6
    loo = T6.loo_pseudo_radii(load_cases())
    LOO_JSON.write_text(json.dumps({s: {"_global": d["_global"]} for s, d in loo.items()}))
    print(f"LOO track records -> {LOO_JSON}")


def build_frame():
    """Feature frame of the active dataset, using anchor knowledge ONLY (deployment)."""
    from eval.pipeline import load_cases
    import experiments.exp_t6_defaults as T6
    from data import store

    loo = json.loads(LOO_JSON.read_text())
    rows = []
    for c in load_cases():
        prov = c["provenance"]
        pts = np.array([[p["lat"], p["lon"]] for p in prov], float)
        n = len(pts)
        w = np.asarray(T6.line_weights_for(prov, "L1"), float)
        w = w / w.sum()
        q = np.array([1.0 / (loo.get(p["source"], {}).get("_global", 100.0) + EPS)
                      for p in prov])
        wq = w * q
        wq = wq / wq.sum()
        est = np.array(T6.estimate(c, "L1", "b", loo, eps=EPS))   # _global fallback
        d = np.array([haversine_error(tuple(p), tuple(est)) for p in pts])
        r = {"nlines": n, "err": haversine_error(tuple(est), c["truth"])}
        for R in RADII:
            r[f"core_w_{R}"] = float(w[d < R].sum())
            r[f"core_qw_{R}"] = float(wq[d < R].sum())
            r[f"kde_w_{R}"] = float(np.sum(w * np.exp(-(d / R) ** 2)))
        rows.append(r)
    df = pd.DataFrame(rows)
    path = OUT / f"s_qw_features_{store.DATASET}.csv"
    df.to_csv(path, index=False)
    print(f"{store.DATASET}: n={len(df)}  miss rate(>{TAU:.0f}km)={(df.err > TAU).mean():.3f}  -> {path}")


# --------------------------------------------------------------------------- #
# Analyse
# --------------------------------------------------------------------------- #
def _prep(df, cols, mu=None, sd=None):
    raw = np.column_stack([np.log1p(df[c].values) if df[c].max() > 1.5 else df[c].values
                           for c in cols])
    if mu is None:
        mu, sd = raw.mean(0), raw.std(0) + 1e-9
    return (raw - mu) / sd, mu, sd


def _oof(df, cols, y, seed=0):
    F = folds(y, seed)
    ph = np.empty(len(y))
    raw = np.column_stack([np.log1p(df[c].values) if df[c].max() > 1.5 else df[c].values
                           for c in cols])
    for fi in range(K):
        tr, te = F != fi, F == fi
        mu, sd = raw[tr].mean(0), raw[tr].std(0) + 1e-9
        Xtr = np.column_stack([np.ones(tr.sum()), (raw[tr] - mu) / sd])
        Xte = np.column_stack([np.ones(te.sum()), (raw[te] - mu) / sd])
        ph[te] = 1.0 / (1.0 + np.exp(-Xte @ _fit(Xtr, y[tr])))
    return ph


def _transfer(A, yA, T, yT, cols):
    """Anchor fit on the target population: raw / intercept-recal. / Platt (each OOF)."""
    Xa, mu, sd = _prep(A, cols)
    w = _fit(np.column_stack([np.ones(len(A)), Xa]), yA)
    Xt, _, _ = _prep(T, cols, mu, sd)
    z = np.column_stack([np.ones(len(T)), Xt]) @ w
    F = folds(yT, 0)
    ph_i, ph_p = np.empty(len(T)), np.empty(len(T))
    for fi in range(K):
        tr, te = F != fi, F == fi
        b0 = 0.0
        for _ in range(100):                       # 1-parameter IRLS with offset z
            p = 1.0 / (1.0 + np.exp(-(z[tr] + b0)))
            b0 -= (p - yT[tr]).sum() / ((p * (1 - p)).sum() + 1e-9)
        ph_i[te] = 1.0 / (1.0 + np.exp(-(z[te] + b0)))
        wp = _fit(np.column_stack([np.ones(tr.sum()), z[tr]]), yT[tr])
        ph_p[te] = 1.0 / (1.0 + np.exp(-(np.column_stack([np.ones(te.sum()), z[te]]) @ wp)))
    return [("raw", 1.0 / (1.0 + np.exp(-z))), ("intercept recal.", ph_i), ("Platt recal.", ph_p)]


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    # Sub-phases in separate processes (dataset switching)
    subprocess.run([sys.executable, __file__, "--dump-loo"], check=True,
                   env={**os.environ, "GEOIP_DATASET": "anchors"}, cwd=ROOT)
    for ds in DATASETS:
        subprocess.run([sys.executable, __file__, "--build"], check=True,
                       env={**os.environ, "GEOIP_DATASET": ds}, cwd=ROOT)

    frames = {ds: pd.read_csv(OUT / f"s_qw_features_{ds}.csv") for ds in DATASETS}
    A = frames["anchors"]
    yA = (A.err > TAU).astype(int).values
    out_rows = []

    print("\n" + "=" * 78)
    print("CANDIDATES on ANCHORS (deployment frame, 10-fold OOF, 5 fold seeds)")
    print("=" * 78)
    for name, cols in CANDIDATES:
        b = [bss(_oof(A, cols, yA, s), yA) for s in range(5)]
        ph0 = _oof(A, cols, yA, 0)
        print(f"{name:16s} BSS {np.mean(b):+.3f} [{min(b):+.3f},{max(b):+.3f}]  "
              f"AP {avg_prec(ph0, yA):.3f}  ECE {ece(ph0, yA):.3f}")
        out_rows.append({"section": "anchors_oof", "candidate": name, "metric": "bss_mean5",
                         "value": round(float(np.mean(b)), 3),
                         "extra": f"[{min(b):+.3f},{max(b):+.3f}] AP={avg_prec(ph0, yA):.3f}"})

    print("\ntau robustness (anti scale matching; OOF seed 0):")
    for name, cols in (CANDIDATES[0], CANDIDATES[4]):
        vals = []
        for tau in (50, 100, 200):
            yt = (A.err > tau).astype(int).values
            vals.append(bss(_oof(A, cols, yt, 0), yt))
        print(f"  {name:16s} " + "  ".join(f"tau{t}={v:+.3f}" for t, v in zip((50, 100, 200), vals)))
        out_rows.append({"section": "tau_grid", "candidate": name, "metric": "bss@tau50/100/200",
                         "value": "", "extra": "/".join(f"{v:+.3f}" for v in vals)})

    for ds in DATASETS[1:]:
        T = frames[ds]
        yT = (T.err > TAU).astype(int).values
        print(f"\nTRANSFER anchors -> {ds.upper()} (n={len(T)}, base rate {yT.mean():.3f})"
              + ("  ** small sample **" if len(T) < 100 else ""))
        for name, cols in (CANDIDATES[0], CANDIDATES[4]):
            print(f"  {name}:")
            for lab, ph in _transfer(A, yA, T, yT, cols):
                print(f"    {lab:18s} BSS {bss(ph, yT):+.3f}  AP {avg_prec(ph, yT):.3f}  "
                      f"ECE {ece(ph, yT):.3f}")
                out_rows.append({"section": f"transfer_{ds}", "candidate": f"{name} / {lab}",
                                 "metric": "bss", "value": round(float(bss(ph, yT)), 3),
                                 "extra": f"AP={avg_prec(ph, yT):.3f} ECE={ece(ph, yT):.3f}"})

    pd.DataFrame(out_rows).to_csv(OUT / "s_quality_weighted.csv", index=False)
    print(f"\nCSV: {OUT}/s_quality_weighted.csv")


if __name__ == "__main__":
    if "--dump-loo" in sys.argv:
        dump_loo()
    elif "--build" in sys.argv:
        build_frame()
    else:
        run()
