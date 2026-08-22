"""S improvement round 2: ceiling analysis + lineage-pattern risk model.

Builds on exp_s_quality_weighted.py (QW multi, OOF-BSS +0.356) and answers
two questions:

(1) HOW HIGH IS THE CEILING? "Silent" misses (entire line mass < 100 km
    around the estimator, error nonetheless > 100 km) are in principle
    invisible to EVERY consensus-geometric measure. Anchors: only 7/129
    misses silent -> oracle upper bound BSS ~ +0.94 (probes ~ +0.82,
    mobile ~ +0.84). The ceiling is NOT the problem.

(2) WHAT REALISTICALLY GAINS MORE? Pre-specified feature levers on QW multi:
    distance quantiles, per-lineage in-core indicators (WHICH source is
    missing from the core?), city consensus, combination. Winner: LINEAGE
    PATTERN (QW + in50 indicators per source, k=13): anchors OOF-BSS +0.599
    [5 seeds +0.587..+0.605], in-sample +0.630 (barely any overfit gap),
    tau-robust (+0.613/+0.605/+0.540 at tau=50/100/200).
    Interpretation: the IDENTITY of the core deviators contributes (learned
    credibility voting per provider line), generalizes hub axis and QW.

Transfer (track records + model from anchors only):
    Probes  raw +0.240 / Platt +0.270 (refitted within-population: +0.331)
    Mobile  n=60: in50 overfits there (-0.14 on refit); best mobile result
            remains QW multi transfer+Platt (+0.168) -> the mobile
            bottleneck is DATA.

Caveats: (a) lineage-pattern weights depend on the DB snapshot ->
recalibration cadence needed; (b) 13 features = risk MODEL, no longer a
single measure (framing: S = definition, QW profile = calibrated score,
lineage model = full triage); (c) city consensus contributes nothing
(+0.352), squared terms diverge in the IRLS (k>200).

Addendum 2026-08-01 (probes_mobile_ext, n=178 instead of 60, only 7 IPs
overlap): the cellular statement holds, but the ceiling there does NOT.
In detail:
  - QW multi stays ahead on cellular (Platt +0.164; adjusted +0.170) and
    thereby reproduces +0.168 from n=60 on an effectively new sample.
  - in50 keeps losing on cellular (Platt +0.123), but the collapse on the
    within-population refit was a small-sample artifact
    (-0.144 at n=60 -> +0.077 at n=178).
  - the CEILING collapses: silent common-mode misses 18/57 instead of 2/16,
    oracle BSS +0.63 instead of +0.84. On a more broadly drawn cellular
    sample, about a third of the misses are invisible to EVERY consensus
    measure -- the statement "consensus geometry is not exhausted" holds for
    anchors/fixed line, not for cellular. There, external evidence is needed
    (RTT/active measurement, ASN type).

Invocation: python experiments/exp_s_lineage_risk.py
Output: tables (stdout) + eval/out/s_lineage_risk.csv
        + feature frames eval/out/s_risk_features_{anchors,probes,probes_mobile,probes_mobile_ext,probes_mobile_extq}.csv
          (no IP column)
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
# Datasets: anchors = training/selection population, rest = transfer targets.
# Overridable via GEOIP_DATASETS (reproduction of the state as of 2026-07-22).
DATASETS = tuple(t.strip() for t in (
    os.environ.get("GEOIP_DATASETS")
    or "anchors,probes,probes_mobile,probes_mobile_ext,probes_mobile_extq"
    ).split(",") if t.strip())
SOURCES = ("dbip_lite", "geojs", "ip2location_lite", "ip_api", "ipapi_co",
           "ipinfo", "ipwho_is", "maxmind_geolite2", "reallyfreegeoip")
QW = [f"core_qw_{r}" for r in RADII]
DQ = ["dq25", "dq50", "dq75", "dq90", "dmax", "dwmean"]
SRC = [f"in50_{s}" for s in SOURCES]
CITY = ["city_mass", "city_n", "city_cov"]


def build_frame():
    """Extended frame (anchor knowledge only): QW profile + distance quantiles +
    per-lineage in-core indicators + city consensus."""
    from eval.pipeline import load_cases
    import experiments.exp_t6_defaults as T6
    from data import store

    loo = json.loads(LOO_JSON.read_text())
    suffix = "" if store.DATASET == "anchors" else f"_{store.DATASET}"
    obs = pd.read_csv(ROOT / "data" / "cache" / f"observations{suffix}.csv", dtype=str)
    city_map = {(r.ip, r.source): (str(r.city).strip().lower()
                                   if isinstance(r.city, str) and str(r.city).strip() else None)
                for r in obs.itertuples()}
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
        est = np.array(T6.estimate(c, "L1", "b", loo, eps=EPS))
        d = np.array([haversine_error(tuple(p), tuple(est)) for p in pts])
        r = {"nlines": n, "err": haversine_error(tuple(est), c["truth"])}
        for R in RADII:
            r[f"core_w_{R}"] = float(w[d < R].sum())
            r[f"core_qw_{R}"] = float(wq[d < R].sum())
        for qq in (25, 50, 75, 90):
            r[f"dq{qq}"] = float(np.percentile(d, qq))
        r["dmax"], r["dwmean"] = float(d.max()), float(np.sum(w * d))
        src_d = {p["source"]: dd for p, dd in zip(prov, d)}
        for s in SOURCES:               # 1=in core, 0=outside, 0.5=source missing
            r[f"in50_{s}"] = 1.0 if (s in src_d and src_d[s] < 50) else (0.0 if s in src_d else 0.5)
        cities = [city_map.get((c["ip"], p["source"])) for p in prov]
        mask = np.array([ci is not None for ci in cities])
        if mask.any():
            wm = w[mask] / w[mask].sum()
            vals: dict[str, float] = {}
            for ci, wi in zip([ci for ci in cities if ci is not None], wm):
                vals[ci] = vals.get(ci, 0.0) + wi
            r["city_mass"], r["city_n"] = float(max(vals.values())), float(len(vals))
        else:
            r["city_mass"], r["city_n"] = 0.5, float(n)
        r["city_cov"] = float(mask.mean())
        rows.append(r)
    df = pd.DataFrame(rows)
    path = OUT / f"s_risk_features_{store.DATASET}.csv"
    df.to_csv(path, index=False)
    print(f"{store.DATASET}: n={len(df)}  miss rate={(df.err > TAU).mean():.3f}  -> {path}")


def _mat(df, cols):
    return np.column_stack([np.log1p(df[c].values) if df[c].max() > 1.5 else df[c].values
                            for c in cols])


def _oof(df, cols, y, seed=0):
    X = _mat(df, cols)
    F = folds(y, seed)
    ph = np.empty(len(y))
    for fi in range(K):
        tr, te = F != fi, F == fi
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        ph[te] = 1.0 / (1.0 + np.exp(-(np.column_stack([np.ones(te.sum()), (X[te] - mu) / sd])
                                       @ _fit(np.column_stack([np.ones(tr.sum()), (X[tr] - mu) / sd]),
                                              y[tr]))))
    return ph


def _transfer(A, yA, T, yT, cols):
    Xa, Xt = _mat(A, cols), _mat(T, cols)
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-9
    w = _fit(np.column_stack([np.ones(len(A)), (Xa - mu) / sd]), yA)
    z = np.column_stack([np.ones(len(T)), (Xt - mu) / sd]) @ w
    F = folds(yT, 0)
    ph_p = np.empty(len(T))
    for fi in range(K):
        tr, te = F != fi, F == fi
        wp = _fit(np.column_stack([np.ones(tr.sum()), z[tr]]), yT[tr])
        ph_p[te] = 1.0 / (1.0 + np.exp(-(np.column_stack([np.ones(te.sum()), z[te]]) @ wp)))
    return 1.0 / (1.0 + np.exp(-z)), ph_p


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    if not LOO_JSON.exists():
        subprocess.run([sys.executable, str(ROOT / "experiments" / "exp_s_quality_weighted.py"),
                        "--dump-loo"], check=True,
                       env={**os.environ, "GEOIP_DATASET": "anchors"}, cwd=ROOT)
    for ds in DATASETS:
        subprocess.run([sys.executable, __file__, "--build"], check=True,
                       env={**os.environ, "GEOIP_DATASET": ds}, cwd=ROOT)

    frames = {ds: pd.read_csv(OUT / f"s_risk_features_{ds}.csv") for ds in DATASETS}
    A = frames["anchors"]
    yA = (A.err > TAU).astype(int).values
    out_rows = []

    print("\nCEILING — silent common-mode misses (invisible to every consensus measure):")
    for ds in DATASETS:
        T = frames[ds]
        yT = (T.err > TAU).astype(int).values
        silent = T["core_w_100"].values >= 0.999
        m = yT == 1
        ph_o = np.where(silent, yT[silent].mean() if silent.any() else 0.0, yT.astype(float))
        print(f"  {ds:14s} {int((silent & m).sum())}/{int(m.sum())} misses silent  "
              f"-> oracle BSS {bss(ph_o, yT):+.3f}")
        out_rows.append({"section": "ceiling", "candidate": ds, "metric": "oracle_bss",
                         "value": round(float(bss(ph_o, yT)), 3),
                         "extra": f"silent {int((silent & m).sum())}/{int(m.sum())}"})

    print("\nCANDIDATES (anchors, 10-fold OOF, 5 seeds):")
    CAND = [("QW multi (reference)", QW), ("+ distance quantiles", QW + DQ),
            ("+ per-lineage in50", QW + SRC), ("+ city consensus", QW + CITY),
            ("all combined", QW + DQ + SRC + CITY)]
    for name, cols in CAND:
        b = [bss(_oof(A, cols, yA, s), yA) for s in range(5)]
        print(f"  {name:22s} BSS {np.mean(b):+.3f} [{min(b):+.3f},{max(b):+.3f}]  (k={len(cols)})")
        out_rows.append({"section": "anchors_oof", "candidate": name, "metric": "bss_mean5",
                         "value": round(float(np.mean(b)), 3),
                         "extra": f"[{min(b):+.3f},{max(b):+.3f}] k={len(cols)}"})

    WIN = QW + SRC
    print("\nWINNER 'QW + in50' — robustness:")
    for tau in (50, 100, 200):
        yt = (A.err > tau).astype(int).values
        v = bss(_oof(A, WIN, yt, 0), yt)
        print(f"  tau={tau}: OOF-BSS {v:+.3f}")
        out_rows.append({"section": "tau_grid", "candidate": "QW+in50",
                         "metric": f"bss@tau{tau}", "value": round(float(v), 3), "extra": ""})
    X = _mat(A, WIN)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = np.column_stack([np.ones(len(A)), (X - mu) / sd])
    gap = bss(1.0 / (1.0 + np.exp(-(Xs @ _fit(Xs, yA)))), yA)
    print(f"  in-sample BSS {gap:+.3f} (overfit gap to the OOF number small)")

    print("\nTRANSFER (model + track records from anchors only):")
    for ds in DATASETS[1:]:
        T = frames[ds]
        yT = (T.err > TAU).astype(int).values
        warn = "  ** small sample **" if len(T) < 100 else ""
        print(f"  -> {ds} (n={len(T)}, base rate {yT.mean():.3f}){warn}")
        for name, cols in (("QW multi", QW), ("QW+in50", WIN)):
            ph_raw, ph_p = _transfer(A, yA, T, yT, cols)
            ph_int = _oof(T, cols, yT, 0)          # within-population refit
            print(f"     {name:9s} raw BSS {bss(ph_raw, yT):+.3f}/ECE {ece(ph_raw, yT):.3f}  "
                  f"Platt {bss(ph_p, yT):+.3f}/ECE {ece(ph_p, yT):.3f}  "
                  f"within-pop {bss(ph_int, yT):+.3f}")
            out_rows.append({"section": f"transfer_{ds}", "candidate": name, "metric": "bss",
                             "value": round(float(bss(ph_p, yT)), 3),
                             "extra": f"raw={bss(ph_raw, yT):+.3f} internal={bss(ph_int, yT):+.3f}"})

    pd.DataFrame(out_rows).to_csv(OUT / "s_lineage_risk.csv", index=False)
    print(f"\nCSV: {OUT}/s_lineage_risk.csv")


if __name__ == "__main__":
    if "--build" in sys.argv:
        build_frame()
    else:
        run()
