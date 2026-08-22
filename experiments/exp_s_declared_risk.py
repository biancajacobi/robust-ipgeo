"""S improvement round 3: three levers against the gap to the ceiling.

Starting point after round 2 (exp_s_lineage_risk.py): QW + in50 reaches
OOF-BSS +0.599 on the anchors against an oracle ceiling of +0.94 -- so there
is still headroom in the VISIBLE geometry. The silent misses are too rare on
anchors (7/129) to explain this gap; it must therefore lie in cases the model
COULD see. Three previously unused levers:

(1) DECLARED UNCERTAINTY (new information channel). All features so far are
    consensus-geometric: they measure how closely the sources cluster.
    What was not used is what the sources say about THEMSELVES:
      - ``rad_mm``  MaxMind's declared accuracy_radius (the only source in
                    the portfolio that ships a radius; median 50 km anchors,
                    100 km cellular)
      - ``dc_mass``/``dc_n``  weight mass resp. number of answers detected
                    as default centroid (country/hub midpoint instead of a
                    real localization)
      - ``ctry_n``/``ctry_mass``  country consensus (coarser than the city
                    consensus from round 2, which contributed nothing)
      - ``nlines``  number of responding SOURCES, not effective lines (the
        GeoLite replicas count individually; was in the frame, in no candidate)
    Mechanism rationale: a case in which all sources report the same default
    centroid looks geometrically tight (high concentration), but is exactly
    the common-mode case. Default flag and declared radius are partly
    independent of the consensus geometry.

(2) REGULARIZATION. In round 2, squared terms diverged in the unregularized
    IRLS (k>200, negative BSS = artifact). Here, lambda is chosen on a grid
    via INNER CV on the training folds (never on the test fold), so that
    richer models -- interactions QW x declared uncertainty -- become fairly
    assessable at all.

(3) THE WINNER ALONG ITS OWN AXIS. in50 asks per source only "within
    50 km?". Never tried was what took S -> QW: the same question at
    MULTIPLE radii (in25/in50/in100 per source) and the per-source distance
    continuous instead of binarized. Selection here runs exclusively on the
    anchors, without any holdout contact -- this lever is protocol-clean.

RESULT: (1) and (2) are dead ends -- declared +0.352 vs. QW alone +0.356,
combined with in50 even worse (+0.584 vs. +0.599); ridge with interactions
stays at +0.587. So the divergence from round 2 concealed no gain. (3)
delivers: **R7 = QW + in25/in50/in100 per source, OOF-BSS +0.625**
[+0.616..+0.632], tau-robust, and better on EVERY target population. Nested
selection (the candidate choice itself cross-validated): +0.608 (since the
imputation-leakage fix; +0.603 before) -- the
honest number for the procedure; R7's +0.625 is slightly optimistic.

PROTOCOL / DISCLOSURE: the candidate set is pre-specified; selection takes
place on the anchors (10-fold OOF, 5 fold seeds). The DECLARED channel
(R2-R6) was added after a pilot analysis that also included
probes/probes_mobile_ext -- for it, those populations are not an untouched
holdout (it contributes nothing anyway). Lever (3)/R7 had no holdout contact
whatsoever. Additionally, a fresh, disjoint sample was drawn AFTER the
selection (``probes_holdout``, n=399, via ``fetch_probes.py
--exclude-datasets``) -- there the ranking is confirmed cleanly:
QW multi +0.227 < QW+in50 +0.282 < R7 **+0.315** (Platt).

Invocation: python experiments/exp_s_declared_risk.py
         (GEOIP_DATASETS overrides the dataset list)
Output: tables (stdout) + eval/out/s_declared_risk.csv
        + feature frames eval/out/s_decl_features_<dataset>.csv (no IP column)
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
    folds, bss, avg_prec, ece, K,
)

OUT = ROOT / "eval" / "out"
RADII = (25, 50, 75, 100)
TAU = 100.0
EPS = 10
LOO_JSON = OUT / "s_qw_loo_anchors.json"
DATASETS = tuple(t.strip() for t in (
    os.environ.get("GEOIP_DATASETS")
    or "anchors,probes,probes_holdout,probes_mobile,probes_mobile_ext,probes_mobile_extq"
    ).split(",") if t.strip())
SOURCES = ("dbip_lite", "geojs", "ip2location_lite", "ip_api", "ipapi_co",
           "ipinfo", "ipwho_is", "maxmind_geolite2", "reallyfreegeoip")

QW = [f"core_qw_{r}" for r in RADII]
DQ = ["dq25", "dq50", "dq75", "dq90", "dmax", "dwmean"]
SRC = [f"in50_{s}" for s in SOURCES]
DECL = ["rad_mm", "rad_mm_missing", "dc_mass", "dc_n", "ctry_mass", "ctry_n", "nlines"]
SRC_MULTI = [f"in{R}_{s}" for R in (25, 100) for s in SOURCES]   # in50_* comes from SRC
DIST = [f"dist_{s}" for s in SOURCES]

LAMBDAS = (1e-6, 1e-2, 1.0, 10.0)
INNER_FOLDS = 3


# --------------------------------------------------------------------------- #
# Connection class from the RIPE probe tags
# --------------------------------------------------------------------------- #
# CAUTION: ``t-mobile`` is a PROVIDER tag, not an access-technology tag -- 17 of
# the 183 probes in probes_mobile_ext entered exclusively through it and at the
# same time carry fibre/ftth/home, i.e. they are fiber home connections. For the
# classification, only the access-technology tags therefore count.
CELLULAR = {"mobile", "3g", "4g", "5g", "lte"}
FIXEDLINE = {"home", "nat"}
DATACENTRE = {"datacentre", "datacenter"}


def load_tags(dataset: str) -> dict[str, set[str]]:
    path = ROOT / "data" / "cache" / f"{dataset}_tags.csv"
    if not path.exists():
        return {}
    import csv
    with open(path, encoding="utf-8") as fh:
        return {r["ip"]: {t for t in r["tags"].split(";") if t}
                for r in csv.DictReader(fh)}


def classify(tags: set[str]) -> str:
    """Connection class; same order as exp_probes_compare.classify, but
    ``mobile`` extended to ``cellular`` (3g/4g/5g/lte) and without provider tags."""
    if DATACENTRE & tags:
        return "datacentre"
    if CELLULAR & tags:
        return "cellular"
    if FIXEDLINE & tags:
        return "home/nat"
    return "other"


# --------------------------------------------------------------------------- #
# Frame
# --------------------------------------------------------------------------- #
def build_frame():
    """Frame from round 2 + declared uncertainty (anchor knowledge only)."""
    from eval.pipeline import load_cases
    import experiments.exp_t6_defaults as T6
    from data import store

    loo = json.loads(LOO_JSON.read_text())
    suffix = "" if store.DATASET == "anchors" else f"_{store.DATASET}"
    obs = pd.read_csv(ROOT / "data" / "cache" / f"observations{suffix}.csv")
    by = {(r.ip, r.source): r for r in obs.itertuples()}
    tags = load_tags(store.DATASET)

    rows = []
    for c in load_cases():
        prov = c["provenance"]
        pts = np.array([[p["lat"], p["lon"]] for p in prov], float)
        w = np.asarray(T6.line_weights_for(prov, "L1"), float)
        w = w / w.sum()
        q = np.array([1.0 / (loo.get(p["source"], {}).get("_global", 100.0) + EPS)
                      for p in prov])
        wq = w * q
        wq = wq / wq.sum()
        est = np.array(T6.estimate(c, "L1", "b", loo, eps=EPS))
        d = np.array([haversine_error(tuple(p), tuple(est)) for p in pts])

        # nlines = number of SOURCES (len(prov)), not line-deduplicated -- name is historical
        r = {"nlines": float(len(prov)), "err": haversine_error(tuple(est), c["truth"])}
        for R in RADII:
            r[f"core_w_{R}"] = float(w[d < R].sum())
            r[f"core_qw_{R}"] = float(wq[d < R].sum())
        for qq in (25, 50, 75, 90):
            r[f"dq{qq}"] = float(np.percentile(d, qq))
        r["dmax"], r["dwmean"] = float(d.max()), float(np.sum(w * d))
        src_d = {p["source"]: dd for p, dd in zip(prov, d)}
        for s in SOURCES:
            r[f"in50_{s}"] = 1.0 if (s in src_d and src_d[s] < 50) else (0.0 if s in src_d else 0.5)
        # Lever (3): the winner from round 2 widened along ITS own axis --
        # in-core indicator per source at multiple radii (analogous to the multi-scale
        # step that took S -> QW) and the per-source distance continuous instead of
        # binarized.
        for R in (25, 100):
            for s in SOURCES:
                r[f"in{R}_{s}"] = (1.0 if (s in src_d and src_d[s] < R)
                                   else (0.0 if s in src_d else 0.5))
        for s in SOURCES:
            r[f"dist_{s}"] = float(src_d[s]) if s in src_d else np.nan

        # --- declared uncertainty ---
        rad, dc, ctry = [], [], []
        for p in prov:
            o = by.get((c["ip"], p["source"]))
            rad.append(float(o.accuracy_radius)
                       if o is not None and pd.notna(o.accuracy_radius) else np.nan)
            dc.append(1.0 if (o is not None
                              and str(o.is_default_centroid).strip().lower() in ("true", "1"))
                      else 0.0)
            ctry.append(str(o.country).strip().lower()
                        if o is not None and isinstance(o.country, str) and str(o.country).strip()
                        else None)
        rad = np.asarray(rad, float)
        dc = np.asarray(dc, float)
        have = ~np.isnan(rad)
        # Only MaxMind ships a radius -> median of the present values = that value.
        # If absent, NaN is kept (imputation per training fold in _impute);
        # the indicator keeps the information "no radius available" separate.
        r["rad_mm"] = float(np.median(rad[have])) if have.any() else np.nan
        r["rad_mm_missing"] = 0.0 if have.any() else 1.0
        r["dc_mass"] = float(w[dc > 0].sum())
        r["dc_n"] = float(dc.sum())
        vals: dict[str, float] = {}
        for ci, wi in zip(ctry, w):
            if ci:
                vals[ci] = vals.get(ci, 0.0) + wi
        r["ctry_mass"] = float(max(vals.values())) if vals else 0.5
        r["ctry_n"] = float(len(vals)) if vals else 1.0
        # Connection class (not a model feature -- only for the evaluation by
        # category; in operation it is NOT known).
        r["cat"] = classify(tags[c["ip"]]) if c["ip"] in tags else "unknown"
        rows.append(r)

    df = pd.DataFrame(rows)
    # rad_mm and dist_*: NaN is PRESERVED -- the median imputation happens in
    # _oof/_transfer per TRAINING fold (_impute), never over the whole frame
    # (otherwise leakage of the test-fold distribution into the features;
    # review 2026-08-18).
    keep_nan = ["rad_mm"] + [f"dist_{s}" for s in SOURCES]
    fill_cols = [c for c in df.columns if c not in keep_nan and df[c].dtype != object]
    df[fill_cols] = df[fill_cols].fillna(0.0)
    path = OUT / f"s_decl_features_{store.DATASET}.csv"
    df.to_csv(path, index=False)
    print(f"{store.DATASET}: n={len(df)}  miss rate={(df.err > TAU).mean():.3f}  -> {path}")


# --------------------------------------------------------------------------- #
# Fit: IRLS with selectable ridge (intercept remains unpenalized)
# --------------------------------------------------------------------------- #
def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def _fit_ridge(X, y, lam, it=200):
    w = np.zeros(X.shape[1])
    P = lam * np.eye(X.shape[1])
    P[0, 0] = 0.0                      # do not regularize the intercept
    for _ in range(it):
        p = _sig(X @ w)
        W = p * (1 - p) + 1e-9
        H = (X * W[:, None]).T @ X + P + 1e-9 * np.eye(X.shape[1])
        w -= np.linalg.solve(H, X.T @ (p - y) + P @ w)
    return w


def _impute(Xtr, Xte):
    """Median imputation ONLY from the training part (rad_mm/dist_* carry NaN).

    The training median (after log1p, see _mat) fills both parts; a column
    completely empty in training falls back to 0 (corresponds to the old
    fillna(0.0) for empty populations). No test-fold influence on the
    imputation (review 2026-08-18)."""
    med = np.array([np.nanmedian(c) if np.any(~np.isnan(c)) else 0.0
                    for c in Xtr.T])
    return (np.where(np.isnan(Xtr), med, Xtr),
            np.where(np.isnan(Xte), med, Xte))


def _mat(df, cols, inter=False):
    # Known latent limitation (review 2026-08-18): the log1p decision is made
    # on the FULL frame and is made ANEW per population. For the feature sets
    # actually transferred (QW/SRC/SRC_MULTI, all in [0,1]) the branch never
    # triggers; for future candidates with ranges around 1.5 the decision
    # would have to be fixed on the anchors.
    X = np.column_stack([np.log1p(df[c].values) if df[c].max() > 1.5 else df[c].values
                         for c in cols])
    if inter:
        qi = [cols.index(c) for c in QW if c in cols]
        di = [cols.index(c) for c in DECL if c in cols]
        if qi and di:
            X = np.column_stack([X] + [X[:, a] * X[:, b] for a in qi for b in di])
    return X


def _logloss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _pick_lambda(Xtr, ytr, seed):
    """Inner lambda choice on the TRAINING data (INNER_FOLDS=3 folds of a
    10-way split) -- the outer test fold is never seen."""
    if len(LAMBDAS) == 1:
        return LAMBDAS[0]
    Fi = folds(ytr, seed + 1000)[: len(ytr)]
    best, best_ll = LAMBDAS[0], np.inf
    for lam in LAMBDAS:
        ll = []
        for fi in range(min(INNER_FOLDS, K)):
            tr, te = Fi != fi, Fi == fi
            if te.sum() == 0 or ytr[tr].sum() == 0:
                continue
            mu, sd = Xtr[tr].mean(0), Xtr[tr].std(0) + 1e-9
            A = np.column_stack([np.ones(tr.sum()), (Xtr[tr] - mu) / sd])
            B = np.column_stack([np.ones(te.sum()), (Xtr[te] - mu) / sd])
            ll.append(_logloss(ytr[te], _sig(B @ _fit_ridge(A, ytr[tr], lam))))
        if ll and np.mean(ll) < best_ll:
            best, best_ll = lam, float(np.mean(ll))
    return best


def _oof(df, cols, y, seed=0, inter=False, ridge=False):
    X = _mat(df, cols, inter)
    F = folds(y, seed)
    ph = np.empty(len(y))
    lams = []
    for fi in range(K):
        tr, te = F != fi, F == fi
        Xtr, Xte = _impute(X[tr], X[te])
        lam = _pick_lambda(Xtr, y[tr], seed) if ridge else 1e-6
        lams.append(lam)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        A = np.column_stack([np.ones(tr.sum()), (Xtr - mu) / sd])
        B = np.column_stack([np.ones(te.sum()), (Xte - mu) / sd])
        ph[te] = _sig(B @ _fit_ridge(A, y[tr], lam))
    return ph, lams


def _transfer(A, yA, T, yT, cols, inter=False, ridge=False):
    """Anchor fit on target population: raw / Platt / within-population."""
    Xa, Xt = _impute(*(_mat(A, cols, inter), _mat(T, cols, inter)))
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-9
    lam = _pick_lambda(Xa, yA, 0) if ridge else 1e-6
    w = _fit_ridge(np.column_stack([np.ones(len(A)), (Xa - mu) / sd]), yA, lam)
    z = np.column_stack([np.ones(len(T)), (Xt - mu) / sd]) @ w
    F = folds(yT, 0)
    ph_p = np.empty(len(T))
    for fi in range(K):
        tr, te = F != fi, F == fi
        wp = _fit_ridge(np.column_stack([np.ones(tr.sum()), z[tr]]), yT[tr], 1e-6)
        ph_p[te] = _sig(np.column_stack([np.ones(te.sum()), z[te]]) @ wp)
    return _sig(z), ph_p


# --------------------------------------------------------------------------- #
# Pre-specified candidate set
# --------------------------------------------------------------------------- #
CANDIDATES = [
    ("R0 QW multi (ref. R1)",   QW,                   False, False),
    ("R1 QW+in50 (winner R2)",  QW + SRC,             False, False),
    ("R2 QW+declared",          QW + DECL,            False, False),
    ("R3 QW+in50+declared",     QW + SRC + DECL,      False, False),
    ("R4 everything (+dists)",  QW + SRC + DQ + DECL, False, False),
    ("R5 = R3, ridge tuned",    QW + SRC + DECL,      False, True),
    ("R6 = R5 + interactions",  QW + SRC + DECL,      True,  True),
    # Lever (3): winner along its own axis -- selection ONLY on anchors,
    # no holdout contact, hence protocol-clean (unlike the declared channel).
    ("R7 QW+in25/50/100",       QW + SRC + SRC_MULTI,        False, True),
    ("R8 QW+dist per source",   QW + DIST,                   False, True),
    ("R9 QW+in50+dist",         QW + SRC + DIST,             False, True),
    ("R10 in-profile+dist",     QW + SRC + SRC_MULTI + DIST, False, True),
]


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    if not LOO_JSON.exists():
        subprocess.run([sys.executable, str(ROOT / "experiments" / "exp_s_quality_weighted.py"),
                        "--dump-loo"], check=True,
                       env={**os.environ, "GEOIP_DATASET": "anchors"}, cwd=ROOT)
    for ds in DATASETS:
        subprocess.run([sys.executable, __file__, "--build"], check=True,
                       env={**os.environ, "GEOIP_DATASET": ds}, cwd=ROOT)

    frames = {ds: pd.read_csv(OUT / f"s_decl_features_{ds}.csv") for ds in DATASETS}
    A = frames["anchors"]
    yA = (A.err > TAU).astype(int).values
    out_rows = []

    print("\n" + "=" * 78)
    print("CANDIDATES (anchors, 10-fold OOF, 5 fold seeds) — selection ONLY here")
    print("=" * 78)
    results = {}
    for name, cols, inter, ridge in CANDIDATES:
        b, lam_used = [], []
        for s in range(5):
            ph, lams = _oof(A, cols, yA, s, inter, ridge)
            b.append(bss(ph, yA))
            lam_used += lams
        ph0, _ = _oof(A, cols, yA, 0, inter, ridge)
        k = _mat(A, cols, inter).shape[1]
        lam_txt = f"  lambda~{np.median(lam_used):g}" if ridge else ""
        print(f"  {name:26s} BSS {np.mean(b):+.3f} [{min(b):+.3f},{max(b):+.3f}]  "
              f"AP {avg_prec(ph0, yA):.3f}  ECE {ece(ph0, yA):.3f}  (k={k}){lam_txt}")
        results[name] = float(np.mean(b))
        out_rows.append({"section": "anchors_oof", "candidate": name, "metric": "bss_mean5",
                         "value": round(float(np.mean(b)), 3),
                         "extra": f"[{min(b):+.3f},{max(b):+.3f}] AP={avg_prec(ph0, yA):.3f} k={k}{lam_txt}"})

    win_name = max(results, key=results.get)
    wcols, winter, wridge = next((c, i, r) for n, c, i, r in CANDIDATES if n == win_name)
    print(f"\nWINNER: {win_name}  (BSS {results[win_name]:+.3f})")

    print("\ntau robustness of the winner (anti scale matching, OOF seed 0):")
    for tau in (50, 100, 200):
        yt = (A.err > tau).astype(int).values
        ph, _ = _oof(A, wcols, yt, 0, winter, wridge)
        v = bss(ph, yt)
        print(f"  tau={tau}: OOF-BSS {v:+.3f}")
        out_rows.append({"section": "tau_grid", "candidate": win_name,
                         "metric": f"bss@tau{tau}", "value": round(float(v), 3), "extra": ""})

    print("\nTRANSFER (model + track records from anchors only):")
    ref = [("R0 QW multi", QW, False, False),
           ("R1 QW+in50", QW + SRC, False, False),
           (win_name, wcols, winter, wridge)]
    for ds in DATASETS[1:]:
        T = frames[ds]
        yT = (T.err > TAU).astype(int).values
        warn = "  ** small sample **" if len(T) < 100 else ""
        print(f"  -> {ds} (n={len(T)}, base rate {yT.mean():.3f}){warn}")
        for name, cols, inter, ridge in ref:
            ph_raw, ph_p = _transfer(A, yA, T, yT, cols, inter, ridge)
            ph_int, _ = _oof(T, cols, yT, 0, inter, ridge)
            print(f"     {name:26s} raw {bss(ph_raw, yT):+.3f}  Platt {bss(ph_p, yT):+.3f}"
                  f"/ECE {ece(ph_p, yT):.3f}  within-pop {bss(ph_int, yT):+.3f}")
            out_rows.append({"section": f"transfer_{ds}", "candidate": name, "metric": "bss",
                             "value": round(float(bss(ph_p, yT)), 3),
                             "extra": f"raw={bss(ph_raw, yT):+.3f} internal={bss(ph_int, yT):+.3f}"})

    if "--nested" in sys.argv:
        print("\nNESTED SELECTION (honest number: the candidate choice itself is"
              " cross-validated)")
        yv = yA
        F = folds(yv, 0)
        ph_nested = np.empty(len(yv))
        picks = []
        for fi in range(K):
            tr, te = F != fi, F == fi
            Atr = A[tr].reset_index(drop=True)
            ytr = yv[tr]
            best, best_b = None, -np.inf
            for name, cols, inter, ridge in CANDIDATES:
                ph_in, _ = _oof(Atr, cols, ytr, 0, inter, False)   # training part only;
                # ranking without inner lambda search (cost), final fit below with lambda
                b = bss(ph_in, ytr)
                if b > best_b:
                    best, best_b = (name, cols, inter, ridge), b
            picks.append(best[0])
            name, cols, inter, ridge = best
            X = _mat(A, cols, inter)
            Xtr, Xte = _impute(X[tr], X[te])
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            lam = _pick_lambda(Xtr, ytr, 0) if ridge else 1e-6
            wgt = _fit_ridge(np.column_stack([np.ones(tr.sum()), (Xtr - mu) / sd]), ytr, lam)
            ph_nested[te] = _sig(
                np.column_stack([np.ones(te.sum()), (Xte - mu) / sd]) @ wgt)
        from collections import Counter
        cnt = Counter(picks)
        print(f"  nested OOF-BSS {bss(ph_nested, yv):+.3f}  "
              f"(naively chosen: {results[win_name]:+.3f})")
        print("  fold choice: " + ", ".join(f"{k}x {n}" for n, k in cnt.most_common()))
        out_rows.append({"section": "nested", "candidate": "nested selection",
                         "metric": "bss", "value": round(float(bss(ph_nested, yv)), 3),
                         "extra": "; ".join(f"{k}x {n}" for n, k in cnt.most_common())})

    print("\nDISCLOSURE: the DECLARED channel (R2-R6) was added after a pilot analysis")
    print("that also included probes/probes_mobile_ext -- for these candidates those")
    print("populations are not an untouched holdout (it contributes nothing anyway).")
    print("Lever (3)/R7 was chosen exclusively on anchors; probes_holdout was drawn")
    print("AFTER the selection (--exclude-datasets) and is clean for all candidates.")

    pd.DataFrame(out_rows).to_csv(OUT / "s_declared_risk.csv", index=False)
    print(f"\nCSV: {OUT}/s_declared_risk.csv")


if __name__ == "__main__":
    if "--build" in sys.argv:
        build_frame()
    else:
        run()
