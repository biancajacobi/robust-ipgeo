"""Hybrid baseline, measured: point = CV-median-best single source, risk signal = S around that point.

Background: a natural hybrid strategy -- "point estimate from the median-best
single source, risk signal from the consensus of the rest" -- is easy to
dismiss qualitatively (breakdown point 0, source drift). The point error of
that strategy is already covered by exp_single_source_cv (CV global median).
This experiment measures the MISSING second half: does an S-style risk
signal around the single-source point carry skill?

Design (identical protocols to the rest of the repo):
  - Point: 10-fold CV choice of the median-best source (fold split, fallback
    chain along the training ranking) -- digit-identical to
    exp_single_source_cv.
  - Risk-signal candidates around the single-source point q (core radius
    r=50 km, the pre-specified city scale, NOT optimised):
      S_all          line-weighted mass of ALL observations within 50 km of q
                     (unchanged S, only the reference point is q instead of
                     L1*b)
      S_rest_obs     like S_all, but WITHOUT the chosen observation itself
                     ("consensus of the rest", naive: lineage siblings stay)
      S_rest_lineage like S_rest_obs, but the ENTIRE line of the chosen
                     source is removed (lineage-aware rest consensus)
  - Event: Y = 1{single-source error > 100 km}; evaluated as an OOF logit
    (10-fold stratified, seed 0, identical to exp_support_concentration):
    BSS / AP / ECE. References in the same run: predecessor label (med+hub)
    on Y, regular S on aggregation misses (+0.210 context), cross view
    (regular S around L1*b as predictor of the hybrid misses).
  - Transfer probes (n=500): source chosen on ALL anchors (transfer as in
    exp_single_source_cv part 3); calibration (a) frozen (logit fitted on
    anchors, applied to probes) and (b) population-internal OOF.

Usage:   python experiments/exp_hybrid_single_source.py
Result:  tables (stdout) + eval/out/hybrid_single_source.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data import store                                         # noqa: E402
from eval.pipeline import load_cases                           # noqa: E402
from eval.metrics import haversine_error                       # noqa: E402
from experiments.exp_t6_defaults import (                      # noqa: E402
    loo_pseudo_radii, estimate, line_weights_for, EPS_GRID,
)
from experiments.exp_bootstrap_ci import SOURCE_LABELS         # noqa: E402
import experiments.exp_support_concentration as SC             # noqa: E402

OUT = ROOT / "eval" / "out"
K = 10
SEED = 0
TAU = 100.0
R_CORE = 50.0          # pre-specified city scale (like S), NOT optimised
rows_out: list[dict] = []


# --------------------------------------------------------------------------- #
# Point strategy: CV-median-best single source (digit-identical to
# exp_single_source_cv, part 1, criterion median)
# --------------------------------------------------------------------------- #
def build_src_tables(cases):
    """Per source: error AND point per IP; plus L1*b error and eps."""
    loo = loo_pseudo_radii(cases)
    eps = min(EPS_GRID, key=lambda e: abs(e - float(np.median(
        [loo[s]["_global"] for s in loo]))))
    agg = {}
    src_err = {s: {} for s, _ in SOURCE_LABELS}
    src_pt = {s: {} for s, _ in SOURCE_LABELS}
    for c in cases:
        ip = c["ip"]
        agg[ip] = haversine_error(estimate(c, "L1", "b", loo, eps=eps), c["truth"])
        for p in c["provenance"]:
            if p["source"] in src_err:
                src_err[p["source"]][ip] = haversine_error((p["lat"], p["lon"]), c["truth"])
                src_pt[p["source"]][ip] = (p["lat"], p["lon"])
    return agg, src_err, src_pt, loo, eps


def ranking(src_err, ips):
    """Source ranking (best first) by median error on the given IPs."""
    scores = []
    for s, _ in SOURCE_LABELS:
        e = [src_err[s][ip] for ip in ips if ip in src_err[s]]
        if e:
            scores.append((float(np.median(e)), s))
    return [s for _, s in sorted(scores)]


def cv_choice(cases, src_err):
    """10-fold CV choice (median): chosen source per IP (fallback chain)."""
    ips = sorted({c["ip"] for c in cases})
    idx_of = {ip: i for i, ip in enumerate(ips)}
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(ips))
    fold = np.empty(len(ips), int)
    fold[perm] = np.arange(len(ips)) % K
    chosen = {}
    picks = []
    for fi in range(K):
        tr_ips = [ip for ip in ips if fold[idx_of[ip]] != fi]
        te_ips = [ip for ip in ips if fold[idx_of[ip]] == fi]
        order = ranking(src_err, tr_ips)
        picks.append(order[0])
        for ip in te_ips:
            for s in order:
                if ip in src_err[s]:
                    chosen[ip] = s
                    break
    return chosen, picks


# --------------------------------------------------------------------------- #
# Hybrid risk signal: S variants around the single-source point
# --------------------------------------------------------------------------- #
def hybrid_features(case, src_name, q):
    """S_all / S_rest_obs / S_rest_lineage within r=50 km of the point q."""
    prov = case["provenance"]
    lin_chosen = next(p["lineage"] for p in prov if p["source"] == src_name)

    def s_of(sub):
        if not sub:
            return 0.0                      # no rest consensus available
        pts = np.array([[p["lat"], p["lon"]] for p in sub], float)
        w = np.asarray(line_weights_for(sub, "L1"), float)
        w = w / w.sum()
        d = np.array([haversine_error(tuple(p), q) for p in pts])
        return float(w[d < R_CORE].sum())

    return {
        "s_all": s_of(prov),
        "s_rest_obs": s_of([p for p in prov if p["source"] != src_name]),
        "s_rest_lineage": s_of([p for p in prov if p["lineage"] != lin_chosen]),
    }


def frame_for(cases, chosen, src_err, src_pt, loo, eps):
    """Per IP: hybrid point error, S variants, predecessor-label features, agg reference."""
    base = SC.build_frame(cases, loo, eps)      # med, hub, core_w_50, err (agg)
    base = base.set_index("ip")
    rows = []
    for c in cases:
        ip = c["ip"]
        s = chosen[ip]
        q = src_pt[s][ip]
        r = {"ip": ip, "source": s, "err_hyb": src_err[s][ip],
             "err_agg": float(base.loc[ip, "err"]),
             "med": float(base.loc[ip, "med"]), "hub": float(base.loc[ip, "hub"]),
             "s_agg": float(base.loc[ip, f"core_w_{int(R_CORE)}"])}
        r.update(hybrid_features(c, s, q))
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def eval_oof(df, cols, y, tag, pop):
    ph = SC.oof_predict(df, cols, y)
    b, a, e = SC.bss(ph, y), SC.avg_prec(ph, y), SC.ece(ph, y)
    print(f"  {tag:44s} BSS {b:+.3f}  AP {a:.3f}  ECE {e:.3f}")
    rows_out.append({"population": pop, "view": "oof", "predictor": tag,
                     "bss": round(b, 3), "ap": round(a, 3), "ece": round(e, 3),
                     "n": len(y), "misses": int(y.sum())})
    return ph


def frozen_fit(df_tr, cols, y_tr):
    """In-sample logit (anchors) with standardisation -- for the transfer."""
    raw = np.column_stack([np.log1p(df_tr[c].values) if df_tr[c].max() > 1.5
                           else df_tr[c].values for c in cols])
    mu, sd = raw.mean(0), raw.std(0) + 1e-9
    X = np.column_stack([np.ones(len(raw)), (raw - mu) / sd])
    w = SC._fit(X, y_tr)
    return mu, sd, w


def frozen_apply(df_te, cols, mu, sd, w):
    raw = np.column_stack([np.log1p(df_te[c].values) if df_te[c].max() > 1.5
                           else df_te[c].values for c in cols])
    X = np.column_stack([np.ones(len(raw)), (raw - mu) / sd])
    return 1.0 / (1.0 + np.exp(-X @ w))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    agg, src_err, src_pt, loo, eps = build_src_tables(cases)
    chosen, picks = cv_choice(cases, src_err)
    df = frame_for(cases, chosen, src_err, src_pt, loo, eps)
    y_hyb = (df.err_hyb.values > TAU).astype(int)
    y_agg = (df.err_agg.values > TAU).astype(int)

    print("=" * 86)
    print("HYBRID BASELINE: point = CV-median-best single source, risk signal = S around the point")
    print(f"Anchors n={len(df)}  (fold choice: "
          + ", ".join(f"{picks.count(u)}x {u}" for u in sorted(set(picks))) + ")")
    print("=" * 86)
    print("\nPoint error (consistency check against exp_single_source_cv):")
    print(f"  Hybrid point:  median {np.median(df.err_hyb):7.2f} km   "
          f"mean {np.mean(df.err_hyb):7.1f}   tail>100 {100*np.mean(df.err_hyb > TAU):5.1f} %")
    print(f"  L1*b (agg.):   median {np.median(df.err_agg):7.2f} km   "
          f"mean {np.mean(df.err_agg):7.1f}   tail>100 {100*np.mean(df.err_agg > TAU):5.1f} %")
    rows_out.append({"population": "anchors", "view": "point", "predictor": "hybrid_point",
                     "bss": "", "ap": "", "ece": "", "n": len(df),
                     "misses": int(y_hyb.sum()),
                     "median_km": round(float(np.median(df.err_hyb)), 2),
                     "tail_pct": round(float(100 * np.mean(df.err_hyb > TAU)), 1)})

    print(f"\nOOF calibration (event: HYBRID error > {TAU:.0f} km; "
          f"base rate {y_hyb.mean():.3f}, misses {int(y_hyb.sum())}):")
    for tag, cols in [("S_all around single-source point", ["s_all"]),
                      ("S_rest_obs (without chosen observation)", ["s_rest_obs"]),
                      ("S_rest_lineage (without chosen line)", ["s_rest_lineage"]),
                      ("predecessor label med+hub", ["med", "hub"]),
                      ("CROSS: regular S (around L1*b)", ["s_agg"])]:
        eval_oof(df, cols, y_hyb, tag, "anchors")

    print(f"\nContext in the same run (event: AGGREGATION error > {TAU:.0f} km; "
          f"base rate {y_agg.mean():.3f}):")
    for tag, cols in [("regular S (around L1*b)", ["s_agg"]),
                      ("predecessor label med+hub", ["med", "hub"])]:
        ph = SC.oof_predict(df, cols, y_agg)
        b = SC.bss(ph, y_agg)
        print(f"  {tag:44s} BSS {b:+.3f}")
        rows_out.append({"population": "anchors", "view": "oof_agg_event",
                         "predictor": tag, "bss": round(b, 3), "ap": "", "ece": "",
                         "n": len(y_agg), "misses": int(y_agg.sum())})

    # ---- Transfer: probes (source chosen on ALL anchors) ---------------------
    order_all = ranking(src_err, sorted(agg))
    print(f"\nTRANSFER probes: anchor choice (median, all anchors) = {order_all[0]}")
    cache = ROOT / "data" / "cache"
    p_cases = load_cases(store.load_anchors_csv(cache / "probes.csv"),
                         store.load_observations_csv(cache / "observations_probes.csv"))
    p_agg, p_src_err, p_src_pt, p_loo, p_eps = build_src_tables(p_cases)
    p_chosen = {}
    for c in p_cases:
        for s in order_all:
            if c["ip"] in p_src_err[s]:
                p_chosen[c["ip"]] = s
                break
    pdf = frame_for(p_cases, p_chosen, p_src_err, p_src_pt, p_loo, p_eps)
    py = (pdf.err_hyb.values > TAU).astype(int)
    print(f"  Point error probes: median {np.median(pdf.err_hyb):7.2f} km   "
          f"tail>100 {100*np.mean(pdf.err_hyb > TAU):5.1f} %   "
          f"(L1*b there: {np.median(pdf.err_agg):.2f} / {100*np.mean(pdf.err_agg > TAU):.1f} %)")
    rows_out.append({"population": "probes", "view": "point", "predictor": "hybrid_point",
                     "bss": "", "ap": "", "ece": "", "n": len(pdf), "misses": int(py.sum()),
                     "median_km": round(float(np.median(pdf.err_hyb)), 2),
                     "tail_pct": round(float(100 * np.mean(pdf.err_hyb > TAU)), 1)})

    # For the frozen fit: anchor features under the DEPLOYED choice
    # (all anchors, no CV) -- consistent with the transfer scenario.
    a_chosen = {}
    for c in cases:
        for s in order_all:
            if c["ip"] in src_err[s]:
                a_chosen[c["ip"]] = s
                break
    adf = frame_for(cases, a_chosen, src_err, src_pt, loo, eps)
    ay = (adf.err_hyb.values > TAU).astype(int)

    print(f"\n  probes, event hybrid error > {TAU:.0f} km "
          f"(base rate {py.mean():.3f}, misses {int(py.sum())}):")
    for tag, cols in [("S_all", ["s_all"]), ("S_rest_obs", ["s_rest_obs"]),
                      ("S_rest_lineage", ["s_rest_lineage"]),
                      ("predecessor label med+hub", ["med", "hub"])]:
        # (a) frozen: logit fitted on anchors (deployed choice), applied to probes
        mu, sd, w = frozen_fit(adf, cols, ay)
        ph = frozen_apply(pdf, cols, mu, sd, w)
        bf = SC.bss(ph, py)
        rows_out.append({"population": "probes", "view": "frozen", "predictor": tag,
                         "bss": round(bf, 3), "ap": round(SC.avg_prec(ph, py), 3),
                         "ece": round(SC.ece(ph, py), 3), "n": len(py),
                         "misses": int(py.sum())})
        # (b) population-internal OOF
        ph2 = SC.oof_predict(pdf, cols, py)
        bo = SC.bss(ph2, py)
        print(f"  {tag:44s} frozen BSS {bf:+.3f}   OOF-internal BSS {bo:+.3f}")
        rows_out.append({"population": "probes", "view": "oof", "predictor": tag,
                         "bss": round(bo, 3), "ap": round(SC.avg_prec(ph2, py), 3),
                         "ece": round(SC.ece(ph2, py), 3), "n": len(py),
                         "misses": int(py.sum())})

    pd.DataFrame(rows_out).to_csv(OUT / "hybrid_single_source.csv", index=False)
    print(f"\nCSV: {OUT}/hybrid_single_source.csv")


if __name__ == "__main__":
    main()
