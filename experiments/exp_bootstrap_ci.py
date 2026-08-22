"""T1 — Paired-bootstrap confidence intervals for the aggregation L1·b.

Reproduces the T1 bootstrap table of the accompanying paper: for each single
source (and the unweighted baseline L0·a), the anchor-wise median difference

    Δ = median(d_{L1·b}) − median(d_{source})

is reported as the point estimate, and a 95% percentile confidence interval is
determined via a *paired* bootstrap (the same resampled anchor indices for
numerator and denominator) with B = 10,000 repetitions and fixed seed = 0.
Negative Δ means the aggregation is more accurate.

The pairing is done per source over the intersection of anchors on which both
L1·b and the source have an error value (relevant only for MaxMind/geojs:
1,076 instead of 1,077).

Invocation:  python experiments/exp_bootstrap_ci.py
Result: table (stdout) + eval/out/t1_bootstrap_ci.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.pipeline import load_cases                       # noqa: E402
from eval.metrics import haversine_error                   # noqa: E402
from experiments.exp_t6_defaults import (                  # noqa: E402
    loo_pseudo_radii, estimate, EPS_GRID,
)

B = 10_000
SEED = 0
N_VERGLEICHE = 8                 # 7 sources + baseline (T1 bootstrap table)
BONF_ALPHA = 0.05 / N_VERGLEICHE  # Bonferroni-adjusted level
OUT = Path("eval/out"); OUT.mkdir(parents=True, exist_ok=True)

# order & display names as in the T1 comparison table
SOURCE_LABELS = [
    ("ip2location_lite", "IP2Location LITE"),
    ("ipinfo",           "ipinfo.io"),
    ("ipwho_is",         "ipwho.is"),
    ("ip_api",           "ip-api.com"),
    ("dbip_lite",        "DB-IP Lite"),
    ("maxmind_geolite2", "MaxMind GeoLite2 / geojs"),
    ("reallyfreegeoip",  "reallyfreegeoip.org"),
]


def paired_bootstrap(d_agg, d_ref, b=B, seed=SEED):
    """Δ = median(d_agg) − median(d_ref) plus paired 95% percentile CI.

    d_agg, d_ref are aligned over the same anchor index (same length).
    """
    d_agg = np.asarray(d_agg, dtype=float)
    d_ref = np.asarray(d_ref, dtype=float)
    n = len(d_agg)
    delta = float(np.median(d_agg) - np.median(d_ref))
    rng = np.random.default_rng(seed)
    boot = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, n)            # paired: same indices on both sides
        boot[i] = np.median(d_agg[idx]) - np.median(d_ref[idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # Bonferroni-adjusted intervals (multiplicity over the 8 comparisons);
    # same bootstrap distribution, just wider quantiles.
    lo_b, hi_b = np.percentile(boot, [100 * BONF_ALPHA / 2,
                                      100 * (1 - BONF_ALPHA / 2)])
    return delta, float(lo), float(hi), float(lo_b), float(hi_b)


def paired_bootstrap_stat(d_agg, d_ref, stat, b=B, seed=SEED):
    """Δ = stat(d_agg) − stat(d_ref) plus paired 95% percentile CI for an
    arbitrary statistic (e.g. mean or tail rate). Negative Δ = aggregation better.
    """
    d_agg = np.asarray(d_agg, dtype=float)
    d_ref = np.asarray(d_ref, dtype=float)
    n = len(d_agg)
    delta = float(stat(d_agg) - stat(d_ref))
    rng = np.random.default_rng(seed)
    boot = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, n)            # paired: same indices on both sides
        boot[i] = stat(d_agg[idx]) - stat(d_ref[idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return delta, float(lo), float(hi)


def main():
    cases = load_cases()
    loo = loo_pseudo_radii(cases)
    globals_ = [loo[s]["_global"] for s in loo]
    anchor_eps = min(EPS_GRID, key=lambda e: abs(e - float(np.median(globals_))))

    # anchor-wise errors: aggregation L1·b, unweighted baseline L0·a, each single source.
    agg_l1b, base_l0a = {}, {}
    src_err = {s: {} for s, _ in SOURCE_LABELS}
    for c in cases:
        ip = c["ip"]
        agg_l1b[ip] = haversine_error(estimate(c, "L1", "b", loo, eps=anchor_eps), c["truth"])
        base_l0a[ip] = haversine_error(estimate(c, "L0", "a", loo, eps=anchor_eps), c["truth"])
        for p in c["provenance"]:
            if p["source"] in src_err:
                src_err[p["source"]][ip] = haversine_error((p["lat"], p["lon"]), c["truth"])

    rows = []
    print("=" * 78)
    print(f"T1 — paired-bootstrap CIs (B={B:,}, seed={SEED}) | Δ = median(L1·b) − median(source)")
    print("=" * 78)
    print(f"{'Source/reference':28s} {'Med source':>10s} {'Δ [km]':>8s}   95% CI")

    def emit(label, ref_map):
        ips = sorted(set(agg_l1b) & set(ref_map))
        d_agg = [agg_l1b[ip] for ip in ips]
        d_ref = [ref_map[ip] for ip in ips]
        med_ref = float(np.median(d_ref))
        delta, lo, hi, lo_b, hi_b = paired_bootstrap(d_agg, d_ref)
        sig = "" if lo <= 0 <= hi else "  *"      # * = CI does not include 0
        sig_b = "" if lo_b <= 0 <= hi_b else " *"  # * = Bonferroni CI without 0 as well
        print(f"{label:28s} {med_ref:10.2f} {delta:+8.2f}   [{lo:+.2f}, {hi:+.2f}]{sig}"
              f"  Bonf[{lo_b:+.2f}, {hi_b:+.2f}]{sig_b} (n={len(ips)})")
        rows.append({"reference": label, "median_source_km": round(med_ref, 2),
                     "delta_km": round(delta, 2), "ci_lo_km": round(lo, 2),
                     "ci_hi_km": round(hi, 2),
                     "bonf_ci_lo_km": round(lo_b, 2), "bonf_ci_hi_km": round(hi_b, 2),
                     "n": len(ips)})

    for key, label in SOURCE_LABELS:
        emit(label, src_err[key])
    print("-" * 78)
    emit("naive unweighted L0·a (baseline)", base_l0a)

    print(f"\n  median(L1·b) overall = {np.median(list(agg_l1b.values())):.2f} km (n={len(agg_l1b)})")
    pd.DataFrame(rows).to_csv(OUT / "t1_bootstrap_ci.csv", index=False)
    print(f"  CSV: {OUT}/t1_bootstrap_ci.csv")

    # --- mean and tail-rate differences against the sources leading in mean/tail ---
    # (RQ1: the aggregation loses there ex post; here backed by inferential statistics)
    print("\n" + "=" * 78)
    print(f"T1 — mean & tail-rate differences (L1·b − source), paired, B={B:,}")
    print("=" * 78)
    print(f"{'Source':18s} {'Δ mean [km]':>22s}   {'Δ tail rate [pp]':>22s}")
    mean_stat = lambda d: float(np.mean(d))
    tail_stat = lambda d: float(100.0 * np.mean(np.asarray(d) > 100.0))
    mt_rows = []
    for key, label in [("ip2location_lite", "IP2Location LITE"), ("ipinfo", "ipinfo.io")]:
        ips = sorted(set(agg_l1b) & set(src_err[key]))
        da = np.array([agg_l1b[ip] for ip in ips])
        dr = np.array([src_err[key][ip] for ip in ips])
        dm, lom, him = paired_bootstrap_stat(da, dr, mean_stat)
        dt, lot, hit = paired_bootstrap_stat(da, dr, tail_stat)
        sm = "" if lom <= 0 <= him else "  *"
        st = "" if lot <= 0 <= hit else "  *"
        print(f"{label:18s} {dm:+8.1f} [{lom:+7.1f},{him:+7.1f}]{sm}   "
              f"{dt:+6.1f} [{lot:+6.1f},{hit:+6.1f}]{st}")
        mt_rows.append({"reference": label, "delta_mean_km": round(dm, 1),
                        "mean_ci_lo": round(lom, 1), "mean_ci_hi": round(him, 1),
                        "delta_tail_pp": round(dt, 1), "tail_ci_lo": round(lot, 1),
                        "tail_ci_hi": round(hit, 1), "n": len(ips)})
    pd.DataFrame(mt_rows).to_csv(OUT / "t1_bootstrap_meantail_ci.csv", index=False)
    print("  (+ = aggregation worse than source; * = CI does not include 0)")
    print(f"  CSV: {OUT}/t1_bootstrap_meantail_ci.csv")


if __name__ == "__main__":
    main()
