"""Probe (MEASUREMENT ONLY, no paper intervention): what changes in the headline
aggregation L1·b and in S if line replicas (geojs, reallyfreegeoip; lineage
maxmind_geolite) receive the accuracy_radius of the local MaxMind point of the
same case instead of the LOO pseudo-radius of their own source?

Background (limitations paragraph of the paper): the local MaxMind point carries
r = 500–1000 km on default centroids, the replicas only r_LOO ≈ 14.5 km, so the
line is down-weighted only in part.

Variants:
  status_quo   as before (point_radius unchanged)
  inherit      replica inherits the accuracy_radius of the MaxMind point in the
               same case (if present), otherwise its own LOO radius
  inherit_same as inherit, but only if replica and MaxMind point are < 1 km apart
               (bit-identical coordinate); otherwise own LOO radius
  line_max     every point of a line receives the largest radius of the line

Usage: python experiments/exp_replicate_radius_probe.py
Output: stdout + eval/out/replicate_radius_probe.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.pipeline import load_cases                      # noqa: E402
from eval.metrics import haversine_error                  # noqa: E402
import experiments.exp_t6_defaults as T6                  # noqa: E402
import experiments.exp_support_concentration as SC        # noqa: E402

OUT = Path("eval/out")
REPLICATES = ("geojs", "reallyfreegeoip")
ORIGIN = "maxmind_geolite2"
N_BOOT, SEED = 2000, 0

_orig_point_radius = T6.point_radius
_case_by_ip: dict = {}


def _maxmind_point(ip):
    c = _case_by_ip[ip]
    return next((q for q in c["provenance"] if q["source"] == ORIGIN), None)


def make_point_radius(mode):
    def pr(p, ip, loo):
        base = _orig_point_radius(p, ip, loo)
        if mode == "status_quo":
            return base
        if mode == "line_max":
            c = _case_by_ip[ip]
            same = [q for q in c["provenance"] if q["lineage"] == p["lineage"]]
            return max(_orig_point_radius(q, ip, loo) for q in same)
        if p["source"] not in REPLICATES:
            return base
        mm = _maxmind_point(ip)
        if mm is None or mm["accuracy_radius"] is None:
            return base
        if mode == "inherit_same":
            if haversine_error((p["lat"], p["lon"]), (mm["lat"], mm["lon"])) >= 1.0:
                return base
        return mm["accuracy_radius"]
    return pr


def run_mode(cases, loo, eps, mode):
    T6.point_radius = make_point_radius(mode)
    try:
        errs = np.array([haversine_error(T6.estimate(c, "L1", "b", loo, eps=eps), c["truth"])
                         for c in cases])
        df = SC.build_frame(cases, loo, eps)
    finally:
        T6.point_radius = _orig_point_radius
    y = (df.err > SC.TAU).astype(int).values
    ph_s = SC.oof_predict(df, [f"core_w_{SC.R_HEADLINE}"], y)
    ph_old = SC.oof_predict(df, ["med", "hub"], y)
    return errs, df, {
        "median_km": float(np.median(errs)), "mean_km": float(errs.mean()),
        "tail_pct": 100 * float((errs > 100).mean()),
        "misses": int(y.sum()),
        "S_bss": SC.bss(ph_s, y), "S_ap": SC.avg_prec(ph_s, y), "S_ece": SC.ece(ph_s, y),
        "old_bss": SC.bss(ph_old, y),
    }


def paired_boot(e0, e1, fn, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(e0)
    d = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        d.append(fn(e1[idx]) - fn(e0[idx]))
    return np.percentile(d, [2.5, 97.5])


def _restrict(cases, sources):
    out = []
    for c in cases:
        prov = [p for p in c["provenance"] if p["source"] in sources]
        if prov:
            out.append(dict(c, provenance=prov, sources=[p["source"] for p in prov],
                            points=np.array([[p["lat"], p["lon"]] for p in prov], float)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="restrict the portfolio to these sources (comma list); LOO radii and eps stay calibrated on the full portfolio as in T11")
    ap.add_argument("--out-suffix", default="", help="suffix for the CSV")
    args = ap.parse_args()
    cases = load_cases()
    _case_by_ip.update({c["ip"]: c for c in cases})
    loo = T6.loo_pseudo_radii(cases)
    eps = min(T6.EPS_GRID, key=lambda e: abs(e - float(np.median([loo[s]["_global"] for s in loo]))))
    print(f"Cases: {len(cases)}, headline eps = {eps} km")
    if args.only:
        cases = _restrict(cases, set(args.only.split(",")))
        _case_by_ip.clear(); _case_by_ip.update({c["ip"]: c for c in cases})
        print(f"portfolio restricted to {args.only}: {len(cases)} cases")

    # Diagnose: wie oft Replikat-Koordinate == MaxMind-Koordinate, Radius-Verteilung
    same = {s: 0 for s in REPLICATES}; tot = {s: 0 for s in REPLICATES}
    for c in cases:
        mm = next((q for q in c["provenance"] if q["source"] == ORIGIN), None)
        if mm is None:
            continue
        for p in c["provenance"]:
            if p["source"] in REPLICATES:
                tot[p["source"]] += 1
                same[p["source"]] += haversine_error((p["lat"], p["lon"]), (mm["lat"], mm["lon"])) < 1.0
    for s in REPLICATES:
        print(f"  {s}: coordinate < 1 km from the local MaxMind point in {same[s]}/{tot[s]} cases "
              f"({100*same[s]/max(tot[s],1):.1f} %); LOO-Radius global {loo[s]['_global']:.1f} km")
    mmr = np.array([q["accuracy_radius"] for c in cases for q in c["provenance"]
                    if q["source"] == ORIGIN and q["accuracy_radius"] is not None])
    print(f"  MaxMind accuracy_radius: Median {np.median(mmr):.0f}, p75 {np.percentile(mmr,75):.0f}, "
          f"p90 {np.percentile(mmr,90):.0f} km; ≥100 km {100*(mmr>=100).mean():.1f} %, ≥500 km {100*(mmr>=500).mean():.1f} %")

    modes = ["status_quo", "inherit", "inherit_same", "line_max"]
    res, errs = {}, {}
    for m in modes:
        e, df, r = run_mode(cases, loo, eps, m)
        res[m], errs[m] = r, e
    e0 = errs["status_quo"]
    rows = []
    print("\n" + "=" * 96)
    print(f"{'Variante':14s} {'Median':>7s} {'Mittel':>7s} {'Tail%':>6s} {'Misses':>6s} | "
          f"{'S-BSS':>7s} {'S-AP':>6s} {'S-ECE':>6s} {'alt-BSS':>7s} | Δ-Median [CI]       Δ-Tail pp [CI]")
    print("=" * 96)
    for m in modes:
        r = res[m]; e1 = errs[m]
        dmed = np.median(e1) - np.median(e0); cmed = paired_boot(e0, e1, np.median)
        dtail = 100 * ((e1 > 100).mean() - (e0 > 100).mean())
        ctail = paired_boot(e0, e1, lambda a: 100 * (a > 100).mean())
        dmean = e1.mean() - e0.mean(); cmean = paired_boot(e0, e1, np.mean)
        print(f"{m:14s} {r['median_km']:7.2f} {r['mean_km']:7.1f} {r['tail_pct']:6.1f} {r['misses']:6d} | "
              f"{r['S_bss']:+7.3f} {r['S_ap']:6.3f} {r['S_ece']:6.3f} {r['old_bss']:+7.3f} | "
              f"{dmed:+5.2f} [{cmed[0]:+.2f},{cmed[1]:+.2f}]  {dtail:+4.1f} [{ctail[0]:+.1f},{ctail[1]:+.1f}]")
        flips = int(((e0 > 100) != (e1 > 100)).sum())
        rows.append({"mode": m, **{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()},
                     "d_median_km": round(dmed, 3), "d_median_lo": round(cmed[0], 3), "d_median_hi": round(cmed[1], 3),
                     "d_mean_km": round(dmean, 2), "d_mean_lo": round(cmean[0], 2), "d_mean_hi": round(cmean[1], 2),
                     "d_tail_pp": round(dtail, 3), "d_tail_lo": round(ctail[0], 3), "d_tail_hi": round(ctail[1], 3),
                     "miss_flips": flips,
                     "est_moved_gt1km_pct": None})
    # How many estimate points move at all?
    T6.point_radius = make_point_radius("inherit")
    est1 = [T6.estimate(c, "L1", "b", loo, eps=eps) for c in cases]
    T6.point_radius = _orig_point_radius
    est0 = [T6.estimate(c, "L1", "b", loo, eps=eps) for c in cases]
    mv = np.array([haversine_error(a, b) for a, b in zip(est0, est1)])
    print(f"\ninherit vs status_quo: estimate identical (<0.01 km) in {100*(mv<0.01).mean():.1f} %, "
          f">1 km bewegt {100*(mv>1).mean():.1f} %, >100 km bewegt {100*(mv>100).mean():.1f} %")
    pd.DataFrame(rows).to_csv(OUT / f"replicate_radius_probe{args.out_suffix}.csv", index=False)
    print(f"CSV: {OUT}/replicate_radius_probe{args.out_suffix}.csv")


if __name__ == "__main__":
    main()
