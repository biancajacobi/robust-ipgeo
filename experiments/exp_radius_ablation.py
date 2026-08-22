"""Ablation of the radius weighting: does MaxMind's native accuracy_radius help -- and how?

A natural objection: the accompanying paper calls
native accuracy fields "not comparable across sources", yet mixes MaxMind's
native accuracy_radius with empirical LOO pseudo-radii in the same weight
function w/(r+eps). Same unit (km) does not mean same calibration. This
ablation tests the tension directly:

  headline       as published: MaxMind native (per IP), others per-IP LOO
  loo_only       native radius DISABLED: MaxMind also gets its per-IP LOO
                 pseudo-radius like all other sources
  native_scaled  native radius RESCALED to the LOO scale
                 (r' = r * LOO median(MaxMind) / median(native radii)):
                 keeps the per-IP ordering information, removes the
                 scale/calibration assumption

Reading: if L1*b stays stable under loo_only, the native radius is
dispensable and the criticism moot. If loo_only degrades, native_scaled
shows whether the contribution hinges on the (attackable) km calibration or
only on the per-IP ordering information "this particular answer is
uncertain". Note (review 2026-08-18): since all w_i enter the SAME objective
function, the absolute MaxMind scale necessarily also determines the relative
influence -- the question can only be answered EMPIRICALLY, and that is
exactly what this ablation does.

Scope note: the pre-signal of the centroid detection (native radius >= 500 km,
methodology section) is a DETECTION criterion and not part of the weighting
ablated here; variant b does not filter.

Invocation: python experiments/exp_radius_ablation.py
Output: tables (stdout) + eval/out/radius_ablation.csv
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
from experiments.exp_bootstrap_ci import (                    # noqa: E402
    B, paired_bootstrap_stat,
)
from experiments.exp_cluster_bootstrap import (               # noqa: E402
    make_groups, cluster_boot,
)
from experiments.exp_support_concentration import (           # noqa: E402
    oof_predict, bss,
)

OUT = ROOT / "eval" / "out"
TAIL_KM = 100.0
S_RADIUS = 50.0
rows: list[dict] = []


def point_radius_mode(p, ip, loo, mode, mm_scale):
    if p["source"] == "maxmind_geolite2":
        r = p.get("accuracy_radius")
        if mode == "headline" and r is not None:
            return float(r)
        if mode == "native_scaled" and r is not None:
            return float(r) * mm_scale
        # loo_only -- or native radius missing: LOO like all other sources
        tab = loo.get("maxmind_geolite2", {})
        return float(tab.get(ip, tab.get("_global", T6.EPS_KM)))
    tab = loo.get(p["source"], {})
    return float(tab.get(ip, tab.get("_global", T6.EPS_KM)))


def estimate_mode(case, loo, eps, mode, mm_scale):
    """L1*b as in exp_t6_defaults.estimate(level='L1', variant='b'), but with
    an interchangeable radius source for MaxMind."""
    prov = case["provenance"]
    pts = np.array([[p["lat"], p["lon"]] for p in prov], dtype=float)
    w = np.asarray(T6.line_weights_for(prov, "L1"), dtype=float)
    rad = np.array([point_radius_mode(p, case["ip"], loo, mode, mm_scale)
                    for p in prov], dtype=float)
    est = weighted_geometric_median(pts, w / (rad + eps))
    return (float(est[0]), float(est[1]))


def s_value(case, est):
    prov = case["provenance"]
    w = np.asarray(T6.line_weights_for(prov, "L1"), dtype=float)
    w = w / w.sum()
    d = np.array([haversine_error((p["lat"], p["lon"]), est) for p in prov])
    return float(w[d < S_RADIUS].sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    loo = T6.loo_pseudo_radii(cases)
    eps = min(T6.EPS_GRID, key=lambda e: abs(e - float(np.median(
        [loo[s]["_global"] for s in loo]))))

    native = [float(p["accuracy_radius"]) for c in cases for p in c["provenance"]
              if p["source"] == "maxmind_geolite2" and p["accuracy_radius"] is not None]
    mm_scale = float(loo["maxmind_geolite2"]["_global"] / np.median(native))
    print(f"MaxMind: native radius median {np.median(native):.0f} km "
          f"(range {min(native):.0f}-{max(native):.0f}), LOO median "
          f"{loo['maxmind_geolite2']['_global']:.1f} km -> scale factor {mm_scale:.3f}")

    asn = [str(c.get("asn")) for c in cases]
    groups = make_groups(asn)
    errs, svals = {}, {}
    for mode in ("headline", "loo_only", "native_scaled"):
        e, s = [], []
        for c in cases:
            est = estimate_mode(c, loo, eps, mode, mm_scale)
            e.append(haversine_error(est, c["truth"]))
            s.append(s_value(c, est))
        errs[mode], svals[mode] = np.asarray(e), np.asarray(s)

    ok = (round(float(np.median(errs["headline"])), 1) == 4.1
          and round(float(np.mean(errs["headline"])), 1) == 183.4)
    print(f"Reproduction of headline (4.1 / 183.4): {'OK' if ok else 'DEVIATION!'}")

    stats = {"median": lambda d: float(np.median(d)),
             "mean":   lambda d: float(np.mean(d)),
             "tail":   lambda d: float(100.0 * np.mean(np.asarray(d) > TAIL_KM))}
    print("\n" + "=" * 84)
    print(f"RADIUS ABLATION on L1*b (n={len(cases)}, B={B:,}) | "
          "delta = headline - variant; NEGATIVE = headline better")
    print("=" * 84)
    e_h = errs["headline"]
    for mode in ("headline", "loo_only", "native_scaled"):
        e = errs[mode]
        # S quality per variant: OOF logit on the S belonging to that variant
        df = pd.DataFrame({"s": svals[mode]})
        y = (e > TAIL_KM).astype(int)
        b_s = bss(oof_predict(df, ["s"], y), y)
        print(f"\n{mode:14s}  Med {np.median(e):5.2f}  Mean {np.mean(e):6.1f}  "
              f"Tail {100*np.mean(e > TAIL_KM):5.1f} %   S-OOF-BSS {b_s:+.3f}")
        rows.append({"mode": mode, "metric": "abs", "median": round(float(np.median(e)), 2),
                     "mean": round(float(np.mean(e)), 1),
                     "tail_pct": round(float(100*np.mean(e > TAIL_KM)), 1),
                     "s_oof_bss": round(float(b_s), 3)})
        if mode == "headline":
            continue
        for mkey, mstat in stats.items():
            delta, lo, hi = paired_bootstrap_stat(e_h, e, mstat)
            boot = cluster_boot(lambda idx, f=mstat: float(f(e_h[idx]) - f(e[idx])),
                                groups, B)
            clo, chi = np.percentile(boot, [2.5, 97.5])
            sig = "*" if not (lo <= 0 <= hi) else " "
            csig = "*" if not (clo <= 0 <= chi) else " "
            print(f"  Delta {mkey:6s} {delta:+9.2f}  iid [{lo:+8.2f},{hi:+8.2f}]{sig} "
                  f" ASN [{clo:+8.2f},{chi:+8.2f}]{csig}")
            rows.append({"mode": mode, "metric": mkey, "delta": round(delta, 3),
                         "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                         "asn_lo": round(float(clo), 3), "asn_hi": round(float(chi), 3),
                         "sig_iid": bool(not (lo <= 0 <= hi)),
                         "sig_asn": bool(not (clo <= 0 <= chi))})

    pd.DataFrame(rows).to_csv(OUT / "radius_ablation.csv", index=False)
    print("\nREADING:")
    print("  loo_only measures the total contribution of the native radius; native_scaled")
    print("  isolates the scale/calibration dependence (ordering information")
    print("  is kept). If native_scaled ~ headline, the absolute scaling of the")
    print("  native radius is empirically not decisive for performance;")
    print("  if additionally loo_only ~ headline, its per-IP ordering information")
    print("  also contributes no measurable amount to the point estimator.")
    print(f"\nCSV: {OUT}/radius_ablation.csv")


if __name__ == "__main__":
    main()
