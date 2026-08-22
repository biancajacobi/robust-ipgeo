"""Transfer of the radius weights: anchor-learned vs. population-specific pseudo-radii.

A natural question: the demonstrator freezes
anchor-learned LOO pseudo-radii (and eps), although source qualities reverse
between populations (tab:probes: MaxMind improves on probes, ipinfo
degrades). The radius ablation only exonerates the native MaxMind radius.
This script measures the missing sensitivity directly:

  fresh      : L1*b with LOO pseudo-radii learned on the TARGET population
               and the eps choice made there (status quo of tab:probes),
  frozen     : L1*b with the radii learned on ALL anchors and the anchor
               eps, applied unchanged (for foreign IPs, the global anchor
               median per source applies, as in the frozen transform
               calibration; MaxMind's native accuracy_radius remains
               observation-bound).

Populations: probes (n=500), probes_holdout (n=399), probes_mobile_ext
(n=178). Delta = frozen - fresh per case (positive = freezing costs),
paired bootstrap (B=10,000, seed 0, iid) + ASN cluster bootstrap.

Invocation: python experiments/exp_radius_transfer.py
Output: table (stdout) + eval/out/radius_transfer.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data import store                                        # noqa: E402
from eval.pipeline import load_cases                          # noqa: E402
from eval.metrics import haversine_error                      # noqa: E402
from experiments.exp_t6_defaults import (                     # noqa: E402
    loo_pseudo_radii, estimate, EPS_GRID,
)
from experiments.exp_bootstrap_ci import (                    # noqa: E402
    B, SEED, paired_bootstrap, paired_bootstrap_stat,
)
from experiments.exp_cluster_bootstrap import (               # noqa: E402
    make_groups, cluster_boot,
)

OUT = ROOT / "eval" / "out"
CACHE = ROOT / "data" / "cache"
TAIL_KM = 100.0
DATASETS = ("probes", "probes_holdout", "probes_mobile_ext")


def eps_for(loo) -> float:
    return min(EPS_GRID, key=lambda e: abs(e - float(np.median(
        [loo[s]["_global"] for s in loo]))))


def _stats(e: np.ndarray) -> tuple[float, float, float]:
    return (float(np.median(e)), float(np.mean(e)),
            100 * float(np.mean(e > TAIL_KM)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    anchors = load_cases()
    loo_a = loo_pseudo_radii(anchors)
    eps_a = eps_for(loo_a)
    print(f"Anchor calibration: eps={eps_a} km, "
          f"{len(loo_a)} sources (global radii frozen)")

    rows = []
    for ds in DATASETS:
        p_cases = load_cases(store.load_anchors_csv(CACHE / f"{ds}.csv"),
                             store.load_observations_csv(
                                 CACHE / f"observations_{ds}.csv"))
        loo_p = loo_pseudo_radii(p_cases)
        eps_p = eps_for(loo_p)

        e_fresh, e_frozen, asn = [], [], []
        for c in p_cases:
            e_fresh.append(haversine_error(
                estimate(c, "L1", "b", loo_p, eps=eps_p), c["truth"]))
            e_frozen.append(haversine_error(
                estimate(c, "L1", "b", loo_a, eps=eps_a), c["truth"]))
            asn.append(str(c.get("asn")))
        e_fresh, e_frozen = np.asarray(e_fresh), np.asarray(e_frozen)

        fm, zm = _stats(e_fresh), _stats(e_frozen)
        print("\n" + "=" * 74)
        print(f"{ds} (n={len(p_cases)}, eps fresh={eps_p} km)")
        print("=" * 74)
        print(f"  fresh      : median {fm[0]:5.1f} | mean {fm[1]:6.1f} | "
              f"tail>100 {fm[2]:4.1f}%")
        print(f"  frozen     : median {zm[0]:5.1f} | mean {zm[1]:6.1f} | "
              f"tail>100 {zm[2]:4.1f}%")

        # Delta = frozen - fresh (positive = freezing costs)
        dmed, lo, hi, _, _ = paired_bootstrap(e_frozen, e_fresh)
        dmean, mlo, mhi = paired_bootstrap_stat(e_frozen, e_fresh, np.mean)
        tail = lambda a: 100 * np.mean(a > TAIL_KM)  # noqa: E731
        dt, tlo, thi = paired_bootstrap_stat(e_frozen, e_fresh, tail)
        groups = make_groups(asn)
        cl = {}
        for key, stat in [
            ("median", lambda i: float(np.median(e_frozen[i]) - np.median(e_fresh[i]))),
            ("mean",   lambda i: float(np.mean(e_frozen[i]) - np.mean(e_fresh[i]))),
            ("tail",   lambda i: float(tail(e_frozen[i]) - tail(e_fresh[i]))),
        ]:
            v = cluster_boot(stat, groups, B)
            cl[key] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        print(f"\n  Paired bootstrap (B={B:,}, seed={SEED}) | "
              f"Δ = frozen − fresh (positive = freezing costs)")
        for key, d, (l, h), unit in [("median", dmed, (lo, hi), "km"),
                                     ("mean", dmean, (mlo, mhi), "km"),
                                     ("tail", dt, (tlo, thi), "pp")]:
            al, ah = cl[key]
            print(f"    Δ {key:6s} = {d:+7.2f} {unit}  iid [{l:+.2f}, {h:+.2f}]  "
                  f"ASN-Cluster [{al:+.2f}, {ah:+.2f}]")

        rows.append({
            "population": ds, "n": len(p_cases),
            "eps_fresh": eps_p, "eps_frozen": eps_a,
            "fresh_median": round(fm[0], 1), "fresh_mean": round(fm[1], 1),
            "fresh_tail": round(fm[2], 1),
            "frozen_median": round(zm[0], 1), "frozen_mean": round(zm[1], 1),
            "frozen_tail": round(zm[2], 1),
            "d_median": round(dmed, 2), "d_median_lo": round(lo, 2),
            "d_median_hi": round(hi, 2),
            "d_mean": round(dmean, 2), "d_mean_lo": round(mlo, 2),
            "d_mean_hi": round(mhi, 2),
            "d_tail": round(dt, 2), "d_tail_lo": round(tlo, 2),
            "d_tail_hi": round(thi, 2),
            **{f"d_{k}_asn_{b}": round(cl[k][i], 2)
               for k in ("median", "mean", "tail")
               for i, b in enumerate(("lo", "hi"))},
        })

    pd.DataFrame(rows).to_csv(OUT / "radius_transfer.csv", index=False)
    print(f"\n  CSV: {OUT}/radius_transfer.csv")


if __name__ == "__main__":
    main()
