#!/usr/bin/env python3
"""Symmetric transfer comparison: FROZEN trust vote vs. FROZEN aggregation
(fairness control: the transfer comparison previously paired asymmetrically — the trust vote frozen on
anchors against the aggregation FRESHLY calibrated on the target population;
that overstates the vote's disadvantage by the aggregation's freezing penalty,
which exp_radius_transfer measures separately).

Here both methods are treated identically: both calibrated on ALL anchors
(trust = inverse median error per source; aggregation = L1*b with anchor LOO
radii and anchor eps) and applied unchanged to probes / probes_holdout /
probes_mobile_ext. Paired bootstrap differences (median, mean, tail>100 km),
iid and ASN-clustered, as in exp_trust_weighted_fusion.

Usage:   python experiments/exp_trust_frozen_symmetric.py
Output:  tables (stdout) + eval/out/trust_frozen_symmetric.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import store                                     # noqa: E402
from eval.pipeline import load_cases                       # noqa: E402
from eval.metrics import haversine_error                   # noqa: E402
from experiments.exp_t6_defaults import (                  # noqa: E402
    loo_pseudo_radii, estimate,
)
from experiments.exp_radius_transfer import eps_for        # noqa: E402
from experiments.exp_trust_weighted_fusion import (        # noqa: E402
    source_errors, trust_from, city_lut_from, trust_weighted_vote,
    _stats, _paired, _print_paired, TRANSFER_DATASETS, CACHE, OUT,
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    anchors = load_cases()
    loo_a = loo_pseudo_radii(anchors)
    eps_a = eps_for(loo_a)
    trust_all = trust_from(source_errors(anchors), [c["ip"] for c in anchors])
    print(f"Anchor calibration frozen: eps={eps_a} km, "
          f"trust over {len(trust_all)} sources")

    rows = []
    for ds in TRANSFER_DATASETS:
        p_cases = load_cases(store.load_anchors_csv(CACHE / f"{ds}.csv"),
                             store.load_observations_csv(
                                 CACHE / f"observations_{ds}.csv"))
        p_lut = city_lut_from(CACHE / f"observations_{ds}.csv")

        e_agg, e_twf, asn = [], [], []
        for c in p_cases:
            e_agg.append(haversine_error(
                estimate(c, "L1", "b", loo_a, eps=eps_a), c["truth"]))
            e_twf.append(haversine_error(
                trust_weighted_vote(c, p_lut, trust_all), c["truth"]))
            asn.append(str(c.get("asn")))
        e_agg, e_twf = np.asarray(e_agg), np.asarray(e_twf)

        print("\n" + "=" * 78)
        print(f"SYMMETRICALLY FROZEN: {ds} (n={len(p_cases)})")
        print("=" * 78)
        for name, e in [("L1*b frozen", e_agg), ("trust vote frozen", e_twf)]:
            m = _stats(e)
            print(f"  {name:20s}: median {m[0]:5.1f} | mean {m[1]:6.1f} | "
                  f"tail>100 {m[2]:4.1f}%")
        res = _paired(e_agg, e_twf, asn)
        print("\n  Paired bootstrap | Δ = L1*b(frozen) − trust vote(frozen) "
              "(negative = aggregation better)")
        _print_paired(res)
        am, tm = _stats(e_agg), _stats(e_twf)
        rows.append({"population": ds, "n": len(p_cases),
                     "agg_frozen_median": am[0], "agg_frozen_mean": am[1],
                     "agg_frozen_tail": am[2],
                     "tw_frozen_median": tm[0], "tw_frozen_mean": tm[1],
                     "tw_frozen_tail": tm[2], **res})
    pd.DataFrame(rows).to_csv(OUT / "trust_frozen_symmetric.csv", index=False)
    print(f"\nCSV: {OUT}/trust_frozen_symmetric.csv")


if __name__ == "__main__":
    main()
