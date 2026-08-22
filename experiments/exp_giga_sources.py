#!/usr/bin/env python3
"""Giga source comparison: are the 122 km an aggregator or an evidence problem?

A natural question: T9 reports L1·b on the 2,127 Giga pairs; it is
legitimate to ask whether the failure hits the aggregation or
the available GeoIP evidence as a whole. Therefore, on the IDENTICAL cases:
each individual source, the naive geometric median (L0·a), the unweighted
city majority vote (exp_city_vote), and the anchor-frozen trust-weighted
vote (exp_trust_weighted_fusion, part-B configuration: trust learned on all
anchors, applied unchanged).

If they all sit in the same band, the input portfolio simply contains no
locally resolved evidence for these address ranges — robust aggregation
cannot invent information (connecting to the hard-bucket finding, T4).

Usage:  python experiments/exp_giga_sources.py
Result: table (stdout) + eval/out/giga_sources.csv
Data: ODbL — Source: Giga (UNICEF) and contributors.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import store                                      # noqa: E402
from eval.pipeline import load_cases                        # noqa: E402
from eval.metrics import haversine_error                    # noqa: E402
import experiments.exp_t6_defaults as T6                    # noqa: E402
from experiments.exp_radius_transfer import eps_for         # noqa: E402
from experiments.exp_pool_transfer import (                 # noqa: E402
    restrict, errors, stats,
)
from experiments.exp_giga_transfer import EIGHT             # noqa: E402
from experiments.exp_city_vote import city_vote             # noqa: E402
from experiments.exp_trust_weighted_fusion import (         # noqa: E402
    source_errors, trust_from, trust_weighted_vote, city_lut_from,
)

CACHE = Path("data/cache")
OUT = Path("eval/out")


def main() -> None:
    anchors9 = load_cases()
    loo_a = T6.loo_pseudo_radii(anchors9)
    eps_a = eps_for(loo_a)
    cases = restrict(load_cases(
        store.load_anchors_csv(CACHE / "giga.csv"),
        store.load_observations_csv(CACHE / "observations_giga.csv")), EIGHT)

    rows = []

    def add(name: str, e) -> None:
        s = stats(np.asarray(e, float))
        rows.append({"estimator": name, **s})
        print(f"  {name:28s} n={s['n']:5d} | median {s['median_km']:6.1f} "
              f"| mean {s['mean_km']:6.1f} | tail {s['tail_pct']:4.1f} %")

    print(f"Giga source comparison on identical cases (n={len(cases)}, "
          f"frozen anchor calibration eps={eps_a} km)")
    add("L1b_frozen", errors(cases, loo_a, eps_a))
    add("L0a_naive_geom_median",
        [haversine_error(T6.estimate(c, "L0", "a", loo_a, eps=eps_a),
                         c["truth"]) for c in cases])
    for src in EIGHT:
        add(f"src:{src}",
            [haversine_error((p["lat"], p["lon"]), c["truth"])
             for c in cases for p in c["provenance"] if p["source"] == src])
    lut = city_lut_from(CACHE / "observations_giga.csv")
    add("city_majority_vote",
        [haversine_error(city_vote(c, lut), c["truth"]) for c in cases])
    trust_all = trust_from(source_errors(anchors9),
                           sorted(c["ip"] for c in anchors9))
    add("trust_vote_anchor_frozen",
        [haversine_error(trust_weighted_vote(c, lut, trust_all), c["truth"])
         for c in cases])

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "giga_sources.csv", index=False)
    print(f"CSV: {OUT}/giga_sources.csv")
    print("Data: ODbL — Source: Giga (UNICEF) and contributors.")


if __name__ == "__main__":
    main()
