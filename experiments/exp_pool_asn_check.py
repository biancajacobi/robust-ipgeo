#!/usr/bin/env python3
"""ASN-cluster verification of the pool key figures (companion to exp_pool_transfer).

8,896 probes are not independent (providers host multiple probes). This script
establishes two things: (1) the ASN STRUCTURE of the pool — 3,057 ASNs, median
1 probe/ASN, top ASN ~2.7 % — is unproblematic (markedly less clumped than the
anchors; cf. the exchangeability point in the calibration discussion);
(2) the core statements hold under ASN cluster bootstrap (cluster = ASN,
B = 10,000, seed 0): non-auto-geoip median/tail, pooled median, the freezing
difference (n.s.) and the circularity gap
auto-geoip − non-auto-geoip (highly significantly negative).

Usage:  python experiments/exp_pool_asn_check.py
        python experiments/exp_pool_asn_check.py --with-ipwhois
          (8 sources = 6 lines, analogous to exp_pool_transfer --with-ipwhois;
           writes pool_asn_check_8src.csv, published artifacts untouched)
Result: table (stdout) + eval/out/pool_asn_check.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import store                                     # noqa: E402
from eval.pipeline import load_cases                       # noqa: E402
import experiments.exp_t6_defaults as T6                   # noqa: E402
from experiments.exp_radius_transfer import eps_for        # noqa: E402
from experiments.exp_pool_transfer import restrict, errors, TAU, SEVEN  # noqa: E402
from experiments.exp_s_declared_risk import load_tags      # noqa: E402

CACHE = Path("data/cache")
OUT = Path("eval/out")
B, SEED = 10_000, 0


def main() -> None:
    with_ipwhois = "--with-ipwhois" in sys.argv
    srcs = SEVEN + ("ipwho_is",) if with_ipwhois else SEVEN
    sfx = "_8src" if with_ipwhois else ""
    anchors = load_cases()
    loo_a = T6.loo_pseudo_radii(anchors)
    eps_a = eps_for(loo_a)
    cases = restrict(load_cases(
        store.load_anchors_csv(CACHE / "probes_pool.csv"),
        store.load_observations_csv(CACHE / "observations_probes_pool.csv")), srcs)
    loo_p = T6.loo_pseudo_radii(cases)
    eps_p = eps_for(loo_p)
    e_fro = errors(cases, loo_a, eps_a)
    e_fre = errors(cases, loo_p, eps_p)
    asn = np.array([str(c.get("asn")) for c in cases])
    tags = load_tags("probes_pool")
    auto = np.array([any(t.startswith("system-auto-geoip")
                         for t in tags.get(c["ip"], set())) for c in cases])
    op = ~auto

    counts = pd.Series(asn).value_counts()
    print(f"ASN structure: {len(counts)} ASNs for {len(cases)} probes | "
          f"median {int(counts.median())} probe(s)/ASN | "
          f"top ASN {counts.iloc[0]} ({100*counts.iloc[0]/len(cases):.1f} %) | "
          f"top 10 {100*counts.iloc[:10].sum()/len(cases):.1f} %")

    rng = np.random.default_rng(SEED)
    uasn = np.unique(asn)
    idx_of = {a: np.where(asn == a)[0] for a in uasn}

    def cboot(stat):
        out = np.empty(B)
        for b in range(B):
            pick = rng.choice(uasn, size=len(uasn), replace=True)
            idx = np.concatenate([idx_of[a] for a in pick])
            out[b] = stat(idx)
        return np.percentile(out, [2.5, 97.5])

    quantities = [
        ("non_auto_geoip_median_km",
         lambda i: np.median(e_fro[i][op[i]]), float(np.median(e_fro[op]))),
        ("non_auto_geoip_tail_pct",
         lambda i: 100 * np.mean(e_fro[i][op[i]] > TAU),
         float(100 * np.mean(e_fro[op] > TAU))),
        ("pooled_median_km",
         lambda i: np.median(e_fro[i]), float(np.median(e_fro))),
        ("delta_median_frozen_minus_fresh_km",
         lambda i: np.median(e_fro[i]) - np.median(e_fre[i]),
         float(np.median(e_fro) - np.median(e_fre))),
        ("gap_autogeoip_minus_non_auto_geoip_median_km",
         lambda i: np.median(e_fro[i][auto[i]]) - np.median(e_fro[i][op[i]]),
         float(np.median(e_fro[auto]) - np.median(e_fro[op]))),
    ]
    rows = []
    print(f"\nASN cluster bootstrap (cluster = ASN, B={B:,}, seed={SEED}):")
    for name, stat, point in quantities:
        lo, hi = cboot(stat)
        print(f"  {name:42s} {point:8.2f}  95%-CI [{lo:+.2f}, {hi:+.2f}]")
        rows.append({"quantity": name, "point": round(point, 3),
                     "ci_lo": round(float(lo), 3), "ci_hi": round(float(hi), 3)})
    pd.DataFrame(rows).to_csv(OUT / f"pool_asn_check{sfx}.csv", index=False)
    print(f"\nCSV: {OUT / f'pool_asn_check{sfx}.csv'}")


if __name__ == "__main__":
    main()
