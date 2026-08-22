"""E4c — Regional breakdown of the headline aggregation L1*b.

A natural question: the RIPE anchor
population is Europe-heavy; the limitation so far only pointed to future
work. This script provides the breakdown: median/mean/tail>100km of the
headline aggregation L1*b per country (all countries with n >= MIN_N
anchors, remainder as a catch-all row) plus coarse continent groups and the
Europe share of the population.

No new data collection, pure reprocessing of the existing cases —
configuration identical to T1 (LOO pseudo-radii, eps choice as in
exp_city_vote/exp_single_source_cv).

Invocation: python experiments/exp_region_headline.py
Output: table (stdout) + eval/out/e4_region_headline.csv
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
from experiments.exp_t6_defaults import (                     # noqa: E402
    loo_pseudo_radii, estimate, EPS_GRID,
)

OUT = ROOT / "eval" / "out"
MIN_N = 30
TAIL_KM = 100.0

# coarse continent assignment of the countries present in the anchors (ISO-2)
EUROPE = {"AL", "AT", "BA", "BE", "BG", "BY", "CH", "CY", "CZ", "DE", "DK",
          "EE", "ES", "FI", "FO", "FR", "GB", "GE", "GI", "GR", "HR", "HU",
          "IE", "IS", "IT", "LI", "LT", "LU", "LV", "MD", "ME", "MK", "MT",
          "NL", "NO", "PL", "PT", "RO", "RS", "RU", "SE", "SI", "SK", "UA"}


def _stats(e: np.ndarray) -> tuple[float, float, float]:
    return (float(np.median(e)), float(np.mean(e)),
            100 * float(np.mean(e > TAIL_KM)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    loo = loo_pseudo_radii(cases)
    eps = min(EPS_GRID, key=lambda e: abs(e - float(np.median(
        [loo[s]["_global"] for s in loo]))))

    err_by_country: dict[str, list[float]] = {}
    for c in cases:
        e = haversine_error(estimate(c, "L1", "b", loo, eps=eps), c["truth"])
        err_by_country.setdefault(str(c.get("country")), []).append(e)

    n_total = sum(len(v) for v in err_by_country.values())
    n_eu = sum(len(v) for k, v in err_by_country.items() if k in EUROPE)
    print("=" * 72)
    print(f"L1*b per country (n_total={n_total}, Europe share "
          f"{100 * n_eu / n_total:.1f} %, countries={len(err_by_country)})")
    print("=" * 72)

    rows = []
    rest: list[float] = []
    for country, errs in sorted(err_by_country.items(),
                                key=lambda kv: -len(kv[1])):
        if len(errs) >= MIN_N:
            m = _stats(np.asarray(errs))
            rows.append({"country": country, "n": len(errs),
                         "median_km": round(m[0], 1), "mean_km": round(m[1], 1),
                         "tail_gt100km_pct": round(m[2], 1),
                         "europe": country in EUROPE})
        else:
            rest.extend(errs)
    if rest:
        m = _stats(np.asarray(rest))
        rows.append({"country": f"other (<{MIN_N})", "n": len(rest),
                     "median_km": round(m[0], 1), "mean_km": round(m[1], 1),
                     "tail_gt100km_pct": round(m[2], 1), "europe": ""})
    for region, keys in [("Europe", EUROPE),
                         ("non-Europe", set(err_by_country) - EUROPE)]:
        errs = [e for k in keys for e in err_by_country.get(k, [])]
        if errs:
            m = _stats(np.asarray(errs))
            rows.append({"country": f"[{region}]", "n": len(errs),
                         "median_km": round(m[0], 1), "mean_km": round(m[1], 1),
                         "tail_gt100km_pct": round(m[2], 1), "europe": ""})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(OUT / "e4_region_headline.csv", index=False)
    print(f"\nCSV: {OUT}/e4_region_headline.csv")


if __name__ == "__main__":
    main()
