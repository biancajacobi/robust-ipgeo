"""Sensitivity of the probe stress test to questionable probe locations.

Motivation (cf. izhikevich2024rooting): probe coordinates are
operator-reported; Izhikevich et al. (arXiv:2409.19109) show via DNS-root
latency violations that at least ~2% of RIPE vantage points are not at their
reported location, and have filed disputes -- which today surface in the RIPE
system tag ``system-geoloc-disputed``. In addition, ~23-27% of the fixed-line
probes carry ``system-auto-geoip-city/-country`` tags; their (not formally
documented) naming suggests that the position was (partly) set based on
GeoIP -- a potential circularity risk for a validation AGAINST GeoIP
databases.

This script computes the headline aggregation L1*b of the stress test
(population-specific LOO radii, as in tab:probes) per population in three
variants, alongside the individual sources contrasted in the paper:

  full          : all cases (status quo of tab:probes),
  excl. disputed: without ``system-geoloc-disputed`` probes,
  conservative  : additionally without ``system-auto-geoip-*`` probes.

Invocation: python experiments/exp_probe_geoloc_sensitivity.py
Output: tables (stdout) + eval/out/probe_geoloc_sensitivity.csv
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

OUT = ROOT / "eval" / "out"
CACHE = ROOT / "data" / "cache"
TAIL_KM = 100.0
DATASETS = (("probes", "probes_tags"),
            ("probes_holdout", "probes_holdout_tags"),
            ("probes_mobile_ext", "probes_mobile_ext_tags"))
SOURCES = ("maxmind_geolite2", "ipinfo", "ip2location_lite")


def _stats(e) -> tuple[float, float, float]:
    e = np.asarray(e, float)
    return (float(np.median(e)), float(np.mean(e)),
            100 * float(np.mean(e > TAIL_KM)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds, tagfile in DATASETS:
        cases = load_cases(store.load_anchors_csv(CACHE / f"{ds}.csv"),
                           store.load_observations_csv(
                               CACHE / f"observations_{ds}.csv"))
        tags = pd.read_csv(CACHE / f"{tagfile}.csv")
        tags["tags"] = tags["tags"].fillna("")
        disputed = set(tags.loc[
            tags.tags.str.contains("system-geoloc-disputed"), "ip"].astype(str))
        autogeo = set(tags.loc[
            tags.tags.str.contains("system-auto-geoip"), "ip"].astype(str))

        # Aggregation is always learned on the FULL population (status quo);
        # only the evaluation set is filtered.
        loo = loo_pseudo_radii(cases)
        eps = min(EPS_GRID, key=lambda e: abs(e - float(np.median(
            [loo[s]["_global"] for s in loo]))))
        err_agg, err_src = {}, {s: {} for s in SOURCES}
        for c in cases:
            err_agg[c["ip"]] = haversine_error(
                estimate(c, "L1", "b", loo, eps=eps), c["truth"])
            for p in c["provenance"]:
                if p["source"] in err_src:
                    err_src[p["source"]][c["ip"]] = haversine_error(
                        (p["lat"], p["lon"]), c["truth"])

        variants = [
            ("full", set()),
            ("excl. disputed", disputed),
            ("conservative", disputed | autogeo),
        ]
        print("\n" + "=" * 78)
        print(f"{ds}: n={len(err_agg)}, disputed={len(disputed & set(err_agg))}, "
              f"auto-geoip={len(autogeo & set(err_agg))}")
        print("=" * 78)
        for name, drop in variants:
            keep = [ip for ip in err_agg if ip not in drop]
            m = _stats([err_agg[ip] for ip in keep])
            line = (f"  {name:14s} (n={len(keep):4d}): L1*b med {m[0]:5.1f} | "
                    f"mean {m[1]:6.1f} | tail {m[2]:4.1f}%")
            row = {"population": ds, "variant": name, "n": len(keep),
                   "l1b_median": round(m[0], 1), "l1b_mean": round(m[1], 1),
                   "l1b_tail": round(m[2], 1)}
            for s in SOURCES:
                e = [err_src[s][ip] for ip in keep if ip in err_src[s]]
                if e:
                    sm = _stats(e)
                    line += f" | {s.split('_')[0]} Med {sm[0]:5.1f}"
                    row[f"{s}_median"] = round(sm[0], 1)
                    row[f"{s}_mean"] = round(sm[1], 1)
                    row[f"{s}_tail"] = round(sm[2], 1)
            print(line)
            rows.append(row)

    pd.DataFrame(rows).to_csv(OUT / "probe_geoloc_sensitivity.csv", index=False)
    print(f"\n  CSV: {OUT}/probe_geoloc_sensitivity.csv")


if __name__ == "__main__":
    main()
