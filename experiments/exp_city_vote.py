"""Additional T1 baseline: naive city-level majority vote + paired bootstrap CIs.

A faithful, deliberately simple representative of the discrete fusion methods
(li2017fusion / xie2018fusion): per anchor, every responding source votes with
its (country, city) label; the city with the most votes wins, and the estimate
is the coordinate median of the winning group. NO line collapse (each source =
one vote) -- the naive fusion approach treats the MaxMind replicates as
independent evidence.

Additionally a LINE-COLLAPSED variant of the same vote: every vote is
weighted by 1/|line among the voters| (the same collapse definition as the
L1 aggregation, cf. eval.pipeline.line_weights: every provenance line carries
total vote mass 1). It separates the discrete-vs-continuous effect from the
collapsed-vs-not-collapsed effect: if the vote's deficit against L1*b
persists after the collapse, it is due to the discretisation, not to
replicate dominance.

Tie-break: highest vote count, then alphabetically by (country, city). Sources
without a city label abstain; if no source point has a city, the fallback is
the coordinate median of all source points.

The vote is computed on the same cases/points as the aggregation L1*b
(load_cases), with city/country looked up per (ip, source) from
observations.csv. This keeps the paired bootstrap against L1*b (median, mean,
tail) properly aligned.

Usage:  python experiments/exp_city_vote.py
Result: table (stdout) + eval/out/city_vote_bootstrap.csv
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.pipeline import load_cases                         # noqa: E402
from eval.metrics import haversine_error                     # noqa: E402
from experiments.exp_t6_defaults import (                    # noqa: E402
    loo_pseudo_radii, estimate, EPS_GRID,
)
from experiments.exp_bootstrap_ci import (                   # noqa: E402
    paired_bootstrap, paired_bootstrap_stat,
)

OBS = Path("data/cache/observations.csv")
OUT = Path("eval/out"); OUT.mkdir(parents=True, exist_ok=True)
TAIL_KM = 100.0

# OBS is pinned to the anchor file, but load_cases() is env-controlled:
# with GEOIP_DATASET != anchors the city LUT would read a DIFFERENT population
# than the cases and every case would silently fall into the coordinate-median
# fallback.
import data.store as _store  # noqa: E402
if _store.DATASET != "anchors":
    raise SystemExit("exp_city_vote is pinned to the anchors dataset "
                     f"(GEOIP_DATASET={_store.DATASET!r} is set -- please unset it).")


def city_lookup() -> dict[tuple[str, str], tuple[str, str]]:
    df = pd.read_csv(OBS, dtype=str)
    lut = {}
    for _, r in df.iterrows():
        city = str(r.get("city") or "").strip()
        country = str(r.get("country") or "").strip()
        lut[(str(r["ip"]), str(r["source"]))] = (country, city)
    return lut


def city_vote(case: dict, lut: dict) -> tuple[float, float]:
    named = []  # (key, lat, lon)
    for p in case["provenance"]:
        country, city = lut.get((case["ip"], p["source"]), ("", ""))
        if city and city.lower() != "nan":
            named.append((f"{country.upper()}|{city.lower()}", p["lat"], p["lon"]))
    if not named:
        pts = case["points"]
        return float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
    counts = Counter(k for k, _, _ in named)
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    win = [(la, lo) for k, la, lo in named if k == top]
    return (float(np.median([w[0] for w in win])),
            float(np.median([w[1] for w in win])))


def city_vote_collapsed(case: dict, lut: dict) -> tuple[float, float]:
    """Line-collapsed city vote: vote weight 1/|line| per voting point.

    Collapse definition as in the L1 aggregation (eval.pipeline.line_weights),
    but counted over the VOTING points (abstentions without a city label do
    not artificially shrink a line towards zero): every line that votes at
    all carries total vote mass 1. Winner = city with the largest vote mass;
    tie-break and estimate point (coordinate median of the winning group)
    identical to the naive vote.
    """
    named = []  # (key, lat, lon, lineage)
    for p in case["provenance"]:
        country, city = lut.get((case["ip"], p["source"]), ("", ""))
        if city and city.lower() != "nan":
            named.append((f"{country.upper()}|{city.lower()}", p["lat"], p["lon"],
                          p.get("lineage") or "unknown"))
    if not named:
        pts = case["points"]
        return float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
    counts = Counter(lin for _, _, _, lin in named)
    mass: Counter = Counter()
    for k, _, _, lin in named:
        mass[k] += 1.0 / counts[lin]
    top = sorted(mass.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    win = [(la, lo) for k, la, lo, _ in named if k == top]
    return (float(np.median([w[0] for w in win])),
            float(np.median([w[1] for w in win])))


def _stats(e: np.ndarray) -> tuple[float, float, float]:
    return (float(np.median(e)), float(np.mean(e)), 100 * float(np.mean(e > TAIL_KM)))


def main() -> None:
    cases = load_cases()
    loo = loo_pseudo_radii(cases)
    globals_ = [loo[s]["_global"] for s in loo]
    anchor_eps = min(EPS_GRID, key=lambda e: abs(e - float(np.median(globals_))))
    lut = city_lookup()

    agg, agga, cv, cvl = {}, {}, {}, {}
    for c in cases:
        ip = c["ip"]
        agg[ip] = haversine_error(estimate(c, "L1", "b", loo, eps=anchor_eps), c["truth"])
        agga[ip] = haversine_error(estimate(c, "L1", "a", loo), c["truth"])
        cv[ip] = haversine_error(city_vote(c, lut), c["truth"])
        cvl[ip] = haversine_error(city_vote_collapsed(c, lut), c["truth"])

    ips = sorted(set(agg) & set(cv))
    da = np.array([agg[ip] for ip in ips])
    daa = np.array([agga[ip] for ip in ips])
    dr = np.array([cv[ip] for ip in ips])
    dl = np.array([cvl[ip] for ip in ips])

    cvm = _stats(dr)
    aggm = _stats(da)
    aggam = _stats(daa)
    cvlm = _stats(dl)
    print("=" * 66)
    print(f"City-level majority vote vs. aggregation L1*b   (n={len(ips)})")
    print("=" * 66)
    print(f"  City vote : median {cvm[0]:5.1f} | mean {cvm[1]:6.1f} | tail>100 {cvm[2]:4.1f}%")
    print(f"  L1*b      : median {aggm[0]:5.1f} | mean {aggm[1]:6.1f} | tail>100 {aggm[2]:4.1f}%")
    print(f"  City vote (line-collapsed)"
          f": median {cvlm[0]:5.1f} | mean {cvlm[1]:6.1f} | tail>100 {cvlm[2]:4.1f}%")

    dmed, lo, hi, lob, hib = paired_bootstrap(da, dr)
    dmean, mlo, mhi = paired_bootstrap_stat(da, dr, np.mean)
    tail = lambda a: 100 * np.mean(a > TAIL_KM)  # noqa: E731
    dt, tlo, thi = paired_bootstrap_stat(da, dr, tail)
    print("\n  Paired bootstrap (B=10,000, seed=0) | Δ = L1*b − city vote (negative = L1*b better)")
    print(f"    Δ median = {dmed:+6.2f} km  95%-CI [{lo:+.2f}, {hi:+.2f}]")
    print(f"    Δ mean   = {dmean:+6.1f} km  95%-CI [{mlo:+.1f}, {mhi:+.1f}]")
    print(f"    Δ tail   = {dt:+6.2f} pp  95%-CI [{tlo:+.2f}, {thi:+.2f}]")

    # Line-collapsed variant: against L1*b (same structure as above) and
    # directly against the naive vote (= share of the deficit due to the
    # missing collapse).
    lmed, llo, lhi, _, _ = paired_bootstrap(da, dl)
    lmean, lmlo, lmhi = paired_bootstrap_stat(da, dl, np.mean)
    lt, ltlo, lthi = paired_bootstrap_stat(da, dl, tail)
    print("\n  Paired bootstrap | Δ = L1*b − city vote (line-collapsed)")
    print(f"    Δ median = {lmed:+6.2f} km  95%-CI [{llo:+.2f}, {lhi:+.2f}]")
    print(f"    Δ mean   = {lmean:+6.1f} km  95%-CI [{lmlo:+.1f}, {lmhi:+.1f}]")
    print(f"    Δ tail   = {lt:+6.2f} pp  95%-CI [{ltlo:+.2f}, {lthi:+.2f}]")

    cmed, clo, chi, _, _ = paired_bootstrap(dl, dr)
    cmean, cmlo, cmhi = paired_bootstrap_stat(dl, dr, np.mean)
    ct, ctlo, cthi = paired_bootstrap_stat(dl, dr, tail)
    print("\n  Paired bootstrap | Δ = collapsed − naive (negative = collapse helps)")
    print(f"    Δ median = {cmed:+6.2f} km  95%-CI [{clo:+.2f}, {chi:+.2f}]")
    print(f"    Δ mean   = {cmean:+6.1f} km  95%-CI [{cmlo:+.1f}, {cmhi:+.1f}]")
    print(f"    Δ tail   = {ct:+6.2f} pp  95%-CI [{ctlo:+.2f}, {cthi:+.2f}]")

    # Clean discretisation comparison: L1*a is line-collapsed like the
    # collapsed vote but WITHOUT radius weighting -- the Δ isolates discrete
    # vs. continuous, without the radius confound of L1*b.
    amed, alo, ahi, _, _ = paired_bootstrap(daa, dl)
    amean, amlo, amhi = paired_bootstrap_stat(daa, dl, np.mean)
    at, atlo, athi = paired_bootstrap_stat(daa, dl, tail)
    print(f"\n  L1*a      : median {aggam[0]:5.1f} | mean {aggam[1]:6.1f} | tail>100 {aggam[2]:4.1f}%")
    print("  Paired bootstrap | Δ = L1*a − city vote (line-collapsed)"
          " (negative = continuous better)")
    print(f"    Δ median = {amed:+6.2f} km  95%-CI [{alo:+.2f}, {ahi:+.2f}]")
    print(f"    Δ mean   = {amean:+6.1f} km  95%-CI [{amlo:+.1f}, {amhi:+.1f}]")
    print(f"    Δ tail   = {at:+6.2f} pp  95%-CI [{atlo:+.2f}, {athi:+.2f}]")

    pd.DataFrame([{
        "n": len(ips),
        "cv_median": round(cvm[0], 1), "cv_mean": round(cvm[1], 1), "cv_tail": round(cvm[2], 1),
        "l1b_median": round(aggm[0], 1), "l1b_mean": round(aggm[1], 1), "l1b_tail": round(aggm[2], 1),
        "d_median": round(dmed, 2), "d_median_lo": round(lo, 2), "d_median_hi": round(hi, 2),
        "d_mean": round(dmean, 1), "d_mean_lo": round(mlo, 1), "d_mean_hi": round(mhi, 1),
        "d_tail": round(dt, 2), "d_tail_lo": round(tlo, 2), "d_tail_hi": round(thi, 2),
        # new columns: line-collapsed variant
        "cvl_median": round(cvlm[0], 1), "cvl_mean": round(cvlm[1], 1), "cvl_tail": round(cvlm[2], 1),
        "dl_median": round(lmed, 2), "dl_median_lo": round(llo, 2), "dl_median_hi": round(lhi, 2),
        "dl_mean": round(lmean, 1), "dl_mean_lo": round(lmlo, 1), "dl_mean_hi": round(lmhi, 1),
        "dl_tail": round(lt, 2), "dl_tail_lo": round(ltlo, 2), "dl_tail_hi": round(lthi, 2),
        "dcvl_median": round(cmed, 2), "dcvl_median_lo": round(clo, 2), "dcvl_median_hi": round(chi, 2),
        "dcvl_mean": round(cmean, 1), "dcvl_mean_lo": round(cmlo, 1), "dcvl_mean_hi": round(cmhi, 1),
        "dcvl_tail": round(ct, 2), "dcvl_tail_lo": round(ctlo, 2), "dcvl_tail_hi": round(cthi, 2),
        # new columns: L1*a (line-collapsed, without radius weighting) vs. collapsed vote
        "l1a_median": round(aggam[0], 1), "l1a_mean": round(aggam[1], 1), "l1a_tail": round(aggam[2], 1),
        "dla_median": round(amed, 2), "dla_median_lo": round(alo, 2), "dla_median_hi": round(ahi, 2),
        "dla_mean": round(amean, 1), "dla_mean_lo": round(amlo, 1), "dla_mean_hi": round(amhi, 1),
        "dla_tail": round(at, 2), "dla_tail_lo": round(atlo, 2), "dla_tail_hi": round(athi, 2),
    }]).to_csv(OUT / "city_vote_bootstrap.csv", index=False)
    print(f"\n  CSV: {OUT}/city_vote_bootstrap.csv")


if __name__ == "__main__":
    main()
