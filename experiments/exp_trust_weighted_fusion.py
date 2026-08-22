"""Trust-weighted discrete fusion (after ivanov2022geoservices) vs. L1*b.

A natural objection: the city-level majority vote is too
simple as a fusion baseline -- the published methods weight the votes with
trust/confidence proxies (li2017fusion, xie2018fusion) or average the
coordinates of the winning city trust-weighted over free geoservices
(ivanov2022geoservices). We recompute this stronger variant here so that the
fusion baseline is not a strawman:

  1. Trust per source learned OUT-OF-FOLD (10-fold CV, seed 0, identical fold
     assignment as exp_single_source_cv): trust_s = 1 / median error of the
     source on the training fold. Ivanov et al. assign fixed trust scores
     from an evaluation dataset; the OOF variant is the fair equivalent on
     this corpus (no test leak).
  2. Vote: every responding source with a city label votes with weight
     trust_s for its (country, city); the city with the largest weight sum
     wins (primary criterion: weight sum; tie-break: alphabetical). Estimate point =
     trust-weighted arithmetic MEAN of the winner coordinates (as in the
     published method; deliberately NO median). Sources without a city label
     abstain; if no source has a city, the case falls back to the
     trust-weighted mean of all source points.
  3. NO line collapse (every source = one vote) -- faithful to the published
     methods; this is exactly where we expect the common-mode effect.

Part A (anchors): OOF trust vote vs. L1*b on the same cases --
median/mean/tail>100km, paired bootstrap (B=10,000, seed 0, iid) and ASN
cluster bootstrap of the differences; unweighted majority vote
(exp_city_vote) for context in the same run.

Part B (TRANSFER, analogous to exp_single_source_cv): trust learned on ALL
anchors and applied unchanged to probes / probes_holdout / probes_mobile_ext,
against the aggregation L1*b NEWLY learned there in each case (identical to
the stress-test configuration, tab:probes). Tests whether the learned trust
weights are a property of the calibration population.

Usage:   python experiments/exp_trust_weighted_fusion.py
Output:  tables (stdout) + eval/out/trust_weighted_fusion.csv
         (long format: one row per population)
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
from experiments.exp_city_vote import city_lookup, city_vote  # noqa: E402

OUT = ROOT / "eval" / "out"
CACHE = ROOT / "data" / "cache"
K = 10
TAIL_KM = 100.0
TRANSFER_DATASETS = ("probes", "probes_holdout", "probes_mobile_ext")


def fold_assignment(ips: list[str]) -> dict[str, int]:
    """Identical fold assignment as exp_single_source_cv (seed 0, 10-fold)."""
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(ips))
    fold = np.empty(len(ips), int)
    fold[perm] = np.arange(len(ips)) % K
    return {ip: int(fold[i]) for i, ip in enumerate(ips)}


def source_errors(cases: list[dict]) -> dict[str, dict[str, float]]:
    """{source: {ip: haversine error}} for the trust training."""
    src: dict[str, dict[str, float]] = {}
    for c in cases:
        for p in c["provenance"]:
            src.setdefault(p["source"], {})[c["ip"]] = haversine_error(
                (p["lat"], p["lon"]), c["truth"])
    return src


def trust_from(src: dict, ips: list[str]) -> dict[str, float]:
    """{source: trust} from the median error on the given IPs."""
    t = {}
    for s, errs in src.items():
        e = [errs[ip] for ip in ips if ip in errs]
        if e:
            t[s] = 1.0 / max(1e-6, float(np.median(e)))
    return t


def city_lut_from(obs_csv: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """(ip, source) -> (country, city) from an observations CSV (as in exp_city_vote)."""
    df = pd.read_csv(obs_csv, dtype=str)
    lut = {}
    for _, r in df.iterrows():
        city = str(r.get("city") or "").strip()
        country = str(r.get("country") or "").strip()
        lut[(str(r["ip"]), str(r["source"]))] = (country, city)
    return lut


def trust_weighted_vote(case: dict, lut: dict, trust: dict) -> tuple[float, float]:
    """Trust-weighted city vote + trust-weighted mean of the winners."""
    named = []  # (city_key, lat, lon, w)
    for p in case["provenance"]:
        w = trust.get(p["source"])
        if w is None:
            continue
        country, city = lut.get((case["ip"], p["source"]), ("", ""))
        if city and city.lower() != "nan":
            named.append((f"{country.upper()}|{city.lower()}", p["lat"], p["lon"], w))
    if not named:  # no city labels: fall back to the trust-weighted mean of all points
        pts, ws = [], []
        for p in case["provenance"]:
            w = trust.get(p["source"])
            if w is not None:
                pts.append((p["lat"], p["lon"])); ws.append(w)
        if not pts:  # population contains a source never seen on the anchors
            pts = [(p["lat"], p["lon"]) for p in case["provenance"]]
            ws = [1.0] * len(pts)
        pts, ws = np.asarray(pts, float), np.asarray(ws, float)
        return tuple((pts * ws[:, None]).sum(0) / ws.sum())
    votes: dict[str, float] = {}
    for k, _, _, w in named:
        votes[k] = votes.get(k, 0.0) + w
    top = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    win = np.asarray([(la, lo, w) for k, la, lo, w in named if k == top], float)
    ws = win[:, 2]
    return tuple((win[:, :2] * ws[:, None]).sum(0) / ws.sum())


def l1b_errors(cases: list[dict]) -> dict[str, float]:
    """Headline aggregation L1*b with the population's own LOO pseudo-radii."""
    loo = loo_pseudo_radii(cases)
    eps = min(EPS_GRID, key=lambda e: abs(e - float(np.median(
        [loo[s]["_global"] for s in loo]))))
    return {c["ip"]: haversine_error(estimate(c, "L1", "b", loo, eps=eps),
                                     c["truth"]) for c in cases}


def _stats(e: np.ndarray) -> tuple[float, float, float]:
    return (float(np.median(e)), float(np.mean(e)),
            100 * float(np.mean(e > TAIL_KM)))


def _paired(da: np.ndarray, db: np.ndarray, asn: list[str]) -> dict:
    """Paired deltas L1*b - baseline (negative = L1*b better): iid + ASN cluster."""
    dmed, lo, hi, _, _ = paired_bootstrap(da, db)
    dmean, mlo, mhi = paired_bootstrap_stat(da, db, np.mean)
    tail = lambda a: 100 * np.mean(a > TAIL_KM)  # noqa: E731
    dt, tlo, thi = paired_bootstrap_stat(da, db, tail)
    groups = make_groups(asn)
    diffs = {"median": lambda i: float(np.median(da[i]) - np.median(db[i])),
             "mean":   lambda i: float(np.mean(da[i]) - np.mean(db[i])),
             "tail":   lambda i: float(tail(da[i]) - tail(db[i]))}
    cl = {}
    for key, stat in diffs.items():
        v = cluster_boot(stat, groups, B)
        cl[key] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return {"d_median": dmed, "d_median_ci": (lo, hi), "d_median_asn": cl["median"],
            "d_mean": dmean, "d_mean_ci": (mlo, mhi), "d_mean_asn": cl["mean"],
            "d_tail": dt, "d_tail_ci": (tlo, thi), "d_tail_asn": cl["tail"]}


def _print_paired(res: dict) -> None:
    for key, unit in [("median", "km"), ("mean", "km"), ("tail", "pp")]:
        d = res[f"d_{key}"]; lo, hi = res[f"d_{key}_ci"]
        alo, ahi = res[f"d_{key}_asn"]
        print(f"    Δ {key:6s} = {d:+7.2f} {unit}  iid [{lo:+.2f}, {hi:+.2f}]  "
              f"ASN-Cluster [{alo:+.2f}, {ahi:+.2f}]")


def _row(population: str, n: int, tm, am, res: dict, cm=None) -> dict:
    row = {"population": population, "n": n,
           "twf_median": round(tm[0], 1), "twf_mean": round(tm[1], 1),
           "twf_tail": round(tm[2], 1),
           "l1b_median": round(am[0], 1), "l1b_mean": round(am[1], 1),
           "l1b_tail": round(am[2], 1),
           **{f"d_{k}": round(res[f"d_{k}"], 2) for k in ("median", "mean", "tail")},
           **{f"d_{k}_{b}": round(res[f"d_{k}_ci"][i], 2)
              for k in ("median", "mean", "tail") for i, b in enumerate(("lo", "hi"))},
           **{f"d_{k}_asn_{b}": round(res[f"d_{k}_asn"][i], 2)
              for k in ("median", "mean", "tail") for i, b in enumerate(("lo", "hi"))}}
    if cm is not None:
        row.update({"cv_median": round(cm[0], 1), "cv_mean": round(cm[1], 1),
                    "cv_tail": round(cm[2], 1)})
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    # ---- Part A: anchors, OOF trust ---------------------------------------
    cases = load_cases()
    lut = city_lookup()
    src = source_errors(cases)
    agg = l1b_errors(cases)
    ips = sorted(agg)
    fold_of = fold_assignment(ips)
    trusts = [trust_from(src, [ip for ip in ips if fold_of[ip] != fi])
              for fi in range(K)]

    twf, cv, asn_of = {}, {}, {}
    by_ip = {c["ip"]: c for c in cases}
    for ip in ips:
        c = by_ip[ip]
        twf[ip] = haversine_error(
            trust_weighted_vote(c, lut, trusts[fold_of[ip]]), c["truth"])
        cv[ip] = haversine_error(city_vote(c, lut), c["truth"])
        asn_of[ip] = str(c.get("asn"))

    da = np.array([agg[ip] for ip in ips])
    dt_ = np.array([twf[ip] for ip in ips])
    dc = np.array([cv[ip] for ip in ips])
    asn = [asn_of[ip] for ip in ips]

    print("=" * 78)
    print(f"A) Trust-gewichtete Fusion (OOF-Trust, {K}-fach) vs. L1*b   (n={len(ips)})")
    print("=" * 78)
    for name, e in [("L1*b", da), ("Trust-Vote", dt_), ("Majority-Vote", dc)]:
        m = _stats(e)
        print(f"  {name:14s}: Median {m[0]:5.1f} | Mittel {m[1]:6.1f} | "
              f"Tail>100 {m[2]:4.1f}%")
    res = _paired(da, dt_, asn)
    print(f"\n  Paired Bootstrap (B={B:,}, seed={SEED}) | "
          f"Δ = L1*b − Trust-Vote (negativ = L1*b besser)")
    _print_paired(res)
    rows.append(_row("anchors_oof", len(ips), _stats(dt_), _stats(da), res,
                     cm=_stats(dc)))

    # ---- Part B: transfer (trust learned on ALL anchors) ------------------
    trust_all = trust_from(src, ips)
    for ds in TRANSFER_DATASETS:
        p_cases = load_cases(store.load_anchors_csv(CACHE / f"{ds}.csv"),
                             store.load_observations_csv(
                                 CACHE / f"observations_{ds}.csv"))
        p_lut = city_lut_from(CACHE / f"observations_{ds}.csv")
        p_agg = l1b_errors(p_cases)
        p_ips = sorted(p_agg)
        p_by_ip = {c["ip"]: c for c in p_cases}
        e_agg = np.array([p_agg[ip] for ip in p_ips])
        e_twf = np.array([haversine_error(
            trust_weighted_vote(p_by_ip[ip], p_lut, trust_all),
            p_by_ip[ip]["truth"]) for ip in p_ips])
        p_asn = [str(p_by_ip[ip].get("asn")) for ip in p_ips]

        print("\n" + "=" * 78)
        print(f"B) TRANSFER: Anchor-Trust -> {ds} (n={len(p_ips)})")
        print("=" * 78)
        for name, e in [("L1*b (dort)", e_agg), ("Trust-Vote", e_twf)]:
            m = _stats(e)
            print(f"  {name:14s}: Median {m[0]:5.1f} | Mittel {m[1]:6.1f} | "
                  f"Tail>100 {m[2]:4.1f}%")
        p_res = _paired(e_agg, e_twf, p_asn)
        print(f"\n  Paired Bootstrap | Δ = L1*b − Trust-Vote (negativ = L1*b besser)")
        _print_paired(p_res)
        rows.append(_row(ds, len(p_ips), _stats(e_twf), _stats(e_agg), p_res))

    pd.DataFrame(rows).to_csv(OUT / "trust_weighted_fusion.csv", index=False)
    print(f"\n  CSV: {OUT}/trust_weighted_fusion.csv")


if __name__ == "__main__":
    main()
