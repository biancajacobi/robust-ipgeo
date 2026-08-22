"""Fair learned single-source baseline: CV-selected source vs. aggregation L1*b.

A natural objection: the aggregation uses corpus knowledge
(LOO pseudo-radii); the previous T1 argument "single-source selection requires
ex-post knowledge" is therefore asymmetric. Fair baseline: a SINGLE-SOURCE
strategy that receives the same corpus knowledge -- in the training fold the
globally best source per criterion (median / mean / tail rate>100km) is
determined, and in the test fold only that source is used (fallback chain along
the training ranking if the source does not answer for an IP).

Computed are:
  1. CV single source (10-fold, seed 0) per criterion on the anchors,
     paired bootstrap CIs of the difference to L1*b (iid + cluster=ASN),
  2. a REGIONAL variant (choice per country in the training fold, fallback to
     global for strata < MIN_STRAT cases) -- checks whether finer selection
     carries or overfits on thin strata,
  3. the TRANSFER test: the source selected on ALL anchors per criterion,
     applied to probes (n=500), probes_holdout (n=399), and
     probes_mobile_ext (n=178) against the aggregation newly
     learned there (identical to the stress-test configuration, tab:probes).

Expectation (honestly noted in advance): on the anchor population, the CV
source wins the metric it was selected on (tab:t1 implicitly concedes that).
The defence of the aggregation lies in transfer (the anchor-selected source is
a property of the population), in the non-stationarity of the sources, in the
breakdown point 0 of the single source, and in the missing ex-ante signal.

Usage:   python experiments/exp_single_source_cv.py
Output:  tables (stdout) + eval/out/single_source_cv.csv
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
    B, SEED, SOURCE_LABELS, paired_bootstrap, paired_bootstrap_stat,
)
from experiments.exp_cluster_bootstrap import (               # noqa: E402
    make_groups, cluster_boot,
)

OUT = ROOT / "eval" / "out"
K = 10
MIN_STRAT = 20          # minimum training cases per country for the regional choice
TAIL_KM = 100.0
CRITERIA = [
    ("median", lambda e: float(np.median(e))),
    ("mean", lambda e: float(np.mean(e))),
    ("tail",   lambda e: float(np.mean(np.asarray(e) > TAIL_KM))),
]
rows: list[dict] = []


# --------------------------------------------------------------------------- #
def build_errors(cases):
    """(agg_err, src_err, meta) per IP: L1*b error, per-source errors, (asn, country)."""
    loo = loo_pseudo_radii(cases)
    eps = min(EPS_GRID, key=lambda e: abs(e - float(np.median(
        [loo[s]["_global"] for s in loo]))))
    agg, meta = {}, {}
    src = {s: {} for s, _ in SOURCE_LABELS}
    for c in cases:
        ip = c["ip"]
        agg[ip] = haversine_error(estimate(c, "L1", "b", loo, eps=eps), c["truth"])
        meta[ip] = (str(c.get("asn")), str(c.get("country")))
        for p in c["provenance"]:
            if p["source"] in src:
                src[p["source"]][ip] = haversine_error((p["lat"], p["lon"]), c["truth"])
    return agg, src, meta


def ranking(src, ips, crit):
    """Source ranking (best first) by criterion on the given IPs."""
    scores = []
    for s, _ in SOURCE_LABELS:
        e = [src[s][ip] for ip in ips if ip in src[s]]
        if e:
            scores.append((crit(e), s))
    return [s for _, s in sorted(scores)]


def strategy_errors(src, order_for_ip, ips):
    """Error vector of the strategy: per IP the first responding source of the ranking."""
    out = []
    for ip in ips:
        for s in order_for_ip(ip):
            if ip in src[s]:
                out.append(src[s][ip])
                break
        else:
            raise RuntimeError(f"no source for {ip}")
    return np.asarray(out)


def compare(label, e_strat, e_agg, labels_asn, crit_key):
    """Paired differences L1*b - strategy on all three measures, iid + ASN cluster."""
    stats = {"median": lambda d: float(np.median(d)),
             "mean": lambda d: float(np.mean(d)),
             "tail":   lambda d: float(100.0 * np.mean(np.asarray(d) > TAIL_KM))}
    print(f"  {label}")
    print(f"    strategy:  Med {np.median(e_strat):7.2f}  Mean {np.mean(e_strat):7.1f}  "
          f"Tail {100*np.mean(e_strat > TAIL_KM):5.1f} %   (L1*b: {np.median(e_agg):.2f} / "
          f"{np.mean(e_agg):.1f} / {100*np.mean(e_agg > TAIL_KM):.1f})")
    groups = make_groups(labels_asn)
    for mkey, mstat in stats.items():
        delta, lo, hi = paired_bootstrap_stat(e_agg, e_strat, mstat)
        boot = cluster_boot(lambda idx, f=mstat: float(f(e_agg[idx]) - f(e_strat[idx])),
                            groups, B)
        clo, chi = np.percentile(boot, [2.5, 97.5])
        sig = "*" if not (lo <= 0 <= hi) else " "
        csig = "*" if not (clo <= 0 <= chi) else " "
        mark = "  <- selection criterion" if mkey == crit_key else ""
        print(f"    Delta {mkey:6s} {delta:+9.2f}  iid [{lo:+8.2f},{hi:+8.2f}]{sig} "
              f" ASN [{clo:+8.2f},{chi:+8.2f}]{csig}{mark}")
        rows.append({"strategy": label, "criterion": crit_key, "metric": mkey,
                     "delta": round(delta, 3), "ci_lo": round(lo, 3),
                     "ci_hi": round(hi, 3), "asn_lo": round(float(clo), 3),
                     "asn_hi": round(float(chi), 3),
                     "sig_iid": bool(not (lo <= 0 <= hi)),
                     "sig_asn": bool(not (clo <= 0 <= chi)), "n": len(e_agg)})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    agg, src, meta = build_errors(cases)
    ips = sorted(agg)
    e_agg = np.array([agg[ip] for ip in ips])
    asn = [meta[ip][0] for ip in ips]
    idx_of = {ip: i for i, ip in enumerate(ips)}

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(ips))
    fold = np.empty(len(ips), int)
    fold[perm] = np.arange(len(ips)) % K

    print("=" * 84)
    print(f"LEARNED SINGLE-SOURCE BASELINE vs. L1*b (anchors n={len(ips)}, "
          f"{K}-fold CV, B={B:,})")
    print("delta = L1*b - strategy; NEGATIVE = aggregation better, POSITIVE = strategy better")
    print("=" * 84)

    # --- 1) global CV choice per criterion ------------------------------------
    for ckey, cfun in CRITERIA:
        picks, e_strat = [], np.empty(len(ips))
        for fi in range(K):
            tr_ips = [ip for ip in ips if fold[idx_of[ip]] != fi]
            te_ips = [ip for ip in ips if fold[idx_of[ip]] == fi]
            order = ranking(src, tr_ips, cfun)
            picks.append(order[0])
            e = strategy_errors(src, lambda ip: order, te_ips)
            for ip, v in zip(te_ips, e):
                e_strat[idx_of[ip]] = v
        uniq = sorted(set(picks))
        print(f"\nCV single source by {ckey.upper()}  (fold choice: "
              + ", ".join(f"{picks.count(u)}x {u}" for u in uniq) + ")")
        compare(f"CV global ({ckey})", e_strat, e_agg, asn, ckey)

    # --- 2) regional choice (per country, fallback to global) -----------------
    for ckey, cfun in [("median", CRITERIA[0][1]), ("mean", CRITERIA[1][1])]:
        e_strat = np.empty(len(ips))
        n_regional = 0
        for fi in range(K):
            tr_ips = [ip for ip in ips if fold[idx_of[ip]] != fi]
            te_ips = [ip for ip in ips if fold[idx_of[ip]] == fi]
            g_order = ranking(src, tr_ips, cfun)
            by_land: dict[str, list[str]] = {}
            for ip in tr_ips:
                by_land.setdefault(meta[ip][1], []).append(ip)
            land_order = {L: ranking(src, lips, cfun)
                          for L, lips in by_land.items() if len(lips) >= MIN_STRAT}
            n_regional += sum(meta[ip][1] in land_order for ip in te_ips)

            def order_for(ip, lo=land_order, go=g_order):
                return lo.get(meta[ip][1], go)
            e = strategy_errors(src, order_for, te_ips)
            for ip, v in zip(te_ips, e):
                e_strat[idx_of[ip]] = v
        print(f"\nREGIONAL CV choice per country by {ckey.upper()} "
              f"(>= {MIN_STRAT} training cases, else global; "
              f"{n_regional/len(ips):.0%} of cases regional)")
        compare(f"CV regional ({ckey})", e_strat, e_agg, asn, ckey)

    # --- 3) transfer: anchor-selected source on never-seen populations --------
    # Deduplicated by selected source (the mean and tail choices coincide on
    # ipinfo); per population the aggregation is newly learned there (like the
    # stress test in tab:probes).
    cache = ROOT / "data" / "cache"
    picks_by_source: dict[str, list[str]] = {}
    for ckey, cfun in CRITERIA:
        picks_by_source.setdefault(ranking(src, ips, cfun)[0], []).append(ckey)
    for ds in ("probes", "probes_holdout", "probes_mobile_ext"):
        p_cases = load_cases(store.load_anchors_csv(cache / f"{ds}.csv"),
                             store.load_observations_csv(
                                 cache / f"observations_{ds}.csv"))
        p_agg, p_src, p_meta = build_errors(p_cases)
        p_ips = sorted(p_agg)
        e_pagg = np.array([p_agg[ip] for ip in p_ips])
        p_asn = [p_meta[ip][0] for ip in p_ips]
        print("\n" + "=" * 84)
        print(f"TRANSFER: source selected on ALL anchors -> {ds} "
              f"(n={len(p_ips)}); L1*b there: Med {np.median(e_pagg):.1f} / "
              f"Mean {np.mean(e_pagg):.1f} / Tail {100*np.mean(e_pagg > TAIL_KM):.1f} %")
        print("=" * 84)
        for source, ckeys in picks_by_source.items():
            order = ranking(src, ips,
                            dict(CRITERIA)[ckeys[0]])   # learned on anchors!
            e_strat = strategy_errors(p_src, lambda ip: order, p_ips)
            print(f"\nanchor choice by {'/'.join(k.upper() for k in ckeys)}: {source}")
            compare(f"Transfer {ds} ({'/'.join(ckeys)})", e_strat, e_pagg,
                    p_asn, ckeys[0])

    pd.DataFrame(rows).to_csv(OUT / "single_source_cv.csv", index=False)
    print("\nREADING:")
    print("  The CV single source expectedly wins the metric it was selected on within")
    print("  the anchor population -- the fair comparison concedes that explicitly. The")
    print("  aggregation defends itself via transfer (population change), the")
    print("  non-stationarity of the sources, breakdown, and the consensus-based")
    print("  ex-ante signal S, which none of the strategies tested here carries.")
    print(f"\nCSV: {OUT}/single_source_cv.csv")


if __name__ == "__main__":
    main()
