#!/usr/bin/env python3
"""Temporal drift at city level: the same anchors, the same sources, four months
later (June 2026 -> October 2026).

Question (2026-10-02): the paper recommends that authorities archive offline
builds because no provider can be queried as of a past date; drift, however,
had only been measured at country level (delegation archives,
exp_registry_drift), and the transfer of the frozen calibration to another
build vintage was "unmeasured beyond the two-month window of T8". This
experiment measures exactly that with the study's own small archive: the June
builds (data/db/, hash-pinned in the ledger) and the October builds
(data/db_2026-10/) plus a fresh query of all web APIs for the same anchor
addresses.

Design (pre-specified):
  * Population = anchors whose IP AND coordinates are identical in both
    metadata pulls (2026-06-01, 2026-10-02) ("stable anchors"); the ground
    truth is therefore the same by construction, every difference is source drift.
  * A  Source level: per source the distance between the June and the October
       answer (identical, >1 km, >100 km, median), error against GT per month,
       sign of the change at the 100-km threshold (hit->miss, miss->hit),
       country changes.
  * B  Aggregation L1*b on the COMMON portfolio (sources with >= 90 % success in
       both months on the stable anchors; web APIs can vanish between two
       dates -- in October 2026 reallyfreegeoip.org blocked with a Cloudflare
       challenge, a portfolio change is not drift): error June vs. October
       (paired bootstrap CI for the median and tail difference); on October
       FROZEN (June LOO radii + eps, calibrated on all June anchors and sources
       as in the paper) vs. FRESH (October radii) as in T8; flips at the 100-km
       threshold; distance between the two aggregate estimates. In addition the
       June row with the full portfolio (paper state) for orientation.
  * C  Risk signal S (r = 50, line-weighted): logit fitted on June and applied
       unchanged to October (BSS, ECE), Platt-OOF on October as control; miss
       rate per S half in both months.

What the experiment does NOT measure: drift on end-user addresses (anchors are
the most stable address class) and periods beyond four months.

Usage:   python experiments/exp_temporal_drift.py
Output:  tables (stdout) + eval/out/temporal_drift_sources.csv
         + eval/out/temporal_drift.csv (aggregated, IP-free)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import store                                     # noqa: E402
from eval.pipeline import load_cases                       # noqa: E402
from eval.metrics import haversine_error                   # noqa: E402
import experiments.exp_t6_defaults as T6                   # noqa: E402
from experiments.exp_radius_transfer import eps_for        # noqa: E402
from experiments.exp_bootstrap_ci import paired_bootstrap_stat  # noqa: E402
from experiments.exp_support_concentration import bss, ece  # noqa: E402
from experiments.exp_country_level import (                # noqa: E402
    country_votes, city_lut_from, _cc,
)
from experiments.exp_pool_transfer import (                # noqa: E402
    errors, s_values, stats, fit_logit, apply_logit, platt_oof, TAU,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "eval" / "out"
OUT.mkdir(parents=True, exist_ok=True)
COORD_TOL = 1e-4          # Grad; ~10 m — „gleiche Koordinate" der Anchor-Metadaten


def _load(anchors_csv: Path, obs_csv: Path):
    anchors = store.load_anchors_csv(anchors_csv)
    obs = store.load_observations_csv(obs_csv)
    return anchors, obs


def stable_ips(a_then: list[dict], a_now: list[dict]) -> tuple[set, dict]:
    then = {a["ip"]: a for a in a_then if a.get("ip")}
    now = {a["ip"]: a for a in a_now if a.get("ip")}
    both = set(then) & set(now)
    same = {ip for ip in both
            if abs(float(then[ip]["lat"]) - float(now[ip]["lat"])) < COORD_TOL
            and abs(float(then[ip]["lon"]) - float(now[ip]["lon"])) < COORD_TOL}
    churn = {"n_then": len(then), "n_now": len(now), "n_same_ip": len(both),
             "n_same_ip_and_coord": len(same), "n_only_then": len(set(then) - set(now)),
             "n_only_now": len(set(now) - set(then)), "n_coord_changed": len(both - same)}
    return same, churn


def _by_src(obs: list[dict], ips: set) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for o in obs:
        if o["ip"] in ips and o.get("status") == "success":
            try:
                lat, lon = float(o["lat"]), float(o["lon"])
            except (TypeError, ValueError):
                continue
            out.setdefault(o["source"], {})[o["ip"]] = {"lat": lat, "lon": lon,
                                                        "country": o.get("country") or ""}
    return out


def source_table(obs_then, obs_now, ips, truth) -> list[dict]:
    then, now = _by_src(obs_then, ips), _by_src(obs_now, ips)
    rows = []
    for s in sorted(set(then) | set(now)):
        a, b = then.get(s, {}), now.get(s, {})
        common = sorted(set(a) & set(b))
        if not common:
            rows.append({"source": s, "n_then": len(a), "n_now": len(b), "n_both": 0})
            continue
        d_pair = np.array([haversine_error((a[ip]["lat"], a[ip]["lon"]),
                                           (b[ip]["lat"], b[ip]["lon"])) for ip in common])
        e_then = np.array([haversine_error((a[ip]["lat"], a[ip]["lon"]), truth[ip]) for ip in common])
        e_now = np.array([haversine_error((b[ip]["lat"], b[ip]["lon"]), truth[ip]) for ip in common])
        miss_then, miss_now = e_then > TAU, e_now > TAU
        cc = np.array([(a[ip]["country"] or "").upper() != (b[ip]["country"] or "").upper()
                       for ip in common])
        rows.append({
            "source": s, "n_then": len(a), "n_now": len(b), "n_both": len(common),
            "identical_pct": round(100 * float((d_pair < 0.01).mean()), 1),
            "moved_gt1km_pct": round(100 * float((d_pair > 1).mean()), 1),
            "moved_gt100km_pct": round(100 * float((d_pair > 100).mean()), 1),
            "pair_dist_median_km": round(float(np.median(d_pair)), 2),
            "country_changed_pct": round(100 * float(cc.mean()), 2),
            "err_median_then_km": round(float(np.median(e_then)), 2),
            "err_median_now_km": round(float(np.median(e_now)), 2),
            "tail_then_pct": round(100 * float(miss_then.mean()), 1),
            "tail_now_pct": round(100 * float(miss_now.mean()), 1),
            "hit_to_miss_n": int((~miss_then & miss_now).sum()),
            "miss_to_hit_n": int((miss_then & ~miss_now).sum()),
        })
    return rows


def _restrict_cases(cases, ips, sources=None):
    out = []
    for c in cases:
        if c["ip"] not in ips:
            continue
        if sources is not None:
            prov = [p for p in c["provenance"] if p["source"] in sources]
            if not prov:
                continue
            c = dict(c, provenance=prov, sources=[p["source"] for p in prov],
                     points=np.array([[p["lat"], p["lon"]] for p in prov], float))
        out.append(c)
    return out


def common_portfolio(src_rows: list[dict], n_stable: int, min_cov: float = 0.9) -> list[str]:
    """Sources that succeeded in BOTH months on >= min_cov of the stable anchors."""
    return sorted(r["source"] for r in src_rows
                  if r.get("n_both", 0) >= min_cov * n_stable)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--then-anchors", default=str(CACHE / "anchors.csv"))
    ap.add_argument("--then-obs", default=str(CACHE / "observations.csv"))
    ap.add_argument("--now-anchors", default=str(CACHE / "anchors_oct.csv"))
    ap.add_argument("--now-obs", default=str(CACHE / "observations_anchors_oct.csv"))
    ap.add_argument("--label-then", default="2026-06")
    ap.add_argument("--label-now", default="2026-10")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--exclude", default="",
                    help="comma list of sources excluded from the common portfolio "
                         "(confounding check, e.g. ipwho_is)")
    ap.add_argument("--only", default="",
                    help="comma list: restrict the portfolio to exactly these sources "
                         "(e.g. offline lines only: maxmind_geolite2,geojs,dbip_lite,ip2location_lite)")
    args = ap.parse_args()

    a_then, o_then = _load(Path(args.then_anchors), Path(args.then_obs))
    a_now, o_now = _load(Path(args.now_anchors), Path(args.now_obs))
    ips, churn = stable_ips(a_then, a_now)
    truth = {a["ip"]: (float(a["lat"]), float(a["lon"])) for a in a_then if a["ip"] in ips}
    print(f"Anchor-Metadaten {args.label_then} → {args.label_now}: {churn}")
    print(f"Stable anchors (same IP and coordinate): n={len(ips)}\n")

    # ---------- A  Quellenebene ----------
    src_rows = source_table(o_then, o_now, ips, truth)
    df_s = pd.DataFrame(src_rows)
    cols = ["source", "n_both", "identical_pct", "moved_gt1km_pct", "moved_gt100km_pct",
            "pair_dist_median_km", "country_changed_pct", "err_median_then_km",
            "err_median_now_km", "tail_then_pct", "tail_now_pct", "hit_to_miss_n", "miss_to_hit_n"]
    print("A  sources: answer June <-> October on stable anchors")
    print(df_s.reindex(columns=cols).to_string(index=False))
    df_s.to_csv(OUT / f"temporal_drift_sources{args.out_suffix}.csv", index=False)

    # ---------- B  Aggregation ----------
    # calibration "then" on ALL anchors of that month (as in the paper), then
    # restriction to the stable set for the paired comparison.
    portfolio = common_portfolio(src_rows, len(ips))
    if args.exclude:
        portfolio = [x for x in portfolio if x not in args.exclude.split(",")]
    if args.only:
        portfolio = [x for x in portfolio if x in args.only.split(",")]
    dropped = sorted({r["source"] for r in src_rows} - set(portfolio))
    print(f"\nCommon portfolio (>= 90 % success in both months): {', '.join(portfolio)}"
          f"\n  not collectable in both months: {', '.join(dropped) or '-'}")
    cases_then_all = load_cases(a_then, o_then)
    loo_then = T6.loo_pseudo_radii(cases_then_all)
    eps_then = eps_for(loo_then)
    cases_then_full = _restrict_cases(cases_then_all, ips)            # Paper-Portfolio
    cases_then = _restrict_cases(cases_then_all, ips, portfolio)
    cases_now_all = load_cases(a_now, o_now)
    loo_now = T6.loo_pseudo_radii(cases_now_all)
    eps_now = eps_for(loo_now)
    cases_now = _restrict_cases(cases_now_all, ips, portfolio)
    ct = {c["ip"]: c for c in cases_then}
    cn = {c["ip"]: c for c in cases_now}
    # paired comparison only where the IP was answered by the SAME portfolio
    # sources in both months (daily quotas of free APIs otherwise drop single
    # sources per IP -- that would be portfolio, not drift).
    common_any = sorted(set(ct) & set(cn))
    common = [ip for ip in common_any if set(ct[ip]["sources"]) == set(cn[ip]["sources"])]
    print(f"  Cases answered in both months: {len(common_any)}; of which identical "
          f"source set per IP: {len(common)} (only these enter B and C)")
    cases_then = [ct[ip] for ip in common]
    cases_now = [cn[ip] for ip in common]
    n = len(common)

    e_then_full = errors([ct[ip] for ip in common
                          for ct in [{c["ip"]: c for c in cases_then_full}]], loo_then, eps_then)
    e_then = errors(cases_then, loo_then, eps_then)          # June, common portfolio
    e_now_frozen = errors(cases_now, loo_then, eps_then)     # October, frozen
    e_now_fresh = errors(cases_now, loo_now, eps_now)        # October, fresh
    est_then = np.array([T6.estimate(c, "L1", "b", loo_then, eps=eps_then) for c in cases_then])
    est_now = np.array([T6.estimate(c, "L1", "b", loo_then, eps=eps_then) for c in cases_now])
    d_est = np.array([haversine_error(tuple(p), tuple(q)) for p, q in zip(est_then, est_now)])
    # Country change of the aggregated estimate: core vote as in T10 (exp_country_level),
    # both months with the frozen June radii; truth = anchor country from the metadata.
    lut_then = city_lut_from(Path(args.then_obs))
    lut_now = city_lut_from(Path(args.now_obs))
    truth_cc = {a["ip"]: _cc(a.get("country") or "") for a in a_then if a["ip"] in ips}
    cc_then = np.array([country_votes(c, lut_then, loo_then, eps_then)[0] for c in cases_then])
    cc_now = np.array([country_votes(c, lut_now, loo_then, eps_then)[0] for c in cases_now])
    cc_truth = np.array([truth_cc[c["ip"]] for c in cases_then])
    cc_changed = cc_then != cc_now
    n_src_then = np.array([len(c["provenance"]) for c in cases_then])
    n_src_now = np.array([len(c["provenance"]) for c in cases_now])

    d_med, lo_m, hi_m = paired_bootstrap_stat(e_now_frozen, e_then, np.median)
    tail = lambda e: 100 * float(np.mean(e > TAU))  # noqa: E731
    d_tail, lo_t, hi_t = paired_bootstrap_stat(e_now_frozen, e_then, tail)
    miss_then, miss_now = e_then > TAU, e_now_frozen > TAU

    print(f"\nB  Aggregation L1·b auf n={n} stabilen Anchors "
          f"(eps then {eps_then} km, fresh {eps_now} km; "
          f"sources per case median {np.median(n_src_then):.0f} -> {np.median(n_src_now):.0f})")
    rows = [{"population": "anchors_stable", "n": n, "n_any_both": len(common_any), **churn,
             "eps_then_km": eps_then, "eps_now_km": eps_now,
             "portfolio": "+".join(portfolio), "dropped_sources": "+".join(dropped)}]
    for name, e in [(f"{args.label_then}_fullportfolio", e_then_full),
                    (f"{args.label_then}", e_then),
                    (f"{args.label_now}_frozen", e_now_frozen),
                    (f"{args.label_now}_fresh", e_now_fresh)]:
        s = stats(e)
        print(f"  {name:16s}: Median {s['median_km']:5.2f} | Mittel {s['mittel_km']:6.1f} "
              f"| Tail>100 {s['tail_pct']:4.1f} %")
        rows.append({"population": "anchors_stable", "config": name, **s})
    print(f"  delta median (Oct frozen - June) {d_med:+.2f} km [{lo_m:+.2f}; {hi_m:+.2f}] | "
          f"Δ Tail {d_tail:+.2f} pp [{lo_t:+.2f}; {hi_t:+.2f}]")
    print(f"  Flips at 100 km: hit->miss {int((~miss_then & miss_now).sum())}, "
          f"miss->hit {int((miss_then & ~miss_now).sum())}, miss both {int((miss_then & miss_now).sum())}")
    print(f"  aggregate distance June<->Oct: identical (<0.01 km) {100*np.mean(d_est < 0.01):.1f} %, "
          f">1 km {100*np.mean(d_est > 1):.1f} %, >100 km {100*np.mean(d_est > 100):.1f} %, "
          f"Median {np.median(d_est):.2f} km")
    print(f"  Country vote changed June<->Oct: {100*cc_changed.mean():.2f} % (n={int(cc_changed.sum())}); "
          f"country correct June {100*np.mean(cc_then == cc_truth):.1f} %, Oct {100*np.mean(cc_now == cc_truth):.1f} %")
    rows.append({"population": "anchors_stable", "config": "delta_frozen_minus_then",
                 "d_median_km": round(d_med, 3), "d_median_lo": round(lo_m, 3), "d_median_hi": round(hi_m, 3),
                 "d_tail_pp": round(d_tail, 3), "d_tail_lo": round(lo_t, 3), "d_tail_hi": round(hi_t, 3),
                 "hit_to_miss_n": int((~miss_then & miss_now).sum()),
                 "miss_to_hit_n": int((miss_then & ~miss_now).sum()),
                 "miss_both_n": int((miss_then & miss_now).sum()),
                 "est_identical_pct": round(100 * float(np.mean(d_est < 0.01)), 1),
                 "est_moved_gt1km_pct": round(100 * float(np.mean(d_est > 1)), 1),
                 "est_moved_gt100km_pct": round(100 * float(np.mean(d_est > 100)), 1),
                 "est_dist_median_km": round(float(np.median(d_est)), 2),
                 "country_vote_changed_pct": round(100 * float(cc_changed.mean()), 2),
                 "country_vote_changed_n": int(cc_changed.sum()),
                 "country_hit_then_pct": round(100 * float(np.mean(cc_then == cc_truth)), 2),
                 "country_hit_now_pct": round(100 * float(np.mean(cc_now == cc_truth)), 2)})

    # ---------- C  risk signal S ----------
    S_then = s_values(cases_then, loo_then, eps_then)
    S_now = s_values(cases_now, loo_then, eps_then)
    y_then = miss_then.astype(float)
    y_now = miss_now.astype(float)
    mu, sd, w = fit_logit(S_then, y_then)
    ph_then_oof = platt_oof(S_then, y_then)
    ph_now_frozen = apply_logit(S_now, mu, sd, w)
    ph_now_oof = platt_oof(S_now, y_now)
    print(f"\nC  risk signal S (r=50), logit {args.label_then} frozen -> {args.label_now}")
    print(f"  June  Platt-OOF : BSS {bss(ph_then_oof, y_then):+.3f} (ECE {ece(ph_then_oof, y_then):.3f}) | miss rate {y_then.mean():.3f}")
    print(f"  Oct   frozen    : BSS {bss(ph_now_frozen, y_now):+.3f} (ECE {ece(ph_now_frozen, y_now):.3f}) | miss rate {y_now.mean():.3f}")
    print(f"  Oct   Platt-OOF : BSS {bss(ph_now_oof, y_now):+.3f} (ECE {ece(ph_now_oof, y_now):.3f})")
    # isolate the base-rate effect (review point 3): keep the June slope, shift only the
    # intercept so that the mean prediction matches the October miss rate
    # matches. Difference to "frozen" = pure base-rate share of the BSS drop.
    from scipy.optimize import brentq
    X_now = (S_now - mu) / sd
    f = lambda b0: float(np.mean(1 / (1 + np.exp(-(b0 + w[1] * X_now))))) - float(y_now.mean())  # noqa: E731
    b0 = brentq(f, w[0] - 10, w[0] + 10)
    ph_now_shift = 1 / (1 + np.exp(-(b0 + w[1] * X_now)))
    # rank discrimination without calibration: AUC (Mann-Whitney) per month
    def auc(S, y):
        pos, neg = S[y == 1], S[y == 0]
        return float(np.mean([(1.0 if a < b else 0.5 if a == b else 0.0)
                              for a in pos for b in neg]))  # low S = miss
    auc_then, auc_now = auc(S_then, y_then), auc(S_now, y_now)
    print(f"  Oct   slope frozen, base rate adjusted: BSS {bss(ph_now_shift, y_now):+.3f} "
          f"(ECE {ece(ph_now_shift, y_now):.3f})")
    print(f"  AUC (low S -> miss): June {auc_then:.3f} | Oct {auc_now:.3f}")
    half = 0.5
    for lab, S, y in [(args.label_then, S_then, y_then), (args.label_now, S_now, y_now)]:
        lo_mask = S < half
        print(f"  {lab}: P(Miss | S<{half}) {100*y[lo_mask].mean():5.1f} % (n={lo_mask.sum()}) | "
              f"P(Miss | S≥{half}) {100*y[~lo_mask].mean():5.2f} % (n={(~lo_mask).sum()})")
        rows.append({"population": "anchors_stable", "config": f"S_halves_{lab}",
                     "n_low": int(lo_mask.sum()), "miss_low_pct": round(100 * float(y[lo_mask].mean()), 2),
                     "n_high": int((~lo_mask).sum()), "miss_high_pct": round(100 * float(y[~lo_mask].mean()), 3)})
    rows.append({"population": "anchors_stable", "config": "S_transfer",
                 "bss_then_oof": round(float(bss(ph_then_oof, y_then)), 3),
                 "ece_then_oof": round(float(ece(ph_then_oof, y_then)), 3),
                 "bss_now_frozen": round(float(bss(ph_now_frozen, y_now)), 3),
                 "ece_now_frozen": round(float(ece(ph_now_frozen, y_now)), 3),
                 "bss_now_oof": round(float(bss(ph_now_oof, y_now)), 3),
                 "ece_now_oof": round(float(ece(ph_now_oof, y_now)), 3),
                 "bss_now_frozen_slope_shifted_intercept": round(float(bss(ph_now_shift, y_now)), 3),
                 "auc_then": round(auc_then, 3), "auc_now": round(auc_now, 3),
                 "missrate_then": round(float(y_then.mean()), 4),
                 "missrate_now": round(float(y_now.mean()), 4),
                 "S_pair_abs_diff_median": round(float(np.median(np.abs(S_now - S_then))), 3)})

    pd.DataFrame(rows).to_csv(OUT / f"temporal_drift{args.out_suffix}.csv", index=False)
    print(f"\nCSV: {OUT}/temporal_drift{args.out_suffix}.csv + temporal_drift_sources{args.out_suffix}.csv")


if __name__ == "__main__":
    main()
