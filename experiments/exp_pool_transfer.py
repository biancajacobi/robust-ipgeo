#!/usr/bin/env python3
"""Pool transfer: L1·b + S on 8,896 never-seen RIPE probes (probes_pool).

The dataset (nabi filter: stable-30d, without geoloc-disputed, without shared
IPs, without all previously used populations; data/fetch_probes.py) is the
scaling validation of the transfer. Sources: the SEVEN fully collected ones
(without ipwho.is — daily quota ~1k, without ipapi.co — Cloudflare block);
that is 5 effective lines. The line ablation covers the omission
quantitatively (exp_min_lines: 5-line subset +0.16 km median / +1.3 pp tail
on anchors; exp_min_lines_s: S BSS of the 5-line subset even above the full
setup). reallyfreegeoip cannot be fully collected server-side (persistent
residual error set despite >12 resume passes) — the actual coverage is
reported below; missing rfg responses leave the MaxMind line intact via
maxmind_geolite2+geojs.

Three pre-specified design points (2026-08-20):
  1. SOURCE CONFOUND CONTROL: the old 500 probes are ADDITIONALLY computed
     restricted to the same 7 sources — pool↔probes comparisons are thereby
     "same sources, different population".
  2. FROZEN AS MAIN COLUMN: anchor calibration (LOO radii + ε as in
     exp_radius_transfer) applied unchanged; fresh refitting on the target
     population only as a control.
  3. AUTO-GEOIP STRATIFICATION instead of filtering: probes with
     system-auto-geoip-* tags are reported separately (circularity risk of
     the ground truth, cf. exp_probe_geoloc_sensitivity).

S transfer: S = line-weighted core share (r=50) around the FROZEN estimate;
logit fitted on the anchors (same 7 sources) and applied unchanged to the
pool (raw), plus Platt recalibration per OOF on the pool (staging as in
exp_s_quality_weighted). Note: the two-branch path (ip-api mobile flag) is
not available here (pool fetched without the fields parameter) — the
one-branch transfer is reported.

Usage:  python experiments/exp_pool_transfer.py
        python experiments/exp_pool_transfer.py --with-ipwhois
          (8 sources = 6 lines; possible since the ipwho.is completion
           2026-08-21 via pro quota; writes *_8src.csv variants,
           the published 7-source artifacts remain untouched)
Result: tables (stdout) + eval/out/pool_transfer.csv
        + eval/out/pool_transfer_strata.csv
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
import experiments.exp_t6_defaults as T6                   # noqa: E402
from experiments.exp_radius_transfer import eps_for        # noqa: E402
from experiments.exp_support_concentration import (        # noqa: E402
    folds, _fit, bss, ece, K,
)
from experiments.exp_s_declared_risk import (              # noqa: E402
    load_tags, classify,
)

CACHE = Path("data/cache")
OUT = Path("eval/out")
OUT.mkdir(parents=True, exist_ok=True)
TAU = 100.0
SEVEN = ("ip_api", "ipinfo", "geojs", "reallyfreegeoip",
         "maxmind_geolite2", "dbip_lite", "ip2location_lite")


def restrict(cases, sources=SEVEN):
    out = []
    for c in cases:
        prov = [p for p in c["provenance"] if p["source"] in sources]
        if prov:
            cc = dict(c)
            cc["provenance"] = prov
            out.append(cc)
    return out


def errors(cases, loo, eps):
    return np.array([haversine_error(
        T6.estimate(c, "L1", "b", loo, eps=eps), c["truth"]) for c in cases])


def s_values(cases, loo, eps):
    out = []
    for c in cases:
        prov = c["provenance"]
        pts = np.array([[p["lat"], p["lon"]] for p in prov], float)
        w = np.asarray(T6.line_weights_for(prov, "L1"), float)
        w = w / w.sum()
        est = np.array(T6.estimate(c, "L1", "b", loo, eps=eps))
        d = np.array([haversine_error(tuple(p), tuple(est)) for p in pts])
        out.append(float(w[d < 50].sum()))
    return np.array(out)


def stats(e):
    return dict(median_km=float(np.median(e)), mean_km=float(np.mean(e)),
                tail_pct=float(100 * np.mean(e > TAU)), n=len(e))


def fit_logit(S, y):
    X = np.column_stack([np.ones(len(S)), (S - S.mean()) / (S.std() + 1e-9)])
    return S.mean(), S.std() + 1e-9, _fit(X, y)


def apply_logit(S, mu, sd, w):
    X = np.column_stack([np.ones(len(S)), (S - mu) / sd])
    return 1.0 / (1.0 + np.exp(-X @ w))


def platt_oof(S, y):
    ph = np.empty(len(y))
    F = folds(y.astype(int), seed=0)
    for fi in range(K):
        tr, te = F != fi, F == fi
        mu, sd, w = fit_logit(S[tr], y[tr])
        ph[te] = apply_logit(S[te], mu, sd, w)
    return ph


def main() -> None:
    # --with-ipwhois: 8 sources (6 lines) instead of the published 7 (5 lines)
    with_ipwhois = "--with-ipwhois" in sys.argv
    srcs = SEVEN + ("ipwho_is",) if with_ipwhois else SEVEN
    sfx = "_8src" if with_ipwhois else ""

    # ---- Calibration: anchors (9 sources -> radii; fit on the target sources) ----
    anchors9 = load_cases()
    loo_a = T6.loo_pseudo_radii(anchors9)
    eps_a = eps_for(loo_a)
    anchors7 = restrict(anchors9, srcs)
    e_a7 = errors(anchors7, loo_a, eps_a)
    S_a7 = s_values(anchors7, loo_a, eps_a)
    y_a7 = (e_a7 > TAU).astype(float)
    mu_a, sd_a, w_a = fit_logit(S_a7, y_a7)
    print(f"Anchor calibration frozen: eps={eps_a} km; "
          f"anchors on {len(srcs)} sources: {stats(e_a7)}")

    rows, strata_rows = [], []
    pops = [("probes_pool", "probes_pool"), (f"probes500_{len(srcs)}src", "probes")]
    for label, ds in pops:
        cases = restrict(load_cases(
            store.load_anchors_csv(CACHE / f"{ds}.csv"),
            store.load_observations_csv(CACHE / f"observations_{ds}.csv")), srcs)
        n_rfg = sum(any(p["source"] == "reallyfreegeoip" for p in c["provenance"])
                    for c in cases)
        loo_p = T6.loo_pseudo_radii(cases)
        eps_p = eps_for(loo_p)
        e_froz = errors(cases, loo_a, eps_a)
        e_fresh = errors(cases, loo_p, eps_p)

        print("\n" + "=" * 78)
        print(f"{label} (n={len(cases)}, rfg coverage {n_rfg}/{len(cases)} "
              f"= {100*n_rfg/len(cases):.1f} %)")
        print("=" * 78)
        for name, e in [("frozen", e_froz), ("fresh", e_fresh)]:
            s = stats(e)
            print(f"  L1·b {name:12s}: median {s['median_km']:5.2f} | "
                  f"mean {s['mean_km']:6.1f} | tail>100 {s['tail_pct']:4.1f} %")
            rows.append({"population": label, "config": name, **s,
                         "rfg_coverage_pct": round(100*n_rfg/len(cases), 1)})

        # S transfer (frozen estimate + anchor logit)
        S_p = s_values(cases, loo_a, eps_a)
        y_p = (e_froz > TAU).astype(float)
        ph_raw = apply_logit(S_p, mu_a, sd_a, w_a)
        ph_platt = platt_oof(S_p, y_p)
        print(f"  S transfer: raw BSS {bss(ph_raw, y_p):+.3f} "
              f"(ECE {ece(ph_raw, y_p):.3f}) | Platt OOF "
              f"BSS {bss(ph_platt, y_p):+.3f} (ECE {ece(ph_platt, y_p):.3f}) "
              f"| miss rate {y_p.mean():.3f}")
        rows.append({"population": label, "config": "S_transfer",
                     "bss_raw": round(float(bss(ph_raw, y_p)), 3),
                     "bss_platt_oof": round(float(bss(ph_platt, y_p)), 3),
                     "ece_raw": round(float(ece(ph_raw, y_p)), 3),
                     "ece_platt_oof": round(float(ece(ph_platt, y_p)), 3),
                     "missrate": round(float(y_p.mean()), 3), "n": len(cases)})

        # Stratification only for the pool
        if ds != "probes_pool":
            continue
        tags = load_tags("probes_pool")
        auto = {ip for ip, ts in tags.items()
                if any(t.startswith("system-auto-geoip") for t in ts)}

        # S transfer per stratum (review 2026-08-20): the miss label depends
        # on the GT and is systematically undercounted on circular auto-geoip
        # probes -> primarily on the stratum WITHOUT auto-GeoIP metadata.
        mask = np.array([c["ip"] not in auto for c in cases])
        for sname, m in [("non-auto-geoip", mask), ("auto-geoip", ~mask)]:
            ph_r = apply_logit(S_p[m], mu_a, sd_a, w_a)
            ph_pl = platt_oof(S_p[m], y_p[m])
            print(f"  S transfer [{sname}]: raw BSS {bss(ph_r, y_p[m]):+.3f} "
                  f"(ECE {ece(ph_r, y_p[m]):.3f}) | Platt OOF "
                  f"BSS {bss(ph_pl, y_p[m]):+.3f} (ECE {ece(ph_pl, y_p[m]):.3f}) "
                  f"| miss rate {y_p[m].mean():.3f} | n={int(m.sum())}")
            rows.append({"population": label, "config": f"S_transfer_{sname}",
                         "bss_raw": round(float(bss(ph_r, y_p[m])), 3),
                         "bss_platt_oof": round(float(bss(ph_pl, y_p[m])), 3),
                         "ece_raw": round(float(ece(ph_r, y_p[m])), 3),
                         "ece_platt_oof": round(float(ece(ph_pl, y_p[m])), 3),
                         "missrate": round(float(y_p[m].mean()), 3),
                         "n": int(m.sum())})
        groups = {"auto-geoip": [], "non-auto-geoip": []}
        classes: dict[str, list] = {}
        for c, e in zip(cases, e_froz):
            groups["auto-geoip" if c["ip"] in auto else "non-auto-geoip"].append(e)
            cat = classify(tags.get(c["ip"], set()))
            classes.setdefault(cat, []).append(e)
        print("\n  Stratification (frozen):")
        for name, es in list(groups.items()) + sorted(classes.items()):
            s = stats(np.array(es))
            print(f"    {name:14s}: n={s['n']:5d} | median {s['median_km']:6.2f} "
                  f"| mean {s['mean_km']:6.1f} | tail {s['tail_pct']:4.1f} %")
            strata_rows.append({"stratum": name, **s})

    pd.DataFrame(rows).to_csv(OUT / f"pool_transfer{sfx}.csv", index=False)
    pd.DataFrame(strata_rows).to_csv(OUT / f"pool_transfer_strata{sfx}.csv", index=False)
    print(f"\nCSV: {OUT}/pool_transfer{sfx}.csv + pool_transfer_strata{sfx}.csv")


if __name__ == "__main__":
    main()
