#!/usr/bin/env python3
"""Hub coordinate signal: quick check for an OPEN research idea (no paper claim).

Idea: record, per source/country, source
coordinates onto which conspicuously many different IPs of a country collapse
("operator hubs", a symptom of coarse geolocation blocks, cf.
exp_giga_mechanism), and use them as a warning/weighting signal — this
operationalizes nabi's recommendation to treat large geolocation prefixes as
an accuracy bound, without needing their prefix data.

This quick check tests ONLY the precursor: does "share of line mass on hub
coordinates" (hub_mass) carry predictive power for misses (>100 km), BEYOND
the existing risk signal S?

Clean design (no in-sample circularity):
  * Split A/B by IP hash (deterministic, ~50/50).
  * Hub lexicon ONLY from split A: per (source, country), coordinates
    (0.01-degree grid) onto which >= MIN_IPS distinct A-IPs fall.
  * Evaluation ONLY on split B: frozen errors (anchor calibration),
    miss rates per hub_mass bucket, and OOF logit S vs. S+hub_mass (10-fold,
    BSS/ECE) on B.

HONEST LIMITS of this check (which is why it is a "research idea", not a
result):
  * One corpus (Giga), 19 countries, half the sample for the evaluation;
    for deployment, hub lexicons would have to come from INDEPENDENT samples
    per country (routed prefixes + local DBs) and be versioned like
    s_calibration.json (build drift!).
  * hub_mass is confounded with S and country severity; this is raw evidence
    only, not a calibrated operational quantity.
  * MaxMind accuracy_radius could carry the same information (reported
    alongside as a correlation).

Usage:  python experiments/exp_hub_signal.py [--dataset giga|probes_pool]
          (probes_pool = mixed operational view with a low miss rate —
           the natural deployment site of a hub warning signal; giga = default)
Result: table (stdout) + eval/out/hub_signal_<dataset>.csv
"""
from __future__ import annotations

import ipaddress
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import store                                     # noqa: E402
from eval.pipeline import load_cases                       # noqa: E402
import experiments.exp_t6_defaults as T6                   # noqa: E402
from experiments.exp_radius_transfer import eps_for        # noqa: E402
from experiments.exp_pool_transfer import (                # noqa: E402
    restrict, errors, s_values, TAU, SEVEN,
)
from experiments.exp_support_concentration import folds, _fit, bss, ece, K  # noqa: E402

CACHE = Path("data/cache")
OUT = Path("eval/out")
OUT.mkdir(parents=True, exist_ok=True)
EIGHT = SEVEN + ("ipwho_is",)
GRID = 2       # 0.01-degree grid
MIN_IPS = 10   # hub = coordinate with >= MIN_IPS distinct A-IPs (per source+country)


def split_a(ip: str) -> bool:
    try:
        return int(ipaddress.ip_address(ip)) % 2 == 0
    except ValueError:
        # pseudonymized identifiers (e.g. node_00042): digit parity.
        # NOTE: the A/B assignment then differs from a run on raw IPs, so
        # split-dependent values can deviate slightly — the qualitative
        # reading (research-idea quick check) is unaffected.
        return int(re.sub(r"\D", "", ip) or "0") % 2 == 0


def hub_mass_for(case, hubs) -> float:
    prov = case["provenance"]
    w = np.asarray(T6.line_weights_for(prov, "L1"), float)
    w = w / w.sum()
    on_hub = np.array([(p["source"], case["country"],
                        round(p["lat"], GRID), round(p["lon"], GRID)) in hubs
                       for p in prov])
    return float(w[on_hub].sum())


def oof_bss(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ph = np.empty(len(y))
    F = folds(y.astype(int), seed=0)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    for fi in range(K):
        tr, te = F != fi, F == fi
        Xtr = np.column_stack([np.ones(tr.sum()), (X[tr] - mu) / sd])
        Xte = np.column_stack([np.ones(te.sum()), (X[te] - mu) / sd])
        w = _fit(Xtr, y[tr])
        ph[te] = 1.0 / (1.0 + np.exp(-Xte @ w))
    return float(bss(ph, y)), float(ece(ph, y))


def main() -> None:
    ds = sys.argv[sys.argv.index("--dataset") + 1] if "--dataset" in sys.argv else "giga"
    anchors = load_cases()
    loo_a = T6.loo_pseudo_radii(anchors)
    eps_a = eps_for(loo_a)
    cases = restrict(load_cases(
        store.load_anchors_csv(CACHE / f"{ds}.csv"),
        store.load_observations_csv(CACHE / f"observations_{ds}.csv")), EIGHT)
    e = errors(cases, loo_a, eps_a)
    S = s_values(cases, loo_a, eps_a)
    is_a = np.array([split_a(c["ip"]) for c in cases])

    # --- Hub lexicon from split A ---
    coord_ips = defaultdict(set)
    for c, a in zip(cases, is_a):
        if not a:
            continue
        for p in c["provenance"]:
            coord_ips[(p["source"], c["country"],
                       round(p["lat"], GRID), round(p["lon"], GRID))].add(c["ip"])
    hubs = {k for k, ips in coord_ips.items() if len(ips) >= MIN_IPS}
    per_src = defaultdict(int)
    for s, *_ in hubs:
        per_src[s] += 1
    print(f"Hub lexicon from split A (n={int(is_a.sum())}): {len(hubs)} hub coordinates "
          f"(>= {MIN_IPS} IPs, per source+country); per source: {dict(sorted(per_src.items()))}")

    # --- Evaluation on split B ---
    b = ~is_a
    hm = np.array([hub_mass_for(c, hubs) for c in cases])
    y = (e > TAU).astype(float)
    print(f"\nSplit B (n={int(b.sum())}): overall miss rate {y[b].mean():.3f}")
    rows = []
    for lo, hi in ((0.0, 1/3), (1/3, 2/3), (2/3, 1.0001)):
        m = b & (hm >= lo) & (hm < hi)
        if m.sum():
            print(f"  hub_mass [{lo:.2f},{hi:.2f}): n={int(m.sum()):4d} | "
                  f"miss rate {y[m].mean():.3f} | median {np.median(e[m]):7.1f} km")
            rows.append({"bucket": f"[{lo:.2f},{hi:.2f})", "n": int(m.sum()),
                         "missrate": round(float(y[m].mean()), 3),
                         "median_km": round(float(np.median(e[m])), 1)})

    b1, e1c = oof_bss(S[b].reshape(-1, 1), y[b])
    b2, e2c = oof_bss(np.column_stack([S[b], hm[b]]), y[b])
    print(f"\nOOF logit on B (10-fold): S alone BSS {b1:+.3f} (ECE {e1c:.3f}) | "
          f"S + hub_mass BSS {b2:+.3f} (ECE {e2c:.3f}) | delta {b2-b1:+.3f}")

    # Redundancy check: does the MaxMind radius carry the same information?
    rad = np.array([next((float(p.get("accuracy_radius") or np.nan)
                          for p in c["provenance"]
                          if p["source"] == "maxmind_geolite2"), np.nan)
                    for c in cases])
    ok = b & ~np.isnan(rad)
    def srho(a_, b_):
        ra = np.argsort(np.argsort(a_)).astype(float)
        rb = np.argsort(np.argsort(b_)).astype(float)
        ra -= ra.mean(); rb -= rb.mean()
        return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))
    print(f"Spearman(hub_mass, MaxMind radius) on B: {srho(hm[ok], rad[ok]):+.3f} "
          f"(n={int(ok.sum())}) — redundancy indicator")

    rows.append({"bucket": "oof_bss", "n": int(b.sum()),
                 "s_alone": round(b1, 3), "s_plus_hub": round(b2, 3),
                 "delta": round(b2 - b1, 3)})
    pd.DataFrame(rows).to_csv(OUT / f"hub_signal_{ds}.csv", index=False)
    print(f"\nCSV: {OUT / f'hub_signal_{ds}.csv'}")
    print("Framing: quick check of a RESEARCH IDEA (split design, one corpus) — "
          "not a calibrated operational signal.")


if __name__ == "__main__":
    main()
