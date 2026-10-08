#!/usr/bin/env python3
"""Negative control: does the registry channel also carry at CITY LEVEL? (No -- measured.)

Question (2026-09-27): the flag "registry country != country vote" stratifies
the country errors of the aggregation (exp_registry_country, pool RR ~ 11). Does
this also hold for the 100-km city error, i.e. the tail from T1/T7? Competing
hypothesis: city errors are misplacements within the right country (block
granularity, hub defaults, T9); the registry has no city information.

Measurement per population (core vote + registry levels as in exp_registry_country;
primarily whois/RDAP, Giga delegation):
  * share of the 100-km errors for which the registry country is wrong;
  * P(100-km error | registry!=vote) vs. P(... | registry==vote), risk ratio with
    iid bootstrap CI (B, SEED as in T1);
  * comparison: P(100-km error | core mass < 0.5) vs. (>= 0.5) -- the S gate;
  * miss rate in the "quiet" stratum (registry right AND vote == registry).

Usage:   GEOIP_DATASET=anchors python experiments/exp_registry_city.py
Output:  table (stdout) + eval/out/registry_city.csv (aggregated, IP-free)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "maltego"))

from data import store                                     # noqa: E402
from eval.pipeline import load_cases                       # noqa: E402
from eval.metrics import haversine_error                   # noqa: E402
import experiments.exp_t6_defaults as T6                   # noqa: E402
from experiments.exp_radius_transfer import eps_for        # noqa: E402
from experiments.exp_pool_transfer import restrict         # noqa: E402
import experiments.exp_country_level as CL                 # noqa: E402
from experiments.exp_registry_country import POP_DATE, load_rdap, state_cc  # noqa: E402
from registry import Delegations                           # noqa: E402

OUT = ROOT / "eval" / "out"
TAU_KM = 100.0


def rr_ci(flag, miss):
    flag = np.asarray(flag, bool); miss = np.asarray(miss, bool)
    n = len(flag)
    p1 = miss[flag].mean() if flag.sum() else float("nan")
    p0 = miss[~flag].mean() if (~flag).sum() else float("nan")
    rr = p1 / p0 if (p0 and np.isfinite(p1)) else float("nan")
    rng = np.random.default_rng(CL.SEED)
    vals = []
    for _ in range(CL.B):
        idx = rng.integers(0, n, n)
        f = flag[idx]; m = miss[idx]
        if f.sum() and (~f).sum() and m[~f].mean() > 0:
            vals.append(m[f].mean() / m[~f].mean())
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (float("nan"), float("nan")))
    return 100 * p1, 100 * p0, rr, lo, hi


def main() -> None:
    if os.environ.get("GEOIP_DATASET", "anchors") != "anchors":
        sys.exit("exp_registry_city: please run with GEOIP_DATASET=anchors.")
    CL._NORMALIZE = True
    anchors9 = load_cases()
    loo = T6.loo_pseudo_radii(anchors9)
    eps = eps_for(loo)
    rdap = load_rdap()
    print(f"Anchor calibration frozen: eps={eps} km; miss threshold {TAU_KM:.0f} km; B={CL.B}")
    print(f"\n{'population':18s} {'n':>5s} | city miss | of which reg country wrong | P(miss|reg!=vote) P(miss|reg==vote) RR [CI] | "
          f"P(miss|S<0.5) P(miss|S>=0.5) | quiet stratum n / miss")
    rows = []
    for label, ds, obs_name, srcs in CL.POPS:
        cases = anchors9 if ds == "anchors" else load_cases(
            store.load_anchors_csv(CL.CACHE / f"{ds}.csv"), store.load_observations_csv(CL.CACHE / obs_name))
        if srcs is not None:
            cases = restrict(cases, srcs)
        lut = CL.city_lut_from(CL.CACHE / obs_name)
        truth = np.array([CL._cc(c["country"]) for c in cases])
        D = Delegations(POP_DATE[label])
        err, vote, mass, reg_d, reg_w = [], [], [], [], []
        for c in cases:
            est = T6.estimate(c, "L1", "b", loo, eps=eps)
            err.append(haversine_error(est, c["truth"]))
            a, m, _, _ = CL.country_votes(c, lut, loo, eps)
            vote.append(a); mass.append(m)
            reg_d.append(state_cc(D.lookup(c["ip"])[0]))
            reg_w.append(state_cc(rdap[c["ip"]][0]) if c["ip"] in rdap else None)
        err = np.array(err); vote = np.array(vote); mass = np.array(mass)
        miss = err > TAU_KM
        n = len(cases)
        levels = [("registry_delegated", np.array(reg_d))]
        if all(r is not None for r in reg_w):
            levels.insert(0, ("registry_rdap", np.array(reg_w)))
        for est_name, reg in levels:
            reg_wrong = reg != truth
            flag = reg != vote
            p1, p0, rr, lo, hi = rr_ci(flag, miss)
            g1, g0 = 100 * miss[mass < CL.S_GATE].mean(), 100 * miss[mass >= CL.S_GATE].mean()
            quiet = (~flag) & (reg == truth)
            share_reg_wrong = 100 * (miss & reg_wrong).sum() / max(1, miss.sum())
            if est_name == levels[0][0]:
                print(f"{label:18s} {n:5d} | {int(miss.sum()):4d} ({100*miss.mean():4.1f} %) | "
                      f"{int((miss & reg_wrong).sum()):4d} = {share_reg_wrong:4.0f} % [{est_name[9:]}] | "
                      f"{p1:5.1f} % {p0:5.1f} % RR {rr:4.2f} [{lo:.2f}, {hi:.2f}] | {g1:5.1f} % {g0:5.1f} % | "
                      f"{int(quiet.sum())} / {100*miss[quiet].mean():4.1f} %")
            rows.append({"population": label, "estimator": est_name, "n": n, "tau_km": TAU_KM,
                         "city_miss_n": int(miss.sum()), "city_miss_pct": round(100 * miss.mean(), 2),
                         "miss_with_registry_country_wrong_n": int((miss & reg_wrong).sum()),
                         "miss_with_registry_country_wrong_pct_of_misses": round(share_reg_wrong, 1),
                         "flag_n": int(flag.sum()), "p_miss_flag_pct": round(p1, 2), "p_miss_noflag_pct": round(p0, 2),
                         "risk_ratio": round(rr, 3) if np.isfinite(rr) else None,
                         "rr_iid_lo": round(lo, 3) if np.isfinite(lo) else None,
                         "rr_iid_hi": round(hi, 3) if np.isfinite(hi) else None,
                         "p_miss_lowmass_pct": round(g1, 2), "p_miss_highmass_pct": round(g0, 2),
                         "gate_ratio": round(g1 / g0, 3) if g0 else None,
                         "quiet_n": int(quiet.sum()), "quiet_miss_pct": round(100 * miss[quiet].mean(), 2)})
    pd.DataFrame(rows).to_csv(OUT / "registry_city.csv", index=False)
    print(f"\nCSV: {OUT / 'registry_city.csv'}")


if __name__ == "__main__":
    main()
