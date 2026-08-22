"""Targeted contamination: attacker hijacks lines in descending weight mass.

A natural objection: T2 contaminates RANDOMLY
chosen nominal sources; the breakdown statement, however, belongs to the
controlled effective WEIGHT MASS. A targeted attacker who takes over the
heaviest lines first reaches 50% mass with considerably fewer nominal sources.
This experiment measures exactly that -- on the headline configuration L1*b
(line x accuracy-radius weights, i.e. the weight vector the estimator actually
uses and which an informed attacker can derive from the published method):

  - "targeted": per anchor the lines are taken over in descending effective
    mass order (k = 1..n_lines); all observations of a taken-over line are
    shifted in a coordinated way by OFFSET_KM (identical shift model as T2:
    latitude-wise towards the equator).
  - "random":   baseline with k randomly chosen lines (B_RAND draws), same
    mechanics -- the direct comparison of nominal count vs. mass.

Measured per case: controlled mass shares, hijack (error > 1,000 km), support
concentration S at the contaminated estimator, and flag status at the
operating-point threshold t = 5/6. Expectation: the transition sits at ~50%
controlled MASS independently of the nominal line count; unflagged hijacking
additionally requires mass >= t (operating-point property, see T6).

Usage:   python experiments/exp_targeted_contamination.py
Output:  tables (stdout) + eval/out/targeted_contamination.csv
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.pipeline import load_cases                          # noqa: E402
from eval.metrics import haversine_error                      # noqa: E402
from estimators.baselines import weighted_geometric_median    # noqa: E402
import experiments.exp_t6_defaults as T6                      # noqa: E402

OUT = ROOT / "eval" / "out"
OFFSET_KM = 2000.0          # as in exp_contamination
HIJACK_KM = 1000.0
S_RADIUS = 50.0
T_FLAG = 5.0 / 6.0
B_RAND = 20
SEED = 20260604             # as in exp_contamination
rows: list[dict] = []


def case_setup(c, loo, eps):
    """Points, effective (unnormalised) weights, and line decomposition of a case."""
    prov = c["provenance"]
    pts = np.array([[p["lat"], p["lon"]] for p in prov], float)
    w_line = np.asarray(T6.line_weights_for(prov, "L1"), float)
    rad = np.array([T6.point_radius(p, c["ip"], loo) for p in prov], float)
    w_eff = w_line / (rad + eps)
    lines: dict[str, list[int]] = {}
    for i, p in enumerate(prov):
        lines.setdefault(p["lineage"] or "unknown", []).append(i)
    return pts, w_line, w_eff, lines


def attack(pts, w_line, w_eff, idxs, truth):
    """Shift the points at the given indices in a coordinated way -> (error, S, mass share)."""
    a = pts.copy()
    dlat = OFFSET_KM / 111.32
    for i in idxs:
        a[i, 0] += -dlat if a[i, 0] >= 0 else dlat
    est = weighted_geometric_median(a, w_eff)
    err = haversine_error((float(est[0]), float(est[1])), truth)
    wl = w_line / w_line.sum()
    d = np.array([haversine_error((x, y), (float(est[0]), float(est[1])))
                  for x, y in a])
    s = float(wl[d < S_RADIUS].sum())
    mass = float(w_eff[list(idxs)].sum() / w_eff.sum())
    line_mass = float(wl[list(idxs)].sum())
    return err, s, mass, line_mass


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    loo = T6.loo_pseudo_radii(cases)
    eps = min(T6.EPS_GRID, key=lambda e: abs(e - float(np.median(
        [loo[s]["_global"] for s in loo]))))
    rng = np.random.default_rng(SEED)

    recs = []
    for ci, c in enumerate(cases):
        pts, w_line, w_eff, lines = case_setup(c, loo, eps)
        names = list(lines)
        mass_of = {L: float(w_eff[lines[L]].sum() / w_eff.sum()) for L in names}
        order = sorted(names, key=lambda L: -mass_of[L])
        for k in range(1, len(names) + 1):
            # targeted: heaviest k lines
            idxs = [i for L in order[:k] for i in lines[L]]
            err, s, mass, lmass = attack(pts, w_line, w_eff, idxs, c["truth"])
            recs.append({"mode": "targeted", "case": ci, "k": k,
                         "n_lines": len(names),
                         "mass": mass, "line_mass": lmass,
                         "hijack": err > HIJACK_KM,
                         "unflagged": s >= T_FLAG, "err": err,
                         "top_line": order[0]})
            # random: k random lines, B_RAND draws
            for b in range(B_RAND):
                pick = rng.choice(len(names), size=k, replace=False)
                idxs = [i for j in pick for i in lines[names[j]]]
                err, s, mass, lmass = attack(pts, w_line, w_eff, idxs, c["truth"])
                recs.append({"mode": "random", "case": ci, "k": k,
                             "n_lines": len(names),
                             "mass": mass, "line_mass": lmass,
                             "hijack": err > HIJACK_KM,
                             "unflagged": s >= T_FLAG, "err": err,
                             "top_line": ""})
    df = pd.DataFrame(recs)

    print("=" * 88)
    print(f"TARGETED vs. RANDOM LINE CONTAMINATION (L1*b, n={len(cases)}, "
          f"shift {OFFSET_KM:.0f} km, hijack > {HIJACK_KM:.0f} km, B_rand={B_RAND})")
    print("=" * 88)
    t1 = df[(df["mode"] == "targeted") & (df["k"] == 1)]
    print(f"Heaviest line per case: {dict(Counter(t1.top_line).most_common(3))}; "
          f"median mass of the top line {t1.mass.median():.2f}")

    print(f"\n{'k':>2} | {'targeted: mass~':>16} {'hijack':>7} {'unflagged|hijack':>18} | "
          f"{'random: mass~':>14} {'hijack':>7}")
    for k in sorted(df.k.unique()):
        t = df[(df["mode"] == "targeted") & (df.k == k)]
        r = df[(df["mode"] == "random") & (df.k == k)]
        uh = t[t.hijack]
        uh_rate = float(uh.unflagged.mean()) if len(uh) else float("nan")
        print(f"{k:2d} | {t.mass.median():16.2f} {t.hijack.mean():7.1%} "
              f"{uh_rate:18.1%} | {r.mass.median():14.2f} {r.hijack.mean():7.1%}")
        rows.append({"k": k, "targeted_mass_med": round(float(t.mass.median()), 3),
                     "targeted_hijack": round(float(t.hijack.mean()), 4),
                     "targeted_unflagged_given_hijack": (round(uh_rate, 4)
                                                         if uh_rate == uh_rate else ""),
                     "random_mass_med": round(float(r.mass.median()), 3),
                     "random_hijack": round(float(r.hijack.mean()), 4),
                     "n_targeted": len(t), "n_random": len(r)})

    # The actual touchstone: hijack rate as a function of the controlled MASS
    print("\nHijack rate per controlled mass share (both modes pooled):")
    bins = [0, .2, .3, .4, .45, .5, .55, .6, .7, .8, 1.0001]
    df["mbin"] = pd.cut(df.mass, bins, right=False)
    for b, g in df.groupby("mbin", observed=True):
        if len(g) == 0:
            continue
        uh = g[g.hijack]
        uf = float(uh.unflagged.mean()) if len(uh) else float("nan")
        print(f"  Mass {str(b):>14}: hijack {g.hijack.mean():7.1%}  "
              f"(n={len(g):6d})  unflagged|hijack {uf:7.1%}" if uf == uf else
              f"  Mass {str(b):>14}: hijack {g.hijack.mean():7.1%}  (n={len(g):6d})")
        rows.append({"mass_bin": str(b), "hijack_rate": round(float(g.hijack.mean()), 4),
                     "unflagged_given_hijack": (round(uf, 4) if uf == uf else ""),
                     "n": len(g)})

    # Key figures for the paper text
    tg = df[df["mode"] == "targeted"]
    kmin_t = tg[tg.mass >= 0.5].groupby("case").k.min()
    n_med = tg.groupby("case").n_lines.first()
    print(f"\nA targeted attacker reaches >= 50 % mass with a median of "
          f"{kmin_t.median():.0f} out of a median of {n_med.median():.0f} lines "
          f"(Q1-Q3: {kmin_t.quantile(.25):.0f}-{kmin_t.quantile(.75):.0f}); "
          f"hijack rate below 50 % mass: "
          f"{tg[tg.mass < 0.5].hijack.mean():.1%}, from 50 % on: "
          f"{tg[tg.mass >= 0.5].hijack.mean():.1%}")
    rows.append({"summary": "kmin_mass50_targeted",
                 "median": float(kmin_t.median()),
                 "q1": float(kmin_t.quantile(.25)), "q3": float(kmin_t.quantile(.75)),
                 "hijack_below_50": round(float(tg[tg.mass < 0.5].hijack.mean()), 4),
                 "hijack_above_50": round(float(tg[tg.mass >= 0.5].hijack.mean()), 4)})

    # The flag threshold acts on LINE mass: minimum line mass of unflagged hijacks
    uh = df[df.hijack & df.unflagged]
    if len(uh):
        print(f"Unflagged hijacks: n={len(uh)}, minimum attacker LINE mass "
              f"{uh.line_mass.min():.3f} (t = {T_FLAG:.3f}); "
              f"minimum effective mass {uh.mass.min():.3f}")
        rows.append({"summary": "unflagged_hijack_min_line_mass",
                     "min_line_mass": round(float(uh.line_mass.min()), 4),
                     "min_eff_mass": round(float(uh.mass.min()), 4), "n": len(uh)})
    else:
        print("No unflagged hijacks.")

    pd.DataFrame(rows).to_csv(OUT / "targeted_contamination.csv", index=False)
    print("\nREADING:")
    print("  The breakdown belongs to the controlled WEIGHT MASS, not to the")
    print("  nominal source count: a targeted attacker reaches ~50 % mass with")
    print("  a few heavy lines (small k), a random one only with ~half of the")
    print("  lines. Unflagged hijacking additionally requires mass >= t = 5/6")
    print("  -- the operating-point property from T6, confirmed adversarially here.")
    print(f"\nCSV: {OUT}/targeted_contamination.csv")


if __name__ == "__main__":
    main()
