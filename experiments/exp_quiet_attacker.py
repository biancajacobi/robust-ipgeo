"""The quiet attacker: small coordinated shifts at fixed line mass.

A natural objection: T2 and the targeted attack test
large shifts (2,000 km, hijack > 1,000 km). At least as relevant forensically
is an attacker who, with 50-83% line mass and a SMALL shift (100-300 km, or
just outside the 50-km core), produces a plausible-but-wrong city while
keeping S high. The operating-point window [1/2; 5/6) is parametrically
labeled "fails loudly"; this script measures the error and flag curves over
the shift Delta and empirically delimits "loud" from "quiet".

Design (deterministic, same mechanics as exp_targeted_contamination):
  - Regime "majority": per anchor, take over the heaviest lines (descending
    effective mass) until the LINE mass >= 1/2 (smallest possible majority;
    within the window in which the estimator is supposed to fail loudly).
  - Regime "stealth":  ditto until line mass >= 5/6 = t (above the flag
    threshold; the only regime in which unflagged hijacking is possible).
  - Delta sweep: all observations of taken-over lines shifted in a
    coordinated way by Delta km toward the equator (T2 shift model),
    Delta in {25, 50, 75, 100, 150, 200, 300, 500, 1000, 2000}.

Measured per (regime, Delta) over all anchors: median error of the
contaminated estimator, miss rate (> 100 km), flag rate (S < t) and the
forensically critical QUIET-miss rate (miss AND unflagged).

Invocation: python experiments/exp_quiet_attacker.py
Output: table (stdout) + eval/out/quiet_attacker.csv
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
from estimators.baselines import weighted_geometric_median    # noqa: E402
import experiments.exp_t6_defaults as T6                      # noqa: E402
from experiments.exp_targeted_contamination import case_setup  # noqa: E402

OUT = ROOT / "eval" / "out"
DELTAS = (25, 50, 75, 100, 150, 200, 300, 500, 1000, 2000)
MISS_KM = 100.0
S_RADIUS = 50.0
T_FLAG = 5.0 / 6.0
REGIMES = (("majority", 0.5), ("stealth", T_FLAG))


def attacked(pts, w_line, w_eff, idxs, truth, delta_km):
    """Shift the given indices by delta_km toward the equator -> (error, S)."""
    a = pts.copy()
    dlat = delta_km / 111.32
    for i in idxs:
        a[i, 0] += -dlat if a[i, 0] >= 0 else dlat
    est = weighted_geometric_median(a, w_eff)
    err = haversine_error((float(est[0]), float(est[1])), truth)
    wl = w_line / w_line.sum()
    d = np.array([haversine_error((x, y), (float(est[0]), float(est[1])))
                  for x, y in a])
    return err, float(wl[d < S_RADIUS].sum())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    loo = T6.loo_pseudo_radii(cases)
    eps = min(T6.EPS_GRID, key=lambda e: abs(e - float(np.median(
        [loo[s]["_global"] for s in loo]))))

    # per case and regime: smallest prefix set of the heaviest lines
    # with line mass >= threshold
    prepared = []
    for c in cases:
        pts, w_line, w_eff, lines = case_setup(c, loo, eps)
        wl = w_line / w_line.sum()
        names = sorted(lines, key=lambda L: -float(w_eff[lines[L]].sum()))
        sel = {}
        for reg, thr in REGIMES:
            idxs, lmass = [], 0.0
            for L in names:
                idxs += lines[L]
                lmass = float(wl[idxs].sum())
                if lmass >= thr:
                    break
            sel[reg] = (idxs, lmass)
        prepared.append((pts, w_line, w_eff, c["truth"], sel))

    rows = []
    print("=" * 86)
    print(f"QUIET ATTACKER: delta sweep at fixed line mass "
          f"(L1*b, n={len(cases)}, t={T_FLAG:.3f}, miss > {MISS_KM:.0f} km)")
    print("=" * 86)
    for reg, thr in REGIMES:
        masses = [sel[reg][1] for *_, sel in prepared]
        print(f"\nRegime {reg!r} (line mass >= {thr:.3f}; "
              f"median controlled mass {np.median(masses):.2f})")
        print(f"{'Delta':>6} | {'med error':>10} {'miss rate':>9} "
              f"{'flag rate':>9} {'QUIET miss':>10}")
        for delta in DELTAS:
            errs, flags, misses = [], [], []
            for pts, w_line, w_eff, truth, sel in prepared:
                idxs, _ = sel[reg]
                err, s = attacked(pts, w_line, w_eff, idxs, truth, delta)
                errs.append(err)
                misses.append(err > MISS_KM)
                flags.append(s < T_FLAG)
            errs = np.asarray(errs)
            misses = np.asarray(misses)
            flags = np.asarray(flags)
            quiet = float(np.mean(misses & ~flags))
            print(f"{delta:6d} | {np.median(errs):10.1f} "
                  f"{misses.mean():9.1%} {flags.mean():9.1%} {quiet:10.1%}")
            rows.append({"regime": reg, "delta_km": delta,
                         "median_err_km": round(float(np.median(errs)), 1),
                         "miss_rate": round(float(misses.mean()), 4),
                         "flag_rate": round(float(flags.mean()), 4),
                         "quiet_miss_rate": round(quiet, 4),
                         "median_controlled_mass": round(float(np.median(masses)), 3)})

    pd.DataFrame(rows).to_csv(OUT / "quiet_attacker.csv", index=False)
    print(f"\n  CSV: {OUT}/quiet_attacker.csv")


if __name__ == "__main__":
    main()
