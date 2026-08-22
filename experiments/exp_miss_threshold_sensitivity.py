"""Sensitivity of the S calibration skill to the MISS DEFINITION.

A natural objection: the flag-threshold sensitivity (the triage operating
point) is analysed, but the definition of the miss EVENT itself
(error > 100 km) is not. Here the event threshold tau is varied and, for each
threshold, the OOF skill of S is compared against the 2D predecessor label.

Protocol IDENTICAL to exp_support_concentration.py (imported from there):
same feature construction (build_frame), same 10-fold stratified folds
(seed=0), same OOF logit (oof_predict), same metrics. S deliberately stays
the pre-specified line-weighted core share r=50 km for ALL tau (anti-tuning:
r does not move with the evaluation threshold, cf. the scale-matching grid
there).

Consistency anchor: tau=100 must reproduce the headline numbers digit-exact
(S +0.210, predecessor label med+hub +0.068).

Usage:   python experiments/exp_miss_threshold_sensitivity.py
Result:  table (stdout) + eval/out/miss_threshold_sensitivity.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.pipeline import load_cases                       # noqa: E402
from eval import report                                    # noqa: E402
import experiments.exp_t6_defaults as T6                   # noqa: E402
import experiments.exp_support_concentration as SC         # noqa: E402

OUT = report.OUT_DIR
TAUS = [25, 50, 75, 100, 150, 250, 500]
WIN = f"core_w_{SC.R_HEADLINE}"          # S: line-weighted core share r=50 (fixed)


def run() -> None:
    cases = load_cases()
    loo = T6.loo_pseudo_radii(cases)
    eps = min(T6.EPS_GRID,
              key=lambda e: abs(e - float(np.median([loo[s]["_global"] for s in loo]))))
    df = SC.build_frame(cases, loo, eps)

    print("=" * 78)
    print(f"MISS-THRESHOLD SENSITIVITY  (S = {WIN} fixed, OOF {SC.K}-fold, seed={SC.SEED})")
    print(f"n={len(df)}")
    print("=" * 78)
    print(f"{'tau/km':>7s} {'misses':>7s} {'base rate':>10s} "
          f"{'BSS S':>8s} {'BSS label':>10s} {'diff':>7s}")

    rows = []
    for tau in TAUS:
        y = (df.err > tau).astype(int).values
        if y.sum() < 2 or y.sum() > len(y) - 2:
            print(f"{tau:7d}  (skipped: {int(y.sum())} misses)")
            continue
        b_s = SC.bss(SC.oof_predict(df, [WIN], y), y)
        b_l = SC.bss(SC.oof_predict(df, ["med", "hub"], y), y)
        print(f"{tau:7d} {int(y.sum()):7d} {y.mean():10.3f} "
              f"{b_s:+8.3f} {b_l:+10.3f} {b_s - b_l:+7.3f}")
        rows.append({"tau_km": tau, "misses": int(y.sum()),
                     "base_rate": round(float(y.mean()), 3),
                     "bss_s": round(float(b_s), 3),
                     "bss_label": round(float(b_l), 3),
                     "bss_diff": round(float(b_s - b_l), 3)})

    out = OUT / "miss_threshold_sensitivity.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nCSV: {out}")

    anchor = next((r for r in rows if r["tau_km"] == 100), None)
    if anchor:
        ok = anchor["bss_s"] == 0.210 and anchor["bss_label"] == 0.068
        print(f"Consistency anchor tau=100: S {anchor['bss_s']:+.3f} / "
              f"label {anchor['bss_label']:+.3f} -> "
              f"{'OK (headline numbers reproduced)' if ok else 'DEVIATION!'}")


if __name__ == "__main__":
    run()
