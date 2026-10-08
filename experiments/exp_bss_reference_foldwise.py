"""Sensitivity of the BSS reference: whole-corpus base rate vs. per-training-fold (Section 3.4).

The Brier Skill Score is computed against the constant base-rate forecast. The
headline uses the miss rate of the whole anchor corpus as base rate (retrospective
climatology). For the strict out-of-fold reading, a reference per test fold is
formed here from the base rate of the corresponding training fold:
Brier_ref = sum over folds, sum over test cases of (p_train - y_j)^2. Same folds
(K=10, stratified, seed 0) and the same OOF logistic on S as in
exp_support_concentration.

Usage:  python experiments/exp_bss_reference_foldwise.py
Output: stdout + eval/out/bss_reference_foldwise.csv (IP-free)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.pipeline import load_cases                   # noqa: E402
import experiments.exp_t6_defaults as T6               # noqa: E402
import experiments.exp_support_concentration as SC     # noqa: E402

OUT = Path("eval/out")


def main() -> None:
    cases = load_cases()
    loo = T6.loo_pseudo_radii(cases)
    eps = min(T6.EPS_GRID, key=lambda e: abs(e - float(np.median([loo[s]["_global"] for s in loo]))))
    df = SC.build_frame(cases, loo, eps)
    y = (df.err > SC.TAU).astype(int).values
    F = SC.folds(y)
    rows = []
    for label, cols in (("S", [f"core_w_{SC.R_HEADLINE}"]), ("predecessor_label", ["med", "hub"])):
        ph = SC.oof_predict(df, cols, y)
        num = float(((ph - y) ** 2).sum())
        den_global = float(len(y) * y.mean() * (1 - y.mean()))
        den_fold = 0.0
        for fi in range(SC.K):
            tr, te = F != fi, F == fi
            den_fold += float(((y[tr].mean() - y[te]) ** 2).sum())
        rows.append({"measure": label, "bss_global_base_rate": round(1 - num / den_global, 4),
                     "bss_foldwise_train_base_rate": round(1 - num / den_fold, 4),
                     "base_rate_global": round(float(y.mean()), 4)})
        print(f"  {label:18s} BSS global {rows[-1]['bss_global_base_rate']:+.4f} | fold-weise {rows[-1]['bss_foldwise_train_base_rate']:+.4f}")
    pd.DataFrame(rows).to_csv(OUT / "bss_reference_foldwise.csv", index=False)
    print(f"CSV: {OUT}/bss_reference_foldwise.csv")


if __name__ == "__main__":
    main()
