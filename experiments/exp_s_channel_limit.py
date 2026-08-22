"""Where does consensus geometry end? Information limit of the S feature channel.

Motivation: after round 3 (exp_s_declared_risk.py), S stands at OOF-BSS +0.625
on the anchors, while the oracle bound from the ceiling analysis is at +0.94.
One might conclude that much is still to be gained. This script shows that
this is a fallacy -- by measurement, not by argument.

Three computations:

(1) WHAT A BSS TARGET DEMANDS. Conversion BSS -> Brier score at the given
    base rate. Makes it tangible that e.g. BSS 0.95 demands a mean distance
    |p - y| of about 0.07: the model would have to answer almost every case
    with 0.02 or 0.98 and almost never be wrong.

(2) THE ORACLE BOUND IS NOT THE MODEL LIMIT. The ceiling analysis from
    round 2 asks: "which misses are invisible to EVERY consensus measure?"
    (entire line mass < 100 km around the estimator, error nonetheless
    > 100 km). It thus answers a question about visibility, NOT about the
    discriminative power of the features actually available.

(3) THE TRUE LIMIT OF THE CHANNEL. All S features are functions of ONE
    quantity: which source lies within which radius around the estimator.
    The script demonstrates this (the QW profile has variance 0 within an
    in-core pattern) and then determines the best possible prediction on
    exactly this channel as a lookup table over the patterns -- once
    in-sample (optimistic upper bound) and once cross-validated (what of it
    generalizes).

Core finding on anchors: pattern oracle in-sample +0.773, cross-validated
+0.449; the model from round 3 lies in between at +0.625. The channel is
thus largely exhausted -- the gap to +0.94 CANNOT be closed by better
features from the source comparison, only by evidence from outside
(latency measurement/active probing, ASN type, rDNS/whois). This also
explains why two of three levers in round 3 yielded nothing.

Invocation: python experiments/exp_s_channel_limit.py
Output: tables (stdout) + eval/out/s_channel_limit.csv
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.exp_support_concentration import folds, K            # noqa: E402
from experiments.exp_s_declared_risk import (                         # noqa: E402
    DATASETS, QW, SRC, SRC_MULTI, TAU, OUT, _oof,
)

WIN = QW + SRC + SRC_MULTI          # R7 = winner of round 3
BSS_TARGETS = (0.60, 0.625, 0.773, 0.90, 0.939, 0.95, 0.99)


def _frame(ds):
    path = OUT / f"s_decl_features_{ds}.csv"
    if not path.exists():
        subprocess.run([sys.executable,
                        str(ROOT / "experiments" / "exp_s_declared_risk.py"), "--build"],
                       check=True, env={**os.environ, "GEOIP_DATASET": ds}, cwd=ROOT)
    return pd.read_csv(path)


def _bss(ph, y, ref):
    return 1.0 - float(np.mean((ph - y) ** 2)) / ref


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    # ------------------------------------------------------------------ (1)
    A = _frame("anchors")
    yA = (A.err > TAU).astype(int).values
    refA = float(np.mean((yA.mean() - yA) ** 2))
    print("=" * 78)
    print(f"(1) WHAT A BSS TARGET DEMANDS  (anchors, n={len(yA)}, "
          f"base rate {yA.mean():.4f}, Brier of climatology {refA:.5f})")
    print("=" * 78)
    print(f"{'BSS':>8}  {'Brier':>9}  mean distance |p-y|")
    for t in BSS_TARGETS:
        br = refA * (1 - t)
        print(f"{t:8.3f}  {br:9.5f}  {np.sqrt(br):.3f}")
        rows.append({"section": "bss_to_brier", "dataset": "anchors", "metric": f"bss={t}",
                     "value": round(br, 5), "extra": f"|p-y|~{np.sqrt(br):.3f}"})

    # ------------------------------------------------------------------ (3a)
    print("\n" + "=" * 78)
    print("(3a) SUFFICIENCY: is the QW profile a function of the in-core pattern?")
    print("=" * 78)
    pat_cols = SRC + SRC_MULTI
    pat = A[pat_cols].astype(str).agg("|".join, axis=1)
    multi = pat.value_counts()
    multi = multi[multi > 1].index
    sel = pat.isin(multi)
    sd = A[sel].groupby(pat[sel])[QW].std().fillna(0.0)
    print(f"{'feature':14} {'std WITHIN pattern':>26} {'overall std':>16}")
    for c in QW:
        print(f"{c:14} {sd[c].mean():26.4f} {A[c].std():16.4f}")
        rows.append({"section": "sufficiency", "dataset": "anchors", "metric": c,
                     "value": round(float(sd[c].mean()), 4),
                     "extra": f"overall={A[c].std():.4f}"})
    print("-> std ~0: QW_r is the weighted sum of exactly these indicators.")
    print("   The S feature family is ONE channel, not a bundle of independent channels.")

    # ------------------------------------------------------------------ (2)+(3b)
    print("\n" + "=" * 78)
    print("(2)+(3b) BOUNDS PER POPULATION")
    print("=" * 78)
    print(f"{'dataset':20} {'n':>5} {'base':>6} | {'oracle':>7} {'pattern':>7} {'pattern':>7} | "
          f"{'model':>7} | {'pattern/n':>9}")
    print(f"{'':20} {'':>5} {'':>6} | {'(silent)':>7} {'in-smp':>7} {'CV':>7} | "
          f"{'R7 s0':>7} | {'':>9}")
    for ds in DATASETS:
        T = _frame(ds)
        y = (T.err > TAU).astype(int).values
        if y.sum() == 0 or y.sum() == len(y):
            continue
        ref = float(np.mean((y.mean() - y) ** 2))

        # (2) oracle: perfect except on the "silent" cases
        silent = T["core_w_100"].values >= 0.999
        ph_o = np.where(silent, y[silent].mean() if silent.any() else 0.0, y.astype(float))
        b_or = _bss(ph_o, y, ref)

        # (3b) pattern lookup table, in-sample and cross-validated
        p = T[pat_cols].astype(str).agg("|".join, axis=1).values
        u, inv = np.unique(p, return_inverse=True)
        rate = np.array([y[inv == i].mean() for i in range(len(u))])
        b_in = _bss(rate[inv], y, ref)
        F = folds(y, 0)
        ph_cv = np.empty(len(y), float)
        for fi in range(K):
            tr, te = F != fi, F == fi
            lut = {i: y[tr][inv[tr] == i].mean() for i in np.unique(inv[tr])}
            ph_cv[te] = [lut.get(i, y[tr].mean()) for i in inv[te]]
        b_cv = _bss(ph_cv, y, ref)

        # model R7, within-population (on anchors = the reported OOF number)
        ph_m, _ = _oof(T, WIN, y, 0, False, True)
        b_m = _bss(ph_m, y, ref)

        sat = len(u) / len(y)
        flag = "  << degenerate" if sat > 0.5 else ""
        print(f"{ds:20} {len(y):5} {y.mean():6.3f} | {b_or:+7.3f} {b_in:+7.3f} {b_cv:+7.3f} | "
              f"{b_m:+7.3f} | {len(u):4}/{len(y):<4}{flag}")
        for name, val in (("oracle_silent", b_or), ("pattern_insample", b_in),
                          ("pattern_cv", b_cv), ("model_r7_internal_seed0", b_m)):
            rows.append({"section": "limits", "dataset": ds, "metric": name,
                         "value": round(float(val), 3),
                         "extra": f"n={len(y)} base={y.mean():.3f} patterns={len(u)} "
                                  f"patterns_per_case={sat:.2f}"
                                  + (" DEGENERATE" if sat > 0.5 else "")})

    print("\nREADING: 'oracle (silent)' only answers which misses would in principle be")
    print("VISIBLE to a consensus measure -- it is not a bound for the available")
    print("features. The true channel limit is 'pattern in-sample' (best possible")
    print("prediction on exactly this channel, optimistic) resp. 'pattern CV' (what of")
    print("it generalizes). The model lies in between: it pools across patterns and thus")
    print("extracts more than the raw pattern statistic generalizes. Further gains")
    print("require evidence from OUTSIDE the source comparison (latency measurement,")
    print("ASN type, rDNS/whois).")
    print("\nCAVEAT: 'pattern in-sample' is only meaningful as long as there are clearly")
    print("more cases than patterns. From pattern/n > 0.5 (rightmost column) almost every")
    print("case sits in its own group, the lookup table memorizes the labels and the")
    print("number is overfitting -- for probes_mobile (51 patterns on 60 cases) it is")
    print("meaningless. The computation is reliable on the anchors.")

    pd.DataFrame(rows).to_csv(OUT / "s_channel_limit.csv", index=False)
    print(f"\nCSV: {OUT}/s_channel_limit.csv")


if __name__ == "__main__":
    main()
