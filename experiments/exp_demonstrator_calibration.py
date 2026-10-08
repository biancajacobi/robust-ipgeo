#!/usr/bin/env python3
"""Skill of the shipped demonstrator calibration: two-branch Platt layer, S only.

Question (chapter-7 pass, 2026-10-08): for exactly the configuration the transforms
ship -- S as the only feature, two Platt branches routed by the ip-api mobile flag,
fitted on the mixed probe population (maltego/make_s_calibration.py) -- the paper
reports no out-of-fold BSS. Section 4.7 measures the R7 profile with 31 features
(+0.263 / +0.287 routed), 4.8 S on the pool, 4.6 S on the stress test; none of
these is the shipped layer.

Measurement (same pool and same fitting routine as make_s_calibration.py):
  (a) two branches, S only: 10-fold stratified per branch (seed 0), predictions
      pooled, BSS against the pooled base rate, ECE (10 bins);
  (b) one branch, S only, same folds -- reference without type knowledge;
  (c) per branch: BSS within the branch against its own base rate
      (= bss_oof10 in s_calibration.json);
  (d) frozen table from s_calibration.json applied in-sample
      (link to the shipped file);
  side finding: share of cases with >= 3 effective lines (withholding rule).

Usage:   python experiments/exp_demonstrator_calibration.py
Output:  eval/out/demonstrator_calibration.csv (IP-free)
Needs the probe datasets (data/cache/) and the cached ip-api flags.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "maltego"))
from experiments.exp_s_pooled_categories import POOL, _frame, _case_ips   # noqa: E402
from make_s_calibration import fit_logistic, bss, TAU_KM, FEATURE, FLAGS_PATH  # noqa: E402

OUT = ROOT / "eval" / "out" / "demonstrator_calibration.csv"
TABLE = ROOT / "maltego" / "s_calibration.json"
K, SEED = 10, 0


def ece(ph: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    e = 0.0
    for i in range(bins):
        m = (ph >= i / bins) & ((ph < (i + 1) / bins) if i < bins - 1 else np.ones_like(ph, bool))
        if m.any():
            e += m.mean() * abs(ph[m].mean() - y[m].mean())
    return float(e)


def strat_folds(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    f = np.empty(len(y), int)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        f[idx] = np.arange(len(idx)) % K
    return f


def oof_predict(s: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    f = strat_folds(y, rng)
    ph = np.empty(len(y))
    for fi in range(K):
        a, b = fit_logistic(s[f != fi], y[f != fi])
        ph[f == fi] = 1.0 / (1.0 + np.exp(-(a + b * s[f == fi])))
    return ph


def load_pool():
    parts, ip_parts, seen = [], [], set()
    for ds in POOL:
        T = _frame(ds)
        ips = _case_ips(ds)
        keep = [ip not in seen for ip in ips]
        seen.update(ips)
        parts.append(T[pd.Series(keep, index=T.index)].reset_index(drop=True))
        ip_parts += [ip for ip, k in zip(ips, keep) if k]
    P = pd.concat(parts, ignore_index=True)
    flags = json.loads(FLAGS_PATH.read_text())
    flag = np.array([bool(flags.get(ip, False)) for ip in ip_parts])
    return P, flag


def main() -> None:
    P, flag = load_pool()
    y = (P.err > TAU_KM).astype(int).values
    s = P[FEATURE].values.astype(float)
    n = len(y)
    rows = []
    print(f"Demonstrator calibration: S -> P(error > {TAU_KM:.0f} km), mixed probe population n={n}, "
          f"base rate {y.mean():.3f}, cases with >= 3 lines: {int((P.nlines >= 3).sum())}")

    # (a) two branches, pooled
    rng = np.random.default_rng(SEED)
    ph2 = np.empty(n)
    for name, mask in (("default", ~flag), ("mobile", flag)):
        ph2[mask] = oof_predict(s[mask], y[mask], rng)
        rows.append({"variant": "two_branch_S_only", "branch": name, "n": int(mask.sum()),
                     "base_rate": round(float(y[mask].mean()), 4),
                     "bss_oof": round(bss(ph2[mask], y[mask]), 3), "ece_oof": round(ece(ph2[mask], y[mask]), 3),
                     "note": "BSS within the branch against its own base rate"})
    rows.append({"variant": "two_branch_S_only", "branch": "pooled", "n": n, "base_rate": round(float(y.mean()), 4),
                 "bss_oof": round(bss(ph2, y), 3), "ece_oof": round(ece(ph2, y), 3),
                 "note": "shipped transform configuration; 10-fold stratified per branch, seed 0"})
    # (b) one branch
    ph1 = oof_predict(s, y, np.random.default_rng(SEED))
    rows.append({"variant": "one_branch_S_only", "branch": "pooled", "n": n, "base_rate": round(float(y.mean()), 4),
                 "bss_oof": round(bss(ph1, y), 3), "ece_oof": round(ece(ph1, y), 3),
                 "note": "reference without type knowledge"})
    # (d) frozen table in-sample
    tab = json.loads(TABLE.read_text(encoding="utf-8"))["branches"]
    phf = np.empty(n)
    for name, mask in (("default", ~flag), ("mobile", flag)):
        a, b = tab[name]["a"], tab[name]["b"]
        phf[mask] = 1.0 / (1.0 + np.exp(-(a + b * s[mask])))
    rows.append({"variant": "frozen_table_insample", "branch": "pooled", "n": n, "base_rate": round(float(y.mean()), 4),
                 "bss_oof": round(bss(phf, y), 3), "ece_oof": round(ece(phf, y), 3),
                 "note": "s_calibration.json applied unchanged (in-sample, link only)"})
    rows.append({"variant": "withholding_rule", "branch": "pooled", "n": int((P.nlines >= 3).sum()),
                 "base_rate": round(float(y[P.nlines.values >= 3].mean()), 4), "bss_oof": None, "ece_oof": None,
                 "note": "cases with >= 3 effective lines (S is reported only there)"})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(OUT, index=False)
    print(f"\nCSV: {OUT}")


if __name__ == "__main__":
    main()
