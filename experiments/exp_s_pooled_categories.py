"""S in operation: ONE model, ONE calibration, mixed population.

All transfer numbers so far are computed separately per target population and
recalibrated per population (Platt). That is right for the methods question,
but does NOT describe deployment: for an arbitrary IP one does not know in
advance whether a cellular router, a NAT home connection or a data center is
behind it. Calibrating per population presumes exactly that knowledge.

This script therefore computes the honest operational case:
  - model R7 (QW + in25/in50/in100 per source), trained ONLY on the anchors,
  - applied to a MIXED population of probes + probes_holdout +
    probes_mobile_ext (3 duplicates, deduplicated n=1074),
  - ONE single Platt layer, fitted out-of-fold on the mixture -- without any
    knowledge of the connection class.
Only afterwards is it broken down by class, to show WHERE the errors sit.

Two BSS per class, because the question is ambiguous:
  - ``bss_vs_pooled``: against the climatology of the MIXED population. This
    is the operational number -- the reference likewise does not know which
    class is present. Can turn negative if a class is much harder than the
    average.
  - ``bss_vs_own``: against the climatology of the class itself. Measures the
    pure discriminative power WITHIN the class, but presumes that the base
    rate of the class is known -- hence not available in operation.
Additionally ``bss_own_cal``: what a class-specific Platt layer would bring,
i.e. the value of knowing "which class is this?".

Classification: exp_s_declared_risk.classify (access-technology tags; the
provider tag ``t-mobile`` deliberately does NOT count as cellular).

Invocation: python experiments/exp_s_pooled_categories.py
Output: tables (stdout) + eval/out/s_pooled_categories.csv
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
from experiments.exp_support_concentration import folds, K, ece         # noqa: E402
from experiments.exp_s_declared_risk import (                           # noqa: E402
    QW, SRC, SRC_MULTI, TAU, OUT, _mat, _fit_ridge, _pick_lambda, _sig,
)

WIN = QW + SRC + SRC_MULTI                       # R7
POOL = ("probes", "probes_holdout", "probes_mobile_ext")


def _frame(ds):
    path = OUT / f"s_decl_features_{ds}.csv"
    if not path.exists() or "cat" not in pd.read_csv(path, nrows=0).columns:
        subprocess.run([sys.executable,
                        str(ROOT / "experiments" / "exp_s_declared_risk.py"), "--build"],
                       check=True, env={**os.environ, "GEOIP_DATASET": ds}, cwd=ROOT)
    return pd.read_csv(path)


def _case_ips(ds):
    """IPs in exactly the order in which build_frame generates the rows."""
    from data import store
    from eval.pipeline import load_cases
    cache = ROOT / "data" / "cache"
    obs = cache / ("observations.csv" if ds == "anchors" else f"observations_{ds}.csv")
    return [c["ip"] for c in load_cases(store.load_anchors_csv(cache / f"{ds}.csv"),
                                        store.load_observations_csv(obs))]


def _brier(ph, y):
    return float(np.mean((ph - y) ** 2))


def _platt_oof(z, y, seed=0):
    """One Platt layer, out-of-fold on the given population."""
    F = folds(y, seed)
    ph = np.empty(len(y))
    for fi in range(K):
        tr, te = F != fi, F == fi
        w = _fit_ridge(np.column_stack([np.ones(tr.sum()), z[tr]]), y[tr], 1e-6)
        ph[te] = _sig(np.column_stack([np.ones(te.sum()), z[te]]) @ w)
    return ph


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    A = _frame("anchors")
    yA = (A.err > TAU).astype(int).values

    # The subsets are NOT guaranteed disjoint: probes_mobile_ext was drawn
    # without excluding probes (--exclude-datasets came only later), three IPs
    # occur twice. Therefore deduplicate via the case order; the IPs stay in
    # memory only and end up in no artifact.
    parts, seen, n_dup = [], set(), 0
    for ds in POOL:
        T = _frame(ds).assign(origin=ds)
        ips = _case_ips(ds)
        if len(ips) != len(T):
            raise RuntimeError(f"{ds}: {len(ips)} cases, but {len(T)} frame rows")
        keep = [ip not in seen for ip in ips]
        seen.update(ips)
        n_dup += len(T) - sum(keep)
        parts.append(T[pd.Series(keep, index=T.index)].reset_index(drop=True))
    P = pd.concat(parts, ignore_index=True)
    if n_dup:
        print(f"(deduplication: {n_dup} duplicate IP(s) removed between the subsets)")
    y = (P.err > TAU).astype(int).values
    ref = float(np.mean((y.mean() - y) ** 2))

    print("=" * 84)
    print("MIXED POPULATION (as in operation: class unknown)")
    print("=" * 84)
    print(f"composed of: " + ", ".join(f"{d} (n={len(p)})" for d, p in zip(POOL, parts)))
    print(f"overall n={len(P)}, miss rate {y.mean():.3f}")
    print("\ncomposition by connection class:")
    for c, g in P.groupby("cat"):
        yc = (g.err > TAU).astype(int).values
        print(f"  {c:12} n={len(g):5}  miss rate {yc.mean():.3f}")

    # --- model: trained on anchors only, ONE Platt layer on the mixture
    Xa, Xp = _mat(A, WIN), _mat(P, WIN)
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-9
    lam = _pick_lambda(Xa, yA, 0)
    w = _fit_ridge(np.column_stack([np.ones(len(A)), (Xa - mu) / sd]), yA, lam)
    z = np.column_stack([np.ones(len(P)), (Xp - mu) / sd]) @ w
    ph_raw = _sig(z)
    ph = _platt_oof(z, y)

    b_raw = 1 - _brier(ph_raw, y) / ref
    b_all = 1 - _brier(ph, y) / ref
    print("\n" + "-" * 84)
    print(f"OVERALL BSS (one calibration, class unknown): raw {b_raw:+.3f}  "
          f"Platt {b_all:+.3f}  ECE {ece(ph, y):.3f}")
    print("-" * 84)
    rows += [{"section": "pooled", "cat": "ALL", "n": len(P), "base_rate": round(y.mean(), 3),
              "metric": "bss_raw", "value": round(b_raw, 3), "extra": ""},
             {"section": "pooled", "cat": "ALL", "n": len(P), "base_rate": round(y.mean(), 3),
              "metric": "bss_platt", "value": round(b_all, 3),
              "extra": f"ECE={ece(ph, y):.3f}"}]

    # --- breakdown by class
    print(f"\n{'class':12} {'n':>5} {'base':>6} {'Brier':>8} {'share':>8} | "
          f"{'BSS vs':>8} {'BSS vs':>8} | {'BSS w/':>8}")
    print(f"{'':12} {'':>5} {'rate':>6} {'':>8} {'of err.':>8} | "
          f"{'pooled':>8} {'own':>8} | {'own cal.':>8}")
    tot_brier = _brier(ph, y)
    for c, g in P.groupby("cat"):
        idx = g.index.values
        yc, phc = y[idx], ph[idx]
        bc = _brier(phc, yc)
        share = len(idx) * bc / (len(P) * tot_brier)
        refc = float(np.mean((yc.mean() - yc) ** 2))
        b_pool = 1 - bc / ref
        b_own = 1 - bc / refc if refc > 0 else float("nan")
        # what would a class-specific calibration bring?
        b_owncal = float("nan")
        if len(idx) >= 40 and 0 < yc.mean() < 1:
            b_owncal = 1 - _brier(_platt_oof(z[idx], yc), yc) / refc
        print(f"{c:12} {len(idx):5} {yc.mean():6.3f} {bc:8.4f} {share:8.1%} | "
              f"{b_pool:+8.3f} {b_own:+8.3f} | {b_owncal:+8.3f}")
        rows.append({"section": "by_category", "cat": c, "n": len(idx),
                     "base_rate": round(float(yc.mean()), 3), "metric": "bss_vs_pooled",
                     "value": round(float(b_pool), 3),
                     "extra": f"brier={bc:.4f} share_of_error={share:.3f} "
                              f"bss_vs_own={b_own:+.3f} bss_own_cal={b_owncal:+.3f}"})

    print("\nREADING:")
    print("  'BSS vs pooled' is the operational number -- reference and model both do")
    print("  not know which class is present. Negative values mean: in this class the")
    print("  model is worse than the POOLED base rate, not that it is useless.")
    print("  'BSS vs own' measures the discriminative power within the class, but")
    print("  presumes knowledge of its base rate. The difference to 'BSS w/ own")
    print("  calibration' quantifies what knowing the class would be worth.")

    pd.DataFrame(rows).to_csv(OUT / "s_pooled_categories.csv", index=False)
    print(f"\nCSV: {OUT}/s_pooled_categories.csv")


if __name__ == "__main__":
    main()
