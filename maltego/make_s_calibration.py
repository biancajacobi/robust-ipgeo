"""Regenerate s_calibration.json: frozen two-branch calibration P(miss | S).

Freezes, for live operation of the Maltego transforms, a Platt calibration
S -> P(aggregation error > 100 km), separated by the two branches of the
operating design (exp_s_type_routing):

  default  ip-api reports NO mobile flag (or flag not available),
  mobile   ip-api reports mobile=true.

The data basis is deliberately the MIXED probe population (probes +
probes_holdout + probes_mobile_ext, deduplicated, n~1074) — not the anchors:
arbitrary field IPs resemble probes (home/NAT/mobile connections) more than
data-center anchors. The feature is ONLY the live-computable S (core_w_50,
line-weighted core share r=50 km) — the R7 features need the per-source
anchor track record and add no live value without recalibration
infrastructure.

Honesty note, which also goes into _meta: In the mobile branch S is nearly
uninformative (measured channel limit; roughly a third of the mislocations is
invisible to any consensus measure). The value of the branch is the correct
BASE RATE (~0.4 instead of ~0.1) — the curve there is expectedly flat. That
is exactly what the transform should display, instead of imputing the
fixed-line curve to mobile cases.

CAUTION: needs the real probe data (not part of this release) plus the
cached ip-api flags (data/cache/ipapi_mobile_flags.json). The bundled
s_calibration.json was generated from the original data.

Usage:  python maltego/make_s_calibration.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.exp_s_pooled_categories import (   # noqa: E402
    POOL, _frame, _case_ips,
)

OUT = Path(__file__).resolve().parent / "s_calibration.json"
FLAGS_PATH = ROOT / "data" / "cache" / "ipapi_mobile_flags.json"
TAU_KM = 100.0
FEATURE = "core_w_50"      # = S of the transforms (geoloc.support_concentration)


def fit_logistic(s: np.ndarray, y: np.ndarray, it: int = 200,
                 ridge: float = 1e-6) -> tuple[float, float]:
    """p = sigmoid(a + b*S) via IRLS (like experiments._fit, only 1 raw feature)."""
    X = np.column_stack([np.ones(len(s)), s])
    w = np.zeros(2)
    for _ in range(it):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        W = p * (1 - p) + 1e-9
        w -= np.linalg.solve((X * W[:, None]).T @ X + ridge * np.eye(2),
                             X.T @ (p - y) + ridge * w)
    return float(w[0]), float(w[1])


def bss(ph: np.ndarray, y: np.ndarray) -> float:
    b = y.mean()
    return float(1 - np.mean((ph - y) ** 2) / (b * (1 - b)))


def oof_bss(s: np.ndarray, y: np.ndarray, k: int = 10, seed: int = 0) -> float:
    """10-fold stratified, only for reporting honesty (the full fit is frozen)."""
    rng = np.random.default_rng(seed)
    f = np.empty(len(y), int)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        f[idx] = np.arange(len(idx)) % k
    ph = np.empty(len(y))
    for fi in range(k):
        tr, te = f != fi, f == fi
        a, b = fit_logistic(s[tr], y[tr])
        ph[te] = 1.0 / (1.0 + np.exp(-(a + b * s[te])))
    return bss(ph, y)


def main() -> None:
    # Pool as in exp_s_pooled_categories/_type_routing (deduplicated over IPs)
    parts, ip_parts, seen = [], [], set()
    for ds in POOL:
        T = _frame(ds)
        ips = _case_ips(ds)
        if len(ips) != len(T):
            raise RuntimeError(f"{ds}: {len(ips)} cases, but {len(T)} frame rows")
        keep = [ip not in seen for ip in ips]
        seen.update(ips)
        parts.append(T[pd.Series(keep, index=T.index)].reset_index(drop=True))
        ip_parts += [ip for ip, k in zip(ips, keep) if k]
    P = pd.concat(parts, ignore_index=True)
    y = (P.err > TAU_KM).astype(int).values
    s = P[FEATURE].values.astype(float)

    flags = json.loads(FLAGS_PATH.read_text())
    flag = np.array([bool(flags.get(ip, False)) for ip in ip_parts])
    n_missing = sum(ip not in flags for ip in ip_parts)
    if n_missing:
        print(f"WARNING: {n_missing} pool IP(s) without a cached flag -> default branch")

    branches = {}
    print(f"Two-branch calibration P(error > {TAU_KM:.0f} km | S={FEATURE}), pool n={len(P)}")
    for name, mask in (("default", ~flag), ("mobile", flag)):
        sb, yb = s[mask], y[mask]
        a, b = fit_logistic(sb, yb)
        ph = 1.0 / (1.0 + np.exp(-(a + b * sb)))
        branches[name] = {
            "a": round(a, 6), "b": round(b, 6),
            "n": int(mask.sum()), "base_rate": round(float(yb.mean()), 4),
            "bss_insample": round(bss(ph, yb), 3),
            "bss_oof10": round(oof_bss(sb, yb), 3),
        }
        print(f"  {name:8s} n={mask.sum():4d}  base rate={yb.mean():.3f}  "
              f"a={a:+.3f} b={b:+.3f}  BSS in-sample {bss(ph, yb):+.3f} / OOF {branches[name]['bss_oof10']:+.3f}")

    payload = {
        "_meta": {
            "description": (
                "Frozen two-branch Platt calibration S -> P(aggregation "
                "error > 100 km) for live operation of the transforms. Branch "
                "'mobile' applies when ip-api reports mobile=true; otherwise (also "
                "when the flag is not retrievable) 'default'. p = sigmoid(a + b*S)."
            ),
            "note_mobile": (
                "In the mobile branch S is nearly uninformative (measured "
                "visibility limit of the consensus geometry); the branch mainly "
                "delivers the correct base rate. Curve there expectedly flat."
            ),
            "note_population": (
                "Fitted on the UNFILTERED pool (incl. cases with < 3 "
                "effective lines); the transforms apply the curve only at "
                ">= 3 lines -- P(miss | S) is thereby slightly conservatively "
                "shifted for the served subpopulation "
                "(p at high S tends to be overestimated)."
            ),
            "generated_with": "maltego/make_s_calibration.py",
            "data_state": (f"mixed probe pool {'+'.join(POOL)}, n={len(P)}, "
                           f"tau={TAU_KM:.0f} km, as of "
                           f"{datetime.now(timezone.utc).date().isoformat()}"),
            "feature": f"{FEATURE} (= S of the transforms, r=50 km, line-weighted)",
        },
        "tau_km": TAU_KM,
        "branches": branches,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{OUT} written.")


if __name__ == "__main__":
    main()
