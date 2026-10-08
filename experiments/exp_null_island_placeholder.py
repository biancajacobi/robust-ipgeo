"""Data-rule control: Null Island placeholders (0,0) in source answers (Section 3.1).

reallyfreegeoip returns the coordinate (0,0) ("Null Island", Gulf of Guinea) for
some anchors as a placeholder for "unknown". The pipeline keeps these answers as
observations; exp_source_correlation.py excludes them only for the pairwise
matrix. This script measures what the placeholders do to the headline
configuration L1*b and to the risk signal S: metrics with and without the (0,0)
observations (LOO radii recomputed each time), the shift of the estimates on the
affected anchors, and S per affected anchor.

Usage:  python experiments/exp_null_island_placeholder.py
Output: stdout + eval/out/null_island_placeholder.csv (IP-free)
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.pipeline import load_cases                   # noqa: E402
from eval.metrics import haversine_error               # noqa: E402
import experiments.exp_t6_defaults as T6               # noqa: E402
import experiments.exp_support_concentration as SC     # noqa: E402

OUT = Path("eval/out")


def _is_null(p) -> bool:
    return p["lat"] == 0.0 and p["lon"] == 0.0


def _run(cases, loo, eps):
    est = [T6.estimate(c, "L1", "b", loo, eps=eps) for c in cases]
    err = np.array([haversine_error(e, c["truth"]) for e, c in zip(est, cases)])
    df = SC.build_frame(cases, loo, eps)
    y = (df.err > SC.TAU).astype(int).values
    ph = SC.oof_predict(df, [f"core_w_{SC.R_HEADLINE}"], y)
    return est, err, SC.bss(ph, y), df


def main() -> None:
    cases = load_cases()
    loo = T6.loo_pseudo_radii(cases)
    eps = min(T6.EPS_GRID, key=lambda e: abs(e - float(np.median([loo[s]["_global"] for s in loo]))))
    nulls = [(c["ip"], p["source"]) for c in cases for p in c["provenance"] if _is_null(p)]
    affected = [i for i, c in enumerate(cases) if any(_is_null(p) for p in c["provenance"])]
    print(f"Cases {len(cases)}, (0,0) observations: {len(nulls)} from {sorted({s for _, s in nulls})}, affected anchors: {len(affected)}")
    cases2 = copy.deepcopy(cases)
    for c in cases2:
        c["provenance"] = [p for p in c["provenance"] if not _is_null(p)]
        c["points"] = [[p["lat"], p["lon"]] for p in c["provenance"]]
    loo2 = T6.loo_pseudo_radii(cases2)
    e0, r0, b0, d0 = _run(cases, loo, eps)
    e1, r1, b1, d1 = _run(cases2, loo2, eps)
    mv = np.array([haversine_error(a, b) for a, b in zip(e0, e1)])
    others = np.delete(mv, affected)
    rows = []
    for label, r, b in (("with_placeholders", r0, b0), ("without_placeholders", r1, b1)):
        rows.append({"config": label, "median_km": round(float(np.median(r)), 4), "mean_km": round(float(r.mean()), 3),
                     "tail_pct": round(100 * float((r > 100).mean()), 3), "s_bss_oof": round(float(b), 4)})
        print(f"  {label:22s} Median {rows[-1]['median_km']:.3f} | Mittel {rows[-1]['mean_km']:.2f} | Tail {rows[-1]['tail_pct']:.3f} % | S-BSS {b:+.4f}")
    rows.append({"config": "shift_affected_anchors", "n": len(affected), "max_km": round(float(mv[affected].max()), 3),
                 "median_km": round(float(np.median(mv[affected])), 3)})
    rows.append({"config": "shift_other_anchors_via_loo_radius", "n": int(len(others)), "max_km": round(float(others.max()), 3),
                 "n_gt_1km": int((others > 1).sum())})
    s0, s1 = d0[f"core_w_{SC.R_HEADLINE}"].values[affected], d1[f"core_w_{SC.R_HEADLINE}"].values[affected]
    rows.append({"config": "S_affected_anchors", "n": len(affected), "s_with_median": round(float(np.median(s0)), 3),
                 "s_without_median": round(float(np.median(s1)), 3), "n_s_lower_with": int((s0 < s1 - 1e-9).sum())})
    for src in sorted({s for _, s in nulls}):
        rows.append({"config": f"loo_radius_{src}", "with_km": round(loo[src]["_global"], 2), "without_km": round(loo2[src]["_global"], 2)})
    print(f"  estimate shift, affected anchors: max {mv[affected].max():.3f} km (median {np.median(mv[affected]):.3f}); "
          f"other anchors (via the LOO radius only): max {others.max():.2f} km, >1 km: {(others > 1).sum()}")
    print(f"  S auf betroffenen Anchors niedriger mit Platzhalter: {(s0 < s1 - 1e-9).sum()} von {len(affected)}")
    pd.DataFrame(rows).to_csv(OUT / "null_island_placeholder.csv", index=False)
    print(f"CSV: {OUT}/null_island_placeholder.csv")


if __name__ == "__main__":
    main()
