"""Sensitivität der Anker-Hub-Achse: MEHRHEIT (>=0.5) vs. MINDESTENS EINE (>0).

Belegt die vorab getroffene Festlegung auf die
Mehrheitsregel. Lädt die Fälle einmal, berechnet Streuung + Schätzfehler (L1+b) einmal
und wertet dann beide Regeln zur Aggregation des per-Beobachtung-Flags
(is_default_centroid) zum Anker-Hub-Flag nebeneinander aus: 2D-Matrix-Zellen +
Recall/Precision/Flag-Rate. Kein Eingriff in die Headline-Experimente; reproduzierbares
Anhang-Artefakt. Aufruf: python experiments/exp_t6_hub_rule.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.pipeline import load_cases          # noqa: E402
from eval.metrics import haversine_error       # noqa: E402
from exp_t6_defaults import (                   # noqa: E402
    loo_pseudo_radii, estimate, EPS_GRID, _bootstrap_tail_ci,
)


def hub_rules(prov):
    flags = [bool(p["is_default_centroid"]) for p in prov]
    frac = float(np.mean(flags)) if flags else 0.0
    return {"Mehrheit (>=0.5)": frac >= 0.5, "Mindestens eine (>0)": frac > 0.0}


def main():
    cases = load_cases()
    loo = loo_pseudo_radii(cases)
    globals_ = [loo[s]["_global"] for s in loo]
    anchor_eps = min(EPS_GRID, key=lambda e: abs(e - float(np.median(globals_))))

    # Pro Fall einmal: Streuung, Schätzfehler, beide Hub-Booleans
    recs = []
    for c in cases:
        pts = c["points"]; n = len(pts)
        pd_ = [haversine_error(tuple(pts[i]), tuple(pts[j]))
               for i in range(n) for j in range(i + 1, n)]
        spread = float(np.median(pd_)) if pd_ else 0.0
        err = haversine_error(estimate(c, "L1", "b", loo, eps=anchor_eps), c["truth"])
        rec = {"spread": spread, "low": spread < 50, "err": err, "n_src": n}
        rec.update(hub_rules(c["provenance"]))
        recs.append(rec)

    RULES = ["Mehrheit (>=0.5)", "Mindestens eine (>0)"]
    total = len(recs)
    miss = sum(1 for r in recs if r["err"] > 100)
    n_multi = sum(1 for r in recs if r["n_src"] >= 3)
    print(f"Faelle: {total}  | Misses (>100km): {miss}  | Anker mit >=3 Quellen: {n_multi} "
          f"({100*n_multi/total:.0f}%; nur dort koennen sich die Regeln unterscheiden)")
    print(f"Headline-eps = {anchor_eps} km")

    for rule in RULES:
        print("\n" + "=" * 78)
        print(f"REGEL: {rule}")
        print("=" * 78)
        # Matrix
        print("2D-Matrix — je Feld: n | Median | Mittel | %>100km")
        labels = {(True, False): "niedrig+kein_hub  belastbar",
                  (True, True):  "niedrig+hub       SICHERHEITS-ILLUSION",
                  (False, False): "hoch+kein_hub     ehrliche Unsicherheit",
                  (False, True):  "hoch+hub          Inkonsistenz"}
        for low in (True, False):
            for hub in (False, True):
                sub = [r["err"] for r in recs if r["low"] == low and r[rule] == hub]
                a = np.array(sub, dtype=float)
                med = np.median(a) if len(a) else float("nan")
                mean = a.mean() if len(a) else float("nan")
                tail = 100 * (a > 100).mean() if len(a) else float("nan")
                print(f"  {labels[(low,hub)]:42s} n={len(a):4d} | {med:5.0f} | {mean:6.0f} | {tail:5.1f}%")
        illusion = np.array([r["err"] for r in recs if r["low"] and r[rule]], dtype=float)
        if len(illusion):
            lo, hi = _bootstrap_tail_ci(illusion)
            print(f"  Bootstrap-95%-CI %>100km der Illusion-Zelle (n={len(illusion)}): [{lo:.1f}, {hi:.1f}] %")

        # Recall / Precision / Flag-Rate  (unsicher = hohe Streuung ODER hub)
        flagged_all = sum(1 for r in recs if r["low"] is False or r[rule])
        flagged_miss = sum(1 for r in recs if (r["low"] is False or r[rule]) and r["err"] > 100)
        hub_rate = 100 * sum(1 for r in recs if r[rule]) / total
        print(f"\n  Hub-Rate gesamt: {hub_rate:.1f}%")
        print(f"  Recall auf Misses: {flagged_miss}/{miss} = {100*flagged_miss/miss:.0f}%")
        print(f"  Geflaggt gesamt: {flagged_all}/{total} = {100*flagged_all/total:.0f}%  "
              f"(False-Alarm {100*(flagged_all-flagged_miss)/flagged_all:.0f}%)")
        print(f"  Praezision der Flags auf Misses: {100*flagged_miss/flagged_all:.0f}%")


if __name__ == "__main__":
    main()
