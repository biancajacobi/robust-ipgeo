"""Zusaetzliche T1-Baseline: naiver City-Level-Majority-Vote + gepaarte Bootstrap-CIs.

Faithfuller, bewusst simpler Vertreter der diskreten Fusions-Verfahren
(li2017fusion / xie2018fusion): je Anchor stimmt jede antwortende Quelle mit ihrem
(Land, Stadt)-Label ab; die Stadt mit den meisten Stimmen gewinnt, der Schaetzpunkt
ist der Koordinaten-Median der Gewinner-Gruppe. KEINE Linien-Kollabierung (jede
Quelle = eine Stimme) -- der naive Fusions-Ansatz behandelt die MaxMind-Replikate
als unabhaengige Evidenz.

Tie-Break: hoechste Stimmenzahl, dann alphabetisch nach (Land, Stadt). Quellen ohne
Stadt-Label enthalten sich; hat kein Quellpunkt eine Stadt, Fallback =
Koordinaten-Median aller Quellpunkte.

Der Vote wird auf denselben Faellen/Punkten wie die Aggregation L1*b gerechnet
(load_cases), Stadt/Land je (ip, source) aus observations.csv nachgeschlagen. So
ist der gepaarte Bootstrap gegen L1*b (Median, Mittel, Tail) sauber ausgerichtet.

Aufruf:  python experiments/exp_city_vote.py
Ergebnis: Tabelle (stdout) + eval/out/city_vote_bootstrap.csv
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.pipeline import load_cases                         # noqa: E402
from eval.metrics import haversine_error                     # noqa: E402
from experiments.exp_t6_defaults import (                    # noqa: E402
    loo_pseudo_radii, estimate, EPS_GRID,
)
from experiments.exp_bootstrap_ci import (                   # noqa: E402
    paired_bootstrap, paired_bootstrap_stat,
)

OBS = Path("data/cache/observations.csv")
OUT = Path("eval/out"); OUT.mkdir(parents=True, exist_ok=True)
TAIL_KM = 100.0


def city_lookup() -> dict[tuple[str, str], tuple[str, str]]:
    df = pd.read_csv(OBS, dtype=str)
    lut = {}
    for _, r in df.iterrows():
        city = str(r.get("city") or "").strip()
        country = str(r.get("country") or "").strip()
        lut[(str(r["ip"]), str(r["source"]))] = (country, city)
    return lut


def city_vote(case: dict, lut: dict) -> tuple[float, float]:
    named = []  # (key, lat, lon)
    for p in case["provenance"]:
        country, city = lut.get((case["ip"], p["source"]), ("", ""))
        if city and city.lower() != "nan":
            named.append((f"{country.upper()}|{city.lower()}", p["lat"], p["lon"]))
    if not named:
        pts = case["points"]
        return float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
    counts = Counter(k for k, _, _ in named)
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    win = [(la, lo) for k, la, lo in named if k == top]
    return (float(np.median([w[0] for w in win])),
            float(np.median([w[1] for w in win])))


def _stats(e: np.ndarray) -> tuple[float, float, float]:
    return (float(np.median(e)), float(np.mean(e)), 100 * float(np.mean(e > TAIL_KM)))


def main() -> None:
    cases = load_cases()
    loo = loo_pseudo_radii(cases)
    globals_ = [loo[s]["_global"] for s in loo]
    anchor_eps = min(EPS_GRID, key=lambda e: abs(e - float(np.median(globals_))))
    lut = city_lookup()

    agg, cv = {}, {}
    for c in cases:
        ip = c["ip"]
        agg[ip] = haversine_error(estimate(c, "L1", "b", loo, eps=anchor_eps), c["truth"])
        cv[ip] = haversine_error(city_vote(c, lut), c["truth"])

    ips = sorted(set(agg) & set(cv))
    da = np.array([agg[ip] for ip in ips])
    dr = np.array([cv[ip] for ip in ips])

    cvm = _stats(dr)
    aggm = _stats(da)
    print("=" * 66)
    print(f"City-Level-Majority-Vote vs. Aggregation L1*b   (n={len(ips)})")
    print("=" * 66)
    print(f"  City-Vote : Median {cvm[0]:5.1f} | Mittel {cvm[1]:6.1f} | Tail>100 {cvm[2]:4.1f}%")
    print(f"  L1*b      : Median {aggm[0]:5.1f} | Mittel {aggm[1]:6.1f} | Tail>100 {aggm[2]:4.1f}%")

    dmed, lo, hi, lob, hib = paired_bootstrap(da, dr)
    dmean, mlo, mhi = paired_bootstrap_stat(da, dr, np.mean)
    tail = lambda a: 100 * np.mean(a > TAIL_KM)  # noqa: E731
    dt, tlo, thi = paired_bootstrap_stat(da, dr, tail)
    print("\n  Paired bootstrap (B=10.000, seed=0) | Δ = L1*b − City-Vote (negativ = L1*b besser)")
    print(f"    Δ median = {dmed:+6.2f} km  95%-CI [{lo:+.2f}, {hi:+.2f}]")
    print(f"    Δ mean   = {dmean:+6.1f} km  95%-CI [{mlo:+.1f}, {mhi:+.1f}]")
    print(f"    Δ tail   = {dt:+6.2f} pp  95%-CI [{tlo:+.2f}, {thi:+.2f}]")

    pd.DataFrame([{
        "n": len(ips),
        "cv_median": round(cvm[0], 1), "cv_mean": round(cvm[1], 1), "cv_tail": round(cvm[2], 1),
        "l1b_median": round(aggm[0], 1), "l1b_mean": round(aggm[1], 1), "l1b_tail": round(aggm[2], 1),
        "d_median": round(dmed, 2), "d_median_lo": round(lo, 2), "d_median_hi": round(hi, 2),
        "d_mean": round(dmean, 1), "d_mean_lo": round(mlo, 1), "d_mean_hi": round(mhi, 1),
        "d_tail": round(dt, 2), "d_tail_lo": round(tlo, 2), "d_tail_hi": round(thi, 2),
    }]).to_csv(OUT / "city_vote_bootstrap.csv", index=False)
    print(f"\n  CSV: {OUT}/city_vote_bootstrap.csv")


if __name__ == "__main__":
    main()
