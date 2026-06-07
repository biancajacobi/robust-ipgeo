"""Streuungsmaß-Auswahl für das Konfidenzlabel — welche Dispersion erkennt Misses?

Hintergrund: Das 2D-Konfidenzlabel (exp_t6_defaults, Teil II) nutzte als Streuungs-
achse die MEDIAN-Paardistanz der Quellpunkte. Die mediane Paardistanz ist robust
gegen einen einzelnen weit abweichenden Punkt — genau deshalb *übersieht* sie den
Fall „eine Quelle fällt auf einen fernen Default-Centroid, die übrigen sind einig":
der Ausreißer hebt die Mediandistanz nicht, der Schätzer kann aber dennoch daneben
liegen. Dieser Test prüft systematisch, ob ein Streuungsmaß, das den Ausreißer
*sieht* (Mittelwert, Maximum, Quantile, mittlere Distanz zum Schätzer), die Misses
besser markiert — und ob der Centroid-Flag (any vs. majority) bei gutem Streuungs-
maß überhaupt noch etwas beiträgt.

Metrik: Miss = Haversine-Fehler des eingesetzten Schätzers (L1+b) > 100 km
(gleiche Definition wie run_confidence_recall). Verglichen wird threshold-frei:
  - Average Precision (Fläche unter Precision/Recall; Boden = Basisrate),
  - Precision bei festem Recall (fairer Arbeitspunkt-Vergleich),
  - Orthogonalität: bringt der Centroid-Flag AP-Gewinn ZUSÄTZLICH zum besten
    Streuungsmaß (logistisches 2-Merkmal-Surrogat)?

Aufruf:  python experiments/exp_spread_measure.py
Ergebnis: Tabelle (stdout) + eval/out/spread_measure_comparison.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.pipeline import load_cases            # noqa: E402
from eval.metrics import haversine_error         # noqa: E402
import experiments.exp_t6_defaults as T6         # noqa: E402

OUT = Path("eval/out"); OUT.mkdir(parents=True, exist_ok=True)
MISS_KM = 100.0
FEATURES = {
    "median pairwise (alt)": "med",
    "mean pairwise":         "mean",
    "max pairwise":          "mx",
    "q75 pairwise":          "q75",
    "q90 pairwise":          "q90",
    "max dist->Schätzer":    "maxd_est",
    "mean dist->Schätzer":   "mean_d_est",
}


def build_frame(cases, loo, eps):
    rows = []
    for c in cases:
        pts = np.array(c["points"], dtype=float)
        n = len(pts)
        pdd = (np.array([haversine_error(tuple(pts[i]), tuple(pts[j]))
                         for i in range(n) for j in range(i + 1, n)])
               if n > 1 else np.array([0.0]))
        est = T6.estimate(c, "L1", "b", loo, eps=eps)
        dse = np.array([haversine_error(tuple(p), est) for p in pts])
        k = int(sum(p["is_default_centroid"] for p in c["provenance"]))
        rows.append(dict(
            ip=c["ip"], med=float(np.median(pdd)), mean=float(np.mean(pdd)),
            mx=float(np.max(pdd)), q75=float(np.percentile(pdd, 75)),
            q90=float(np.percentile(pdd, 90)), maxd_est=float(np.max(dse)),
            mean_d_est=float(np.mean(dse)), any_c=int(k > 0), maj_c=int(k / n >= 0.5),
            err=haversine_error(est, c["truth"]),
        ))
    d = pd.DataFrame(rows)
    d["y"] = (d.err > MISS_KM).astype(int)
    return d


def average_precision(y, s):
    o = np.argsort(-np.asarray(s, dtype=float))
    y = np.asarray(y)[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    P = tp / (tp + fp)
    R = tp / y.sum()
    R = np.concatenate([[0.0], R])
    return float(np.sum((R[1:] - R[:-1]) * P))


def prec_at_recall(y, s, target):
    o = np.argsort(-np.asarray(s, dtype=float))
    y = np.asarray(y)[o]
    tp = np.cumsum(y)
    R = tp / y.sum()
    P = tp / np.arange(1, len(y) + 1)
    idx = np.where(R >= target)[0]
    if not len(idx):
        return float("nan"), float("nan"), 0
    i = idx[0]
    return float(P[i]) * 100, float(R[i]) * 100, int(i + 1)


def _logit_ap(d, y, cols):
    """AP eines logistischen 2-/n-Merkmal-Surrogats (IRLS, dependency-frei)."""
    X = np.column_stack([np.log1p(d[c].values) if d[c].max() > 1 else d[c].values for c in cols])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    X = np.column_stack([np.ones(len(d)), X])
    w = np.zeros(X.shape[1])
    for _ in range(200):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        W = p * (1 - p) + 1e-9
        w -= np.linalg.solve((X * W[:, None]).T @ X + 1e-6 * np.eye(X.shape[1]),
                             X.T @ (p - y) + 1e-6 * w)
    return average_precision(y, 1.0 / (1.0 + np.exp(-X @ w)))


def run():
    cases = load_cases()
    loo = T6.loo_pseudo_radii(cases)
    eps = min(T6.EPS_GRID, key=lambda e: abs(e - float(np.median([loo[s]["_global"] for s in loo]))))
    d = build_frame(cases, loo, eps)
    y = d.y.values

    print("=" * 88)
    print(f"STREUUNGSMASS-VERGLEICH als Miss-Detektor (Miss = Fehler > {MISS_KM:.0f} km)")
    print(f"n={len(d)}  Misses={int(y.sum())}  Basisrate={y.mean():.3f} (= AP-Boden)")
    print("=" * 88)
    print(f"{'Streuungsmaß':24s} {'AP':>6s}  {'Prec@Rec=0.90':>20s}  {'Prec@Rec=0.72':>20s}")
    out_rows = []
    for name, col in FEATURES.items():
        ap = average_precision(y, d[col].values)
        p9 = prec_at_recall(y, d[col].values, 0.90)
        p7 = prec_at_recall(y, d[col].values, 0.72)
        print(f"{name:24s} {ap:6.3f}  {p9[0]:5.1f}% @{p9[1]:4.1f}% (n{p9[2]:4d})  "
              f"{p7[0]:5.1f}% @{p7[1]:4.1f}% (n{p7[2]:4d})")
        out_rows.append({"measure": name, "ap": round(ap, 3),
                         "prec_at_rec90": round(p9[0], 1), "n_flag_rec90": p9[2],
                         "prec_at_rec72": round(p7[0], 1), "n_flag_rec72": p7[2]})

    print("\nOrthogonalität — bringt der Centroid-Flag AP ZUSÄTZLICH zum besten Streuungsmaß?")
    best = max(FEATURES.values(), key=lambda c: average_precision(y, d[c].values))
    combos = {f"{best}": [best], f"{best} + any": [best, "any_c"],
              f"{best} + majority": [best, "maj_c"], "median + any (alte Achsen)": ["med", "any_c"]}
    for label, cols in combos.items():
        ap = _logit_ap(d, y, cols)
        print(f"  AP[{label:30s}] = {ap:.3f}")
        out_rows.append({"measure": f"logit:{label}", "ap": round(ap, 3),
                         "prec_at_rec90": "", "n_flag_rec90": "",
                         "prec_at_rec72": "", "n_flag_rec72": ""})

    print("\nQuadranten-Reinheit: belastbar-Zelle (low_spread & kein hub), Schwelle 50 km")
    for col, tag in (("med", "median (alt)"), ("mean", "mean (neu)")):
        m = (d[col] < 50) & (d.maj_c == 0)
        e = d[m].err
        print(f"  {tag:12s}: n={m.sum():4d}  q95={np.quantile(e, .95):7.1f} km  miss%={100*(e>100).mean():4.1f}")
        out_rows.append({"measure": f"belastbar-Zelle ({tag})", "ap": "",
                         "prec_at_rec90": f"q95={np.quantile(e,.95):.0f}",
                         "n_flag_rec90": int(m.sum()),
                         "prec_at_rec72": f"miss={100*(e>100).mean():.1f}%", "n_flag_rec72": ""})

    pd.DataFrame(out_rows).to_csv(OUT / "spread_measure_comparison.csv", index=False)
    print(f"\nCSV: {OUT}/spread_measure_comparison.csv")


if __name__ == "__main__":
    run()
