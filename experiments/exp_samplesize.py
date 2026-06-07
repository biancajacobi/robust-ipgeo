"""E3 / T3 — Stichprobengröße n: Quellenzahl senken (FF3).

Sub-Sampling über die bestehenden Beobachtungen: je Anchor zufällig n ∈ {3,5,7}
Quellen ziehen, Schätzer rechnen, Distanzfehler messen. Bootstrap über die zufällige
Auswahl (B Wiederholungen) → IQR-Bänder. Referenz „all" = alle verfügbaren Quellen.

Methodischer Zweck: Brätz' klassen-/dichtebasiertes Verfahren arbeitet auf n≈15 und
unterstellt Normalverteilung — hier wird gezeigt, ob die robusten Schätzer auch bei
n=3 noch brauchbare Referenzpunkte liefern (eine konkrete Lücke der Brätz-Annahme).
Hinweis: trimmed_mean (20 %) trimmt erst ab n≥5 etwas; bei n=3 == Mittel.

Aufruf:  python experiments/exp_samplesize.py
Ergebnis: eval/out/e3_samplesize.{csv,png}
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.pipeline import load_cases, line_weights   # noqa: E402
from eval.metrics import haversine_error              # noqa: E402
from estimators.baselines import (centroid, coordinate_median, trimmed_mean,  # noqa: E402
                                  geometric_median, weighted_geometric_median)

N_VALUES = [3, 5, 7]
N_BOOT = 60
SEED = 20260604
OUT = Path("eval/out"); OUT.mkdir(parents=True, exist_ok=True)

ESTS = ["centroid", "median", "trimmed_mean", "geometric_median", "geom_median_perline"]
LABEL = {"centroid": "naiver Mittelwert", "median": "Koord.-Median", "trimmed_mean": "getr. Mittel",
         "geometric_median": "geom. Median", "geom_median_perline": "geom. Median (Linien-gew.)"}
COLOR = {"centroid": "#d62728", "median": "#2ca02c", "trimmed_mean": "#8c564b",
         "geometric_median": "#1f77b4", "geom_median_perline": "#9467bd"}


def make_plot(rows, n_eligible):
    """Forestplot: y = (n, Schätzer) gruppiert, x = Median-Fehler mit IQR-Whiskern."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ypos, ylab, y = [], [], 0
    for n in N_VALUES:
        for e in ESTS:
            m, q1, q3 = rows[(e, n)]
            ax.errorbar(m, y, xerr=[[m - q1], [q3 - m]], fmt="o", color=COLOR[e],
                        capsize=3, markersize=6)
            ypos.append(y); ylab.append(f"n={n} · {LABEL[e]}"); y += 1
        y += 0.6
    ax.set_yticks(ypos, ylab, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Median-Distanzfehler / km  (Whisker = IQR über Bootstrap)")
    ax.set_title(f"E3 — Stichprobengröße: Fehler bei n ∈ {N_VALUES}  "
                 f"(n_Fälle≈{n_eligible[N_VALUES[0]]}, B={N_BOOT})")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "e3_samplesize.png", dpi=150); plt.close(fig)


def load_results_csv(path=OUT / "e3_samplesize.csv"):
    rows, n_eligible = {}, {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            n = int(r["n"])
            rows[(r["estimator"], n)] = (float(r["median_km"]), float(r["q1_km"]), float(r["q3_km"]))
            n_eligible[n] = int(r["n_cases"])
    return rows, n_eligible


def estimate_all(pts, weights):
    return {
        "centroid": centroid(pts),
        "median": coordinate_median(pts),
        "trimmed_mean": trimmed_mean(pts),
        "geometric_median": geometric_median(pts),
        "geom_median_perline": weighted_geometric_median(pts, weights),
    }


def run():
    cases = load_cases()
    # je Fall: Punkte, Linien (für Linien-Gewicht der Teilstichprobe), Truth
    prepared = [(c["points"], [p["lineage"] for p in c["provenance"]], c["truth"]) for c in cases]
    rng = np.random.default_rng(SEED)
    print(f"E3 — {len(cases)} Fälle, n∈{N_VALUES}, B={N_BOOT}")

    results = {e: {n: [] for n in N_VALUES} for e in ESTS}
    n_eligible = {}
    for n in N_VALUES:
        elig = [(pts, lin, t) for (pts, lin, t) in prepared if len(pts) >= n]
        n_eligible[n] = len(elig)
        for _ in range(N_BOOT):
            per_est = {e: [] for e in ESTS}
            for pts, lin, truth in elig:
                idx = rng.choice(len(pts), size=n, replace=False)
                sub, sublin = pts[idx], [lin[i] for i in idx]
                w = line_weights(sublin)
                for e, est in estimate_all(sub, w).items():
                    per_est[e].append(haversine_error((est[0], est[1]), truth))
            for e in ESTS:
                results[e][n].append(float(np.median(per_est[e])))
        print(f"  n={n} fertig ({len(elig)} geeignete Fälle)")

    rows = {}
    with open(OUT / "e3_samplesize.csv", "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["estimator", "n", "median_km", "q1_km", "q3_km", "n_cases"])
        for e in ESTS:
            for n in N_VALUES:
                arr = np.array(results[e][n])
                m, q1, q3 = np.median(arr), np.percentile(arr, 25), np.percentile(arr, 75)
                rows[(e, n)] = (m, q1, q3)
                wr.writerow([e, n, f"{m:.2f}", f"{q1:.2f}", f"{q3:.2f}", n_eligible[n]])

    make_plot(rows, n_eligible)

    print("\nMedian-Fehler km je n:")
    for n in N_VALUES:
        print(f"  n={n}: " + "  ".join(f"{e.split('_')[0]}={rows[(e,n)][0]:.1f}" for e in ESTS))
    print(f"\nTabelle: {OUT}/e3_samplesize.csv   Plot: {OUT}/e3_samplesize.png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="E3 Stichprobengröße.")
    ap.add_argument("--replot", action="store_true",
                    help="nur Plot aus e3_samplesize.csv neu rendern (kein Neu-Rechnen)")
    args = ap.parse_args()
    if args.replot:
        rows, n_eligible = load_results_csv()
        make_plot(rows, n_eligible)
        print(f"Neu gerendert: {OUT}/e3_samplesize.png")
    else:
        run()
