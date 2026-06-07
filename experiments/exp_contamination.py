"""E2 / T2 — Kontamination & empirischer Breakdown-Point (FF2).

Pro Anchor wird ein Anteil α der Quellen *vergiftet* (geodätisch um offset_km weit
verschoben). Gemessen wird der Distanzfehler je Schätzer als Funktion von α — der
empirische Breakdown-Point. Bootstrap über die zufällige Auswahl der vergifteten
Quellen (B Wiederholungen) liefert IQR-Bänder statt einer einzelnen Realisierung.

Erwartung (Theorie): naiver Mittelwert (centroid) wächst ~linear ab α>0; der
geometrische Median bleibt robust bis ~50 % und bricht dann (Breakdown ~0,5).

Aufruf:  python experiments/exp_contamination.py
Ergebnis: eval/out/e2_breakdown.{csv,png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.pipeline import load_cases, line_weights   # noqa: E402
from eval.metrics import haversine_error              # noqa: E402
from estimators.baselines import (centroid, coordinate_median,   # noqa: E402
                                  geometric_median, weighted_geometric_median)

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
N_BOOT = 60
OFFSET_KM = 2000.0
SEED = 20260604
OUT = Path("eval/out"); OUT.mkdir(parents=True, exist_ok=True)

# Schätzer-Farben (s/w-tauglich zusätzlich über Linientyp/Marker)
ESTS = ["centroid", "median", "geometric_median", "geom_median_perline"]
STYLE = {"centroid": ("#d62728", ":", "o"), "median": ("#2ca02c", "-.", "v"),
         "geometric_median": ("#1f77b4", "-", "s"), "geom_median_perline": ("#9467bd", "--", "^")}
LABEL = {"centroid": "naiver Mittelwert", "median": "Koordinaten-Median",
         "geometric_median": "geom. Median", "geom_median_perline": "geom. Median (Linien-gew.)"}


def _k(alpha, n):
    """Anzahl zu vergiftender Quellen: kaufmännisch (half-up) gerundet."""
    return int(np.floor(alpha * n + 0.5))


def contaminate(points, k, rng, offset_km=OFFSET_KM):
    """k zufällige Punkte geodätisch um offset_km verschieben (breitengradweise,
    vom Pol weg → kein Clamping). Coordinated/adversarial in der Hemisphäre."""
    if k <= 0:
        return points
    pts = points.copy()
    idx = rng.choice(len(pts), size=k, replace=False)
    dlat = offset_km / 111.32
    for i in idx:
        pts[i, 0] += -dlat if pts[i, 0] >= 0 else dlat   # Richtung Äquator/andere Hemisphäre
    return pts


def make_plot(rows, eff_stats, n_cases):
    """2-Panel-Plot aus den aggregierten Ergebnissen (rows + α_eff-Statistik)."""
    fig, (ax, axe) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [4, 1]})
    x = np.array(ALPHAS) * 100
    for e in ESTS:
        col, ls, mk = STYLE[e]
        med = [rows[(e, a)][0] for a in ALPHAS]
        q1 = [rows[(e, a)][1] for a in ALPHAS]; q3 = [rows[(e, a)][2] for a in ALPHAS]
        ax.plot(x, med, ls, color=col, marker=mk, label=LABEL[e], linewidth=1.8)
        ax.fill_between(x, q1, q3, color=col, alpha=0.15)
    ax.axvline(50, color="grey", linewidth=1, alpha=0.6)
    ax.set_ylabel("Median-Distanzfehler / km")
    ax.set_title(f"E2 — Breakdown unter Kontamination  (n = {n_cases} Anchors, "
                 f"B = {N_BOOT}, Δ = {OFFSET_KM:.0f} km)")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=8)

    # unteres Panel: tatsächlicher Kontaminationsanteil α_eff je nominaler Stufe
    em = np.array([eff_stats[a][0] for a in ALPHAS]) * 100
    eq1 = np.array([eff_stats[a][1] for a in ALPHAS]) * 100
    eq3 = np.array([eff_stats[a][2] for a in ALPHAS]) * 100
    axe.errorbar(x, em, yerr=[em - eq1, eq3 - em], fmt="o", color="#333333", capsize=3, markersize=4)
    axe.plot([0, 60], [0, 60], "--", color="grey", linewidth=0.8, alpha=0.7)  # α_eff = α_nom
    axe.axhline(50, color="grey", linewidth=0.8, alpha=0.5)
    axe.set_xlabel("nominaler Kontaminationsanteil α_nom / %")
    axe.set_ylabel("α_eff / %", fontsize=8)
    axe.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "e2_breakdown.png", dpi=150); plt.close(fig)


def load_results_csv(path=OUT / "e2_breakdown.csv"):
    """Aggregierte Ergebnisse für ein Re-Rendering aus der CSV laden."""
    import csv
    rows, eff_stats = {}, {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            a, e = float(r["alpha_nom"]), r["estimator"]
            rows[(e, a)] = (float(r["median_km"]), float(r["q1_km"]), float(r["q3_km"]))
            eff_stats[a] = (float(r["alpha_eff_med"]), float(r["alpha_eff_q1"]), float(r["alpha_eff_q3"]))
    return rows, eff_stats


def estimate_all(pts, weights):
    return {
        "centroid": centroid(pts),
        "median": coordinate_median(pts),
        "geometric_median": geometric_median(pts),
        "geom_median_perline": weighted_geometric_median(pts, weights),
    }


def run():
    cases = load_cases()
    prepared = [(c["points"], line_weights([p["lineage"] for p in c["provenance"]]), c["truth"])
                for c in cases]
    rng = np.random.default_rng(SEED)
    print(f"E2 — {len(cases)} Fälle, α={ALPHAS}, B={N_BOOT}, offset={OFFSET_KM:.0f} km")

    # results[est][alpha] = Liste der B Bootstrap-Mediane (über Fälle)
    results = {e: {a: [] for a in ALPHAS} for e in ESTS}
    for a in ALPHAS:
        for _ in range(N_BOOT):
            per_est = {e: [] for e in ESTS}
            for pts, w, truth in prepared:
                k = _k(a, len(pts))
                cp = contaminate(pts, k, rng)
                for e, est in estimate_all(cp, w).items():
                    per_est[e].append(haversine_error((est[0], est[1]), truth))
            for e in ESTS:
                results[e][a].append(float(np.median(per_est[e])))
        print(f"  α={a:.1f} fertig")

    # α_eff = k/n_i je Anchor und α_nom (rep-unabhängig) — macht sichtbar, dass die
    # nominale α-Stufe über die unterschiedlichen Quellenzahlen ein schmales Intervall ist.
    eff = {a: np.array([_k(a, len(pts)) / len(pts) for pts, _, _ in prepared]) for a in ALPHAS}
    eff_stats = {a: (np.median(v), np.percentile(v, 25), np.percentile(v, 75)) for a, v in eff.items()}

    import csv
    rows = {}
    with open(OUT / "e2_breakdown.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["estimator", "alpha_nom", "alpha_eff_med", "alpha_eff_q1", "alpha_eff_q3",
                     "median_km", "q1_km", "q3_km"])
        for e in ESTS:
            for a in ALPHAS:
                arr = np.array(results[e][a])
                m, q1, q3 = np.median(arr), np.percentile(arr, 25), np.percentile(arr, 75)
                rows[(e, a)] = (m, q1, q3)
                em, eq1, eq3 = eff_stats[a]
                wr.writerow([e, a, f"{em:.3f}", f"{eq1:.3f}", f"{eq3:.3f}",
                             f"{m:.2f}", f"{q1:.2f}", f"{q3:.2f}"])

    make_plot(rows, eff_stats, len(cases))

    print("\nα_eff (Median [Q1–Q3]) je nominaler Stufe:")
    for a in ALPHAS:
        em_, q1_, q3_ = eff_stats[a]
        print(f"  α_nom={a:.1f} → α_eff={em_:.3f} [{q1_:.3f}–{q3_:.3f}]")

    print("\nMedian-Fehler km je α (naiver Mittelwert vs. geom. Median):")
    for a in ALPHAS:
        print(f"  α={a:.1f}:  centroid={rows[('centroid',a)][0]:7.1f}  "
              f"geom_median={rows[('geometric_median',a)][0]:6.1f}  "
              f"perline={rows[('geom_median_perline',a)][0]:6.1f}")
    print(f"\nTabelle: {OUT}/e2_breakdown.csv   Plot: {OUT}/e2_breakdown.png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="E2 Kontamination/Breakdown.")
    ap.add_argument("--replot", action="store_true",
                    help="nur Plot aus e2_breakdown.csv neu rendern (kein Neu-Rechnen)")
    args = ap.parse_args()
    if args.replot:
        rows, eff_stats = load_results_csv()
        from eval.pipeline import load_cases as _lc
        make_plot(rows, eff_stats, len(_lc()))
        print(f"Neu gerendert: {OUT}/e2_breakdown.png")
    else:
        run()
