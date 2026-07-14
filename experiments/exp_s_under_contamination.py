#!/usr/bin/env python3
"""Stütz-Konzentration S unter koordinierter Kontamination (T2-Angriffsmodell).

Frage (§4.6/§5.1): „Scheitert laut, nicht leise" ist auf dem natürlichen
Fehler-Regime kalibriert — bleibt der Wächter auch dann laut, wenn ein
Angreifer den Schätzer jenseits des Breakdown kapert?

Aufbau: identisches Kontaminationsmodell wie T2 (zufällige Quellen-Teilmenge,
koordinierte Breitengrad-Verschiebung um 2.000 km Richtung Äquator), Schätzer
L1·b (Headline), S = linien-gewichtete Stütz-Konzentration (r = 50 km) um die
kontaminierte Schätzung. Flag-Schwelle = 41-%-Quantil der natürlichen
S-Verteilung (gleicher Arbeitspunkt wie in T6). Berichtet je Kontaminations-
stufe: Fehler-Median, S-Median gesamt, S-Median der GEKAPERTEN Schätzungen
(Fehler > 1.000 km) und deren Anteil OBERHALB der Flag-Schwelle
(= „selbstbewusst falsch": gekapert, aber nicht geflaggt).

Ergebnis: Tabelle (stdout) + eval/out/s_under_contamination.csv
          + eval/out/s_under_contamination.png
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.pipeline import load_cases                     # noqa: E402
from eval.metrics import haversine_error                 # noqa: E402
from experiments.exp_t6_defaults import (                # noqa: E402
    loo_pseudo_radii, estimate, line_weights_for,
)

OUT = Path("eval/out")
OUT.mkdir(parents=True, exist_ok=True)
EPS = 10
OFFSET_KM = 2000.0
ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
B = 3                    # Wiederholungen je Stufe (Median über 1.077 Anchors stabil)
FLAG_QUOTE = 0.41        # T6-Arbeitspunkt (Flag-Quote des Vorgängerlabel-Vergleichs)
CAPTURED_KM = 1000.0     # „gekapert": Fehler jenseits des natürlichen Regimes


def s_value(case, est):
    prov = case["provenance"]
    w = line_weights_for(prov, "L1")
    w = w / w.sum()
    d = np.array([haversine_error((p["lat"], p["lon"]), est) for p in prov])
    return float(w[d < 50].sum())


def contaminate_case(case, alpha, rng):
    cc = copy.deepcopy(case)
    n = len(cc["provenance"])
    k = int(np.floor(alpha * n + 0.5))          # kaufmännisch, wie T2
    if k:
        dlat = OFFSET_KM / 111.32
        for i in rng.choice(n, size=k, replace=False):
            p = cc["provenance"][i]
            p["lat"] += -dlat if p["lat"] >= 0 else dlat
    return cc


def main():
    cases = load_cases()
    loo = loo_pseudo_radii(cases)

    nat_S = np.array([s_value(c, estimate(c, "L1", "b", loo, eps=EPS))
                      for c in cases])
    thr = float(np.quantile(nat_S, FLAG_QUOTE))
    print(f"natürlich: Median S = {np.median(nat_S):.2f}, "
          f"Flag-Schwelle (Quote {FLAG_QUOTE:.0%}) t = {thr:.3f}\n")

    rows = []
    for ai, alpha in enumerate(ALPHAS):
        rng = np.random.default_rng([0, ai])    # je Stufe reproduzierbar
        errs, ss = [], []
        for _ in range(B):
            for c in cases:
                cc = contaminate_case(c, alpha, rng)
                est = estimate(cc, "L1", "b", loo, eps=EPS)
                errs.append(haversine_error(est, cc["truth"]))
                ss.append(s_value(cc, est))
        errs, ss = np.array(errs), np.array(ss)
        cap = ss[errs > CAPTURED_KM]
        rows.append({
            "alpha_nom": alpha,
            "err_median_km": round(float(np.median(errs)), 1),
            "s_median_all": round(float(np.median(ss)), 3),
            "n_captured": int(len(cap)),
            "s_median_captured": round(float(np.median(cap)), 3) if len(cap) else None,
            "captured_unflagged_pct": round(100 * float((cap >= thr).mean()), 1) if len(cap) else None,
            "flag_threshold": round(thr, 3),
        })
        r = rows[-1]
        print(f"alpha={alpha:.1f}  errMed={r['err_median_km']:>8}  "
              f"S-Med={r['s_median_all']:.2f}  gekapert n={r['n_captured']:>5}  "
              f"S-Med(gek.)={r['s_median_captured']}  "
              f"ungeflaggt={r['captured_unflagged_pct']} %")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "s_under_contamination.csv", index=False)

    # ---- Plot: S entlang der Kontaminationskurve, Flag-Schwelle als Grenze
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.array(ALPHAS) * 100
    ax.plot(x, df["s_median_all"], "o-", color="#1f77b4",
            label="S-Median (alle Schätzungen)")
    m = df["s_median_captured"].notna()
    ax.plot(x[m.values], df.loc[m, "s_median_captured"], "s--", color="#d62728",
            label="S-Median (gekaperte Schätzungen, Fehler > 1.000 km)")
    ax.axhline(thr, color="0.3", ls=":",
               label=f"Flag-Schwelle t = {thr:.2f} (Flag-Quote 41 %)")
    ax.axvspan(50, x.max(), color="0.9",
               label="jenseits des Schätzer-Breakdown")
    ax.set_xlabel(r"nominaler Kontaminationsanteil $\alpha_{\mathrm{nom}}$ [%]")
    ax.set_ylabel("Stütz-Konzentration S")
    ax.set_ylim(0, 1.0)
    ax.set_title("S unter koordinierter Kontamination")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "s_under_contamination.png", dpi=160)
    print(f"\nCSV: {OUT}/s_under_contamination.csv")
    print(f"PNG: {OUT}/s_under_contamination.png")


if __name__ == "__main__":
    main()
