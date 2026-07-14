#!/usr/bin/env python3
"""Antimeridian-Randfall: Kartenabhängigkeit der Grad-Raum-Aggregation (§5.4.2).

Im Rohgrad-Raum ist die Aggregation bei ±180° kartenabhängig (ein Paar bei
179°/−179° erscheint 358° statt 2° entfernt). Dieses Skript prüft die
empirische Relevanz auf der Anchor-Stichprobe:

  (a) betroffene Anchors: Beobachtungs-Längengrade mit Rohgrad-Spanne > 180°,
  (b) wahre (zirkuläre) Längenspanne je Fall (> 180° ⇒ keine kanonische Karte),
  (c) Kontrollrechnung: L1·b auf pro Anchor re-zentrierten Längengraden
      (Unwrap um die erste Beobachtung) gegen den Rohgrad-Status-quo.

Befund: Alle betroffenen Fälle entstehen durch weit entfernte Default-
Ausreißer (der Rohgrad-Raum ÜBERZEICHNET deren Distanz → stärkere robuste
Dämpfung, pro-robust); ehrliche Cluster am Antimeridian treten nicht auf.
Die Re-Zentrierung ändert die Schätzungen nur um Kilometer — außer dort, wo
die wahre Spanne > 180° ist: Dort ist Re-Zentrierung selbst nicht wohl-
definiert und kann drastisch verschlechtern.

Ergebnis: Tabelle (stdout) + eval/out/antimeridian_check.csv
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.pipeline import load_cases                     # noqa: E402
from eval.metrics import haversine_error                 # noqa: E402
from experiments.exp_t6_defaults import (                # noqa: E402
    loo_pseudo_radii, estimate,
)

OUT = Path("eval/out")
OUT.mkdir(parents=True, exist_ok=True)
EPS = 10


def wrap(x):
    """Längengrad nach (−180, 180]."""
    return (x + 180.0) % 360.0 - 180.0


def true_span(lons):
    """Zirkuläre Längenspanne: 360° minus größte Lücke zwischen den Längengraden."""
    s = np.sort(np.mod(lons, 360.0))
    gaps = np.diff(np.concatenate([s, [s[0] + 360.0]]))
    return float(360.0 - gaps.max())


def main():
    cases = load_cases()
    loo = loo_pseudo_radii(cases)

    rows = []
    for c in cases:
        lons = np.array([p["lon"] for p in c["provenance"]])
        raw_span = float(lons.max() - lons.min())
        if raw_span <= 180.0:
            continue
        e_raw = haversine_error(estimate(c, "L1", "b", loo, eps=EPS), c["truth"])
        cc = copy.deepcopy(c)
        ref = cc["provenance"][0]["lon"]
        for p in cc["provenance"]:
            p["lon"] = ref + wrap(p["lon"] - ref)
        est = estimate(cc, "L1", "b", loo, eps=EPS)
        e_rec = haversine_error((est[0], wrap(est[1])), cc["truth"])
        rows.append({"ip": c["ip"],
                     "raw_span_deg": round(raw_span, 1),
                     "true_span_deg": round(true_span(lons), 1),
                     "err_roh_km": round(e_raw, 1),
                     "err_rezentriert_km": round(e_rec, 1),
                     "delta_km": round(e_rec - e_raw, 1)})

    df = pd.DataFrame(rows).sort_values("delta_km", ascending=False)
    df.to_csv(OUT / "antimeridian_check.csv", index=False)
    print(f"Anchors mit Rohgrad-Spanne > 180°: {len(df)} von {len(cases)}")
    print(f"davon wahre Spanne > 180° (keine kanonische Karte): "
          f"{int((df['true_span_deg'] > 180).sum())}")
    print(df.to_string(index=False))
    print(f"\nCSV: {OUT}/antimeridian_check.csv")


if __name__ == "__main__":
    main()
