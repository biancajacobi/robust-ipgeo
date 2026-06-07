"""Brätz' klassen-/dichtebasiertes Schätzverfahren (zu untersuchendes Verfahren).

Referenz: Brätz, M. — "Entwicklung eines Schätzverfahrens zur Bestimmung
robuster Referenzwerte auf geringer Datenbasis unbekannter Güte"
(siehe references.bib, ``braetz``).

Das Verfahren läuft eindimensional pro Koordinate. Für einen 2D-Standort wird
es je einmal auf die Breiten- und die Längen-Reihe angewandt (``estimate``).

Schritte (rekonstruiert aus Braetz 2009; kritisch geprueft —
Auswertung in experiments/exp_braetz.py):
  1. Werte in k Klassen einteilen (empirisch gewählte Klassenzahl k).
  2. Dichte/Häufigkeit je Klasse bestimmen, dichteste Klasse(n) wählen.
  3. optional Fuzzy-Zugehörigkeit an Klassengrenzen (``fuzzy``).
  4. robusten Lageschätzer innerhalb der gewählten Klasse(n) bilden.
  5. Konfidenz/Streuungsmaß ableiten (Verteilungsannahme dokumentieren!).

Kritikpunkte, die als Experiment zu belegen sind: Nutzen der Fuzzy-Stufe,
Normalverteilungsannahme (Q-Q), Sensitivität gegenüber k, Validität der
Konfidenz, Zirkularität.
"""

from __future__ import annotations

import numpy as np


K_MAX = 30              # Brätz-Default (Konvergenz im Bereich k≈26–30, Tab. 4.1)
DENSITY_THRESHOLD = 2   # Hauptlauf: Brätz' Untergrenze aus dem Abbruchkriterium
                        # (Original 4 ist für n≈30 gedacht und bei n=8 zu hart → Sensitivity)


def braetz_1d(values, k_max: int = K_MAX, density_threshold: int = DENSITY_THRESHOLD,
              return_sequence: bool = False):
    """Brätz' Fuzzy-Dichte-Schätzer für eine 1D-Wertereihe (Braetz 2009, Kap. 4).

    Verfahren (1:1 nach Brätz 2009):
      1. Klassenzahl k = 1..k_max sukzessiv erhöhen; Spanne [min,max] in k gleich
         breite Intervalle, Werte je Intervall zählen.
      2. Pro k das dichteste Intervall (höchste Kardinalität) wählen und durch seine
         **Klassen-Mitte** repräsentieren -> ein Mitten-Wert je k.
      3. Abbruch, sobald das dichteste Intervall nur noch ≤1 Wert enthält (Intervalle
         zu klein -> Scheingenauigkeit; „Unschärfe" muss erhalten bleiben).
      4. Bereinigung: nur Mitten behalten, deren Intervall ≥ ``density_threshold``
         Werte hatte (Brätz entfernt lokale Dichte < 4; hier parametrisiert).
      5. Schätzwert = Mittelwert der bereinigten Mitten-Folge.

    Die Mitten-Folge ist (bei normalverteilter Stichprobe) approximativ normalverteilt
    um den Ort höchster Dichte — darauf beruht das Student-t-KI (s. ``braetz_ci``).

    Mit ``return_sequence`` zusätzlich die bereinigte Mitten-Folge (für Diagnose/KI).
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        raise ValueError("leere Wertereihe")
    lo, hi = float(v.min()), float(v.max())
    if hi == lo:                                   # entartet: alle Werte gleich
        return (lo, np.array([lo])) if return_sequence else lo

    mids, dens = [], []
    for k in range(1, k_max + 1):
        width = (hi - lo) / k
        idx = np.minimum(((v - lo) / width).astype(int), k - 1)
        counts = np.bincount(idx, minlength=k)
        max_count = int(counts.max())
        if max_count <= 1:                         # Abbruchkriterium (≤1 Wert/Klasse)
            break
        winner = int(counts.argmax())              # Tie: niedrigster Index (deterministisch)
        mids.append(lo + (winner + 0.5) * width)   # Klassen-Mitte
        dens.append(max_count)

    mids = np.asarray(mids, dtype=float)
    dens = np.asarray(dens, dtype=int)
    kept = mids[dens >= density_threshold]
    if kept.size == 0:                             # Fallback: nie leer zurückgeben
        kept = mids if mids.size else np.array([(lo + hi) / 2])
    est = float(kept.mean())
    return (est, kept) if return_sequence else est


def braetz_ci(values, alpha: float = 0.05, **kw):
    """(Schätzwert, KI-Halbbreite, Mitten-Folge) — Student-t auf der Mitten-Folge.

    Das ist Brätz' Konfidenz: die *interne* Sicherheit der Mitten-Folge (df = m−1),
    NICHT die Distanz zur Ground Truth — zentrale Unterscheidung für die T5-Kritik.
    """
    from scipy import stats
    est, seq = braetz_1d(values, return_sequence=True, **kw)
    m = seq.size
    if m < 2:
        return est, float("nan"), seq
    se = float(seq.std(ddof=1)) / np.sqrt(m)
    half = float(stats.t.ppf(1 - alpha / 2, df=m - 1)) * se
    return est, half, seq


def estimate(points, k_max: int = K_MAX, density_threshold: int = DENSITY_THRESHOLD) -> np.ndarray:
    """Standort-Schätzer (lat, lon): Brätz je Koordinate separat.

    Einheitliche Schnittstelle wie in ``estimators.baselines`` (fn(points) -> (2,)).
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"erwarte Form (n, 2) [lat, lon], erhalten {pts.shape}")
    lat = braetz_1d(pts[:, 0], k_max=k_max, density_threshold=density_threshold)
    lon = braetz_1d(pts[:, 1], k_max=k_max, density_threshold=density_threshold)
    return np.array([lat, lon])
