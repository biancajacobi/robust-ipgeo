#!/usr/bin/env python3
"""Sensitivität der Linien-Definition: reallyfreegeoip in der MaxMind-Linie?

Hintergrund (§3.2.4): reallyfreegeoip.org folgt der MaxMind-Abstammung mit
abweichendem Datenstand (48 % identisch, 17 % > 100 km divergent). Dieses
Skript prüft, ob der volle Linien-Kollaps die Headline-Ergebnisse verzerrt:

  voll          : Status quo — rfg gehört zur MaxMind-GeoLite-Linie (k=3)
  eigene_linie  : rfg zählt als eigene Linie (Familie nur maxmind+geojs, k=2)
  bedingt       : rfg nur dort in der Familie, wo Koordinaten < 1 km an MaxMind

Befund: Median/Tail praktisch invariant; Mittelwert-Differenzen sind reine
Fern-Tail-Umsortierung bereits gescheiterter Anchors; ohne Kollaps kippt
mindestens ein zuvor korrekter Anchor durch die volle Replikat-Stimme.

Ergebnis: Tabellen (stdout) + eval/out/line_sensitivity_rfg.csv
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
EPS = 10          # Headline-Konfiguration
SPLIT_KM = 1.0    # Schwelle für den bedingten Kollaps
RFG = "reallyfreegeoip"
MM = "maxmind_geolite2"


def errors(cases, loo):
    return {c["ip"]: haversine_error(estimate(c, "L1", "b", loo, eps=EPS),
                                     c["truth"]) for c in cases}


def summarize(errs):
    a = np.array(list(errs.values()))
    return dict(median_km=round(float(np.median(a)), 2),
                mittel_km=round(float(a.mean()), 1),
                tail_pct=round(float(100 * (a > 100).mean()), 1))


def variant_own_line(cases):
    cs = copy.deepcopy(cases)
    for c in cs:
        for p in c["provenance"]:
            if p["source"] == RFG:
                p["lineage"] = "rfg_eigene_linie"
    return cs


def variant_conditional(cases):
    cs = copy.deepcopy(cases)
    n_split = 0
    for c in cs:
        src = {p["source"]: p for p in c["provenance"]}
        mm, rf = src.get(MM), src.get(RFG)
        if mm and rf and haversine_error((mm["lat"], mm["lon"]),
                                         (rf["lat"], rf["lon"])) >= SPLIT_KM:
            rf["lineage"] = "rfg_eigene_linie"
            n_split += 1
    return cs, n_split


def main():
    cases = load_cases()
    loo = loo_pseudo_radii(cases)   # Radien identisch für alle Varianten

    e_full = errors(cases, loo)
    e_own = errors(variant_own_line(cases), loo)
    cond_cases, n_split = variant_conditional(cases)
    e_cond = errors(cond_cases, loo)

    rows = []
    for name, e in (("voll", e_full), ("eigene_linie", e_own),
                    (f"bedingt_<{SPLIT_KM:g}km", e_cond)):
        rows.append({"variante": name, **summarize(e)})
        print(f"{name:16s} {rows[-1]}")

    # Fern-Tail-Charakter der Differenz voll vs. eigene Linie
    d = {ip: e_own[ip] - e_full[ip] for ip in e_full}
    big = {ip: x for ip, x in d.items() if abs(x) > 10}
    flips = [(ip, e_full[ip], e_own[ip]) for ip in big
             if e_full[ip] < 100 < e_own[ip]]
    print(f"\n|Delta|>10 km: {len(big)} von {len(d)} Anchors "
          f"(Median |Delta| {np.median(np.abs(list(d.values()))):.3f} km)")
    for ip, ef, eo in flips:
        print(f"  KIPPT ohne Kollaps: {ip}  {ef:.1f} -> {eo:.1f} km")
        rows.append({"variante": "kipp_anchor_ohne_kollaps", "anchor": ip,
                     "err_voll_km": round(ef, 1), "err_eigene_km": round(eo, 1)})
    rows.append({"variante": "delta_stat",
                 "n_delta_gt10km": len(big),
                 "median_abs_delta_km": round(float(np.median(np.abs(list(d.values())))), 3)})

    pd.DataFrame(rows).to_csv(OUT / "line_sensitivity_rfg.csv", index=False)
    print(f"\nCSV: {OUT}/line_sensitivity_rfg.csv  (bedingt: {n_split} IPs getrennt)")


if __name__ == "__main__":
    main()
