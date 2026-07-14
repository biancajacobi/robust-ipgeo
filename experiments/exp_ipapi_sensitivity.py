#!/usr/bin/env python3
"""ipapi.co-Sensitivität der Headline-Konfiguration (L1·b, eps=10, linear).

Hintergrund (§3.1.3 / §4.3): Die 7 erfolgreichen ipapi.co-Beobachtungen
verschieben den Headline-Mittelwert von 172,9 km (ohne sie) auf 183,4 km.
Dieses Skript belegt den Mechanismus reproduzierbar:
  (a) Gesamt-Kennzahlen mit/ohne ipapi.co (Median, Mittel, Tail),
  (b) Anchor-genaue Differenzen der 7 betroffenen Anchors,
  (c) Gewichtsmassen-Tabelle des gekippten Anchors:
      4 von 6 effektiven Linien teilen denselben Ashburn-Default
      (W_C = 47,5 % ohne ipapi); die ipapi-Antwort (6,8 % Gewichtsmasse) hebt
      W_C auf 51,0 % -> gewichteter geometrischer Median kippt auf
      den Default (Fehler 18,8 -> 11.354,5 km). KEIN Gewichts-Fallback:
      ipapi erhaelt regulaeren Leave-one-out-Pseudo-Radius (27,9 km).

Ergebnis: Tabellen (stdout) + eval/out/t6_ipapi_sensitivity.csv
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
    loo_pseudo_radii, estimate, point_radius, line_weights_for,
)

OUT = Path("eval/out")
OUT.mkdir(parents=True, exist_ok=True)
EPS = 10  # Headline-Konfiguration (anchor_eps)


def headline_errors(cases):
    loo = loo_pseudo_radii(cases)
    return loo, {c["ip"]: haversine_error(estimate(c, "L1", "b", loo, eps=EPS),
                                          c["truth"]) for c in cases}


def summarize(errs):
    a = np.array(list(errs.values()))
    return dict(median_km=float(np.median(a)), mean_km=float(a.mean()),
                tail_pct=float(100 * (a > 100).mean()))


def main():
    cases = load_cases()
    ipapi_ips = [c["ip"] for c in cases
                 if any(p["source"] == "ipapi_co" for p in c["provenance"])]

    loo_full, err_full = headline_errors(cases)
    cases_no = copy.deepcopy(cases)
    for c in cases_no:
        c["provenance"] = [p for p in c["provenance"]
                           if p["source"] != "ipapi_co"]
    _, err_no = headline_errors(cases_no)

    rows = []
    for tag, e in (("mit_ipapi", err_full), ("ohne_ipapi", err_no)):
        rows.append({"zeile": tag, "anchor": "", **summarize(e)})
    for ip in ipapi_ips:
        rows.append({"zeile": "anchor_delta", "anchor": ip,
                     "median_km": "", "mean_km": "",
                     "tail_pct": "",
                     "err_mit": round(err_full[ip], 1),
                     "err_ohne": round(err_no[ip], 1),
                     "delta_km": round(err_full[ip] - err_no[ip], 1)})

    # Gewichtsmassen des Kipp-Anchors
    tipped = max(ipapi_ips, key=lambda ip: abs(err_full[ip] - err_no[ip]))
    c = next(c for c in cases if c["ip"] == tipped)
    lw = line_weights_for(c["provenance"], "L1")
    w = []
    for p, l in zip(c["provenance"], lw):
        r = point_radius(p, tipped, loo_full)
        w.append((p["source"], l / (r + EPS),
                  haversine_error((p["lat"], p["lon"]), c["truth"]), r))
    tot = sum(x[1] for x in w)
    wc_with = sum(x[1] for x in w if x[2] > 10_000) / tot
    tot_no = tot - next(x[1] for x in w if x[0] == "ipapi_co")
    wc_no = sum(x[1] for x in w
                if x[2] > 10_000 and x[0] != "ipapi_co") / tot_no

    print(f"Kipp-Anchor: {tipped}")
    print(f"{'Quelle':18s} {'Fehler_km':>10s} {'r_km':>7s} {'Anteil':>7s}")
    for src, wi, e, r in w:
        print(f"{src:18s} {e:10.1f} {r:7.1f} {100*wi/tot:6.1f}%")
    print(f"\nW_C (Default-Seite) mit ipapi: {100*wc_with:.1f} %"
          f"  |  ohne ipapi: {100*wc_no:.1f} %")
    rows.append({"zeile": "gewichtsmasse_default", "anchor": tipped,
                 "median_km": "", "mean_km": "", "tail_pct": "",
                 "wc_mit_pct": round(100 * wc_with, 1),
                 "wc_ohne_pct": round(100 * wc_no, 1)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t6_ipapi_sensitivity.csv", index=False)
    print(f"\nCSV: {OUT}/t6_ipapi_sensitivity.csv")
    for tag, e in (("mit ", err_full), ("ohne", err_no)):
        s = summarize(e)
        print(f"{tag} ipapi: Median {s['median_km']:.2f}  "
              f"Mittel {s['mean_km']:.1f}  Tail {s['tail_pct']:.1f} %")


if __name__ == "__main__":
    main()
