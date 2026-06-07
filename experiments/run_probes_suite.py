"""Treiber: die GESAMTE Experiment-Suite auf dem Probe-Datensatz laufen lassen.

NICHT-INVASIV: aendert KEINE der bestehenden Experiment-Dateien. Setzt nur zur
Laufzeit das (hartkodierte) ``OUT``-Modulglobal der betroffenen Experimente auf den
dataset-aware ``report.OUT_DIR`` (= eval/out_probes) und ruft deren Entry-Funktion auf.

Sicherheits-Eigenschaften:
  * Lauf NUR mit GEOIP_DATASET=probes (sonst Abbruch) -> liest probes.csv /
    observations_probes.csv, schreibt eval/out_probes/.
  * Kein Experiment schreibt jemals eine Provenance-Kette (nur die fetch_*-Skripte
    und exp_whois_mechanism tun das; letzteres ist hier bewusst NICHT enthalten).
    Die Anchor-Hashkette (data/provenance.jsonl) wird also nicht beruehrt.

Aufruf:  GEOIP_DATASET=probes python experiments/run_probes_suite.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from data import store          # noqa: E402
from eval import report         # noqa: E402

if store.DATASET != "probes":
    print("Abbruch: bitte mit  GEOIP_DATASET=probes python experiments/run_probes_suite.py  starten.")
    print(f"  (aktuell DATASET={store.DATASET!r} -> wuerde die Anchor-Artefakte ueberschreiben)")
    sys.exit(2)

OUTP = report.OUT_DIR
OUTP.mkdir(parents=True, exist_ok=True)

# Experimente importieren (kein Code laeuft beim Import; alles unter __main__-Guard)
import exp_accuracy, exp_source_correlation, exp_bootstrap_ci          # noqa: E402
import exp_t6_defaults as t6, plot_t6, exp_t6_hub_rule                 # noqa: E402
import exp_label_calibration, exp_contamination, exp_samplesize        # noqa: E402
import exp_stratification, exp_braetz, exp_support_concentration, exp_spread_measure  # noqa: E402

# Hartkodiertes OUT der betroffenen Module auf den Probe-Output umlenken
for m in (t6, plot_t6, exp_braetz, exp_contamination, exp_samplesize,
          exp_stratification, exp_spread_measure, exp_bootstrap_ci):
    m.OUT = OUTP


def _t6_main():
    cases = t6.load_cases()
    loo = t6.loo_pseudo_radii(cases)
    eps = min(t6.EPS_GRID, key=lambda e: abs(e - float(np.median([loo[s]["_global"] for s in loo]))))
    t6.run_point_estimators(cases, loo, eps)
    t6.run_sensitivity_lines(cases, loo, eps)
    t6.run_eps_grid(cases, loo, eps)
    t6.run_ff1_baseline(cases, loo, eps)
    t6.run_confidence_matrix(cases, loo, eps)
    t6.run_confidence_recall(cases, loo, eps)


def _plot_main():
    cases = plot_t6.load_cases()
    loo = plot_t6.t6.loo_pseudo_radii(cases)
    eps = min(plot_t6.t6.EPS_GRID, key=lambda e: abs(e - np.median([loo[s]["_global"] for s in loo])))
    plot_t6.fig_ecdf(cases, loo, eps)
    plot_t6.fig_heatmap(cases, loo, eps)
    plot_t6.fig_forest(cases, loo, eps)


# Reihenfolge wie im Anhang-Runbook; t6_defaults VOR label_calibration (liest dessen CSV)
STEPS = [
    ("E1  Grundgenauigkeit", exp_accuracy.run),
    ("    Quellen-Korrelation", exp_source_correlation.run),
    ("T1  Bootstrap-CIs", exp_bootstrap_ci.main),
    ("T6  Punktschaetzer + Matrix", _t6_main),
    ("T6  Abbildungen", _plot_main),
    ("T6  Hub-Regel", exp_t6_hub_rule.main),
    ("T6  Label-Kalibrierung", exp_label_calibration.run),
    ("E2  Kontamination/Breakdown", exp_contamination.run),
    ("E3  Stichprobengroesse", exp_samplesize.run),
    ("E4  Stratifizierung", exp_stratification.main),
    ("E5/T5 Braetz", exp_braetz.main),
    ("T6  Stuetz-Konzentration", exp_support_concentration.run),
    ("    Streuungsmass-Auswahl", exp_spread_measure.run),
]


def main():
    print("#" * 80)
    print(f"# PROBE-SUITE  ->  {OUTP}   (Anchor-Artefakte unberuehrt)")
    print("#" * 80)
    ok, fail = [], []
    for name, fn in STEPS:
        print("\n" + "=" * 80 + f"\n>>> {name}\n" + "=" * 80)
        try:
            fn()
            ok.append(name)
        except Exception as e:
            traceback.print_exc()
            fail.append((name, repr(e)))
    print("\n" + "#" * 80)
    print(f"# FERTIG: {len(ok)} ok, {len(fail)} fehlgeschlagen  ->  {OUTP}")
    for n, e in fail:
        print(f"#   FAIL  {n}: {e}")
    print("#" * 80)


if __name__ == "__main__":
    main()
