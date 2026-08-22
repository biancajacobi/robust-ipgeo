"""Driver: run the ENTIRE experiment suite on a NON-anchor dataset.

NON-INVASIVE: changes NONE of the existing experiment files. Only redirects, at
runtime, the (hard-coded) ``OUT`` module global of the affected experiments to the
dataset-aware ``report.OUT_DIR`` (= eval/out_<dataset>) and calls their entry function.

Safety properties:
  * runs ONLY with GEOIP_DATASET=<non-anchors> (aborts otherwise) -> reads
    <dataset>.csv / observations_<dataset>.csv, writes eval/out_<dataset>/.
  * No experiment ever writes a provenance chain (only the fetch_* scripts
    and exp_whois_mechanism do that; the latter is deliberately NOT included here).
    The anchor hash chain (data/provenance.jsonl) is therefore untouched.

Framing: this is the DESCRIPTIVE mirroring of the suite onto a target population
(dashboard "next to the anchors"). It does NOT replace the transfer experiments
with frozen anchor calibration (exp_pool_transfer, exp_giga_transfer): all
calibrations here are refitted within the population.

Invocations:
  GEOIP_DATASET=probes      python experiments/run_probes_suite.py
  GEOIP_DATASET=probes_pool python experiments/run_probes_suite.py
  GEOIP_DATASET=giga        python experiments/run_probes_suite.py
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

if store.DATASET == "anchors":
    print("Abort: please start with  GEOIP_DATASET=<probes|probes_pool|giga|...>")
    print(f"  (currently DATASET={store.DATASET!r} -> would overwrite the anchor artifacts)")
    sys.exit(2)
if not store.GROUND_TRUTH_CSV.exists():
    print(f"Abort: {store.GROUND_TRUTH_CSV} does not exist (fetch the dataset first).")
    sys.exit(2)

OUTP = report.OUT_DIR
OUTP.mkdir(parents=True, exist_ok=True)

# import the experiments (no code runs at import time; everything under a __main__ guard)
import exp_accuracy, exp_source_correlation, exp_bootstrap_ci          # noqa: E402
import exp_t6_defaults as t6, plot_t6, exp_t6_hub_rule                 # noqa: E402
import exp_label_calibration, exp_contamination, exp_samplesize        # noqa: E402
import exp_stratification, exp_braetz, exp_support_concentration, exp_spread_measure  # noqa: E402

# redirect the hard-coded OUT of the affected modules to the probe output
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
    t6.run_rq1_baseline(cases, loo, eps)
    # NOTE: artifact/function name kept for stability; 'confidence' is the legacy name of the 2D predecessor label (paper terminology: predecessor label / risk signal).
    t6.run_confidence_matrix(cases, loo, eps)
    t6.run_confidence_recall(cases, loo, eps)


def _plot_main():
    cases = plot_t6.load_cases()
    loo = plot_t6.t6.loo_pseudo_radii(cases)
    eps = min(plot_t6.t6.EPS_GRID, key=lambda e: abs(e - np.median([loo[s]["_global"] for s in loo])))
    plot_t6.fig_ecdf(cases, loo, eps)
    plot_t6.fig_heatmap(cases, loo, eps)
    plot_t6.fig_forest(cases, loo, eps)


# order as in the runbook; t6_defaults BEFORE label_calibration (reads its CSV)
STEPS = [
    ("E1  base accuracy", exp_accuracy.run),
    ("    source correlation", exp_source_correlation.run),
    ("T1  bootstrap CIs", exp_bootstrap_ci.main),
    ("T6  point estimators + matrix", _t6_main),
    ("T6  figures", _plot_main),
    ("T6  hub rule", exp_t6_hub_rule.main),
    ("T6  label calibration", exp_label_calibration.run),
    ("E2  contamination/breakdown", exp_contamination.run),
    ("E3  sample size", exp_samplesize.run),
    ("E4  stratification", exp_stratification.main),
    ("E5/T5 Braetz", exp_braetz.main),
    ("T6  support concentration", exp_support_concentration.run),
    ("    spread-measure selection", exp_spread_measure.run),
]


def main():
    print("#" * 80)
    print(f"# SUITE on DATASET={store.DATASET}  ->  {OUTP}   (anchor artifacts untouched)")
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
    print(f"# DONE: {len(ok)} ok, {len(fail)} failed  ->  {OUTP}")
    for n, e in fail:
        print(f"#   FAIL  {n}: {e}")
    print("#" * 80)


if __name__ == "__main__":
    main()
