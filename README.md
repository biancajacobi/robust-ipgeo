# robust-ipgeo

**Robust reference-location estimation from multiple heterogeneous IP-geolocation
sources of unknown quality — with ground-truth evaluation, a per-estimate confidence
measure, and a tamper-evident provenance chain.**

## What this is

Free IP-geolocation databases disagree — often by hundreds of kilometres — for the
same IP address, and are only moderately accurate at city level. This project studies
how to robustly aggregate several unreliable sources into a single reference location,
how to qualify each estimate with a confidence measure, and how to keep the whole
pipeline forensically documentable.

It is a self-contained, reproducible reference implementation with result artifacts. Individual IP
addresses are not included; see *Data & privacy* below.

## Approach

- **Estimators** (continuous 2-D): coordinate mean, component-wise median, trimmed
  mean, geometric median (Weiszfeld / L1-median), and the headline
  **accuracy-radius- and provider-line-weighted geometric median**.
- **Provider-line weighting** collapses correlated sources (e.g. databases that resell
  the same upstream data) so they cannot dominate the aggregate — without discarding them.
- **Confidence measure `S`**: the line-weighted *support concentration* — the share of
  effective, line-deduplicated source mass within a 50 km core around the estimate;
  out-of-fold calibrated as a risk score for aggregation failure.
- **Comparison method**: a class-/density-based estimator after Brätz, re-implemented and
  critically examined.
- **Ground truth**: RIPE Atlas anchors (primary) plus an exploratory RIPE Atlas *probe*
  stress test (residential / NAT / mobile). Error metric: Haversine distance.

## Repository layout

```
estimators/   aggregation estimators (baselines + Brätz)
eval/         metrics (Haversine), evaluation pipeline, reporting
eval/out/     result tables (CSV) and figures (PNG) — anchors
eval/out_probes/  result tables/figures — probe stress test
experiments/  studies T1–T6, RIPE-Atlas-probe comparison + sensitivity/edge-case checks
data/         data acquisition (RIPE Atlas, geolocation sources), default detection, provenance store
tests/        unit tests (46, pytest)
notebooks/    results dashboard (reads eval/out, runs without raw data)
docs/         data sources & experiment mapping
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- The **pseudonymized input datasets** are included under `data/cache/` (`anchors.csv`,
  `observations.csv`, `probes.csv`, `observations_probes.csv`, …): IP addresses, RIPE node
  IDs and hostnames are replaced/removed (stable pseudonyms `node_NNNNN`), coordinates
  retained. The experiment scripts in `experiments/` therefore run **directly** on them
  (mapping in `docs/experimente.md`).
- Result tables and figures are also provided in `eval/out/` (anchors) and `eval/out_probes/`
  (probe stress test); the dashboard `notebooks/explore.ipynb` runs on those alone.
- To re-fetch the raw data from scratch: copy `.env.example` to `.env`, add the
  offline-database download keys, then run the `data/fetch_*.py` scripts.
- **Dataset switch**: set `GEOIP_DATASET=probes` to run the pipeline against the probe
  dataset (separate cache/output), e.g.
  `GEOIP_DATASET=probes python experiments/run_probes_suite.py`.

## Selected findings

- The accuracy-radius- and line-weighted geometric median beats most single sources in
  median error and degrades gracefully; the geometric median stays robust up to an
  effective contamination of ≈ 37.5 % (breakdown between 37.5 % and 50 %), while the
  naive mean breaks immediately.
- The line-weighted support concentration `S` triples the out-of-fold forecast skill of
  an earlier two-axis label and surfaces the majority of aggregation failures ex ante
  (recall 72 % → 92 % at a 41 % flag rate). Under coordinated contamination, hijacked
  estimates stay below the flag threshold up to α = 0.5 — the guard has its own, higher
  breakdown (≈ 0.83 line mass).
- Sensitivity and edge-case checks are scripted and artifact-backed (`docs/experimente.md`):
  ipapi.co inclusion (a single weight-mass knife-edge anchor), the `reallyfreegeoip`
  line definition (same lineage, differing data state), antimeridian handling, and
  Bonferroni-adjusted bootstrap CIs.
- The Brätz comparison method is dominated by the robust estimators, and its confidence
  interval is badly miscalibrated — driven by correlated source replicates.
- A probe stress test confirms an access-class difficulty gradient
  (data centre < home/NAT < mobile); the genuinely hard mobile/CGNAT population remains
  beyond this test (see `docs/datenquellen.md`).

## Provenance / chain of custody

The acquisition layer (`data/store.py`) implements an append-only, **tamper-evident
SHA-256 hash chain**: each fetch is recorded with source, version, timestamp and content
hashes, every entry chained to its predecessor (`store.verify_chain()`; CLI
`python data/fetch_anchors.py --verify`). This establishes internal integrity of the
acquisition record; it is not a substitute for an external timestamping authority. The
concrete ledger files and raw evidence attest the original (non-redacted) corpus and are
therefore **not included** in this privacy-preserving release.

## Data & privacy

- Input data are derived from **RIPE Atlas**, public network-measurement infrastructure.
  **Individual IP addresses are not included in this release** — node identifiers are
  pseudonymized. Source coordinates (public infrastructure locations) are retained so the
  pipeline remains reproducible.
- Only public infrastructure IPs are processed; no personal data.

## License

Code: MIT (see [`LICENSE`](LICENSE)). Data are derived from RIPE Atlas and subject to
RIPE Atlas terms of use.
