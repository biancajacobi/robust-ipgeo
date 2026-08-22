# robust-ipgeo

**Robust reference-location estimation from multiple heterogeneous IP-geolocation
sources of unknown quality — with ground-truth evaluation, a per-estimate risk
signal, and a tamper-evident provenance chain.**

## What this is

Free IP-geolocation databases disagree — often by hundreds of kilometres — for the
same IP address, and are only moderately accurate at city level. This project studies
how to robustly aggregate several unreliable sources into a single reference location,
how to qualify each estimate with a risk signal, and how to keep the whole
pipeline forensically documentable.

It is a self-contained, reproducible reference implementation with result artifacts. Individual IP
addresses are not included; see *Data & privacy* below.

## Approach

- **Estimators** (continuous 2-D): coordinate mean, component-wise median, trimmed
  mean, geometric median (Weiszfeld / L1-median), and the headline
  **radius- and provider-line-weighted geometric median**.
- **Provider-line weighting** collapses correlated sources (e.g. databases that resell
  the same upstream data) so they cannot dominate the aggregate — without discarding them.
- **Risk signal `S`**: the line-weighted *support concentration* — the share of
  effective, line-deduplicated source mass within a 50 km core around the estimate;
  out-of-fold calibrated as a risk score for aggregation failure.
- **Comparison method**: a class-/density-based estimator after Braetz, re-implemented and
  critically examined.
- **Ground truth**: RIPE Atlas anchors (primary), an exploratory RIPE Atlas *probe*
  stress test (residential / NAT / mobile), an 8,896-probe external scale-up pool, and
  an end-user stress test on UNICEF Giga school connections (19 countries, Global
  South; reproduction via the official Giga APIs — no Giga data ships with this repo).
  Error metric: Haversine distance.

## Repository layout

```
estimators/   aggregation estimators (baselines + Braetz)
eval/         metrics (Haversine), evaluation pipeline, reporting
eval/out/     result tables (CSV) and figures (PNG) — anchors, plus the
              transfer results (pool `*_8src`, Giga, hub signal, freshness)
eval/out_probes/  result tables/figures — probe stress test
eval/out_<ds>/    full suite dashboards per dataset (reproducible via
                  GEOIP_DATASET=<ds> experiments/run_probes_suite.py)
experiments/  studies T1–T9 + RIPE-Atlas-probe comparison
data/         data acquisition (RIPE Atlas, geolocation sources), default detection, provenance store
maltego/      local Maltego transforms (consensus + S per IP) + lineage checker
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
  `observations.csv`, `probes.csv`, `observations_probes.csv`, `probes_pool.csv`,
  `observations_probes_pool.csv`, …): IP addresses, RIPE node
  IDs and hostnames are replaced/removed (stable pseudonyms `node_NNNNN`), coordinates
  retained. The experiment scripts in `experiments/` therefore run **directly** on them
  (mapping in `docs/experiments.md`).
- Result tables and figures are also provided in `eval/out/` (anchors) and `eval/out_probes/`
  (probe stress test); the dashboard `notebooks/explore.ipynb` runs on those alone.
- To re-fetch the raw data from scratch: copy `.env.example` to `.env`, add the
  offline-database download keys, then run the `data/fetch_*.py` scripts.
- **Dataset switch**: set `GEOIP_DATASET=<name>` to run the pipeline against another
  dataset (separate cache/output `eval/out_<name>`), e.g.
  `GEOIP_DATASET=probes_pool python experiments/run_probes_suite.py`. The `giga`
  dataset must be fetched first (`data/fetch_giga.py`, own API keys required).

## Selected findings

- The radius- and line-weighted geometric median beats most single sources in median
  error and degrades gracefully; the geometric median stays robust up to ~50 % coordinated
  contamination, while the naive mean breaks immediately.
- The line-weighted support concentration `S` triples the out-of-fold calibration skill of
  an earlier two-axis label and surfaces the majority of aggregation failures ex ante.
- The Braetz comparison method is dominated by the robust estimators, and its confidence
  interval is badly miscalibrated — driven by correlated source replicates.
- A probe stress test confirms an access-class difficulty gradient
  (data centre < home/NAT < mobile); the end-user regime is measured directly on
  UNICEF Giga school connections (see below), while roaming mobile/CGNAT populations
  remain insufficiently covered (see `docs/data-sources.md`).

## Recent additions (line ablations, fusion baselines, external scale-up)

Newer experiments harden and extend the above findings (details and key numbers in
`docs/experiments.md`):

- **Line ablation** (`experiments/exp_min_lines.py`, `exp_min_lines_s.py`): exhaustive
  subsets of the six fully covered provider lines. The point estimator's median
  saturates from about three lines and the worst-case tail falls monotonically with
  more lines; the risk signal `S` degenerates with one line, is a lottery with
  two, and is usable from three lines in *every* subset — empirically covering the
  operating rule `MIN_LINES_FOR_S=3`.
- **Trust-vote fusion baselines** (`exp_trust_weighted_fusion.py`,
  `exp_trust_frozen_symmetric.py`): a trust-weighted city vote (out-of-fold trust
  weights) as the strongest fusion baseline. On the anchor set it is statistically
  indistinguishable from the line-weighted geometric median; under transfer to the
  probe population it is significantly worse in mean and tail — also in the symmetric
  frozen-vs-frozen comparison that removes the freezing penalty.
- **Cross-fitting control** (`exp_crossfit_radii.py`): recomputing the calibration
  track records per outer fold instead of globally reproduces the published
  out-of-fold numbers (slightly better, +0.210 → +0.214) — they do not depend on the
  global shortcut.
- **External scale-up** (`exp_pool_transfer.py`, `exp_pool_asn_check.py`): the frozen
  pipeline transfers to `probes_pool`, a pool of 8,896 never-seen RIPE Atlas probes
  (7 sources / 5 lines). Median and tail replicate at 18× scale, frozen vs. fresh is
  practically identical, and `S` keeps its skill. A stratification by auto-GeoIP
  metadata exposes a circularity signal; the stratum *without* auto-GeoIP metadata is
  reported as the honest headline. ASN-cluster bootstrap CIs confirm the pool numbers.
  With `--with-ipwhois` the sixth line (completed after the pre-specified analysis)
  improves the point estimates but makes the freeze delta detectable — the five-line
  analysis stays primary, the completion is reported as an addendum (`*_8src.csv`).
- **End-user stress test on UNICEF Giga school connections** (`data/fetch_giga.py`,
  `exp_giga_transfer.py`, `exp_giga_mechanism.py`, `exp_giga_sources.py`): 2,127
  PTR-verified (IP, school) pairs across 19 countries, school GPS from government
  records (GeoIP-independent). The frozen pipeline degrades to a ~122 km median and a
  ~55 % >100 km tail — consistent with the Global-South failure band reported by
  Nabi et al. and with coarse operator-block geolocation (many school IPs collapse
  onto a handful of provider locations). On the identical pairs every single source
  and every fusion baseline sits in the same band: the input portfolio carries no
  locally resolved evidence, robust aggregation cannot create information. The raw
  anchor calibration of `S` is actively misleading after this population shift; a
  per-population Platt refit restores a modest, calibrated warning signal (also under
  school-/ASN-grouped cross-validation). Giga data are ODbL (source: Giga/UNICEF and
  contributors) and are **not redistributed** here — reproduction needs own API keys.
- **Hybrid baseline: single-source point + consensus risk signal**
  (`exp_hybrid_single_source.py`): the CV-median-best single source as point
  estimator with the line-weighted support concentration of the remaining lines
  around that point as the risk signal. The warning signal works (out-of-fold BSS
  +0.302 on anchors against the hybrid's own >100 km misses, frozen transfer to
  the probes +0.169) — what the hybrid gives up is the estimator side: breakdown
  point 0 of the chosen source and undiluted exposure to one provider's drift.
- **Miss-threshold sensitivity** (`exp_miss_threshold_sensitivity.py`): the
  skill advantage of `S` over the two-dimensional predecessor label is not an
  artifact of the 100 km miss definition — for thresholds between 25 and 250 km
  the out-of-fold gap stays between +0.10 and +0.23; only an extreme 500 km
  definition (56 misses) closes it.
- **Line-collapsed city vote** (variant inside `exp_city_vote.py`): a stepwise
  decomposition of the vote's deficit. Naive vote -> line-collapsed vote isolates
  the replicate collapse (significant only in the mean, -31.2 km); collapsed
  vote -> L1*a isolates discreteness (small and not significant on all three
  metrics); L1*a -> L1*b isolates the accuracy-radius weighting. The deficit is
  driven by replicate dominance and the missing radius weighting — not by
  discreteness itself.
- **Hub-signal quick check** (`exp_hub_signal.py`): research-idea probe whether
  per-source hub-coordinate lexica (coordinates many distinct IPs collapse onto) add
  warning signal beyond `S`; split design, framed strictly as future work.
- **Freshness audit** (`exp_freshness_audit.py`): ground-truth vs. source-fetch
  timestamps for every dataset and pair-age strata for Giga — staleness/churn ruled
  out as a driver of the reported errors.
- **Lineage checker** (`maltego/check_lineage.py`): coverage listing and redundancy
  check (pairwise distance + identical rate on shared IPs) for vetting newly added
  sources before assigning them a provider line.

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
- The **UNICEF Giga** evaluation uses school-connectivity measurements (ODbL). Client
  IPs are recovered from the hostname field the public Giga API itself exposes; they
  identify institutional school connections rather than individuals, are processed
  locally for aggregate error statistics only, and are **not part of this release** —
  neither raw measurements nor derived per-IP rows ship here, only aggregate result
  tables. Reproduction requires own Giga API keys (`data/fetch_giga.py`).

## License

Code: MIT (see [`LICENSE`](LICENSE)). Data are derived from RIPE Atlas and subject to
RIPE Atlas terms of use; the Giga-based result tables are derived from data made
available by the Giga project under the Open Database License (ODbL) — source: Giga
(UNICEF) and contributors.
