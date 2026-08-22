# Experiments — file → study → research question → key finding → output

Overview of which file runs which study, which research question (RQ) it serves,
what comes out of it (key finding), and which artifacts it produces. All experiments
live in `experiments/` and use the measurement infrastructure from `eval/` + the
estimators from `estimators/` on the data from `data/`. The detailed interpretation
is given in the accompanying paper; this repo is self-contained and reproducible.

| File | Study | RQ | Key finding | Output (`eval/out/`) |
|---|---|---|---|---|
| `exp_accuracy.py` | **E1** baseline accuracy | RQ1 | median per source/estimator; geometric median ~5.5 km, naive mean ~38 km | `e1_accuracy.csv`, `e1_accuracy_cdf.png` |
| `exp_source_correlation.py` | source redundancy / effective lines + family drift | methodology | `maxmind_geolite2` ↔ `geojs` 0.0 km (bit-identical) / ↔ `reallyfreegeoip` 0.05 km median → **one** line; all remaining pairs ≥ 0.7 km (n = 1,066). rfg = same lineage, diverging data snapshot (48 % identical, 17 % > 100 km) | `source_correlation.{csv,png}`, `source_family_drift.csv` |
| `exp_bootstrap_ci.py` | **T1** paired bootstrap CIs (B = 10,000, seed 0) | RQ1 | aggregation `L1·b` beats 7/8 single sources significantly in the median; Bonferroni-adjusted (α = 0.05/8) all CIs still exclude 0 | `t1_bootstrap_ci.csv`, `t1_bootstrap_meantail_ci.csv` |
| `exp_t6_defaults.py` | **T6** line scale L0/L1/L2 × aggregator + ε grid + 2-D predecessor-label matrix | RQ1, RQ4 | 12 configs; `L1·b` best (mean 183 km, tail 12.0 %); line collapse −20 %; predecessor-label recall 72 % on misses | `t6_point_estimators.csv`, `t6_confidence_matrix.csv`, `t6_epsilon_grid.csv` |
| `plot_t6.py` | visualization for T6 (ECDF, heatmap, forest) | — | figures for T6 | `t6_ecdf.png`, `t6_confidence_heatmap.png`, `t6_forest.png` |
| `exp_t6_hub_rule.py` | **T6** hub-rule sensitivity (appendix) | RQ4 | majority (≥ 0.5) more discriminative than "at least one"; the latter flags 63.5 % for only ~9 % extra precision | (stdout) |
| `exp_whois_mechanism.py` | validation of the DB-IP↔IP2Location error coupling (RDAP) | mechanism for T6/T5 | shared line only when both sides show a default (justifies the conditional L2) | (RDAP cache + provenance) |
| `exp_label_calibration.py` | **T6** predecessor-label calibration (out-of-fold, 10-fold) | RQ4 | 2-D label usably calibrated (ECE 0.019), discrimination moderate (BSS +0.068) | `label_calibration_*.csv`, `label_calibration.png` |
| `exp_spread_measure.py` | pre-study on spread/concentration measures | RQ4 | pre-selection of measure candidates for S | `spread_measure_comparison.csv` |
| `exp_support_concentration.py` | **T6** support concentration S (out-of-fold) | RQ4 | S triples forecast skill (BSS +0.068 → +0.210, AP 0.25 → 0.42); triage at matched flag rate (41 %): recall on misses 72 → ~91 % unbiased (tie band 86–94 %, since S is massively tied at the threshold; the 92 % reported in the paper corresponds to one concrete tie-breaking); strict threshold: 86 % recall at 29 % flag rate | `support_concentration_oof.csv`, `support_concentration_triage.csv`, `support_concentration.png` |
| `exp_s_under_contamination.py` | guard limit under coordinated contamination | RQ4 | hijacked estimates stay below the flag threshold (0 % unflagged up to α = 0.5; 1.6 % at 0.6); guard breakdown ~0.83 line mass vs. 0.5 for the estimator | `s_under_contamination.{csv,png}` |
| `exp_miss_threshold_sensitivity.py` | is the S skill advantage an artifact of the 100 km miss definition? OOF BSS of S vs. predecessor label for thresholds 25-500 km | RQ4 | robust across 25-250 km (gap +0.10 to +0.23, S itself +0.16 to +0.31, maximum at 50 km = the measure's city scale); only an extreme 500 km definition (56 misses) weakens both predictors and closes the gap; tau=100 reproduces +0.210/+0.068 exactly | `miss_threshold_sensitivity.csv` |
| `exp_ipapi_sensitivity.py` | ipapi.co sensitivity of the headline | RQ1 | the entire mean difference 172.9 ↔ 183.4 km comes from ONE anchor at the weight-mass boundary (47.5 → 51.0 %) | `t6_ipapi_sensitivity.csv` |
| `exp_line_sensitivity.py` | line sensitivity `reallyfreegeoip` | methodology | own/conditional rfg line barely changes median/tail; without the collapse 6 anchors flip into the tail (up to 31 → 1,783 km) | `line_sensitivity_rfg.csv` |
| `exp_antimeridian.py` | antimeridian edge case | methodology | 17 cases with raw-degree span > 180°, all of them default outliers (raw degrees overstate → pro-robust); re-centering not well-defined for 4 cases > 180° true span | `antimeridian_check.csv` |
| `exp_contamination.py` | **E2** contamination / breakdown (`--replot`) | RQ2 | geometric median robust up to α_eff ≈ 37.5 %; naive mean breaks from 10 % | `e2_breakdown.{csv,png}` |
| `exp_samplesize.py` | **E3** sample size n (`--replot`) | RQ3 | robust estimators ~constant (n = 3..7); naive mean deteriorates (17 → 35 km) | `e3_samplesize.{csv,png}` |
| `exp_min_lines.py` | how many effective lines does L1·b need? (exhaustive, 63 subsets of the 6 fully covered lines) | RQ3 | median saturates from k ≈ 3 (every subset 3.9–6.1 km, full 4.13); worst-case tail falls monotonically 35.5 % (k=1) → 11.9 % (k=6); every subset with tail < 10 % contains ipinfo (not knowable ex ante); pool-relevant 5-line subset without ipwhois: 4.29 km / 13.2 % | `min_lines.csv` |
| `exp_min_lines_s.py` | does S hold up with fewer lines? OOF BSS per line subset (S counterpart to `exp_min_lines`) | RQ4 | k=1: S degenerates (constant ≈ 1, BSS ≈ 0); k=2: lottery (worst +0.02, best +0.63 — two ipinfo pairs as upward outliers, 11/15 below +0.15); from k=3: EVERY subset BSS ≥ +0.10 (worst: +0.101 at k=4) → operating rule MIN_LINES_FOR_S=3 empirically covered (worst-case argument); k=6 reproduces the headline +0.210 | `min_lines_s.csv` |
| `exp_trust_weighted_fusion.py` | trust-weighted city vote (ivanov2022 analogue, OOF trust) as strongest fusion baseline | RQ1/RQ3 | on anchors n.s. vs. L1·b (4.3/187.8/12.5 — learned weights imitate the line collapse implicitly); asymmetric transfer (frozen vote vs. fresh aggregation): mean −20.5/tail −3.8 sig. | `trust_weighted_fusion.csv` |
| `exp_trust_frozen_symmetric.py` | SYMMETRIC transfer: frozen vote vs. frozen aggregation (fairness control: symmetry) | RQ1/RQ3 | finding holds after removing the freezing penalty: probes mean **−13.0** [−25.3;−3.8] / tail **−2.8 pp** [−4.8;−1.0] sig.+ASN-robust, median n.s.; holdout/mobile n.s. — the paper cites these numbers | `trust_frozen_symmetric.csv` |
| `exp_city_vote.py` | discrete city-level majority vote (naive + line-collapsed variant) as fusion baseline | RQ1/RQ3 | naive vote 5.1/248.9/14.4 loses all three metrics significantly to L1*b (median -0.93, mean -65.5, tail -2.41 pp); the line-collapsed variant (one vote mass per provenance line) reaches 4.8/217.8/13.9 and decomposes the deficit stepwise: replicate collapse -31.2 [-67.4;-3.7] mean (median/tail n.s.); discreteness (collapsed vote vs. L1*a) small and n.s. on all three metrics (-0.25 [-0.71;+0.18] / -15.5 [-68.7;+37.8] / -0.19 pp [-1.39;+1.02]); the rest is the accuracy-radius weighting (L1*a vs. L1*b) -- the deficit is replicate dominance plus missing radius weighting, not discreteness | `city_vote_bootstrap.csv` |
| `exp_hybrid_single_source.py` | hybrid baseline: point from the CV-median-best single source, risk signal from the support concentration of the remaining lines around that point | RQ1/RQ4 | the warning signal works: OOF BSS +0.302 on anchors (base rate 0.100; predecessor label +0.019 on the same event), frozen transfer to probes +0.169; not directly comparable to the aggregation's +0.210 (different miss events); the hybrid keeps breakdown point 0 and single-provider drift exposure, probes mean 11.0 km worse (n.s.) | `hybrid_single_source.csv` |
| `exp_crossfit_radii.py` | cross-fitting control: track records per outer fold instead of global (control: strict OOF isolation) | RQ4/methodology | arm A reproduces the headline exactly (+0.210); fold-isolated it gets slightly BETTER (S50 +0.214, R7 feature set Δ+0.013 with identical fit code) → the published OOF numbers do not rest on the global shortcut; noted alongside T7 in the paper | `crossfit_radii.csv` |
| `exp_pool_transfer.py` | **pool transfer**: L1·b + S on 8,896 never-seen probes (`probes_pool`, nabi filter, 7 sources/5 lines, rfg 97.4 %) | RQ1/RQ4 (scale-up) | frozen 5.20 km/92.3/15.6 % ≈ fresh (Δ 0.04!); the 500 probes on the same 7 sources: 5.19/136.7/15.6 → median/tail replicate at 18× scale; S transfer raw +0.192, Platt-OOF +0.223; **auto-geoip stratum median 0.83 km vs. non-auto-geoip 7.15 km = massive circularity signal → report the stratum WITHOUT auto-GeoIP metadata (7.15/85.9/17.0) as the honest headline**; `--with-ipwhois` (sixth line completed after the pre-specified analysis): 6.55 [5.80;7.42]/15.2 [12.3;18.8], but the freeze delta becomes significant (+0.22 [+0.12;+0.38]) and Platt drops to +0.15 → reported as a T8 addendum in the paper, the five-line analysis stays primary | `pool_transfer.csv`, `pool_transfer_strata.csv`, `*_8src.csv` |
| `exp_giga_transfer.py` | **Giga transfer (T9)**: frozen pipeline on 2,127 UNICEF Giga (IP, school) pairs, 19 countries (end-user/Global South; `data/fetch_giga.py`, PTR-verified IP extraction, 8 sources/6 lines) | RQ1/RQ4 (end-user) | frozen **121.6 km [31.7;259.9] / tail 54.9 % [39.2;69.8]** (ASN-cluster CIs; consistent with the 54–61 % Global-South band of Nabi et al.); freezing n.s.; S raw breaks (miss rate 12→55 %, BSS −0.62), **Platt-OOF +0.070/ECE 0.028** (school-grouped folds +0.056/0.031, ASN-grouped +0.047/0.056; group bootstrap B=2,000: school CI [+0.030;+0.105] separable from zero, ASN CI [-0.091;+0.161] not); country gradient AL 4.9 → ZA 600 km; churn probe negative (pair age 5.5–37.4 d, ρ −0.003); extraction-method stratum (literal 321 vs. rdns 68) reads as country composition | `giga_transfer.csv`, `giga_transfer_strata.csv` |
| `exp_giga_mechanism.py` | mechanism evidence for the Giga country gradient: top ASNs, shared IPs before filter 4, location concentration of estimates, ClientInfo reference, ip-api access-type flags | mechanism | observations consistent with **coarse operator-block geolocation** (ZA: 22 estimate locations for 86 school locations; UZ, a fixed-line incumbent: 7 for 256); pooled access-technology association is largely compositional (pooled mobile 237 vs. fixed 71 km; inside ZA it reverses: mobile 406 < fixed 566 < hosting 663); CGNAT-style shared IPs only FJ (33 % before filter 4; ZA 5 %); the platform's own GeoIP reference sees the same errors (ZA 612 km) = no pipeline artefact | `giga_mechanism.csv`, `giga_mechanism_flags.csv` |
| `exp_giga_sources.py` | is 122 km an aggregator problem or an evidence problem? Every single source, naive geometric median (L0·a), city majority vote and the anchor-frozen trust vote on the identical 2,127 pairs | RQ1 (end-user) | all eight sources sit at 122.5–174.0 km median (tails 55.8–61.6 %); naive GM 121.6 · trust vote 122.0 · city vote 122.2 ≈ L1·b 121.6/54.9 → **the input portfolio carries no locally resolved evidence; robust aggregation cannot create information the sources do not contain** | `giga_sources.csv` |
| `exp_hub_signal.py` | **quick check of a RESEARCH IDEA** (no paper claim): hub-coordinate lexica (source×country, ≥10 IPs/coordinate) as a warning/weighting signal; split design (lexicon from half A, evaluation on half B) | outlook | Giga: no gain over S (Δ+0.008, not monotone — hubs are ubiquitous there); **pool: ΔBSS +0.017 (S +0.188→+0.205) with better ECE** — promising, but the confounder "legitimate city centroids vs. operator blocks" is unresolved; hub lexica differ per source and drift across builds → framed only as cautious future work | `hub_signal_giga.csv`, `hub_signal_probes_pool.csv` |
| `exp_freshness_audit.py` | freshness audit: Δ(GT timestamp, source fetch) per dataset/source + Giga pair age (motivation: "IPs change hands") | methodology | all populations Δ 0–3 days; pool additionally covered by stable-30d; Giga pair age median 19.5/p90 32.9/max 37.4 d — churn impact ruled out via the age stratum in `exp_giga_transfer` | `freshness_audit.csv`, `freshness_audit.md` |
| `exp_pool_asn_check.py` | ASN-cluster verification of the pool headline numbers (cluster=ASN, B=10k) | methodology | ASN structure unproblematic (3,057 ASNs, median 1 probe/ASN, top 2.7 %); non-auto-geoip 7.15 [6.40;8.23] / tail 17.0 [14.0;20.6]; freezing Δ n.s. [−0.06;+0.12]; circularity gap −6.3 [−7.4;−5.5] sig. | `pool_asn_check.csv` |
| `exp_stratification.py` | **E4** stratification (difficulty + region) | stratification | buckets 426/626/25 (easy/disagree/hard); added value mainly in the disagree bucket | `e4_difficulty.{csv,png}`, `e4_region.csv` |
| `exp_braetz.py` | **E5 / T5** Braetz estimator critically examined | RQ3 (T5) | inferior to the geometric median (15–17 vs. ~5.5 km); CI 94.9 % non-coverage; robust to density threshold 2 vs. 4; line-collapse control: disagree 64 → 40 km | `e5_braetz.csv`, `e5_braetz_threshold.csv`, `e5_braetz_linecollapse.csv`, `e5_braetz_ci_vs_gt.png`, `e5_braetz_qq.png` |
| `run_probes_suite.py` / `exp_probes_compare.py` | probe stress test (separate dataset, `GEOIP_DATASET=probes`) | limitation of RQ1 | aggregation carries over to end-user probes (median 5.4 km); tag gradient up to mobile 41 km / tail 25 % | `eval/out_probes/*`, `probes_vs_anchors.csv`, `probes_by_tag.csv` |

*Artifact names `t6_confidence_matrix.csv` / `t6_confidence_heatmap.png` (and the functions `run_confidence_matrix` / `run_confidence_recall`) are kept for stability; 'confidence' is the legacy name of the 2D predecessor label (paper terminology: predecessor label / risk signal).*

## Layers (what lives where)

- **`data/`** — acquisition + provenance + default/centroid detection (*what comes in*):
  `fetch_anchors` (ground truth), `fetch_sources` (sources per IP), `fetch_geodbs`
  (LITE DB download), `store` (hash-chain provenance), `centroids` (hub-default detection).
- **`estimators/`** — the estimation **algorithms** (*what is tested*): `baselines`
  (centroid, median, trimmed mean, geometric/weighted median), `braetz`.
- **`eval/`** — measurement **infrastructure** (*how it is measured*): `metrics` (Haversine),
  `pipeline` (load → estimate → evaluate, line weights), `report` (tables/plots).
- **`experiments/`** — the **studies** E1–E5/T6 (*what the results say*) — combine
  data + estimators + eval.
- **`tests/`** — **unit tests** of the modules (code correctness, pytest, fast) — *not* studies.
- **`notebooks/explore.ipynb`** — narrative results dashboard across all findings.

## Reproduction (order)

```bash
python data/fetch_anchors.py                  # ground truth
python data/fetch_geodbs.py --all             # local LITE DBs (keys in .env)
python data/fetch_sources.py --all            # sources per IP (+ default annotation)
python experiments/exp_source_correlation.py  # effective lines
python experiments/exp_accuracy.py            # E1
python experiments/exp_bootstrap_ci.py        # T1 bootstrap CIs (+ Bonferroni)
python experiments/exp_ipapi_sensitivity.py   # ipapi.co sensitivity
python experiments/exp_line_sensitivity.py    # rfg line sensitivity
python experiments/exp_antimeridian.py        # antimeridian edge case
python experiments/exp_t6_defaults.py         # T6 (+ ε grid)
python experiments/plot_t6.py                 # T6 figures
python experiments/exp_t6_hub_rule.py         # T6 hub-rule sensitivity (appendix)
python experiments/exp_label_calibration.py   # T6 predecessor-label calibration
python experiments/exp_support_concentration.py  # T6 support concentration S
python experiments/exp_s_under_contamination.py  # guard limit under contamination
python experiments/exp_contamination.py       # E2
python experiments/exp_samplesize.py          # E3
python experiments/exp_min_lines.py           # line ablation (min. effective lines)
python experiments/exp_min_lines_s.py         # S calibration per line subset
python experiments/exp_trust_weighted_fusion.py    # trust-vote baseline (OOF + transfer)
python experiments/exp_trust_frozen_symmetric.py   # symmetric frozen transfer
python experiments/exp_city_vote.py            # city vote (naive + line-collapsed)
python experiments/exp_hybrid_single_source.py # hybrid single-source baseline
python experiments/exp_miss_threshold_sensitivity.py  # miss-definition sensitivity
python experiments/exp_stratification.py      # E4
python experiments/plot_populations.py        # population dashboard figure (all suites first)
python experiments/exp_braetz.py              # E5 / T5 (+ density-threshold sensitivity)
```
