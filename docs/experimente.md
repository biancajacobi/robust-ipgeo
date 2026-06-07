# Experiments — file → study → research question → key finding → output

Which file runs which study, which research question (FF) it serves, the key finding, and
the artifacts it produces. All experiments live in `experiments/` and use the measurement
infrastructure in `eval/`, the estimators in `estimators/`, and the data in `data/`.

| File | Study | FF | Key finding | Output (`eval/out/`) |
|---|---|---|---|---|
| `exp_accuracy.py` | E1 base accuracy | FF1 | per-source/estimator median; geometric median ≈ 5.5 km, naive mean ≈ 38 km | `e1_accuracy.csv`, `e1_accuracy_cdf.png` |
| `exp_source_correlation.py` | source redundancy / effective lines | method | `maxmind_geolite2` ↔ `geojs` 0.0 km / ↔ `reallyfreegeoip` 0.05 km → one line; all other pairs ≥ 0.7 km | `source_correlation.{csv,png}` |
| `exp_bootstrap_ci.py` | T1 paired-bootstrap CIs (B = 10,000) | FF1 | aggregation `L1·b` beats 7/8 single sources in median (CIs exclude 0); mean/tail loss vs. ipinfo significant, vs. IP2Location not | `t1_bootstrap_ci.csv`, `t1_bootstrap_meantail_ci.csv` |
| `exp_t6_defaults.py` | T6 line scale L0/L1/L2 × aggregator + ε-grid + 2-D confidence matrix | FF1, FF4 | 12 configs; `L1·b` best (mean 183 km, tail 12.0 %); line collapse −20 % | `t6_point_estimators.csv`, `t6_confidence_matrix.csv`, `t6_epsilon_grid.csv` |
| `plot_t6.py` | T6 figures (ECDF, heatmap, forest) | — | figures for T6 | `t6_ecdf.png`, `t6_confidence_heatmap.png`, `t6_forest.png` |
| `exp_t6_hub_rule.py` | T6 hub-rule sensitivity | FF4 | majority (≥ 0.5) sharper than "at least one" | (stdout) |
| `exp_label_calibration.py` | T6 confidence-label calibration (out-of-fold) | FF4 | predecessor label calibrated but moderate skill | `label_calibration_*.csv`, `label_calibration.png` |
| `exp_support_concentration.py` | T6 confidence measure `S` (out-of-fold) | FF4 | line-weighted support concentration triples calibration skill (BSS +0.068 → +0.210) | `support_concentration_oof.csv`, `support_concentration.png` |
| `exp_whois_mechanism.py` | RDAP validation of the DB-IP↔IP2Location coupling | mechanism | common line only on mutual default | (RDAP cache + provenance) |
| `exp_contamination.py` | E2 contamination / breakdown | FF2 | geometric median robust to α_eff ≈ 37.5 %; naive mean breaks from 10 % | `e2_breakdown.{csv,png}` |
| `exp_samplesize.py` | E3 sample size n | FF3 | robust estimators ≈ constant (n = 3..7); naive mean worsens (17 → 35 km) | `e3_samplesize.{csv,png}` |
| `exp_stratification.py` | E4 stratification (difficulty + region) | — | buckets easy/uneinig/hart; value mainly in the "uneinig" bucket | `e4_difficulty.{csv,png}`, `e4_region.csv` |
| `exp_braetz.py` | E5 / T5 Brätz estimator (critical review + line-collapse control) | FF3 | dominated by geometric median; CI 94.9 % non-coverage; replicate dominance confirmed but partial | `e5_braetz.csv`, `e5_braetz_threshold.csv`, `e5_braetz_linecollapse.csv`, `e5_braetz_ci_vs_gt.png`, `e5_braetz_qq.png` |
| `exp_probes_compare.py` | probe stress test (anchors vs. probes + tag stratification) | scope/limitation | aggregation holds across populations; gradient data centre 4.6 < home/NAT 8.5 < mobile 41.2 km | `eval/out_probes/probes_vs_anchors.csv`, `probes_by_tag.csv` |
| `run_probes_suite.py` | full suite on the probe dataset (driver) | scope | runs all experiments with `GEOIP_DATASET=probes` → `eval/out_probes/` | `eval/out_probes/*.csv` |

## Layers

- **`data/`** — acquisition + provenance + default/centroid detection (`fetch_anchors`,
  `fetch_probes`, `fetch_sources`, `fetch_geodbs`, `store`, `centroids`).
- **`estimators/`** — the estimation algorithms (`baselines`, `braetz`).
- **`eval/`** — measurement infrastructure (`metrics`, `pipeline`, `report`).
- **`experiments/`** — the studies (combine data + estimators + eval).
- **`tests/`** — unit tests of the modules (46, pytest; includes a determinism test).
- **`notebooks/explore.ipynb`** — results dashboard over `eval/out` (no raw data needed).

## Reproduction order

```bash
python data/fetch_anchors.py                  # ground truth
python data/fetch_geodbs.py --all             # local LITE databases (keys in .env)
python data/fetch_sources.py --all            # sources per IP
python experiments/exp_source_correlation.py
python experiments/exp_accuracy.py            # E1
python experiments/exp_bootstrap_ci.py        # T1
python experiments/exp_t6_defaults.py         # T6
python experiments/plot_t6.py
python experiments/exp_label_calibration.py
python experiments/exp_support_concentration.py
python experiments/exp_contamination.py       # E2
python experiments/exp_samplesize.py          # E3
python experiments/exp_stratification.py      # E4
python experiments/exp_braetz.py              # E5 / T5

# optional probe stress test (separate dataset)
GEOIP_DATASET=probes python data/fetch_probes.py --n 500
GEOIP_DATASET=probes python data/fetch_sources.py --all
GEOIP_DATASET=probes python experiments/run_probes_suite.py
python experiments/exp_probes_compare.py
```
