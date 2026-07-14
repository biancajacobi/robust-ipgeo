# Experimente — Datei → Studie → Forschungsfrage → Kernbefund → Output

Übersicht, welche Datei welche Studie ausführt, welche Forschungsfrage sie bedient,
was dabei herauskommt (Kernbefund) und welche Artefakte sie erzeugt. Alle Experimente
liegen in `experiments/` und nutzen die Mess-Infrastruktur aus `eval/` + die Schätzer aus
`estimators/` auf den Daten aus `data/`. Die ausführliche Interpretation steht in der
zugehörigen Ausarbeitung; dieses Repo ist eigenständig nachvollziehbar.

| Datei | Studie | FF | Kernbefund | Output (`eval/out/`) |
|---|---|---|---|---|
| `exp_accuracy.py` | **E1** Grundgenauigkeit | FF1 | Median je Quelle/Schätzer; geom. Median ~5,5 km, naiver Mittelwert ~38 km | `e1_accuracy.csv`, `e1_accuracy_cdf.png` |
| `exp_source_correlation.py` | Quellen-Redundanz / effektive Linien + Familien-Drift | Methodik | `maxmind_geolite2` ↔ `geojs` 0,0 km (bit-identisch) / ↔ `reallyfreegeoip` 0,05 km Median → **eine** Linie; alle übrigen Paare ≥ 0,7 km (n = 1.066). rfg = gleiche Abstammung, abweichender Datenstand (48 % identisch, 17 % > 100 km) | `source_correlation.{csv,png}`, `source_family_drift.csv` |
| `exp_bootstrap_ci.py` | **T1** Paired-Bootstrap-CIs (B = 10.000, seed 0) | FF1 | Aggregation `L1·b` schlägt 7/8 Einzelquellen im Median signifikant; auch Bonferroni-adjustiert (α = 0,05/8) schließen alle CIs 0 aus | `t1_bootstrap_ci.csv`, `t1_bootstrap_meantail_ci.csv` |
| `exp_t6_defaults.py` | **T6** Linien-Skala L0/L1/L2 × Aggregator + ε-Grid + 2D-Konfidenz-Matrix | FF1, FF4 | 12 Konfigs; `L1·b` best (Mittel 183 km, Tail 12,0 %); Linien-Kollaps −20 %; Konfidenz-Recall 72 % auf Misses | `t6_point_estimators.csv`, `t6_confidence_matrix.csv`, `t6_epsilon_grid.csv` |
| `plot_t6.py` | Visualisierung zu T6 (ECDF, Heatmap, Forest) | — | Abbildungen zu T6 | `t6_ecdf.png`, `t6_confidence_heatmap.png`, `t6_forest.png` |
| `exp_t6_hub_rule.py` | **T6** Hub-Regel-Sensitivität (Anhang) | FF4 | Mehrheit (≥ 0,5) trennschärfer als „mind. eine"; letztere flaggt 63,5 % bei nur ~9 % Zusatz-Präzision | (stdout) |
| `exp_whois_mechanism.py` | Validierung der DB-IP↔IP2Location-Fehlerkopplung (RDAP) | Mechanismus zu T6/T5 | gemeinsame Linie nur bei beidseitigem Default (begründet das bedingte L2) | (RDAP-Cache + Provenance) |
| `exp_label_calibration.py` | **T6** Vorgängerlabel-Kalibrierung (out-of-fold, 10-fach) | FF4 | 2D-Label brauchbar kalibriert (ECE 0,019), Trennschärfe moderat (BSS +0,068) | `label_calibration_*.csv`, `label_calibration.png` |
| `exp_spread_measure.py` | Vorstudie Streu-/Konzentrationsmaße | FF4 | Vorauswahl der Maß-Kandidaten für S | `spread_measure_comparison.csv` |
| `exp_support_concentration.py` | **T6** Stütz-Konzentration S (out-of-fold) | FF4 | S verdreifacht die Forecast-Güte (BSS +0,068 → +0,210, AP 0,25 → 0,42); Recall auf Misses 72 → 92 % bei gleicher Flag-Quote (41 %) | `support_concentration_oof.csv`, `support_concentration.png` |
| `exp_s_under_contamination.py` | Wächter-Grenze unter koordinierter Kontamination | FF4 | gekaperte Schätzungen bleiben unter der Flag-Schwelle (0 % ungeflaggt bis α = 0,5; 1,6 % bei 0,6); Wächter-Breakdown ~0,83 Linienmasse vs. 0,5 des Schätzers | `s_under_contamination.{csv,png}` |
| `exp_ipapi_sensitivity.py` | ipapi.co-Sensitivität der Headline | FF1 | gesamte Mittel-Differenz 172,9 ↔ 183,4 km stammt von EINEM Anchor an der Gewichtsmassen-Grenze (47,5 → 51,0 %) | `t6_ipapi_sensitivity.csv` |
| `exp_line_sensitivity.py` | Linien-Sensitivität `reallyfreegeoip` | Methodik | eigene/bedingte rfg-Linie ändert Median/Tail praktisch nicht; ohne Kollaps kippen 6 Anchors in den Tail (bis 31 → 1.783 km) | `line_sensitivity_rfg.csv` |
| `exp_antimeridian.py` | Antimeridian-Randfall | Methodik | 17 Fälle mit Rohgrad-Spanne > 180°, sämtlich Default-Ausreißer (Rohgrad überzeichnet → pro-robust); Re-Zentrierung bei 4 Fällen > 180° wahrer Spanne nicht wohldefiniert | `antimeridian_check.csv` |
| `exp_contamination.py` | **E2** Kontamination / Breakdown (`--replot`) | FF2 | geom. Median robust bis α_eff ≈ 37,5 %; naiver Mittelwert bricht ab 10 % | `e2_breakdown.{csv,png}` |
| `exp_samplesize.py` | **E3** Stichprobengröße n (`--replot`) | FF3 | robuste Schätzer ~konstant (n = 3..7); naiver Mittelwert verschlechtert sich (17 → 35 km) | `e3_samplesize.{csv,png}` |
| `exp_stratification.py` | **E4** Stratifizierung (Schwierigkeit + Region) | FF (Stratifizierung) | Buckets 426/626/25 (easy/uneinig/hart); Mehrwert v. a. im uneinig-Bucket | `e4_difficulty.{csv,png}`, `e4_region.csv` |
| `exp_braetz.py` | **E5 / T5** Brätz-Schätzer kritisch geprüft | FF3 (T5) | unterliegt geom. Median (15–17 vs. ~5,5 km); CI 94,9 % Nicht-Abdeckung; robust ggü. Dichteschwelle 2 vs. 4; Linien-Kollaps-Kontrolle: uneinig 64 → 40 km | `e5_braetz.csv`, `e5_braetz_threshold.csv`, `e5_braetz_linecollapse.csv`, `e5_braetz_ci_vs_gt.png`, `e5_braetz_qq.png` |
| `run_probes_suite.py` / `exp_probes_compare.py` | Probe-Stresstest (separater Datensatz, `GEOIP_DATASET=probes`) | Limitation zu FF1 | Aggregation trägt auf Endkunden-Probes (Median 5,4 km); Tag-Gradient bis Mobilfunk 41 km / Tail 25 % | `eval/out_probes/*`, `probes_vs_anchors.csv`, `probes_by_tag.csv` |

## Schichten (wo liegt was)

- **`data/`** — Beschaffung + Provenance + Default-/Centroid-Detektion (*was kommt rein*):
  `fetch_anchors` (Ground Truth), `fetch_sources` (Quellen je IP), `fetch_geodbs`
  (LITE-DB-Download), `store` (Hash-Ketten-Provenance), `centroids` (Hub-Default-Erkennung).
- **`estimators/`** — die Schätz-**Algorithmen** (*was wird getestet*): `baselines`
  (centroid, Median, getrimmtes Mittel, geom./gewichteter Median), `braetz`.
- **`eval/`** — Mess-**Infrastruktur** (*wie wird gemessen*): `metrics` (Haversine),
  `pipeline` (load → estimate → evaluate, Linien-Gewichte), `report` (Tabellen/Plots).
- **`experiments/`** — die **Studien** E1–E5/T6 (*was sagen die Ergebnisse*) — kombinieren
  data + estimators + eval.
- **`tests/`** — **Unit-Tests** der Module (Code-Korrektheit, pytest, schnell) — *keine* Studien.
- **`notebooks/explore.ipynb`** — narratives Ergebnis-Dashboard über alle Befunde.

## Reproduktion (Reihenfolge)

```bash
python data/fetch_anchors.py                  # Ground Truth
python data/fetch_geodbs.py --all             # lokale LITE-DBs (Keys in .env)
python data/fetch_sources.py --all            # Quellen je IP (+ Default-Annotation)
python experiments/exp_source_correlation.py  # effektive Linien
python experiments/exp_accuracy.py            # E1
python experiments/exp_bootstrap_ci.py        # T1 Bootstrap-CIs (+ Bonferroni)
python experiments/exp_ipapi_sensitivity.py   # ipapi.co-Sensitivität
python experiments/exp_line_sensitivity.py    # rfg-Linien-Sensitivität
python experiments/exp_antimeridian.py        # Antimeridian-Randfall
python experiments/exp_t6_defaults.py         # T6 (+ ε-Grid)
python experiments/plot_t6.py                 # T6-Abbildungen
python experiments/exp_t6_hub_rule.py         # T6 Hub-Regel-Sensitivität (Anhang)
python experiments/exp_label_calibration.py   # T6 Vorgängerlabel-Kalibrierung
python experiments/exp_support_concentration.py  # T6 Stütz-Konzentration S
python experiments/exp_s_under_contamination.py  # Wächter-Grenze unter Kontamination
python experiments/exp_contamination.py       # E2
python experiments/exp_samplesize.py          # E3
python experiments/exp_stratification.py      # E4
python experiments/exp_braetz.py              # E5 / T5 (+ Dichteschwellen-Sensitivität)
```
