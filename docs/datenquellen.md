# Data sources — ground truth & geolocation estimators

Overview of all data used: the **ground truth** (known locations) and the
**estimation sources** (free geolocation APIs/databases whose per-IP value series is
robustly aggregated). Acquisition, normalization and provenance live in
`data/fetch_anchors.py`, `data/fetch_probes.py`, `data/fetch_sources.py`, `data/store.py`.

> **Privacy:** individual IP addresses are removed from this release; node identifiers are
> pseudonymized (`node_NNNNN`). Source coordinates (public infrastructure locations) are
> retained for reproducibility. Only public infrastructure IPs are processed.

---

## Ground truth

Two RIPE Atlas datasets are used, with clearly different evidentiary status.

### Primary: RIPE Atlas Anchors

| Source | Endpoint | Provides | Notes |
|---|---|---|---|
| RIPE Atlas Anchors | `https://atlas.ripe.net/api/v2/anchors/` | IP (`ip_v4`), coordinates (`geometry.coordinates` = **[lon, lat]**), `city`, `country`, `as_v4` | paginated; filter `is_disabled=false`; swap the GeoJSON order on read |

- Anchors are fixed, professionally operated measurement nodes in data centres / IXPs /
  ISPs with **independently documented** coordinates — a strong, non-personal ground truth.
- Sample: ~1,077 active anchors, broad country spread (>60 countries).
- Evidence: raw response + SHA-256 + hash-chain entry in `data/provenance.jsonl`.

### Secondary (stress test): RIPE Atlas Probes

Probes — unlike anchors — are largely operated on **end-user access lines**
(residential, NAT, mobile), which approximates the forensically typical population.
They are used for an **exploratory stress test only**,
not as primary ground truth, because:

- **Self-reported, rounded coordinates:** a probe's location is set by its host and
  privacy-rounded by RIPE — noisier than anchor ground truth.
- **Selection bias:** probes are volunteer-operated and skew to well-registered fixed-line
  prefixes; restricting to public IPv4 excludes CGNAT-bound lines. The set is therefore
  biased toward the more easily locatable subset.

A random sample of 500 connected non-anchor probes with public IPv4 (`seed=0`) plus a
targeted mobile bucket (all available RIPE-`mobile` probes, ~60) are fetched via
`data/fetch_probes.py` into a **separate dataset** (`GEOIP_DATASET=probes` /
`probes_mobile`) that never touches the anchor artifacts or their hash chain.

---

## Estimation sources (9 sources → ~6 effective lines)

One `(lat, lon)` estimate per source per IP → a small, heterogeneous value series.
Registry in code: `data/fetch_sources.py` → `SOURCES`. Two kinds: local LITE databases
(read offline, full coverage; fetched provenance-tracked via `data/fetch_geodbs.py`,
credentials in `.env`) and throttled web APIs (`ipinfo` needs a token).

| Key (`source`) | Kind / endpoint | `lineage` |
|---|---|---|
| `maxmind_geolite2` | local mmdb `GeoLite2-City.mmdb` | `maxmind_geolite` |
| `dbip_lite` | local mmdb `dbip-city-lite-*.mmdb` | `dbip_lite` |
| `ip2location_lite` | local BIN `IP2LOCATION-LITE-DB5.BIN` | `ip2location_lite` |
| `ipinfo` | `https://ipinfo.io/{ip}/json?token=…` | `ipinfo` |
| `ip_api` | `http://ip-api.com/json/{ip}` | `ipapi_com_unverified` |
| `ipwho_is` | `https://ipwho.is/{ip}` | `ipwhois_unverified` |
| `geojs` | `https://get.geojs.io/v1/ip/geo/{ip}.json` | `maxmind_geolite` |
| `reallyfreegeoip` | `https://reallyfreegeoip.org/json/{ip}` | `maxmind_geolite` |
| `ipapi_co` | `https://ipapi.co/{ip}/json/` | `ipapi_co_unverified` (rate-limited, ~unusable) |

> **Effective lines:** `geojs` ≡ `reallyfreegeoip` ≡ `maxmind_geolite2` form **one** line
> `maxmind_geolite` (measured 0 km pairwise distance). So: MaxMind · DB-IP · IP2Location ·
> ipinfo · ip-api · ipwho.is ≈ 6 lines. `lineage` is the **documented** origin, not derived
> from distances; `ip-api`/`ipwho.is` do not disclose their origin → treated as separate,
> unverified lines. Per-observation quality columns: `accuracy_radius` (MaxMind),
> `is_default_centroid`, `centroid_match`, `db_version` (see `data/centroids.py`).

### Status / error handling per API
- **ip-api:** success = `status == "success"`. **ipwho.is:** `success == true`.
  **geojs / reallyfreegeoip:** `latitude`/`longitude` present (strings → float).
  **ipapi.co:** no `error` field.
- **"Null Island" (0,0):** `reallyfreegeoip` sometimes returns `0,0` for unknown — treated
  as an outlier and excluded from the correlation matrix.
- **Minimum sources:** aggregation is meaningful from `n ≥ 3` (at `n = 2` median, trimmed
  mean and geometric median collapse to the midpoint); `trimmed_mean` trims only from `n ≥ 5`.

---

## Source (in)dependence — the methodological core

What matters for robust fusion is the **effective number of independent lines**, not the
API count. Two things are kept separate:

- **Measured redundancy** (`experiments/exp_source_correlation.py`): pairwise median
  distance of source outputs on the common intersection → `eval/out/source_correlation.{csv,png}`.
- **Documented origin** (`lineage`): from provider docs/ToS, not from distances.

**Finding:** `maxmind_geolite2` ↔ `geojs` = 0.0 km and ↔ `reallyfreegeoip` = 0.05 km
(effectively one line); all other pairs ≥ 0.7 km. → three sources, one line.

**Consequence (instead of deletion):** duplicates are kept; correlation is handled via
**line weights** — each `lineage` gets total weight 1, split evenly among its members
(`eval.pipeline.line_weights`), and the **weighted geometric median**
(`estimators.baselines.weighted_geometric_median`) uses them.

---

## Storage & provenance (lightweight)

| File | Content | versioned |
|---|---|:---:|
| `data/cache/sources_raw*.jsonl` | append-only raw cache (one line per (source, IP)) | no (gitignored) |
| `data/cache/observations*.csv` | normalized working copy (`OBSERVATION_COLUMNS`) | probe set only (pseudonymized) |
| `data/provenance*.jsonl` | one audit-ledger entry per fetch, in a tamper-evident hash chain | acquisition record — **not part of this redacted release** |

The hash-chain mechanism is implemented in `data/store.py` (`verify_chain()`); the concrete
ledgers and raw evidence attest the original (non-redacted) corpus and are not shipped here.
Only successful responses count as cached; transient errors are retried on the next run.
