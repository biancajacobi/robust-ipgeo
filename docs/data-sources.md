# Data sources — ground truth & geolocation estimate sources

Overview of all data sources used in this project: the **ground truth**
(known locations) and the **estimate sources** (free geolocation APIs whose
per-IP value series is aggregated robustly). Acquisition, normalization and
provenance: `data/fetch_anchors.py`, `data/fetch_sources.py`, `data/store.py`.

> Scope (deliberate): The ground truth consists of **infrastructure/hosting IPs**
> (RIPE Atlas anchors in data centres/IXPs/ISPs). This matches the forensic
> scenario (VPN exits, cloud, hosting, CGN). Residential geolocation is
> **out of scope** for lack of reliable ground truth (limitation).

---

## Ground truth

| Source | Endpoint | Provides | Notes |
|--------|----------|----------|-------|
| **RIPE Atlas anchors** | `https://atlas.ripe.net/api/v2/anchors/` | IP (`ip_v4`), coordinates (`geometry.coordinates` = **[lon, lat]**), `city`, `country`, `as_v4` | paginated; filter on `is_disabled=false`. Swap the GeoJSON order when reading! |

- **Status:** 1077 active anchors fetched (`data/cache/anchors.csv`).
- **Evidence:** raw response + SHA-256 + hash-chain entry in `data/provenance.jsonl`.
- BibTeX: `ripeatlas`.

**Stress-test populations** (transfer, not calibration ground truth): RIPE Atlas
*probes* (n=500 exploratory; operator-reported, privacy-rounded coordinates), a
never-seen stable-probe pool (n=8,896, nabi-style filters), and **UNICEF Giga
school connections** (`data/fetch_giga.py`): 2,127 PTR-verified (IP, school)
pairs across 19 countries, school GPS from government records —
GeoIP-independent, but government-supplied rather than field-validated, and the
network attachment point need not coincide with the school building. Client IPs
are recovered from the hostname field the public Giga API exposes (declared
provider-schema selection → ASN-cluster CIs). Giga data are ODbL (source: Giga
(UNICEF) and contributors) and are **not part of this repo** — reproduction
requires own API keys (see `.env.example`).

### Rejected ground-truth extension (deliberate — no "second anchor list")
A **second reference source** was examined and rejected. The obvious candidate
would be the **RIPE Atlas probes** (instead of the anchors), which often sit in
residential/SOHO connections. Arguments against:
- **Location self-reported & coarser:** probe coordinates are set by the operator,
  they are not infrastructure-attested like anchors → no reliable ground truth.
- **Personal data:** residential connections fall outside the deliberately chosen
  infrastructure scope (see above) and would reintroduce the excluded residential
  case (cf. E4a spread / E4b region **instead of** a residential split).

The **1077 active anchors suffice** as ground truth (broad country spread — US 158,
DE 119, NL 63, FR 46, … > 60 countries; carries E1 and E4b). The binding bottleneck
is in any case **not** the number of reference IPs, but the **source** coverage on the
common intersection — and that was addressed by the locally read LITE DBs (full
coverage): the common intersection of all eight active sources now comprises
**n = 1,066** anchors (see below). Another anchor list would not have changed this;
only more independent **lines** (LITE DBs) help.

---

## Estimate sources (9 sources → ~6 effective lines)

One `(lat, lon)` estimate per source per IP → a small, heterogeneous value series.
Registry in the code: `data/fetch_sources.py` → `SOURCES`. **Two kinds:** local LITE DBs
(read offline, full coverage; downloaded with provenance evidence via
`data/fetch_geodbs.py`, credentials in `.env`) and web APIs (rate-limited;
`ipinfo` needs a token).

| Key (`source`) | Kind / endpoint | `lineage` | Success (/1077) |
|---|---|---|:---:|
| `maxmind_geolite2` | local mmdb `GeoLite2-City.mmdb` | `maxmind_geolite` (**confirmed**) | 1076 |
| `dbip_lite` | local mmdb `dbip-city-lite-*.mmdb` | `dbip_lite` (WHOIS/geofeeds) | 1077 |
| `ip2location_lite` | local BIN `IP2LOCATION-LITE-DB5.BIN` | `ip2location_lite` | 1077 |
| `ipinfo` | `https://ipinfo.io/{ip}/json?token=…` | `ipinfo` (own data) | 1077 |
| `ip_api` | `http://ip-api.com/json/{ip}` | `ipapi_com_unverified` | 1077 |
| `ipwho_is` | `https://ipwho.is/{ip}` (with `IPWHOISIO` key: `ipwhois.pro`, same data basis/line, higher quota) | `ipwhois_unverified` | 1077 |
| `geojs` | `https://get.geojs.io/v1/ip/geo/{ip}.json` | `maxmind_geolite` (**confirmed**) | 1076 |
| `reallyfreegeoip` | `https://reallyfreegeoip.org/json/{ip}` | `maxmind_geolite` (freegeoip fork) | 1077 |
| `ipapi_co` | `https://ipapi.co/{ip}/json/` | `ipapi_co_unverified` | 7 ⚠️ rate limit |

> **Effective lines:** `geojs` ≡ `reallyfreegeoip` ≡ `maxmind_geolite2` form **one**
> line `maxmind_geolite` (measured 0 km distance, identical default rate 44 %). Thus:
> MaxMind · DB-IP · IP2Location · ipinfo · ip-api · ipwho.is = ~6 lines (ipapi_co dead).
>
> **Adding a new source?** Assign it its own `*_unverified` token, then run
> `python maltego/check_lineage.py --list` (coverage/evidence status) or
> `--ips <file>` (redundancy check: pairwise distance + identical rate on shared
> IPs, without ground truth). The distinguishing feature is the **identical rate**,
> not the median alone. The assignment remains a documented decision
> (precedent rfg: 0.05 km/48 % identical → MaxMind line; see the accompanying
> paper + `exp_line_sensitivity`).
> `lineage` = **documented** origin, not derived from distances. `ip-api`/`ipwho.is`
> do not disclose their origin → conservatively treated as their own, unverified lines.
> Quality columns per observation: `accuracy_radius` (MaxMind), `is_default_centroid`,
> `centroid_match`, `db_version` (see `data/centroids.py`).

### Status fields / error detection per API
- **ip-api:** success = `status == "success"`; otherwise `message`.
- **ipwho.is:** success = `success == true`; otherwise `message`.
- **geojs:** success = `latitude`/`longitude` present (as strings, → float).
- **reallyfreegeoip:** success = `latitude`/`longitude` present; empty `city` → `None`.
  Sometimes returns `0,0` for unknown (**"Null Island"** — a classic geolocation error,
  treated as an outlier by the robust estimator).
- **ipapi.co:** success = no `error`; otherwise `reason`.

### Observations & caveats (relevant for the evaluation)
- **`ipapi_co` practically unusable:** the free tier rate-limits almost everything
  away in the data centre (3/1077). Kept in the registry for transparency, but
  contributes little. Transient errors are honestly logged as `status=http_error`.
- **At least 3 usable sources required:** at `n = 2`, median, trimmed mean and
  geometric median collapse onto the midpoint — aggregation only becomes
  meaningful from `n ≥ 3` (motivation for the additional sources).
- **`trimmed_mean` needs `n ≥ 5`:** with four sources, trimming 20 % removes nothing (= mean).
- **"Null Island" (0,0):** `reallyfreegeoip` sometimes returns `0,0` for unknown —
  treated as an outlier; excluded from the correlation matrix.
- **ip-api free tier is HTTP only** and forbids commercial use (academic use ok).

### Rejected source
- **`freeipapi.com`** (`https://freeipapi.com/api/json/{ip}`): behind Cloudflare it
  returns a `302 Found` instead of JSON → not readily usable, not integrated.

---

## Source (in)dependence — the methodological core

For robust fusion, what counts is the **effective number of independent lines**, not
the API count. Two things are kept cleanly separated:

- **Measured redundancy** (`experiments/exp_source_correlation.py`): pairwise
  median distance of the source outputs on the **common intersection** (same IP
  basis for all pairs, without `(0,0)`) → `eval/out/source_correlation.{csv,png}`.
  Identical outputs = no information gain for the fusion.
- **Documented origin** (`lineage`, see above): from provider docs/ToS, *not*
  derived from distances.

**Finding (n = 1,066 shared IPs, all eight active sources):** `maxmind_geolite2` ↔
`geojs` = **0.0 km median** and ↔ `reallyfreegeoip` = **0.05 km** (effectively the
same line); all remaining pair distances ≥ 0.7 km (smallest `dbip_lite` ↔ `ip_api`
0.7 km, `ip_api` ↔ `ipwho_is` ≈ 2.7 km). → **three sources (MaxMind / `geojs` /
`reallyfreegeoip`), but only one line.** Agreement ≠ shared line (two sources can be
close because both are *right*) — the line claim therefore rests on `lineage`, the
matrix only on measured redundancy.

**Consequence for the aggregation (instead of deleting):** duplicates are kept; the
correlation is handled via **line weights** — each `lineage` receives a total weight
of 1, distributed evenly across its members (`eval.pipeline.line_weights`),
and the **weighted geometric median** (RFA: `g(v)=Σ αᵢ‖v−wᵢ‖`,
`estimators.baselines.weighted_geometric_median`) computes with them. E1 reports
**both**: naive per source (`geometric_median`) vs. per line (`geometric_median_perline`).

### Genuinely independent lines (LITE DBs, local) — integrated
Genuine independence (crucial for RQ2/breakdown) is provided by the local LITE DBs.
They are downloaded with provenance evidence via `data/fetch_geodbs.py` (SHA-256 +
build per run in the hash chain) and read offline (`data/fetch_sources.py`,
`geoip2`/`IP2Location`):

| Source | File | Build (snapshot) | Origin | Access |
|---|---|---|---|---|
| `maxmind_geolite2` | `GeoLite2-City.mmdb` | 2026-06-02 | MaxMind (own data) | account + license key |
| `dbip_lite` | `dbip-city-lite-2026-06.mmdb` | 2026-06 | DB-IP (WHOIS/geofeeds) | monthly download (no login) |
| `ip2location_lite` | `IP2LOCATION-LITE-DB5.BIN` | 2026-06-01 | IP2Location (own data) | download token |

> **Evidenced instead of assumed:** the correlation matrix measures that
> `geojs`/`reallyfreegeoip` correspond exactly to the local GeoLite2 (0 km, identical
> 44 % default rate) — the "are they reselling GeoLite?" conjecture is thereby
> confirmed; all three = one line.

### Mentioned but NOT implemented signals (possible extension)
Signals methodologically independent of the DBs; noted here only as an outlook,
not implemented:
- **RDAP / RIR WHOIS:** provides the registration **country/HQ**, not the host → as
  a coordinate systematically coarse (country centroid). Sensible only as an
  independent **plausibility/cross-check layer** ("does the estimate lie in the
  RDAP country?", serving RQ4), **not** as a point in the geometric median.
- **rDNS hostname codes** (airport/city codes; Hoiho/DRoP, cf. `dan2022geographic`).
- **Geofeeds (RFC 8805):** location data published by ISPs, DB-independent.

---

## Storage & provenance (lightweight)

| File | Content | versioned |
|------|---------|:---------:|
| `data/cache/sources_raw.jsonl` | append-only raw cache: 1 line per (source, IP) with the verbatim response — evidence **and** double-fetch protection | no (gitignored) |
| `data/cache/observations.csv` | normalized working copy (`OBSERVATION_COLUMNS`); cumulative view of the entire cache | no (gitignored) |
| `data/provenance.jsonl` | 1 audit-ledger entry per run (sources, counters, SHA-256 of cache + observations), in a tamper-evident hash chain | **yes** |

Only successful responses (with a body) count as cached; transient errors are
retried on the next run.

---

## Reproduction

```bash
python data/fetch_anchors.py                                  # ground truth
python data/fetch_sources.py --all                            # all sources, all IPs
python data/fetch_sources.py --all --sources geojs,reallyfreegeoip   # targeted top-up
python data/fetch_anchors.py --verify                         # verify provenance chain
```
