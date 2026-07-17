# Maltego integration (local transforms)

Two local Maltego transforms that apply this repo's method to a single IP —
live, with no server, no cache, and no data leaving your machine:

| Transform | Input | Output |
|---|---|---|
| `robustgeolocate` | IPv4 Address entity | **one** Location entity: robust consensus location (line- and radius-weighted geometric median, "L1+b"), with the support concentration **S**, the number of effective lines, and warnings attached as properties/notes |
| `persourceestimates` | IPv4 Address entity | **one Location entity per source** (disagreement view): the raw per-source estimates with their lineage and line weight |

The entity weight of `robustgeolocate` is `S × 100` — so the graph shows at a
glance whether the sources support each other (S close to 1) or contradict
each other (S small).

## Installation

```bash
git clone <this-repo> && cd <repo>
pip install -r requirements.txt -r maltego/requirements.txt
```

Test without Maltego (queries the sources live):

```bash
python maltego/geoloc.py 9.9.9.9
cd maltego && python project.py local robustgeolocate 9.9.9.9
```

## API keys / data sources (bring your own)

The transforms redistribute **no** geolocation data and no credentials — every
source is queried with the user's own accounts/downloads. Sources without a
key/database file simply drop out; the estimator keeps running on the
remaining lines (with a warning).

| Source | Requirement |
|---|---|
| ip-api.com, ipwho.is, geojs.io, reallyfreegeoip.org, ipapi.co | none (free endpoints, mind the rate limits) |
| ipinfo.io | free token → `IPINFO_TOKEN=...` in `.env` at the repo root |
| GeoLite2 / DB-IP Lite / IP2Location Lite (offline) | download the DBs yourself: `python data/fetch_geodbs.py` (MaxMind account required; the providers' license terms apply) |

Complying with the sources' terms of service (in particular the commercial-use
clauses of the free tiers and the GeoLite2 license) is the user's
responsibility.

## Registering in Maltego

1. Maltego → **Transforms → New Local Transform**
2. Display name e.g. "Robust GeoIP (consensus + S)", input entity type: **IPv4 Address**
3. Command: `python` (or the full path to your venv's Python)
4. Parameters: `project.py local robustgeolocate`
5. Working directory: `<repo>/maltego`
6. Repeat once more with `project.py local persourceestimates`.

**No pin on the map view?** The transforms set `latitude`/`longitude` as
properties of the built-in `maltego.Location` entity. If the coordinates show
up in the entity's detail view but no pin is rendered on the map, your Maltego
version expects different property field IDs — check the actual field IDs of
`maltego.Location` under **Entities → Manage Entities** and adjust the
`addProperty` field names in `transforms/*.py` accordingly.

Note on the entity label: it shows the city/country of the *closest*
observation as a display heuristic — the actual result is the coordinate pair
(the weighted median), which can in principle fall into a different city.

## Method in brief

For each IP, all reachable sources are queried; sources with the same data
provenance (`lineage`, e.g. GeoLite2 redistributors) are collapsed into **one
effective line**. Each point is additionally weighted by
`1 / (accuracy radius + ε)` (MaxMind: the `accuracy_radius` it reports; other
sources: a frozen track-record radius from `radius_table.json`). The location
is the weighted geometric median. **S** is the line-weighted share of source
mass within 50 km of the estimate (city scale, pre-specified).

## Limitations — please read before use

* **A triage aid and investigative lead, not forensic evidence.** The result
  is meant for prioritization and pre-sorting. It is not a court-ready
  determination of location and does not replace independent verification.
* **Calibration holds for the evaluation dataset.** The calibration quality
  of S reported in the thesis was measured on RIPE Atlas anchors
  (data-center/infrastructure IPs). Live queries — especially for residential
  or mobile IPs — may deviate; treat S as an ordinal signal rather than a
  probability in that case.
* **Frozen pseudo-radii.** There is no ground truth at query time; the radius
  weighting therefore uses each source's global track record from the anchor
  evaluation (`radius_table.json`), not per-IP values.
* **No dataset-based hub detection.** The thesis' centroid/hub detection
  needs frequencies across many IPs. Live, the transform instead warns about
  country centroids when MaxMind reports `accuracy_radius ≥ 500 km`.
* **S measures precision (source consensus), not accuracy.** Sources can
  agree and still be wrong — for instance when several of them share the same
  outdated dataset.
