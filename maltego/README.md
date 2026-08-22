# Maltego integration (local transforms)

Two local Maltego transforms that apply this repo's method to a single IP —
live, with no server, no cache, and no data leaving your machine:

| Transform | Input | Output |
|---|---|---|
| `robustgeolocate` | IPv4 Address entity | **one** Location entity: robust consensus location (line- and radius-weighted geometric median, "L1+b"), with the support concentration **S**, a calibrated **P(error > 100 km)** (two-branch: default / mobile, routed by the ip-api `mobile` flag), the number of effective lines, and warnings attached as properties/notes |
| `persourceestimates` | IPv4 Address entity | **one Location entity per source** (disagreement view): the raw per-source estimates with their lineage and line weight |

The entity weight of `robustgeolocate` is `S × 100` — so the graph shows at a
glance whether the sources support each other (S close to 1) or contradict
each other (S small). If fewer than 3 effective lines answer, S and P(miss)
are **withheld** (reported as such, with a warning) instead of showing a
number the consensus cannot support. *Effective lines*, not raw sources, are
what counts: three GeoLite2 redistributors answering is **one** line (k=1),
and the warning states both numbers. The threshold of 3 is measured, not
guessed: in an exhaustive line ablation (`experiments/exp_min_lines_s.py`),
S carries no skill at k=1, is a lottery at k=2 (out-of-fold BSS between
+0.02 and +0.63 depending on which pair of lines happens to answer), and is
consistently informative from k=3 on (every subset ≥ +0.10).

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
| ipinfo.io | free token ([sign up](https://ipinfo.io/signup)) → `IPINFO_TOKEN=...` in `.env` at the repo root |
| GeoLite2 (offline) | free MaxMind account ([sign up](https://www.maxmind.com/en/geolite2/signup)) → `MAXMIND_LICENSE_KEY=...` in `.env`, then `python data/fetch_geodbs.py` |
| IP2Location Lite (offline) | free account ([sign up](https://lite.ip2location.com/sign-up)) → `IP2LOCATION_DOWNLOAD_TOKEN=...` in `.env`, then `python data/fetch_geodbs.py --ip2location` |
| DB-IP Lite (offline) | no account: `python data/fetch_geodbs.py` downloads it directly |

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

**macOS on Apple Silicon: `ImportError` from numpy when run from Maltego, but
the same command works in the terminal?** Maltego (an Intel/Java build running
under Rosetta) spawns your Python in x86_64 mode, where the arm64 numpy wheels
cannot load — numpy then shows a misleading "do not import from source
directory" error. Fix: force arm64 in the transform registration —
Command `/usr/bin/arch`, Parameters
`-arm64 /path/to/.venv/bin/python project.py local robustgeolocate`
(working directory unchanged). Verified on macOS with Maltego CE.

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

**P(error > 100 km)** maps S to a calibrated miss probability via a frozen
two-branch Platt layer (`s_calibration.json`, fitted on a mixed RIPE-probe
population): a *default* branch and a *mobile* branch, routed by the ip-api
`mobile` flag (requested live via the `fields` parameter; if the flag is not
retrievable, the default branch is used and a warning is shown). The mobile
branch mainly contributes the honest base rate — on cellular connections
roughly a third of large errors are invisible to any consensus measure, so a
flagged IP gets a structural warning and a markedly higher floor on P(miss)
even at S = 1.

## Extending the source portfolio (please read before adding sources)

The line assignment (`lineage` token per source in `data/fetch_sources.SOURCES`)
is a **maintained, documented decision — nothing is computed at runtime.** A
newly added source you know nothing about must get its own `*_unverified`
token; but if it secretly replicates an existing line (many free APIs
redistribute GeoLite2; several regional providers share a common upstream),
the effective-line count k gets **overestimated** and S is reported as
reliable too early. Before trusting k with a new source:

```
python maltego/check_lineage.py --list              # current sources -> lines, evidence status
python maltego/check_lineage.py --ips my_ips.txt    # redundancy check on >= 30 common IPs
```

The check computes pairwise median distance and the share of bit-identical
coordinates on the common intersection (no ground truth needed). A useful
redundancy-screening signal is the **identical-coordinate rate**, not the
median alone: independent-but-accurate sources typically sit single-digit
kilometres apart and rarely share bit-identical coordinates (reference points
on the evaluation corpus: geojs↔GeoLite2 100 % identical — a live check may
show less due to database-build drift; reallyfreegeoip↔GeoLite2 0.05 km
median, 48 % identical; DB-IP↔IP2Location 2.6 km median, 0 % identical —
independent). Identical coordinates alone do not *prove* shared lineage,
however: independent providers may return the same city centroid, country
default, or other well-known coordinate. **The check is diagnostic rather
than a lineage classifier; lineage assignments remain provenance decisions**
— a replicate finding means: investigate the provenance, and if confirmed,
assign the shared lineage token and document the decision.

Two limits remain even with a verified lineage. First, independence says
nothing about **quality**: a new source without a track-record entry in
`radius_table.json` is weighted with the conservative fallback radius (56 km,
the strongest down-weighting) until you can measure one against ground truth.
Second, the shipped **P(miss) layer (`s_calibration.json`) was fitted on this
portfolio**; if you substantially change the source set, re-fit it with
`make_s_calibration.py` — the S→P(miss) mapping does not automatically
transfer to a different portfolio.

## Limitations — please read before use

* **A triage aid and investigative lead, not forensic evidence.** The result
  is meant for prioritization and pre-sorting. It is not a court-ready
  determination of location and does not replace independent verification.
* **Calibration holds for the fitting population.** S itself was evaluated on
  RIPE Atlas anchors; the frozen P(miss) layer was fitted on a mixed RIPE
  *probe* population (residential/NAT/mobile), which is closer to arbitrary
  query IPs but still not identical to them. Treat P(miss) as a calibrated
  triage signal, not as a per-case guarantee; for a specific deployment
  population, refit with `maltego/make_s_calibration.py` on a small labeled
  sample. That this refit is both necessary and sufficient under a strong
  population shift is demonstrated empirically on end-user school connections
  (`experiments/exp_giga_transfer.py`: the raw anchor calibration becomes
  actively misleading, the per-population refit restores calibration).
* **The mobile flag is a live third-party signal.** ip-api may answer
  differently over time; a missing flag silently degrades to the default
  branch (with a warning), never to a crash.
* **Frozen pseudo-radii.** There is no ground truth at query time; the radius
  weighting therefore uses each source's global track record from the anchor
  evaluation (`radius_table.json`), not per-IP values.
* **No dataset-based hub detection.** The evaluation pipeline's dataset-based centroid/hub detection
  needs frequencies across many IPs. Live, the transform instead warns about
  country centroids when MaxMind reports `accuracy_radius ≥ 500 km`.
* **S measures precision (source consensus), not accuracy.** Sources can
  agree and still be wrong — for instance when several of them share the same
  outdated dataset.
