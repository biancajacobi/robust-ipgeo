"""Live geolocation of ONE IP using the method from the accompanying paper (for the Maltego transforms).

Thin wrapper around the existing code: source retrieval and normalization come
unchanged from ``data/fetch_sources.py``, the estimator from
``estimators/baselines.py``. Only the single-IP wiring happens here:

  query all sources -> line weights (L1) -> accuracy-radius weighting
  -> weighted geometric median (L1+b) -> support concentration S
  -> two-branch calibration S -> P(error > 100 km) (branch via the ip-api
     mobile flag, frozen table s_calibration.json; hardening 2026-08).

Operating rules (hardening): with fewer than MIN_LINES_FOR_S effective lines,
S and P(miss) are withheld (consensus not reliable); a set mobile flag
additionally raises a structural warning (visibility limit of the consensus
geometry on mobile networks); an unavailable flag falls back to the default
branch.

Deviations from the evaluation pipeline (deliberate, because no ground truth
exists live — see maltego/README.md, section Limitations):
  * Pseudo-radii per source come from the FROZEN table
    ``radius_table.json`` (global LOO median of the anchor evaluation) instead
    of per-IP leave-one-out. MaxMind uses the live-delivered accuracy_radius.
  * No dataset-based hub/centroid detection (needs frequencies across many
    IPs); instead a warning when MaxMind accuracy_radius >= 500 km.

No cache, no provenance ledger: every call queries live and writes nothing.

Standalone test without Maltego:  python maltego/geoloc.py 9.9.9.9
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# maltego-trx configures the root logger to DEBUG; urllib3 would then write
# every request URL (incl. the ipinfo token) to the stderr log.
logging.getLogger("urllib3").setLevel(logging.WARNING)

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.fetch_sources import SOURCES, query_source, observation_from_raw  # noqa: E402
from estimators.baselines import weighted_geometric_median                  # noqa: E402
from eval.metrics import haversine_error                                    # noqa: E402

EPS_KM = 10.0        # damping of the radius weighting -- identical to the headline
                     # configuration (anchor_eps: grid value at the LOO median of
                     # 7.0 km, see exp_t6_defaults.main) AND to the frames on which
                     # s_calibration.json was fitted. NOT 56 (which in
                     # exp_t6_defaults is only the fallback default).
FALLBACK_RADIUS_KM = 56.0  # pseudo-radius for sources without a radius_table entry:
                     # deliberately CONSERVATIVE, above the largest tabulated
                     # value (42.3 km), i.e. strongest down-weighting.
S_RADIUS_KM = 50.0   # city scale for S, pre-specified (not optimized)
HUB_RADIUS_KM = 500  # MaxMind accuracy_radius from which hub/default suspicion arises
MIN_LINES_FOR_S = 3  # below this, S is withheld (consensus not reliable).
# Empirically supported (experiments/exp_min_lines_s.py, exhaustive line ablation):
# k=1 -> S is ~constant by construction, no signal; k=2 -> out-of-fold quality
# ranges from +0.02 to +0.63 BSS depending on the retained line pair (a lottery);
# from k=3 on, EVERY subset yields BSS >= +0.10. What is counted are EFFECTIVE
# lines (distinct lineage tokens of the responding sources), not raw sources:
# three MaxMind descendants = k=1.
# Detection limit: a newly configured source without verified provenance
# necessarily counts as its own line and can overestimate k —
# maintain the lineage assignment in data/fetch_sources.SOURCES (cf.
# experiments/exp_source_correlation.py for the empirical check).

# Two-branch design (exp_s_type_routing): ip-api only delivers the mobile flag
# when it is explicitly requested. Process-local extension of the field list —
# the evaluation pipeline does not import this module and remains untouched.
SOURCES["ip_api"] = {**SOURCES["ip_api"],
                     "url": "http://ip-api.com/json/{ip}"
                            "?fields=status,message,lat,lon,city,countryCode,mobile"}

_RADIUS_TABLE = json.loads(
    (Path(__file__).resolve().parent / "radius_table.json").read_text(encoding="utf-8")
)["radii_km"]

# Frozen two-branch calibration S -> P(error > 100 km); optional:
# if the file is missing, the transforms keep running without a probability
# statement.
try:
    _S_CALIBRATION = json.loads(
        (Path(__file__).resolve().parent / "s_calibration.json").read_text(encoding="utf-8"))
except FileNotFoundError:
    _S_CALIBRATION = None


def collect(ip: str, sources: list[str] | None = None,
            ) -> tuple[list[dict], list[tuple[str, str]], dict]:
    """Query all sources live -> (valid observations, failure list, extras).

    Extras: ``mobile_flag`` (True/False/None) from the ip-api response — None
    if ip-api did not respond (the default branch then applies later).

    Web sources run in PARALLEL (thread pool): total duration is thus that of
    the slowest source instead of the sum; a hanging source blocks only up to
    its own timeout (fetch_sources.TIMEOUT, 15 s) and then goes onto the
    failure list. Deliberately WITHOUT a shared requests.Session (not
    thread-safe; one call per source does not justify connection pooling).
    The offline DBs are read serially afterwards — they are fast, and the lazy
    reader initialization in fetch_sources is not built for parallel first
    access.

    Sources without a key/DB file degrade to entries on the failure list —
    the estimator keeps running with the remaining lines.
    """
    names = list(sources or SOURCES)
    web = [n for n in names if "reader" not in SOURCES[n]]

    def _safe_query(name: str) -> dict:
        # query_source catches network/parse errors itself (returns error dicts);
        # this is the defensive line for the rest (e.g. the KeyError guard) —
        # ex.map would otherwise raise a propagated exception during iteration
        # and drop the remaining sources from `entries`.
        try:
            return query_source(name, ip)
        except Exception as e:
            return {"source": name, "ip": ip, "fetched_at_utc": None,
                    "http_status": None, "url": None, "body": None, "error": repr(e)}

    entries: dict[str, dict] = {}
    if web:
        with ThreadPoolExecutor(max_workers=len(web)) as ex:
            for name, entry in zip(web, ex.map(_safe_query, web)):
                entries[name] = entry
    for name in names:
        if "reader" in SOURCES[name]:
            entries[name] = _safe_query(name)

    obs, failed = [], []
    for name in names:                       # stable order as in SOURCES
        try:
            o = observation_from_raw(entries[name])
        except Exception as e:
            # Parsers expect a JSON object; resp.json() however accepts any
            # valid JSON (string/array, e.g. rate-limit responses) -> AttributeError.
            failed.append((name, f"parse_error: {e!r}"))
            continue
        if o["status"] == "success" and o["lat"] is not None and o["lon"] is not None:
            obs.append(o)
        else:
            err = entries[name].get("error")
            failed.append((name, f"{o['status']} ({err})" if err else str(o["status"])))

    body = (entries.get("ip_api") or {}).get("body")
    mobile = bool(body["mobile"]) if isinstance(body, dict) and "mobile" in body else None
    return obs, failed, {"mobile_flag": mobile}


def line_weights(lineages: list[str]) -> np.ndarray:
    """L1: each line (same ``lineage``) receives total weight 1 (cf. eval.pipeline)."""
    counts = Counter(lineages)
    return np.array([1.0 / counts[lin] for lin in lineages], dtype=float)


def point_radius(o: dict) -> float:
    """MaxMind: live-delivered accuracy_radius; otherwise frozen pseudo-radius."""
    if o["source"] == "maxmind_geolite2" and o.get("accuracy_radius") is not None:
        return float(o["accuracy_radius"])
    return float(_RADIUS_TABLE.get(o["source"], FALLBACK_RADIUS_KM))


def robust_estimate(obs: list[dict]) -> tuple[float, float]:
    """L1+b: line- and radius-weighted geometric median."""
    pts = np.array([[o["lat"], o["lon"]] for o in obs], dtype=float)
    w = line_weights([o["lineage"] for o in obs])
    rad = np.array([point_radius(o) for o in obs], dtype=float)
    est = weighted_geometric_median(pts, w / (rad + EPS_KM))
    return (float(est[0]), float(est[1]))


def support_concentration(obs: list[dict], est: tuple[float, float],
                          radius_km: float = S_RADIUS_KM) -> float:
    """S: line-weighted share of the source mass within ``radius_km`` around the
    estimate (cf. exp_support_concentration, core_w)."""
    w = line_weights([o["lineage"] for o in obs])
    w = w / w.sum()
    d = np.array([haversine_error((o["lat"], o["lon"]), est) for o in obs])
    return float(w[d < radius_km].sum())


def miss_probability(s: float, mobile_flag: bool | None) -> tuple[float, str] | None:
    """Frozen two-branch calibration: (P(error > tau), branch name).

    Branch 'mobile' only on explicit mobile=true; False AND None (flag not
    available) end up in the default branch — identical to the fitting in
    make_s_calibration.py. None if no calibration table is present.
    """
    if _S_CALIBRATION is None:
        return None
    branch = "mobile" if mobile_flag is True else "default"
    c = _S_CALIBRATION["branches"][branch]
    p = 1.0 / (1.0 + np.exp(-(c["a"] + c["b"] * s)))
    return float(p), branch


def locate(ip: str, sources: list[str] | None = None) -> dict:
    """Complete run for one IP -> result dict for the transforms.

    Keys: ``estimate`` (lat, lon) | None, ``s`` (support concentration),
    ``s_reliable`` (False with < MIN_LINES_FOR_S effective lines: withhold S
    and P(miss)), ``p_miss``/``branch`` (two-branch calibration,
    None without a reliable S or without a table), ``mobile_flag`` (True/False/
    None), ``n_sources``, ``n_lines``, ``observations`` (per source incl.
    distance to the estimate and radius), ``failed``, ``warnings``.
    """
    obs, failed, extras = collect(ip, sources)
    result = {"ip": ip, "estimate": None, "s": None, "s_reliable": False,
              "p_miss": None, "branch": None, "mobile_flag": extras["mobile_flag"],
              "n_sources": len(obs), "n_lines": 0, "observations": obs,
              "failed": failed, "warnings": []}
    if not obs:
        result["warnings"].append("no source delivered a valid position")
        return result

    est = robust_estimate(obs)
    result["estimate"] = est
    result["n_lines"] = len({o["lineage"] for o in obs})
    result["s"] = support_concentration(obs, est)
    result["s_reliable"] = result["n_lines"] >= MIN_LINES_FOR_S
    for o in obs:
        o["dist_to_estimate_km"] = haversine_error((o["lat"], o["lon"]), est)
        o["radius_km"] = point_radius(o)
        o["line_weight"] = None  # set below (needs all lineages)
    w = line_weights([o["lineage"] for o in obs])
    for o, wi in zip(obs, w):
        o["line_weight"] = float(wi)

    if result["s_reliable"]:
        pm = miss_probability(result["s"], result["mobile_flag"])
        if pm is not None:
            result["p_miss"], result["branch"] = pm
    else:
        result["warnings"].append(
            f"only {result['n_lines']} effective line(s) from "
            f"{result['n_sources']} responding source(s) — S and P(miss) "
            "withheld (measured as not reliable below 3 lines: "
            "forecast quality between ~0 and strong depending on the line pair)")
    if result["mobile_flag"] is True:
        result["warnings"].append(
            "ip-api reports a mobile connection — consensus geometry is structurally "
            "weaker here (about a third of the mislocations are agreeing-but-wrong "
            "consensus, which consensus measures cannot indicate); base rate of large "
            "errors in the mobile branch ~44 %")
    elif result["mobile_flag"] is None:
        result["warnings"].append(
            "ip-api mobile flag not retrievable — default branch as fallback")
    mm = next((o for o in obs if o["source"] == "maxmind_geolite2"), None)
    if mm and mm.get("accuracy_radius") and float(mm["accuracy_radius"]) >= HUB_RADIUS_KM:
        result["warnings"].append(
            f"MaxMind accuracy_radius = {mm['accuracy_radius']} km — suspicion of a "
            "country centroid/hub default (position possibly only country-accurate)")
    if failed:
        result["warnings"].append(
            "no response: " + ", ".join(f"{s} ({st})" for s, st in failed))
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python maltego/geoloc.py <ip>")
    r = locate(sys.argv[1])
    print(json.dumps(r, indent=2, ensure_ascii=False, default=str))
