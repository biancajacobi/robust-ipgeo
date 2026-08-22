"""Pipeline: resolve observations per IP → compute estimators → evaluate
against ground truth.

The pipeline is itself a forensic artifact (RQ4): every estimate is traceable
via ``Case.provenance`` to the underlying, provenance-tracked source values
(source, coordinate, timestamp, status). Raw evidence + hash chain live in
``data/`` (see ``data/store``); here the chain
source value → estimate → error is established.

Convention throughout: coordinates as (lat, lon) in decimal degrees.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import store  # noqa: E402
from eval.metrics import haversine_error  # noqa: E402

logger = logging.getLogger(__name__)


def _to_float(x):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def load_cases(anchors: list[dict] | None = None,
               observations: list[dict] | None = None,
               statuses=("success",)) -> list[dict]:
    """Join anchors + observations into cases per IP.

    A *case* (case dict) bundles: ``ip``, ``truth=(lat, lon)`` (anchor),
    ``points`` (np.ndarray (n, 2) of the valid source estimates), ``sources``
    and ``provenance`` (per source value: source/coordinate/time/status).

    Only observations with ``status`` in ``statuses`` and valid coordinates
    become points. Anchors without a usable observation are dropped.
    """
    anchors = store.load_anchors_csv() if anchors is None else anchors
    observations = store.load_observations_csv() if observations is None else observations

    by_ip: dict[str, dict] = {}
    for a in anchors:
        ip = a.get("ip")
        lat, lon = _to_float(a.get("lat")), _to_float(a.get("lon"))
        if not ip or lat is None or lon is None:
            continue
        by_ip[ip] = {"ip": ip, "truth": (lat, lon), "city": a.get("city"),
                     "country": a.get("country"), "asn": a.get("asn"), "_obs": []}

    for o in observations:
        case = by_ip.get(o.get("ip"))
        if case is not None:
            case["_obs"].append(o)

    cases = []
    for case in by_ip.values():
        prov = []
        for o in case.pop("_obs"):
            lat, lon = _to_float(o.get("lat")), _to_float(o.get("lon"))
            if o.get("status") in statuses and lat is not None and lon is not None:
                idc = o.get("is_default_centroid")
                prov.append({"source": o.get("source"),
                             "lineage": o.get("lineage") or "unknown",
                             "lat": lat, "lon": lon,
                             "accuracy_radius": _to_float(o.get("accuracy_radius")),
                             "is_default_centroid": idc in (True, "True", "true"),
                             "fetched_at_utc": o.get("fetched_at_utc"),
                             "status": o.get("status")})
        if not prov:
            continue
        case["provenance"] = prov
        case["sources"] = [p["source"] for p in prov]
        case["points"] = np.array([[p["lat"], p["lon"]] for p in prov], dtype=float)
        cases.append(case)
    return cases


def line_weights(lineages) -> np.ndarray:
    """Weight αᵢ per point: each line (same ``lineage``) receives total weight
    1, distributed evenly across its members. This way correlated sources count
    together as one — without discarding them (for the weighted GM)."""
    from collections import Counter
    counts = Counter(lineages)
    return np.array([1.0 / counts[lin] for lin in lineages], dtype=float)


def estimate_case(points: np.ndarray, estimators: dict) -> dict:
    """Apply every estimator to a case's point cloud.

    Returns: {name: (lat, lon)} or {name: None} if the estimator is not
    defined for this case (e.g. too few points, not yet implemented) —
    errors are caught, not propagated.
    """
    out = {}
    for name, fn in estimators.items():
        try:
            est = np.asarray(fn(points), dtype=float)
            out[name] = (float(est[0]), float(est[1]))
        except Exception as exc:
            logger.warning("Estimator %r failed (%d points): %s",
                           name, len(points), exc)
            out[name] = None
    return out


def evaluate(cases: list[dict], estimators: dict,
             include_sources: bool = False,
             line_weighted: dict | None = None) -> pd.DataFrame:
    """Compute all estimators on all cases and determine haversine errors.

    Returns a long-format DataFrame (one row per case×estimator):
    ``ip, estimator, kind, n_sources, n_lines, est_lat, est_lon, error_km, asn, country``.

    - ``estimators``: dict ``{name: fn(points)}`` — equally weighted per source.
    - ``line_weighted``: optional dict ``{name: fn(points, weights)}`` — with
      line weights (1 per ``lineage``) so correlated sources do not
      dominate (cf. ``line_weights``).
    - ``include_sources``: additionally one ``kind="source"`` row per individual
      source (``estimator="src:<source>"``) — for "best single source" in E1.
    """
    rows = []
    for case in cases:
        pts, truth, n = case["points"], case["truth"], len(case["points"])
        lineages = [p["lineage"] for p in case["provenance"]]
        n_lines = len(set(lineages))

        def _row(name, est, kind="aggregate"):
            return {
                "ip": case["ip"], "estimator": name, "kind": kind,
                "n_sources": n, "n_lines": n_lines,
                "est_lat": est[0] if est else None,
                "est_lon": est[1] if est else None,
                "error_km": haversine_error(est, truth) if est else float("nan"),
                "asn": case["asn"], "country": case["country"],
            }

        for name, est in estimate_case(pts, estimators).items():
            rows.append(_row(name, est))

        if line_weighted:
            w = line_weights(lineages)
            for name, fn in line_weighted.items():
                try:
                    e = np.asarray(fn(pts, w), dtype=float)
                    est = (float(e[0]), float(e[1]))
                except Exception as exc:
                    logger.warning("Line-weighted estimator %r failed (ip=%s): %s",
                                   name, case["ip"], exc)
                    est = None
                rows.append(_row(name, est))

        if include_sources:
            for p in case["provenance"]:
                rows.append(_row(f"src:{p['source']}", (p["lat"], p["lon"]),
                                 kind="source"))
    return pd.DataFrame(rows)
