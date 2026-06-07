"""Pipeline: Beobachtungen je IP auflösen → Schätzer rechnen → gegen Ground
Truth bewerten.

Die Pipeline ist selbst ein forensisches Artefakt (FF4): jeder Schätzwert ist
über ``Case.provenance`` auf die zugrunde liegenden, provenancierten Quellwerte
(Quelle, Koordinate, Zeitpunkt, Status) zurückführbar. Roh-Beleg + Hash-Kette
liegen in ``data/`` (siehe ``data/store``); hier wird die Kette
Quellwert → Schätzung → Fehler hergestellt.

Konvention durchgängig: Koordinaten als (lat, lon) in Dezimalgrad.
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
    """Anchors + Observations zu Fällen je IP verbinden.

    Ein *Fall* (Case-Dict) bündelt: ``ip``, ``truth=(lat, lon)`` (Anchor),
    ``points`` (np.ndarray (n, 2) der gültigen Quell-Schätzungen), ``sources``
    und ``provenance`` (je Quellwert Quelle/Koordinate/Zeit/Status).

    Nur Beobachtungen mit ``status`` in ``statuses`` und gültigen Koordinaten
    werden zu Punkten. Anchors ohne verwertbare Beobachtung entfallen.
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
    """Gewicht αᵢ je Punkt: jede Linie (gleiche ``lineage``) erhält Gesamtgewicht
    1, gleichmäßig auf ihre Mitglieder verteilt. So zählen korrelierte Quellen
    zusammen wie eine — ohne sie zu verwerfen (für den gewichteten GM)."""
    from collections import Counter
    counts = Counter(lineages)
    return np.array([1.0 / counts[lin] for lin in lineages], dtype=float)


def estimate_case(points: np.ndarray, estimators: dict) -> dict:
    """Jeden Schätzer auf die Punktwolke eines Falls anwenden.

    Rückgabe: {name: (lat, lon)} bzw. {name: None}, falls der Schätzer für
    diesen Fall nicht definiert ist (z. B. zu wenige Punkte, noch nicht
    implementiert) — Fehler werden gefangen, nicht propagiert.
    """
    out = {}
    for name, fn in estimators.items():
        try:
            est = np.asarray(fn(points), dtype=float)
            out[name] = (float(est[0]), float(est[1]))
        except Exception as exc:
            logger.warning("Schätzer %r schlug fehl (%d Punkte): %s",
                           name, len(points), exc)
            out[name] = None
    return out


def evaluate(cases: list[dict], estimators: dict,
             include_sources: bool = False,
             line_weighted: dict | None = None) -> pd.DataFrame:
    """Alle Schätzer auf allen Fällen rechnen und Haversine-Fehler bestimmen.

    Liefert ein Long-Format-DataFrame (eine Zeile je Fall×Schätzer):
    ``ip, estimator, kind, n_sources, n_lines, est_lat, est_lon, error_km, asn, country``.

    - ``estimators``: dict ``{name: fn(points)}`` — pro Quelle gleich gewichtet.
    - ``line_weighted``: optional dict ``{name: fn(points, weights)}`` — mit
      Linien-Gewichten (1 pro ``lineage``), damit korrelierte Quellen nicht
      dominieren (vgl. ``line_weights``).
    - ``include_sources``: zusätzlich je Einzelquelle eine ``kind="source"``-Zeile
      (``estimator="src:<quelle>"``) — für „beste Einzelquelle" in E1.
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
                    logger.warning("Linien-gew. Schätzer %r schlug fehl (ip=%s): %s",
                                   name, case["ip"], exc)
                    est = None
                rows.append(_row(name, est))

        if include_sources:
            for p in case["provenance"]:
                rows.append(_row(f"src:{p['source']}", (p["lat"], p["lon"]),
                                 kind="source"))
    return pd.DataFrame(rows)
