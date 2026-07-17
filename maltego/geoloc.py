"""Live-Lokalisierung EINER IP mit dem Verfahren der Arbeit (fuer die Maltego-Transforms).

Duenner Wrapper um den bestehenden Code: Quellen-Abruf und Normalisierung kommen
unveraendert aus ``data/fetch_sources.py``, der Schaetzer aus
``estimators/baselines.py``. Hier passiert nur die Einzel-IP-Verdrahtung:

  Abruf aller Quellen -> Linien-Gewichte (L1) -> Genauigkeitsradius-Gewichtung
  -> gewichteter geometrischer Median (L1+b) -> Stuetz-Konzentration S.

Abweichungen gegenueber der Evaluations-Pipeline (bewusst, weil live keine
Ground Truth existiert — siehe maltego/README.md, Abschnitt Limitationen):
  * Pseudo-Radien je Quelle kommen aus der EINGEFRORENEN Tabelle
    ``radius_table.json`` (globaler LOO-Median der Anchor-Evaluation) statt
    aus per-IP-Leave-one-out. MaxMind nutzt den live gelieferten accuracy_radius.
  * Keine datensatzbasierte Hub-/Centroid-Detektion (braucht Haeufigkeiten ueber
    viele IPs); stattdessen Warnung bei MaxMind-accuracy_radius >= 500 km.

Kein Cache, kein Provenance-Ledger: jeder Aufruf fragt live ab und schreibt nichts.

Standalone-Test ohne Maltego:  python maltego/geoloc.py 9.9.9.9
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

# maltego-trx konfiguriert den Root-Logger auf DEBUG; urllib3 wuerde dann jede
# Request-URL (inkl. ipinfo-Token) ins stderr-Log schreiben.
logging.getLogger("urllib3").setLevel(logging.WARNING)

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.fetch_sources import SOURCES, query_source, observation_from_raw  # noqa: E402
from estimators.baselines import weighted_geometric_median                  # noqa: E402
from eval.metrics import haversine_error                                    # noqa: E402

EPS_KM = 56.0        # Daempfung der Radius-Gewichtung, wie exp_t6_defaults.EPS_KM
S_RADIUS_KM = 50.0   # City-Skala fuer S, vorab spezifiziert (Thesis, nicht optimiert)
HUB_RADIUS_KM = 500  # MaxMind-accuracy_radius ab dem Hub-/Default-Verdacht besteht

_RADIUS_TABLE = json.loads(
    (Path(__file__).resolve().parent / "radius_table.json").read_text(encoding="utf-8")
)["radii_km"]


def collect(ip: str, sources: list[str] | None = None) -> tuple[list[dict], list[tuple[str, str]]]:
    """Alle Quellen live abfragen -> (gueltige Beobachtungen, Fehlliste).

    Quellen ohne Key/DB-Datei degradieren zu Eintraegen in der Fehlliste —
    der Schaetzer laeuft mit den verbleibenden Linien weiter.
    """
    session = requests.Session()
    obs, failed = [], []
    for name in (sources or list(SOURCES)):
        o = observation_from_raw(query_source(name, ip, session=session))
        if o["status"] == "success" and o["lat"] is not None and o["lon"] is not None:
            obs.append(o)
        else:
            failed.append((name, str(o["status"])))
    return obs, failed


def line_weights(lineages: list[str]) -> np.ndarray:
    """L1: jede Linie (gleiche ``lineage``) erhaelt Gesamtgewicht 1 (vgl. eval.pipeline)."""
    counts = Counter(lineages)
    return np.array([1.0 / counts[lin] for lin in lineages], dtype=float)


def point_radius(o: dict) -> float:
    """MaxMind: live gelieferter accuracy_radius; sonst eingefrorener Pseudo-Radius."""
    if o["source"] == "maxmind_geolite2" and o.get("accuracy_radius") is not None:
        return float(o["accuracy_radius"])
    return float(_RADIUS_TABLE.get(o["source"], EPS_KM))


def robust_estimate(obs: list[dict]) -> tuple[float, float]:
    """L1+b: linien- und radiusgewichteter geometrischer Median."""
    pts = np.array([[o["lat"], o["lon"]] for o in obs], dtype=float)
    w = line_weights([o["lineage"] for o in obs])
    rad = np.array([point_radius(o) for o in obs], dtype=float)
    est = weighted_geometric_median(pts, w / (rad + EPS_KM))
    return (float(est[0]), float(est[1]))


def support_concentration(obs: list[dict], est: tuple[float, float],
                          radius_km: float = S_RADIUS_KM) -> float:
    """S: linien-gewichteter Anteil der Quellen-Masse innerhalb ``radius_km`` um den
    Schaetzer (vgl. exp_support_concentration, core_w)."""
    w = line_weights([o["lineage"] for o in obs])
    w = w / w.sum()
    d = np.array([haversine_error((o["lat"], o["lon"]), est) for o in obs])
    return float(w[d < radius_km].sum())


def locate(ip: str, sources: list[str] | None = None) -> dict:
    """Komplettlauf fuer eine IP -> Ergebnis-Dict fuer die Transforms.

    Schluessel: ``estimate`` (lat, lon) | None, ``s`` (Stuetz-Konzentration),
    ``n_sources``, ``n_lines``, ``observations`` (je Quelle inkl. Distanz zum
    Schaetzer und Radius), ``failed`` [(quelle, status)], ``warnings`` [str].
    """
    obs, failed = collect(ip, sources)
    result = {"ip": ip, "estimate": None, "s": None, "n_sources": len(obs),
              "n_lines": 0, "observations": obs, "failed": failed, "warnings": []}
    if not obs:
        result["warnings"].append("keine Quelle lieferte eine gueltige Position")
        return result

    est = robust_estimate(obs)
    result["estimate"] = est
    result["n_lines"] = len({o["lineage"] for o in obs})
    result["s"] = support_concentration(obs, est)
    for o in obs:
        o["dist_to_estimate_km"] = haversine_error((o["lat"], o["lon"]), est)
        o["radius_km"] = point_radius(o)
        o["line_weight"] = None  # unten gesetzt (braucht alle lineages)
    w = line_weights([o["lineage"] for o in obs])
    for o, wi in zip(obs, w):
        o["line_weight"] = float(wi)

    if result["n_lines"] < 3:
        result["warnings"].append(
            f"nur {result['n_lines']} effektive Linie(n) — Konsens wenig aussagekraeftig")
    mm = next((o for o in obs if o["source"] == "maxmind_geolite2"), None)
    if mm and mm.get("accuracy_radius") and float(mm["accuracy_radius"]) >= HUB_RADIUS_KM:
        result["warnings"].append(
            f"MaxMind accuracy_radius = {mm['accuracy_radius']} km — Verdacht auf "
            "Laendercentroid/Hub-Default (Position ggf. nur landesgenau)")
    if failed:
        result["warnings"].append(
            "ohne Antwort: " + ", ".join(f"{s} ({st})" for s, st in failed))
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Aufruf: python maltego/geoloc.py <ip>")
    r = locate(sys.argv[1])
    print(json.dumps(r, indent=2, ensure_ascii=False, default=str))
