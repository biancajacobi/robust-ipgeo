import sys
from pathlib import Path

from maltego_trx.maltego import UIM_INFORM, UIM_PARTIAL
from maltego_trx.transform import DiscoverableTransform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import geoloc  # noqa: E402


class RobustGeolocate(DiscoverableTransform):
    """IP -> robuster Konsens-Standort (L1+b) mit Stuetz-Konzentration S.

    Fragt alle konfigurierten Quellen live ab, kollabiert korrelierte Anbieter
    per Linien-Gewicht, gewichtet nach Genauigkeitsradius und liefert den
    gewichteten geometrischen Median als Location-Entity. S, effektive
    Linienzahl und Warnungen (Hub-Verdacht, ausgefallene Quellen) haengen als
    Properties/Notizen an der Entity. Triage-Indiz, kein forensischer Beweis.
    """

    @classmethod
    def create_entities(cls, request, response):
        ip = request.Value.strip()
        r = geoloc.locate(ip)

        if r["estimate"] is None:
            response.addUIMessage(
                f"{ip}: keine Quelle lieferte eine gueltige Position", UIM_PARTIAL)
            return

        lat, lon = r["estimate"]
        near = min(r["observations"], key=lambda o: o["dist_to_estimate_km"])
        label = ", ".join(x for x in (near.get("city"), near.get("country")) if x) \
            or f"{lat:.4f}, {lon:.4f}"

        ent = response.addEntity("maltego.Location", label)
        ent.addProperty("latitude", "Latitude", "strict", f"{lat:.6f}")
        ent.addProperty("longitude", "Longitude", "strict", f"{lon:.6f}")
        ent.addProperty("robustgeo.s", "Stuetz-Konzentration S (0-1)", "loose",
                        f"{r['s']:.3f}")
        ent.addProperty("robustgeo.n_lines", "effektive Linien", "loose",
                        str(r["n_lines"]))
        ent.addProperty("robustgeo.n_sources", "Quellen mit Antwort", "loose",
                        str(r["n_sources"]))
        ent.addProperty("robustgeo.method", "Methode", "loose",
                        "linien- u. radiusgewichteter geometrischer Median (L1+b)")
        ent.setWeight(int(round(r["s"] * 100)))

        per_source = "\n".join(
            f"{o['source']:>18}: ({o['lat']:.4f}, {o['lon']:.4f})  "
            f"{o['dist_to_estimate_km']:7.1f} km zum Schaetzer  "
            f"[Linie {o['lineage']}, w={o['line_weight']:.2f}, r={o['radius_km']:.0f} km]"
            for o in sorted(r["observations"], key=lambda o: o["dist_to_estimate_km"]))
        ent.addDisplayInformation(
            f"<pre>S = {r['s']:.3f} (Anteil Quellen-Masse &lt; {geoloc.S_RADIUS_KM:.0f} km "
            f"um den Schaetzer)\n\n{per_source}</pre>", "Robust-GeoIP: Quellenlage")

        for wmsg in r["warnings"]:
            response.addUIMessage(f"{ip}: {wmsg}", UIM_PARTIAL)
        response.addUIMessage(
            f"{ip}: S={r['s']:.2f} bei {r['n_lines']} effektiven Linien — "
            "Triage-Indiz, kein forensischer Beweis", UIM_INFORM)
