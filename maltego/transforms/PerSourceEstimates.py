import sys
from pathlib import Path

from maltego_trx.maltego import UIM_PARTIAL
from maltego_trx.transform import DiscoverableTransform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import geoloc  # noqa: E402


class PerSourceEstimates(DiscoverableTransform):
    """IP -> eine Location-Entity je Quelle (Widerspruchs-Ansicht).

    Zeigt die rohen Einzel-Schaetzungen aller Quellen nebeneinander im Graphen —
    nuetzlich, um zu sehen, WORAUF der Konsens beruht und wie stark die Quellen
    fuer diese IP auseinanderliegen. Linien-Zugehoerigkeit als Link-Label.
    """

    @classmethod
    def create_entities(cls, request, response):
        ip = request.Value.strip()
        obs, failed = geoloc.collect(ip)

        if not obs:
            response.addUIMessage(
                f"{ip}: keine Quelle lieferte eine gueltige Position", UIM_PARTIAL)
            return

        w = geoloc.line_weights([o["lineage"] for o in obs])
        for o, wi in zip(obs, w):
            label = ", ".join(x for x in (o.get("city"), o.get("country")) if x) \
                or f"{o['lat']:.4f}, {o['lon']:.4f}"
            ent = response.addEntity("maltego.Location", f"{label} [{o['source']}]")
            ent.addProperty("latitude", "Latitude", "strict", f"{float(o['lat']):.6f}")
            ent.addProperty("longitude", "Longitude", "strict", f"{float(o['lon']):.6f}")
            ent.addProperty("robustgeo.source", "Quelle", "loose", o["source"])
            ent.addProperty("robustgeo.lineage", "Linie (lineage)", "loose", o["lineage"])
            ent.addProperty("robustgeo.line_weight", "Linien-Gewicht", "loose", f"{wi:.2f}")
            if o.get("accuracy_radius") is not None:
                ent.addProperty("robustgeo.accuracy_radius", "accuracy_radius (km)",
                                "loose", str(o["accuracy_radius"]))
            ent.setLinkLabel(f"{o['source']} ({o['lineage']})")
            ent.setWeight(int(round(float(wi) * 100)))

        if failed:
            response.addUIMessage(
                f"{ip}: ohne Antwort: " + ", ".join(f"{s} ({st})" for s, st in failed),
                UIM_PARTIAL)
