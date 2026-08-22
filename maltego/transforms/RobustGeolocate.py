import sys
from pathlib import Path

from maltego_trx.maltego import UIM_INFORM, UIM_PARTIAL
from maltego_trx.transform import DiscoverableTransform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import geoloc  # noqa: E402


class RobustGeolocate(DiscoverableTransform):
    """IP -> robust consensus location (L1+b) with support concentration S.

    Queries all configured sources live, collapses correlated providers via
    line weights, weights by accuracy radius, and returns the weighted
    geometric median as a Location entity. S, the P(error > 100 km) calibrated
    from it (two-branch design: default/mobile via the ip-api flag), the
    effective line count, and warnings (mobile structure, hub suspicion,
    failed sources) are attached to the entity as properties/notes. With < 3
    effective lines, S and P(miss) are withheld.
    Triage lead, not forensic evidence.
    """

    @classmethod
    def create_entities(cls, request, response):
        ip = request.Value.strip()
        r = geoloc.locate(ip)

        if r["estimate"] is None:
            response.addUIMessage(
                f"{ip}: no source delivered a valid position", UIM_PARTIAL)
            return

        lat, lon = r["estimate"]
        # Label = display heuristic: city/country of the observation closest to the
        # estimate. The actual statement is the median COORDINATES — the median can
        # in theory lie in a different city than the label point.
        near = min(r["observations"], key=lambda o: o["dist_to_estimate_km"])
        label = ", ".join(x for x in (near.get("city"), near.get("country")) if x) \
            or f"{lat:.4f}, {lon:.4f}"

        ent = response.addEntity("maltego.Location", label)
        ent.addProperty("latitude", "Latitude", "strict", f"{lat:.6f}")
        ent.addProperty("longitude", "Longitude", "strict", f"{lon:.6f}")
        # Hardening: only report S (and the P(miss) calibrated from it) when
        # enough independent lines respond — otherwise withhold explicitly,
        # instead of writing an unreliable number into the graph.
        if r["s_reliable"]:
            ent.addProperty("robustgeo.s", "Support concentration S (0-1)", "loose",
                            f"{r['s']:.3f}")
            ent.setWeight(int(round(r["s"] * 100)))
        else:
            ent.addProperty("robustgeo.s", "Support concentration S (0-1)", "loose",
                            f"withheld (only {r['n_lines']} lines)")
        if r["p_miss"] is not None:
            ent.addProperty("robustgeo.p_miss",
                            "P(error > 100 km), calibrated", "loose",
                            f"{r['p_miss']:.0%}")
            ent.addProperty("robustgeo.branch", "Calibration branch", "loose",
                            r["branch"])
        if r["mobile_flag"] is not None:
            ent.addProperty("robustgeo.mobile", "ip-api mobile flag", "loose",
                            str(r["mobile_flag"]))
        ent.addProperty("robustgeo.n_lines", "Effective lines", "loose",
                        str(r["n_lines"]))
        ent.addProperty("robustgeo.n_sources", "Sources with a response", "loose",
                        str(r["n_sources"]))
        ent.addProperty("robustgeo.method", "Method", "loose",
                        "line- and radius-weighted geometric median (L1+b)")

        s_line = (f"S = {r['s']:.3f} (share of source mass &lt; {geoloc.S_RADIUS_KM:.0f} km "
                  f"around the estimate)" if r["s_reliable"]
                  else f"S withheld: only {r['n_lines']} effective line(s)")
        p_line = (f"\nP(error &gt; 100 km) = {r['p_miss']:.0%}  "
                  f"[branch: {r['branch']}, frozen calibration on a mixed "
                  f"probe population]" if r["p_miss"] is not None else "")
        per_source = "\n".join(
            f"{o['source']:>18}: ({o['lat']:.4f}, {o['lon']:.4f})  "
            f"{o['dist_to_estimate_km']:7.1f} km to the estimate  "
            f"[line {o['lineage']}, w={o['line_weight']:.2f}, r={o['radius_km']:.0f} km]"
            for o in sorted(r["observations"], key=lambda o: o["dist_to_estimate_km"]))
        ent.addDisplayInformation(
            f"<pre>{s_line}{p_line}\n\n{per_source}</pre>", "Robust GeoIP: source situation")

        for wmsg in r["warnings"]:
            response.addUIMessage(f"{ip}: {wmsg}", UIM_PARTIAL)
        head = (f"S={r['s']:.2f}" if r["s_reliable"] else "S withheld")
        if r["p_miss"] is not None:
            head += f", P(miss)={r['p_miss']:.0%} ({r['branch']})"
        response.addUIMessage(
            f"{ip}: {head} at {r['n_lines']} effective lines — "
            "triage lead, not forensic evidence", UIM_INFORM)
