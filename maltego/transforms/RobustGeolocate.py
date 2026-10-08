import sys
from pathlib import Path

from maltego_trx.maltego import UIM_INFORM, UIM_PARTIAL
from maltego_trx.transform import DiscoverableTransform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import geoloc    # noqa: E402
import registry  # noqa: E402

# Entity properties a record date is read from (the first one found counts).
# In Maltego: give the IPv4Address entity a dynamic property "as_of"
# (Detail View -> Properties -> +), value e.g. 2024-03-15 or 2024-03-15T10:22:00Z.
AS_OF_PROPERTIES = ("as_of", "robustgeo.as_of", "timestamp", "datetime", "date", "seen_at",
                    "properties.as_of", "properties.timestamp", "properties.date")


class RobustGeolocate(DiscoverableTransform):
    """IP -> robust consensus location (L1+b) with support concentration S and registry country.

    Queries all configured sources live, collapses correlated providers via
    line weights, weights by accuracy radius, and returns the weighted
    geometric median as a Location entity. S, the P(error > 100 km) calibrated
    from it (two-branch design: default/mobile via the ip-api flag), the
    effective line count, and warnings (mobile structure, hub suspicion,
    failed sources) are attached to the entity as properties/notes. With < 3
    effective lines, S and P(miss) are withheld.

    Additionally (2026-09): the registry country according to the RIR
    delegation (today) and according to whois/RDAP (most specific object) —
    the registry is authoritative for the ADDRESSEE of an order, the country
    vote for the location; a contradiction is a measurably useful warning
    signal. If the entity carries a property ``as_of`` (record date), the
    registry country as of that date is read from the archived delegation file
    and a warning is raised if the record is old or the address has been
    re-delegated since: GeoIP sources have no history, the estimate holds for
    the current holder. Triage lead, not forensic evidence.
    """

    @classmethod
    def create_entities(cls, request, response):
        ip = request.Value.strip()
        r = geoloc.locate(ip)

        as_of = None
        props = {k.lower(): v for k, v in (getattr(request, "Properties", {}) or {}).items()}
        for key in AS_OF_PROPERTIES:
            if props.get(key):
                as_of = registry.parse_timestamp(props[key])
                if as_of is not None:
                    break
                response.addUIMessage(f"{ip}: record date '{props[key]}' not readable (ISO 8601 expected)", UIM_PARTIAL)
        try:
            ctx = registry.registry_context(ip, as_of, rdap=True, log=lambda *_: None)
        except Exception as e:                                   # the registry must never block the geolocation
            ctx = {"rdap_error": repr(e)}
        reg_warnings = registry.registry_warnings(ctx, r.get("country"), country_mass=r.get("country_mass"))
        flag = registry.registry_flag(ctx, r.get("country"), r.get("country_mass"))

        if r["estimate"] is None:
            response.addUIMessage(
                f"{ip}: no source delivered a valid position", UIM_PARTIAL)
            for wmsg in reg_warnings:
                response.addUIMessage(f"{ip}: {wmsg}", UIM_PARTIAL)
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
        if r["country"]:
            ent.addProperty("robustgeo.country", "Country (line-weighted core vote)", "loose",
                            f"{r['country']} (mass {r['country_mass']:.2f}{', fallback: nearest source' if r['country_fallback'] else ''})")

        # Registry context
        if ctx.get("registry_snapshot_today"):
            ent.addProperty("robustgeo.registry_country", "Registry country (RIR delegation, today)", "loose",
                            f"{ctx['registry_country_today'] or '--'} "
                            f"[{ctx.get('registry_rir') or '?'}, snapshot {ctx['registry_snapshot_today']}]")
        if ctx.get("registry_snapshot_asof"):
            ent.addProperty("robustgeo.registry_country_asof", "Registry country as of record date", "loose",
                            f"{ctx['registry_country_asof'] or 'unallocated'} "
                            f"[snapshot {ctx['registry_snapshot_asof']}, record {ctx['record_age_days']} days old]")
        if ctx.get("rdap_country") or ctx.get("rdap_handle"):
            ent.addProperty("robustgeo.whois", "Whois/RDAP (most specific object)", "loose",
                            f"{ctx.get('rdap_country') or '--'} {ctx.get('rdap_handle') or ''} "
                            f"({ctx.get('rdap_type') or '?'}, {ctx.get('rdap_rir') or '?'})".strip())
        # Registry flag (registry country != vote, primarily the whois object) + frozen
        # country-error rate of its cell flag x S gate — the country-level counterpart of P(miss).
        if flag["registry_flag"] is not None:
            lvl = "whois object" if flag["registry_level"] == "whois" else "RIR delegation"
            ent.addProperty("robustgeo.registry_flag", "Registry flag (registry country != vote)", "loose",
                            (f"YES: {lvl} {flag['registry_flag_country']} vs. vote {r['country']}"
                             if flag["registry_flag"] else f"no ({lvl} {flag['registry_flag_country']})"))
            if flag["p_country_miss"] is not None:
                ent.addProperty("robustgeo.p_country_miss", "P(country wrong), frozen", "loose",
                                f"{flag['p_country_miss']:.1%} [cell {flag['p_country_miss_cell']}, "
                                f"{flag['registry_level']}, pool n={flag['p_country_miss_n']}]")
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
        reg_line = ""
        if ctx.get("registry_snapshot_today"):
            reg_line = (f"\nRegistry (RIR delegation, {ctx.get('registry_rir') or '?'}): today "
                        f"{ctx['registry_country_today'] or '--'} [snapshot {ctx['registry_snapshot_today']}]")
            if ctx.get("registry_snapshot_asof"):
                reg_line += (f"; as of record date {ctx['registry_country_asof'] or 'unallocated'} "
                             f"[snapshot {ctx['registry_snapshot_asof']}]")
        if ctx.get("rdap_country") or ctx.get("rdap_handle"):
            reg_line += (f"\nWhois/RDAP: {ctx.get('rdap_country') or '--'} {ctx.get('rdap_handle') or ''} "
                         f"({ctx.get('rdap_type') or '?'})")
        if flag["registry_flag"] is not None:
            reg_line += (f"\nRegistry flag: {'RAISED' if flag['registry_flag'] else 'not raised'} "
                         f"[{flag['registry_level']}]")
            if flag["p_country_miss"] is not None:
                reg_line += (f" — P(country wrong) = {flag['p_country_miss']:.1%} "
                             f"[frozen rate of cell {flag['p_country_miss_cell']}, probe pool n={flag['p_country_miss_n']}]")
        per_source = "\n".join(
            f"{o['source']:>18}: ({o['lat']:.4f}, {o['lon']:.4f})  "
            f"{o['dist_to_estimate_km']:7.1f} km to the estimate  "
            f"[line {o['lineage']}, w={o['line_weight']:.2f}, r={o['radius_km']:.0f} km]"
            for o in sorted(r["observations"], key=lambda o: o["dist_to_estimate_km"]))
        ent.addDisplayInformation(
            f"<pre>{s_line}{p_line}{reg_line}\n\n{per_source}</pre>", "Robust GeoIP: source situation")

        for wmsg in reg_warnings + list(r["warnings"]):
            response.addUIMessage(f"{ip}: {wmsg}", UIM_PARTIAL)
        head = (f"S={r['s']:.2f}" if r["s_reliable"] else "S withheld")
        if r["p_miss"] is not None:
            head += f", P(miss)={r['p_miss']:.0%} ({r['branch']})"
        if r.get("country"):
            head += f", vote {r['country']}"
            reg = ctx.get("registry_country_asof") if as_of else ctx.get("registry_country_today")
            if reg:
                head += f", registry {reg}"
            if flag["registry_flag"]:
                head += ", REGISTRY FLAG"
                if flag["p_country_miss"] is not None:
                    head += f" P(country wrong)={flag['p_country_miss']:.0%}"
        response.addUIMessage(
            f"{ip}: {head} at {r['n_lines']} effective lines — "
            "triage lead, not forensic evidence", UIM_INFORM)
