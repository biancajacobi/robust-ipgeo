"""Hub-/Centroid-Default-Erkennung für Geo-DB-Ausgaben.

Viele Geo-DBs setzen für schlecht lokalisierbare IPs einen **Fallback-Punkt**
(Länder-/Regions-Schwerpunkt oder einen großen Hub wie Frankfurt/Amsterdam/
Singapur), statt zuzugeben „weiß ich nicht". Solche Pseudo-Punkte sind keine
Messung, sondern eine Heuristik der Quelle — und sie verzerren einen Punkt-
schätzer systematisch. Dieses Modul flaggt sie.

Weil DB-IP/IP2Location (anders als MaxMind) **keinen** ``accuracy_radius`` liefern,
gibt es zwei bewusst unterschiedlich strenge Wächter; beide werden berichtet:

  WÄCHTER A (konservativ, Untergrenze): ein Hub-Punkt zählt nur als Default, wenn
    zusätzlich der ASN auf ihn *kollabiert* (>= ASN_COLLAPSE der IPs eines ASN auf
    demselben Punkt). Verpasst Per-IP-/Sub-ASN-Defaults, flaggt aber keine echten
    regionalen Stadt-ISPs falsch.
  WÄCHTER B (liberal, Obergrenze): jeder Punkt auf einem Frequenz-Hub oder einem
    bekannten Geo-Mittelpunkt zählt als Default (ohne ASN-Bedingung).

Beide Wächter teilen zwei *saubere Per-IP-Signale*: ``city is None`` (Quelle gibt
keine Stadt zurück) und — nur MaxMind — ``accuracy_radius >= MM_RADIUS_DEFAULT``.

Die Hub-Liste ist **datenabgeleitet** (Koordinaten-Häufigkeit über den Datensatz),
ergänzt um eine kleine manuelle Liste geografischer Mittelpunkte. Schwellen sind
hier **eingefroren** (Stand 2026-06-04) und VOR der Messung festgelegt, damit die
Auswertung nicht nach Heuristik-Tuning aussieht.
"""

from __future__ import annotations

from collections import Counter, defaultdict

# --- eingefrorene Schwellen (vor der Messung festgelegt) ---
FREQ_HUB = 10            # >= N IPs auf exakt derselben Koordinate -> Frequenz-Hub
HUB_STRONG = 50          # >= N -> dominanter datenabgeleiteter Mittelpunkt (Doku)
ASN_COLLAPSE = 0.90      # Anteil der IPs eines ASN auf demselben Punkt -> Kollaps (A)
CENTROID_TOL = 0.10      # Grad-Toleranz für Match gegen bekannte Geo-Mittelpunkte
MM_RADIUS_DEFAULT = 500  # MaxMind accuracy_radius (km), ab dem ein Treffer grob ist

# Bekannte geografische Mittelpunkte (manueller Startpunkt; die eigentliche
# Hub-Liste wird datenabgeleitet ergänzt, s. build_index).
KNOWN_CENTROIDS = [
    (37.751, -97.822),   # MaxMind US-Fallback (~Kansas)
    (39.828, -98.579),   # geografischer US-Mittelpunkt
    (51.165, 10.452),    # DE geografische Mitte
    (46.603, 1.888),     # FR geografische Mitte
    (47.001, 8.014),     # CH geografische Mitte
]


def round_coord(lat, lon):
    return (round(float(lat), 3), round(float(lon), 3))


def _near_known(c):
    return any(abs(c[0] - k[0]) <= CENTROID_TOL and abs(c[1] - k[1]) <= CENTROID_TOL
               for k in KNOWN_CENTROIDS)


def build_index(records):
    """records: iterable von dict(source, ip, lat, lon, city, country, radius, asn).

    Rückgabe: (freq, asn_dom)
      freq[source]    Counter{coord: n}  — Häufigkeit jeder Koordinate je Quelle
      asn_dom[s][asn] (dominante_coord, anteil) — größter Koordinaten-Anteil im ASN
    """
    freq = defaultdict(Counter)
    asn_coords = defaultdict(lambda: defaultdict(Counter))
    for r in records:
        if r.get("lat") is None:
            continue
        c = round_coord(r["lat"], r["lon"])
        freq[r["source"]][c] += 1
        if r.get("asn"):
            asn_coords[r["source"]][r["asn"]][c] += 1
    asn_dom = {}
    for s, amap in asn_coords.items():
        asn_dom[s] = {}
        for asn, cc in amap.items():
            coord, n = cc.most_common(1)[0]
            asn_dom[s][asn] = (coord, n / sum(cc.values()))
    return freq, asn_dom


def hub_list(freq, source, threshold=HUB_STRONG):
    """Datenabgeleitete dominante Mittelpunkte einer Quelle (für Doku/Reporting)."""
    return {c: n for c, n in freq[source].items() if n >= threshold}


def is_default(r, guard, freq, asn_dom):
    """True/False, ob (source, ip) ein Hub-/Centroid-Default ist; None ohne Daten.

    guard: 'A' (konservativ, mit ASN-Kollaps) oder 'B' (liberal, nur Hub).
    """
    if r.get("lat") is None:
        return None
    c = round_coord(r["lat"], r["lon"])
    # gemeinsame, saubere Per-IP-Signale (beide Wächter)
    if not r.get("city"):
        return True
    if r["source"] == "maxmind_geolite2" and r.get("radius") and r["radius"] >= MM_RADIUS_DEFAULT:
        return True
    is_hub = freq[r["source"]][c] >= FREQ_HUB or _near_known(c)
    if guard == "B":
        return is_hub
    # Wächter A: Hub UND ASN-Kollaps auf genau diesen Punkt
    dom = asn_dom.get(r["source"], {}).get(r.get("asn"))
    asn_collapsed = dom is not None and dom[0] == c and dom[1] >= ASN_COLLAPSE
    return is_hub and asn_collapsed


def detect(r, freq):
    """Für die Annotation der Beobachtungen: (is_default_centroid, centroid_match).

    Verwendet Wächter B (liberal) — der per-Beobachtung sinnvolle „liegt auf einem
    bekannten Hub?"-Flag (genau die Hub-Achse der 2D-Konfidenzmatrix). ``centroid_match``
    erklärt den Treffer (Grund bzw. Hub-Koordinate). Wächter A (ASN-bedingt) wird in
    den Experimenten aus den Rohfeldern nachgerechnet.
    """
    if r.get("lat") is None:
        return False, ""
    c = round_coord(r["lat"], r["lon"])
    if not r.get("city"):
        return True, "no_city"
    if r["source"] == "maxmind_geolite2" and r.get("radius") and r["radius"] >= MM_RADIUS_DEFAULT:
        return True, f"r>={MM_RADIUS_DEFAULT}km"
    if freq[r["source"]][c] >= FREQ_HUB:
        return True, f"hub:{c[0]},{c[1]}"
    if _near_known(c):
        return True, f"centroid:{c[0]},{c[1]}"
    return False, ""
