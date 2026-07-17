"""radius_table.json neu erzeugen (nach einem neuen Anchor-Fetch/Reindex).

Berechnet je Quelle den globalen Leave-one-out-Median des Quellfehlers auf dem
Anchor-Datensatz (identisch zu experiments/exp_t6_defaults.loo_pseudo_radii,
Eintrag "_global") und friert ihn als statische Tabelle fuer den Live-Betrieb
der Maltego-Transforms ein.

ACHTUNG: braucht den ECHTEN Anchor-Datenbestand (data/cache/ + anchors, im
oeffentlichen Repo nicht enthalten bzw. pseudonymisiert). Die mitgelieferte
radius_table.json wurde aus den Originaldaten erzeugt — normalerweise gibt es
keinen Grund, sie neu zu bauen; auf pseudonymisierten/fremden Daten entstuende
eine stillschweigend falsche Tabelle.

Aufruf:  python maltego/make_radius_table.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.pipeline import load_cases                       # noqa: E402
from experiments.exp_t6_defaults import loo_pseudo_radii   # noqa: E402

OUT = Path(__file__).resolve().parent / "radius_table.json"


def main() -> None:
    cases = load_cases()
    loo = loo_pseudo_radii(cases)
    table = {s: round(d["_global"], 1) for s, d in sorted(loo.items())}
    payload = {
        "_meta": {
            "beschreibung": (
                "Eingefrorene Pseudo-Radien je Quelle (km): globaler Leave-one-out-"
                "Median des Quellfehlers aus der Anchor-Evaluation der Arbeit. "
                "Ersetzt im Live-Betrieb die per-IP-LOO-Radien, da fuer eine "
                "unbekannte IP keine Ground Truth vorliegt. MaxMind nutzt weiterhin "
                "den live gelieferten accuracy_radius."
            ),
            "erzeugt_mit": "maltego/make_radius_table.py",
            "datenstand": (f"anchors-Datensatz, n={len(cases)} Faelle, Stand "
                           f"{datetime.now(timezone.utc).date().isoformat()}"),
            "einheit": "km",
        },
        "radii_km": table,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{OUT} geschrieben ({len(table)} Quellen, n={len(cases)} Faelle).")


if __name__ == "__main__":
    main()