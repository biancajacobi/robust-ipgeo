"""Regenerate radius_table.json (after a fresh anchor fetch/reindex).

Computes, per source, the global leave-one-out median of the source error on
the anchor dataset (identical to experiments/exp_t6_defaults.loo_pseudo_radii,
entry "_global") and freezes it as a static table for live operation of the
Maltego transforms.

CAUTION: requires the REAL anchor data (data/cache/ + anchors). The bundled
radius_table.json was generated from the original data -- there is normally no
reason to rebuild it; running this on pseudonymized or third-party data would
silently produce a wrong table.

Usage:  python maltego/make_radius_table.py
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
            "description": (
                "Frozen per-source pseudo radii (km): global leave-one-out "
                "median of the source error from the anchor evaluation of the "
                "accompanying paper. Replaces the per-IP LOO radii in live "
                "operation, since no ground truth exists for an unknown IP. "
                "MaxMind keeps using its live-reported accuracy_radius."
            ),
            "generated_by": "maltego/make_radius_table.py",
            "data_state": (f"anchors dataset, n={len(cases)} cases, as of "
                           f"{datetime.now(timezone.utc).date().isoformat()}"),
            "unit": "km",
        },
        "radii_km": table,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(table)} sources, n={len(cases)} cases).")


if __name__ == "__main__":
    main()
