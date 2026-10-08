#!/usr/bin/env python3
"""Downloads dated RIR delegation files (delegated-*-extended, RIR Statistics
Exchange Format) of all five registries into data/cache/rir/<YYYYMMDD>/.

Thin wrapper around maltego/registry.py (a single implementation for the
experiment, the batch tool and the transforms). Purpose here: registry-country
baseline for exp_registry_country.py -- per population the snapshot from the
query day of the GeoIP observations in the cache, plus "today" (drift check).
No IP leaves the machine; only the public bulk files are downloaded.

Usage:   python data/fetch_rir.py [YYYYMMDD ...]     (without argument: all dates below)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "maltego"))
from registry import ensure_snapshot, today_str  # noqa: E402

# query days of the GeoIP observations (fetched_at_utc in data/cache/observations*.csv)
POP_DATES = {
    "anchors": "20260604",            # 2026-06-03/04
    "probes": "20260607",
    "probes_holdout": "20260801",
    "probes_mobile_ext": "20260801",
    "probes_pool": "20260820",        # 2026-08-19..21
    "giga": "20260821",
}


def main(argv: list[str]) -> None:
    dates = argv or sorted(set(POP_DATES.values()))
    for d in dates + [today_str()]:
        print(f"== {d}")
        ok = ensure_snapshot(d, fetch=True)
        print(f"   {'complete' if ok else 'MISSING'}")


if __name__ == "__main__":
    main(sys.argv[1:])
