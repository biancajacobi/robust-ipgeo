#!/usr/bin/env python3
"""Drift of the registry country (RIR delegation) over years -- basis of the
statement on temporal validity in the application chapter.

Question: how much does the country to which an address is delegated according
to the delegation file change over 1, 2, 5 and 10 years? This bounds how far a
present-day registry entry carries for old records -- and, conversely, how far
the delegation archives (APNIC from 2001, RIPE/ARIN/LACNIC from 2003, AFRINIC
from 2005) can reproduce the reference date of a record. GeoIP sources have no
history; the delegation is the only datable country statement.

Procedure: for every population, compare the registry country from the most
recent local snapshot ("today") with the one from snapshots 1/2/5/10 years
earlier (same calendar day): share of changed countries (normalised via
STATE_OF), share of blocks not yet allocated back then, hit rate against
today's ground truth back then (illustration only -- the GT is from today).
Offline; snapshots via data/fetch_rir.py or maltego/registry.py fetch.

Usage:   GEOIP_DATASET=anchors python experiments/exp_registry_drift.py
Output:  table (stdout) + eval/out/registry_drift.csv (aggregated, IP-free)
"""
from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "maltego"))

from data import store                                     # noqa: E402
import experiments.exp_country_level as CL                 # noqa: E402
from experiments.exp_registry_country import state_cc, RIR_DIR  # noqa: E402
from registry import Delegations, ensure_snapshot          # noqa: E402

OUT = ROOT / "eval" / "out"
YEARS_BACK = (1, 2, 5, 10)
POPS = ["anchors", "probes", "probes_holdout", "probes_mobile_ext", "probes_pool", "giga"]


def main() -> None:
    if os.environ.get("GEOIP_DATASET", "anchors") != "anchors":
        sys.exit("exp_registry_drift: please run with GEOIP_DATASET=anchors.")
    CL._NORMALIZE = True
    latest = sorted(d.name for d in RIR_DIR.iterdir() if d.is_dir() and d.name.isdigit())[-1]
    ref = dt.datetime.strptime(latest, "%Y%m%d").date()
    dates = {y: ref.replace(year=ref.year - y).strftime("%Y%m%d") for y in YEARS_BACK}
    for d in dates.values():
        if not ensure_snapshot(d, fetch=False):
            sys.exit(f"snapshot {d} missing -- python maltego/registry.py fetch {d}")
    print(f"reference (today): {latest}; comparison: " + ", ".join(f"{y} y = {d}" for y, d in dates.items()))
    D_now = Delegations(latest)
    D_old = {y: Delegations(d) for y, d in dates.items()}

    rows = []
    print(f"\n{'population':18s} {'n':>5s} | {'right today':>13s} | changed vs. today: "
          + " ".join(f"{y:>2d}y" for y in YEARS_BACK) + " | not allocated then: "
          + " ".join(f"{y:>2d}y" for y in YEARS_BACK) + " | hits then: " + " ".join(f"{y:>2d}y" for y in YEARS_BACK))
    for pop in POPS:
        recs = store.load_anchors_csv(ROOT / "data" / "cache" / f"{pop}.csv")
        ips = [r["ip"] for r in recs]
        truth = np.array([CL._cc(r["country"]) for r in recs])
        now = np.array([state_cc(D_now.lookup(ip)[0]) for ip in ips])
        hit_now = float(np.mean((now == truth) & (now != "")))
        n = len(ips)
        row = {"population": pop, "n": n, "snapshot_today": latest, "hit_today_pct": round(100 * hit_now, 2)}
        ch, un, hit = [], [], []
        for y in YEARS_BACK:
            old = [D_old[y].lookup(ip) for ip in ips]
            old_cc = np.array([state_cc(o[0]) for o in old])
            unalloc = np.array([o[2] in ("none", "available", "reserved") for o in old])
            changed = old_cc != now
            h = (old_cc == truth) & (old_cc != "")
            ch.append(changed.mean()); un.append(unalloc.mean()); hit.append(h.mean())
            row.update({f"snapshot_{y}y": dates[y], f"changed_{y}y_pct": round(100 * changed.mean(), 2),
                        f"changed_allocated_{y}y_pct": round(100 * (changed & ~unalloc).mean(), 2),
                        f"unallocated_{y}y_pct": round(100 * unalloc.mean(), 2),
                        f"hit_then_{y}y_pct": round(100 * h.mean(), 2)})
        rows.append(row)
        print(f"{pop:18s} {n:5d} | {100*hit_now:12.1f} % | " + " ".join(f"{100*c:4.1f}%" for c in ch)
              + " | " + " ".join(f"{100*u:4.1f}%" for u in un) + " | " + " ".join(f"{100*h:4.1f}%" for h in hit))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "registry_drift.csv", index=False)
    print(f"\nCSV: {OUT / 'registry_drift.csv'}")


if __name__ == "__main__":
    main()
