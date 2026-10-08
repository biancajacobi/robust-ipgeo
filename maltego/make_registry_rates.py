"""Regenerate registry_rates.json: frozen country-error rates per cell
registry flag x S gate (the country-level counterpart of s_calibration.json).

Freezes, for live operation of the batch tool and the transform, how often the
aggregation's country vote was WRONG on the reference population, split into
the four cells

  flag    = registry country != core vote   (registry flag raised)
  noflag  = registry country == core vote
  lowmass = core mass of the vote < S_GATE  (S gate fires)
  highmass= core mass >= S_GATE

at two registry levels: 'whois' (RDAP, most specific object; PRIMARY, the flag
the transform raises live) and 'delegation' (delegation file; offline fallback
and the only datable level). Plus the irreducible floor (vote follows the
registration, both wrong -- invisible to flag AND gate), the lift as a risk
ratio with iid / ASN-cluster CI, and the recall/precision rows of the flag.

The data basis is DELIBERATELY the 8,896-address probe pool (probes_pool, T8)
and not the anchors: arbitrary operational IPs resemble probes; the anchors are
hosting-heavy (ARIN whois 82 %), where the flag carries only RR 2.3 (n.s.) --
that contrast is frozen alongside for honesty (_meta.contrast_anchors).
Coding 'state' (country field without a state -- EU/ZZ/AP/empty -- counts as no
registry country), core vote L1·b with the frozen anchor calibration, R_CORE
50 km, S_GATE 0.5: all identical to T10 (exp_country_level) and pre-specified
(exp_registry_country, docstring).

The generator does NOT recompute anything; it reads the aggregated, IP-free
eval/out/registry_country.csv of the experiment -- the measurement stays in the
experiment, here it is only frozen and given provenance. That is why it also
runs in the public repository (the CSV ships with it).

Usage:  python maltego/make_registry_rates.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "eval" / "out" / "registry_country.csv"
OUT = Path(__file__).resolve().parent / "registry_rates.json"

POPULATION = "probes_pool"
CODING = "state"
S_GATE, R_CORE_KM = 0.5, 50.0            # as exp_country_level.S_GATE / R_CORE
LEVELS = {"whois": "registry_rdap", "delegation": "registry_delegated"}   # order = live priority
CELLS = ("flag_lowmass", "flag_highmass", "noflag_lowmass", "noflag_highmass")


def _row(df: pd.DataFrame, pop: str, est: str) -> pd.Series:
    m = df[(df.coding == CODING) & (df.population == pop) & (df.estimator == est)]
    if len(m) != 1:
        raise RuntimeError(f"{SRC.name}: expected exactly 1 row for {CODING}/{pop}/{est}, got {len(m)}")
    return m.iloc[0]


def _num(v, nd: int = 3):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return round(float(v), nd)


def level_block(df: pd.DataFrame, pop: str, est: str) -> dict:
    main = _row(df, pop, est)
    cm = _row(df, pop, f"{est}__common_mode")
    lift = _row(df, pop, f"{est}__flag_lift")
    cells = _row(df, pop, f"{est}__flag_x_gate")
    flag = _row(df, pop, f"{est}__flag_ne_vote")
    n = int(main.n)
    block = {
        "estimator": est,
        "snapshot": str(main.snapshot),
        "n": n,
        "registry_hit_pct": _num(main.hit_pct, 2),
        "agg_hit_pct": _num(main.agg_hit_pct_paired, 2),
        "no_state_n": int(main.no_state_n),
        "flag": {"rate_pct": _num(flag.flag_pct, 2), "flag_n": int(lift.flag_n),
                 "recall_pct": _num(flag.recall_pct, 1), "precision_pct": _num(flag.precision_pct, 1)},
        "floor": {"agg_miss_n": int(cm.agg_miss_n),
                  "registry_following_n": int(cm.cm_registry_following_n),
                  "registry_following_pct_of_n": _num(cm.cm_registry_following_pct_of_n),
                  "dev_registry_wrong_n": int(cm.dev_registry_wrong_n),
                  "dev_registry_right_n": int(cm.dev_registry_right_n)},
        "lift": {"p_miss_flag_pct": _num(lift.p_miss_flag_pct), "p_miss_noflag_pct": _num(lift.p_miss_noflag_pct),
                 "risk_ratio": _num(lift.risk_ratio, 2),
                 "rr_ci95_iid": [_num(lift.rr_iid_lo, 2), _num(lift.rr_iid_hi, 2)],
                 "rr_ci95_asn": [_num(lift.rr_asn_lo, 2), _num(lift.rr_asn_hi, 2)]},
        "cells": {},
    }
    total = 0
    for c in CELLS:
        cn = int(cells[f"{c}_n"])
        total += cn
        pct = _num(cells[f"{c}_miss_pct"])
        block["cells"][c] = {"n": cn, "miss_pct": pct,
                             "p_country_miss": (round(pct / 100.0, 5) if pct is not None else None)}
    if total != n:
        raise RuntimeError(f"{est}: cells sum to {total}, population n={n}")
    return block


def main() -> None:
    df = pd.read_csv(SRC)
    levels = {name: level_block(df, POPULATION, est) for name, est in LEVELS.items()}
    gate = _row(df, POPULATION, "registry_rdap__gate_low_mass")
    contrast = {}
    for name, est in LEVELS.items():
        a = _row(df, "anchors", f"{est}__flag_lift")
        contrast[name] = {"n": int(a.n), "flag_n": int(a.flag_n), "risk_ratio": _num(a.risk_ratio, 2),
                          "rr_ci95_iid": [_num(a.rr_iid_lo, 2), _num(a.rr_iid_hi, 2)],
                          "rr_ci95_asn": [_num(a.rr_asn_lo, 2), _num(a.rr_asn_hi, 2)]}

    print(f"Registry flag x S gate, population {POPULATION} (n={levels['whois']['n']}), coding {CODING}")
    for name, b in levels.items():
        print(f"  {name:10s} ({b['estimator']}, {b['snapshot']}): registry hit {b['registry_hit_pct']} %, "
              f"agg {b['agg_hit_pct']} %; flag on {b['flag']['rate_pct']} % of addresses, recall {b['flag']['recall_pct']} %; "
              f"floor {b['floor']['registry_following_pct_of_n']} %; RR {b['lift']['risk_ratio']} "
              f"iid {b['lift']['rr_ci95_iid']} ASN {b['lift']['rr_ci95_asn']}")
        for c in CELLS:
            z = b["cells"][c]
            print(f"      {c:16s} n={z['n']:5d}  country miss {z['miss_pct']} %")
    print(f"  Gate alone (core mass < {S_GATE}): {_num(gate.flag_pct, 2)} % flagged, recall {_num(gate.recall_pct, 1)} %")
    print(f"  Contrast anchors (hosting-heavy): RR whois {contrast['whois']['risk_ratio']} "
          f"iid {contrast['whois']['rr_ci95_iid']}, delegation {contrast['delegation']['risk_ratio']}")

    payload = {
        "_meta": {
            "description": (
                "Frozen country-error rates of the aggregation (core vote L1·b) per cell "
                "registry flag x S gate for live operation of the batch tool and the transform. "
                "flag = registry country != core vote; lowmass = core mass < s_gate. "
                "p_country_miss of a cell = share of wrong country votes in that cell on the "
                "reference population. Level 'whois' (RDAP, most specific object) is the primary flag, "
                "'delegation' (RIR delegation file) the offline fallback and the only datable level."
            ),
            "note_floor": (
                "The floor (vote follows the registration, both wrong) is by construction invisible to flag "
                "AND gate; it is the lower bound of the country error that no consensus or registry signal "
                "indicates."
            ),
            "note_population": (
                "Reference population is the 8,896-address probe pool (RIPE Atlas probes, T8), not the anchors: "
                "operational IPs resemble probes; on the hosting-heavy anchors the flag carries only weakly "
                "(see contrast_anchors). School IPs of the giga population were not queried via whois."
            ),
            "note_cells": (
                "The cell flag & lowmass is small (whois n=20); its rate is an order-of-magnitude hint, "
                "not a precise measure. For calibration, the rows p_miss_flag / p_miss_noflag "
                "(lift, with CI) are the robust statement."
            ),
            "generated_with": "maltego/make_registry_rates.py",
            "source": f"{SRC.relative_to(ROOT).as_posix()} (experiments/exp_registry_country.py, aggregated, IP-free)",
            "data_state": (f"{POPULATION} n={levels['whois']['n']}, coding {CODING}, delegation as of "
                           f"{levels['delegation']['snapshot']}, RDAP cache 2026-09-27, frozen "
                           f"{datetime.now(timezone.utc).date().isoformat()}"),
            "contrast_anchors": contrast,
        },
        "population": POPULATION,
        "coding": CODING,
        "s_gate": S_GATE,
        "r_core_km": R_CORE_KM,
        "level_priority": list(LEVELS),
        "gate_only": {"rate_pct": _num(gate.flag_pct, 2), "recall_pct": _num(gate.recall_pct, 1),
                      "precision_pct": _num(gate.precision_pct, 1)},
        "levels": levels,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{OUT} written.")


if __name__ == "__main__":
    main()
