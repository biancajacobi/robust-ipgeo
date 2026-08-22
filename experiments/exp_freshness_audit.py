"""Freshness audit: how far apart are the ground-truth timestamp and the source fetch?

Motivation: IPs change hands. An (IP, location)
pair is only a valid ground truth for the source fetch if little time passed
between the moment the assignment was valid (anchor/probe metadata draw or
Giga measurement) and the fetch of the GeoIP sources. This audit makes the gap
VISIBLE per dataset and source, instead of asserting it:

  * GT reference timestamp from the provenance ledger (kind=probes/giga or the
    first anchor fetch); for Giga additionally the measurement window and --
    if data/cache/giga_pair_times.csv exists -- the PAIR AGE per (IP, school)
    (last measurement -> source fetch) as a distribution.
  * Source fetch timestamps from observations_<dataset>.csv (fetched_at_utc
    per row).

Structural arguments that the audit does NOT replace, but puts in context
(column note):
  * anchors: RIPE anchors = institutional measurement infrastructure, static
    addresses.
  * probes_pool: draw with --require-tags system-ipv4-stable-30d (IP provably
    stable for 30 days BEFORE the draw) and --drop-shared-ips.
  * giga: filter 4 removes IPs seen at >1 school within the window; a DHCP
    change AFTER the last measurement remains possible -> age stratification
    in exp_giga_transfer.

Output: eval/out/freshness_audit.csv (+ .md), columns:
  dataset, source, n_obs, gt_ref, obs_first, obs_median, obs_last,
  delta_median_days, delta_max_days, note

Usage:  python experiments/exp_freshness_audit.py [--datasets anchors,probes,...]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "data" / "cache"
OUT = ROOT / "eval" / "out"

DEFAULT_DATASETS = ["anchors", "probes", "probes_holdout", "probes_mobile_ext",
                    "probes_pool", "giga"]

NOTES = {
    "anchors": "institutional measurement infrastructure, static addresses",
    "probes": "volunteer probes, no stability filter at draw time (June run)",
    "probes_holdout": "same as probes (holdout draw)",
    "probes_mobile_ext": "mobile tag union; mobile IPs are inherently volatile",
    "probes_pool": "draw with system-ipv4-stable-30d (30 days stable BEFORE the draw) + drop-shared-ips",
    "giga": "GT per pair = measurement window; pair age reported separately (see _pair_age rows)",
}


def _suffix(ds: str) -> str:
    return "" if ds == "anchors" else f"_{ds}"


def _parse_ts(s: str) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def gt_reference(ds: str) -> tuple[datetime | None, str]:
    """GT reference timestamp (+ description) from the provenance ledger."""
    ledger = ROOT / "data" / f"provenance{_suffix(ds)}.jsonl"
    if not ledger.exists():
        return None, "no ledger"
    first_any, hit = None, None
    with open(ledger, encoding="utf-8") as fh:
        for line in fh:
            e = json.loads(line)
            ts = _parse_ts(e.get("fetched_at_utc", ""))
            if ts and first_any is None:
                first_any = e
            if e.get("kind") in ("probes", "giga") and ts:
                hit = e
                break
    e = hit or first_any
    if not e:
        return None, "no dated entry"
    ts = _parse_ts(e["fetched_at_utc"])
    if e.get("kind") == "giga":
        return ts, (f"giga draw {e['fetched_at_utc'][:10]}, measurement window "
                    f"{e.get('window_start')}..{e.get('window_end')}")
    return ts, f"{e.get('kind') or 'anchors'} draw {e['fetched_at_utc'][:10]}"


def audit_dataset(ds: str) -> list[dict]:
    obs_path = CACHE / f"observations{_suffix(ds)}.csv"
    if not obs_path.exists():
        return []
    gt_ts, gt_desc = gt_reference(ds)
    per_source: dict[str, list[datetime]] = {}
    with open(obs_path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ts = _parse_ts(r.get("fetched_at_utc", ""))
            if ts:
                per_source.setdefault(r["source"], []).append(ts)
    rows = []
    for src in sorted(per_source):
        tss = sorted(per_source[src])
        med = tss[len(tss) // 2]
        # Calendar days (not timedelta.days, which rounds 1.9 days down to 1)
        d_med = (med.date() - gt_ts.date()).days if gt_ts else None
        d_max = (tss[-1].date() - gt_ts.date()).days if gt_ts else None
        rows.append({
            "dataset": ds, "source": src, "n_obs": len(tss), "gt_ref": gt_desc,
            "obs_first": tss[0].date().isoformat(),
            "obs_median": med.date().isoformat(),
            "obs_last": tss[-1].date().isoformat(),
            "delta_median_days": d_med, "delta_max_days": d_max,
            "note": NOTES.get(ds, "")})
    return rows


def giga_pair_age_rows() -> list[dict]:
    """Pair age for Giga: last measurement per (IP, school) -> source fetch."""
    times_path = CACHE / "giga_pair_times.csv"
    obs_path = CACHE / "observations_giga.csv"
    if not (times_path.exists() and obs_path.exists()):
        return []
    fetch_by_ip: dict[str, datetime] = {}
    with open(obs_path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ts = _parse_ts(r.get("fetched_at_utc", ""))
            if ts:
                prev = fetch_by_ip.get(r["ip"])
                fetch_by_ip[r["ip"]] = min(prev, ts) if prev else ts   # earliest fetch
    ages = []
    with open(times_path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            last = _parse_ts(r.get("last_created_at", ""))
            fetch = fetch_by_ip.get(r["ip"])
            if last and fetch:
                ages.append((fetch - last).total_seconds() / 86400.0)
    if not ages:
        return []
    ages.sort()
    q = lambda p: ages[min(len(ages) - 1, int(p * len(ages)))]
    return [{
        "dataset": "giga", "source": "_pair_age_days", "n_obs": len(ages),
        "gt_ref": "last measurement per pair -> earliest source fetch of the IP",
        "obs_first": f"min {ages[0]:.1f}", "obs_median": f"p50 {statistics.median(ages):.1f}",
        "obs_last": f"max {ages[-1]:.1f}",
        "delta_median_days": round(statistics.median(ages), 1),
        "delta_max_days": round(ages[-1], 1),
        "note": f"p90 {q(0.90):.1f} d; age stratification: exp_giga_transfer"}]


def main() -> None:
    ap = argparse.ArgumentParser(description="Freshness audit: GT timestamp vs. source fetch.")
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS),
                    help="comma-separated list (default: all known)")
    args = ap.parse_args()

    rows: list[dict] = []
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        got = audit_dataset(ds)
        if not got:
            print(f"  (skipped: {ds} — no observations)")
            continue
        rows.extend(got)
    rows.extend(giga_pair_age_rows())

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["dataset", "source", "n_obs", "gt_ref", "obs_first", "obs_median",
            "obs_last", "delta_median_days", "delta_max_days", "note"]
    with open(OUT / "freshness_audit.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "freshness_audit.md", "w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(cols) + " |\n")
        fh.write("|" + "---|" * len(cols) + "\n")
        for r in rows:
            fh.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")

    print(f"{len(rows)} rows -> {OUT / 'freshness_audit.csv'}")
    for r in rows:
        print(f"  {r['dataset']:<18} {r['source']:<18} n={r['n_obs']:<6} "
              f"Δmedian={r['delta_median_days']}d Δmax={r['delta_max_days']}d")


if __name__ == "__main__":
    main()
