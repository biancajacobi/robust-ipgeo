#!/usr/bin/env python3
"""Batch mode: robust geolocation of a LIST of IPs (CSV in, CSV out).

The Maltego transforms serve the single, interactive case. Bulk triage
(referral lists, log extracts, login histories returned under a production
order) needs the same estimate for hundreds or thousands of addresses in one
run. This tool is the thin list wrapper around ``geoloc.locate`` and
``registry.registry_context`` — nothing is computed here that the transforms
do not compute as well.

Per record (IP, optional timestamp) one output row: coordinates of the line-
and radius-weighted geometric median (L1+b), country vote and its mass (T10),
registry country according to the RIR delegation (as of the record's date,
otherwise today) and today, optionally the whois/RDAP country, support
concentration S, calibrated P(error > 100 km) with branch, effective lines,
responding sources, mobile flag, failed sources and warnings. Optionally the
complete per-source provenance as JSONL (one object per record).

TIMESTAMPS (important for historical holdings): The GeoIP sources have no
history — the estimate describes the CURRENT holder of the address. If a
record carries a timestamp (column ``timestamp``/``time``/``date``/... or
``--as-of`` for the whole file), the registry country is additionally taken
from the archived delegation file of the record date (monthly snapshot,
offline; download of the public bulk file on demand, ~50 MB per month) and a
warning is raised if (a) the record is older than ``--stale-days``, (b) the
address has been re-delegated since, (c) registry country and country vote
contradict each other. Evidence: experiments/exp_registry_country.py.

REGISTRY FLAG (columns ``registry_flag``/``registry_level``/``p_country_miss``):
registry country != country vote at the best available level (whois object
with --rdap, otherwise today's delegation). On the 8,896-address probe pool
the country vote is wrong in 8.1 % (whois) resp. 4.9 % (delegation) of the
flagged cases versus 0.7 % resp. 0.5 % without the flag (risk ratio 11.5
resp. 9.6, ASN-cluster-robust); ``p_country_miss`` is the frozen rate of the
cell flag x S gate (maltego/registry_rates.json) — the country-level
counterpart of ``p_miss``. The flag is a pure COUNTRY channel: it does not
carry for the 100-km city error (exp_registry_city). What stays invisible is
the registry-following floor (vote == registry, both wrong): 0.67 % of all
addresses.

ARCHIVE MODE (--db-archive <folder>): a folder with subfolders ``YYYY-MM/``
that contain the three offline DBs per month (GeoLite2-City.mmdb, dbip-city-
lite-*.mmdb, IP2LOCATION-LITE-DB5.BIN — the same files as data/db/, put aside
monthly). Dated records are then NOT estimated live but from the DB builds of
their month (three offline lines = exactly the minimum for S); column
``basis`` = ``archive:YYYY-MM``. If the month is missing (tolerance +-1 month),
the record falls back to live (``basis`` = ``live``) and a warning is raised.
Why: no provider delivers historical builds (MaxMind 30 days, DB-IP only the
current month) — only an archive of your own allows the method retroactively.
Radii and S calibration remain the frozen ones of the anchor evaluation; their
transfer to older DB builds is not measured.

Operating rules:
  * RESUMABLE: records (IP, timestamp) already present in the output file are
    skipped (unless --no-resume).
  * THROTTLE: --sleep seconds between IPs (free API tiers carry daily quotas;
    paid tiers of the same providers do not; the offline databases never do).
    The same IP with several timestamps is queried live only ONCE.
  * Rows are appended after every record (crash-safe).
  * No cache, no provenance ledger of the evaluation pipeline is touched:
    every call queries live.

Input: a CSV with a column ``ip`` (optional timestamp column) or a text file
with ``ip[,timestamp]`` per line (``#`` comments allowed). Duplicates (same IP
AND same timestamp) are processed once.

Usage:
  python maltego/batch.py ips.txt out.csv
  python maltego/batch.py logins.csv out.csv --timestamp-column login_time --rdap
  python maltego/batch.py ips.txt out.csv --as-of 2024-03-15 --stale-days 90
  python maltego/batch.py logins.csv out.csv --db-archive /archive/geoip-dbs     # offline DBs of the record's month
  python maltego/batch.py ips.csv out.csv --provenance out.jsonl --sleep 0.5
  python maltego/batch.py ips.txt out.csv --sources maxmind_geolite2,dbip_lite,ip2location_lite

Triage lead, not forensic evidence — the caveats of maltego/README.md apply.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geoloc    # noqa: E402
import registry  # noqa: E402

FIELDS = ["ip", "timestamp", "record_age_days", "basis", "fetched_at_utc", "lat", "lon",
          "country", "country_mass", "country_fallback",
          "registry_country", "registry_snapshot", "registry_country_today", "registry_rir",
          "rdap_country", "rdap_handle",
          "registry_flag", "registry_level", "p_country_miss",
          "s", "s_reliable", "p_miss", "branch", "mobile_flag",
          "n_lines", "n_sources", "failed", "warnings"]
TS_COLUMNS = ("timestamp", "time", "datetime", "date", "ts", "seen_at", "seen", "first_seen",
              "last_seen", "event_time", "login_time", "zeit", "zeitstempel", "datum")


def _ts_iso(ts: datetime | None) -> str:
    return ts.isoformat(timespec="seconds") if ts else ""


def read_records(path: Path, ts_column: str | None = None, as_of: datetime | None = None
                 ) -> list[tuple[str, datetime | None]]:
    """(ip, timestamp|None) per record; order kept, duplicates (ip, ts) removed.

    CSV: column ``ip`` (case-insensitive) + timestamp column (``ts_column`` or the
    first matching one from TS_COLUMNS). Text: ``ip[,ts]`` or ``ip ts`` per line.
    ``as_of`` fills in missing timestamps."""
    text = path.read_text(encoding="utf-8")
    recs: list[tuple[str, datetime | None]] = []
    first = text.splitlines()[0] if text.strip() else ""
    header = [c.strip().lower() for c in first.split(",")]
    if "," in first and "ip" in header:
        rows = list(csv.DictReader(text.splitlines()))
        keys = {k.strip().lower(): k for k in (rows[0].keys() if rows else header)}
        ipk = keys["ip"]
        tsk = None
        if ts_column:
            tsk = keys.get(ts_column.strip().lower())
            if tsk is None:
                raise SystemExit(f"timestamp column '{ts_column}' not in {path.name}: {list(keys)}")
        else:
            tsk = next((keys[c] for c in TS_COLUMNS if c in keys), None)
        for r in rows:
            ip = (r.get(ipk) or "").strip()
            ts = registry.parse_timestamp(r.get(tsk)) if tsk else None
            recs.append((ip, ts or as_of))
    else:
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = [p for p in line.replace(",", " ").split() if p]
            ip = parts[0]
            ts = registry.parse_timestamp(" ".join(parts[1:])) if len(parts) > 1 else None
            if ts is None and len(parts) > 1:
                ts = registry.parse_timestamp(parts[1])
            recs.append((ip, ts or as_of))
    seen: set[tuple[str, str]] = set()
    out = []
    for ip, ts in recs:
        key = (ip, _ts_iso(ts))
        if ip and key not in seen:
            seen.add(key); out.append((ip, ts))
    return out


def read_ips(path: Path) -> list[str]:
    """Backwards compatible: only the IPs (unique, order kept)."""
    seen: set[str] = set()
    return [ip for ip, _ in read_records(path) if not (ip in seen or seen.add(ip))]


def done_keys(out: Path) -> set[tuple[str, str]]:
    if not out.exists() or out.stat().st_size == 0:
        return set()
    with open(out, encoding="utf-8", newline="") as fh:
        return {(r["ip"], r.get("timestamp") or "") for r in csv.DictReader(fh)}


def row_from(r: dict, ts: datetime | None, ctx: dict | None, warnings: list[str],
             flag: dict | None = None) -> dict:
    lat, lon = r["estimate"] if r["estimate"] else (None, None)
    fmt = lambda x, nd: ("" if x is None else f"{x:.{nd}f}")  # noqa: E731
    ctx = ctx or {}
    flag = flag or {}
    return {
        "ip": r["ip"],
        "timestamp": _ts_iso(ts),
        "record_age_days": "" if ctx.get("record_age_days") is None else str(ctx["record_age_days"]),
        "basis": r.get("basis") or "live",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lat": fmt(lat, 6), "lon": fmt(lon, 6),
        "country": r.get("country") or "",
        "country_mass": fmt(r.get("country_mass"), 3),
        "country_fallback": str(bool(r.get("country_fallback"))),
        "registry_country": (ctx.get("registry_country_asof") if ts else ctx.get("registry_country_today")) or "",
        "registry_snapshot": (ctx.get("registry_snapshot_asof") if ts else ctx.get("registry_snapshot_today")) or "",
        "registry_country_today": ctx.get("registry_country_today") or "",
        "registry_rir": ctx.get("registry_rir") or "",
        "rdap_country": ctx.get("rdap_country") or "",
        "rdap_handle": ctx.get("rdap_handle") or "",
        "registry_flag": "" if flag.get("registry_flag") is None else str(flag["registry_flag"]),
        "registry_level": flag.get("registry_level") or "",
        "p_country_miss": fmt(flag.get("p_country_miss"), 4),
        "s": fmt(r["s"], 3) if r["s_reliable"] else "",
        "s_reliable": str(bool(r["s_reliable"])),
        "p_miss": fmt(r["p_miss"], 3), "branch": r["branch"] or "",
        "mobile_flag": "" if r["mobile_flag"] is None else str(r["mobile_flag"]),
        "n_lines": r["n_lines"], "n_sources": r["n_sources"],
        "failed": ";".join(f"{s}:{st}" for s, st in r["failed"]),
        "warnings": " | ".join(warnings),
    }


def run(records, out: Path, sources: list[str] | None = None, sleep: float = 0.0,
        provenance: Path | None = None, resume: bool = True, *, use_registry: bool = True,
        rdap: bool = False, stale_days: int = registry.STALE_DAYS_DEFAULT,
        fetch_rir: bool = True, granularity: str = "month", db_archive: Path | None = None,
        log=print) -> dict:
    """records: list of IPs (strings) or (ip, timestamp|None) tuples.
    db_archive: root of the offline-DB archive (subfolders YYYY-MM) or None."""
    records = [(r, None) if isinstance(r, str) else tuple(r) for r in records]
    skip = done_keys(out) if resume else set()
    todo = [(ip, ts) for ip, ts in records if (ip, _ts_iso(ts)) not in skip]
    log(f"{len(records)} record(s) in input, {len(skip)} already in output, {len(todo)} to run")
    new_file = not out.exists() or out.stat().st_size == 0
    stats = {"processed": 0, "with_estimate": 0, "s_reliable": 0, "fallback": 0,
             "countries": {}, "stale": 0, "redelegated": 0, "registry_disagrees": 0,
             "registry_flagged": 0, "archive": 0, "archive_missing": 0}
    located: dict[str, dict] = {}
    if db_archive is not None and not Path(db_archive).is_dir():
        raise SystemExit(f"--db-archive: not a directory: {db_archive}")
    with open(out, "a", encoding="utf-8", newline="") as fh, \
            (open(provenance, "a", encoding="utf-8") if provenance else _Null()) as ph:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        n_live = 0
        for i, (ip, ts) in enumerate(todo, 1):
            archive_note = None
            month_dir = None
            if db_archive is not None and ts is not None:
                month_dir, month, dist = geoloc.resolve_archive_month(db_archive, ts)
                if month_dir is None:
                    archive_note = (f"no archive month for {ts.strftime('%Y-%m')} in {Path(db_archive).name} "
                                    f"(tolerance +-{geoloc.ARCHIVE_TOLERANCE_MONTHS}) — estimated live, "
                                    "i.e. today's holder")
                    stats["archive_missing"] += 1
                elif dist:
                    archive_note = f"archive month {month} used instead of {ts.strftime('%Y-%m')} ({dist} month(s) apart)"
            if month_dir is not None:
                key = f"{ip}|{month_dir.name}"
                if key in located:
                    r = dict(located[key])
                else:
                    r = geoloc.locate_archive(ip, month_dir)
                    located[key] = r
                stats["archive"] += 1
            elif ip in located:
                r = dict(located[ip])
            else:
                if sleep and n_live:
                    time.sleep(sleep)
                r = geoloc.locate(ip, sources)
                located[ip] = r
                n_live += 1
            ctx = None
            flag = None
            warnings = ([archive_note] if archive_note else []) + list(r["warnings"])
            if use_registry:
                ctx = registry.registry_context(ip, ts, rdap=rdap, granularity=granularity,
                                                fetch=fetch_rir, log=log)
                rw = registry.registry_warnings(ctx, r.get("country"),
                                                10**9 if month_dir is not None else stale_days,
                                                country_mass=r.get("country_mass"))
                warnings = rw + warnings
                flag = registry.registry_flag(ctx, r.get("country"), r.get("country_mass"))
                if ctx.get("record_age_days") is not None and ctx["record_age_days"] > stale_days:
                    stats["stale"] += 1
                if ctx.get("registry_redelegated"):
                    stats["redelegated"] += 1
                ref = (ctx.get("registry_country_asof") if ts else ctx.get("registry_country_today"))
                if r.get("country") and ref and ref != r["country"]:
                    stats["registry_disagrees"] += 1
                if flag["registry_flag"]:
                    stats["registry_flagged"] += 1
            row = row_from(r, ts, ctx, warnings, flag)
            w.writerow(row); fh.flush()
            if provenance:
                ph.write(json.dumps({**r, "timestamp": _ts_iso(ts), "registry": ctx, "warnings": warnings},
                                    ensure_ascii=False, default=str) + "\n"); ph.flush()
            stats["processed"] += 1
            if r["estimate"] is not None:
                stats["with_estimate"] += 1
            if r["s_reliable"]:
                stats["s_reliable"] += 1
            if r.get("country_fallback"):
                stats["fallback"] += 1
            if r.get("country"):
                stats["countries"][r["country"]] = stats["countries"].get(r["country"], 0) + 1
            head = (f"S={r['s']:.2f}" if r["s_reliable"] else "S withheld")
            reg = f" reg={row['registry_country'] or '--'}" if use_registry else ""
            if month_dir is not None:
                reg += f" [{row['basis']}]"
            flags = "".join(f" [{f}]" for f, on in (
                ("stale", ctx and ctx.get("record_age_days") is not None and ctx["record_age_days"] > stale_days),
                ("redelegated", ctx and ctx.get("registry_redelegated")),
                (f"registry-flag:{row['registry_level']}"
                 + (f" P(country miss)={row['p_country_miss']}" if row["p_country_miss"] else ""),
                 flag and flag["registry_flag"])) if on)
            log(f"[{i}/{len(todo)}] {ip}{' @' + row['timestamp'] if ts else ''}: {row['country'] or '--'} "
                f"({row['country_mass'] or '--'}){reg} {head} lines={r['n_lines']}{flags}")
    return stats


class _Null:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def write(self, *_): pass
    def flush(self): pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input", type=Path, help="CSV with column 'ip' (+ optional timestamp column) or text file, one 'ip[,timestamp]' per line")
    ap.add_argument("output", type=Path, help="output CSV (appended; resumable)")
    ap.add_argument("--sources", help="comma-separated source names (default: all configured)")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between live IP lookups")
    ap.add_argument("--provenance", type=Path, help="JSONL with the full per-source result per record")
    ap.add_argument("--no-resume", action="store_true", help="do not skip records already in output")
    ap.add_argument("--timestamp-column", help="name of the timestamp column (default: auto-detect)")
    ap.add_argument("--as-of", help="timestamp for records without one (ISO 8601 or epoch)")
    ap.add_argument("--stale-days", type=int, default=registry.STALE_DAYS_DEFAULT,
                    help=f"warn when a record is older than this (default {registry.STALE_DAYS_DEFAULT})")
    ap.add_argument("--rdap", action="store_true", help="also query the registry's RDAP (most specific whois object) per IP")
    ap.add_argument("--no-registry", action="store_true", help="skip the RIR delegation lookup entirely")
    ap.add_argument("--no-fetch-rir", action="store_true", help="never download delegation snapshots (use local ones only)")
    ap.add_argument("--rir-granularity", choices=("month", "day"), default="month",
                    help="delegation snapshot per record month (default) or per day")
    ap.add_argument("--db-archive", type=Path,
                    help="root of an offline-DB archive with YYYY-MM/ subfolders; dated records are "
                         "estimated from their month's databases instead of live")
    a = ap.parse_args(argv)
    as_of = registry.parse_timestamp(a.as_of) if a.as_of else None
    if a.as_of and as_of is None:
        print(f"--as-of not readable: {a.as_of}", file=sys.stderr)
        return 2
    records = read_records(a.input, a.timestamp_column, as_of)
    if not records:
        print("no IPs in input", file=sys.stderr)
        return 2
    sources = [s.strip() for s in a.sources.split(",") if s.strip()] if a.sources else None
    st = run(records, a.output, sources, a.sleep, a.provenance, resume=not a.no_resume,
             use_registry=not a.no_registry, rdap=a.rdap, stale_days=a.stale_days,
             fetch_rir=not a.no_fetch_rir, granularity=a.rir_granularity, db_archive=a.db_archive)
    top = ", ".join(f"{c}:{n}" for c, n in sorted(st["countries"].items(),
                                                   key=lambda kv: (-kv[1], kv[0]))[:5])
    print(f"done: {st['processed']} processed, {st['with_estimate']} with estimate, "
          f"{st['s_reliable']} with reliable S, {st['fallback']} country fallback(s); "
          f"{st['stale']} stale record(s), {st['redelegated']} re-delegated, "
          f"{st['registry_disagrees']} registry/vote disagreement(s) at the record date, "
          f"{st['registry_flagged']} registry flag(s) raised, "
          f"{st['archive']} from archive, {st['archive_missing']} without archive month; "
          f"top countries: {top or '--'}\nCSV: {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
