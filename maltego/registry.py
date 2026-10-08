"""Registry context for an IP: country according to the RIR delegation (datable,
offline) and according to whois/RDAP (most specific object, "now" only).

Why alongside the GeoIP sources? Two reasons, both measured
(experiments/exp_registry_country.py, 2026-09-27):

  * TIME. None of the GeoIP sources has a history; the aggregation says where
    the CURRENT holder of an address sits. For records that are weeks to years
    old (log extracts, stored referral lists), the registry country from the
    daily archived delegation files of the five RIRs is the only country
    statement reproducible as of the record date (archives: APNIC since 2001,
    RIPE/ARIN/LACNIC since 2003, AFRINIC since 2005). Measured drift of the
    registry country on the study populations: 1 year 0-2 %, 2 years 1-5 %,
    5 years 2-14 %, 10 years 8-37 %.
  * CONTRADICTION. Registry country != country vote of the aggregation is a
    usable flag: on the 8,896-probe pool it catches 57 % of the aggregation's
    country misses while flagging 12 % of addresses (delegation level). The
    S gate alone catches 15 % there.

What the registry country is NOT: the user's location. It is the country of
the organisation the block was delegated (allocation) or assigned (whois
object) to. On the study populations it matches the true country in 78-99 %
(delegation) resp. 90-95 % (whois object; mobile probes 100 %) of cases, the
aggregation in 97-100 %.
For the ADDRESSEE of an order (provider seat), however, the registry is the
authoritative source. Both are therefore reported side by side.

Data flow:
  * Delegation files (RIR Statistics Exchange Format, delegated-*-extended)
    live under SNAPSHOT_DIR/<YYYYMMDD>/ and are downloaded on demand
    (~50 MB per date, gzip). Evaluation is offline — no IP leaves the machine.
  * RDAP queries the responsible registry live (one request per IP) — only on
    request (batch: --rdap; transform: always, a single case).

Standalone:  python maltego/registry.py 9.9.9.9 [--as-of 2024-03-15] [--rdap]
             python maltego/registry.py fetch 20240301 [20230301 ...]
"""
from __future__ import annotations

import bisect
import bz2
import datetime as dt
import gzip
import ipaddress
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = ROOT / "data" / "cache" / "rir"
RATES_PATH = Path(__file__).resolve().parent / "registry_rates.json"   # make_registry_rates.py
UA = "robust-ipgeo/registry (research; contact via repo)"
RIRS = ("ripencc", "arin", "apnic", "lacnic", "afrinic")
NO_STATE = {"", "EU", "ZZ", "AP"}          # country field without a state (RIPE uses EU for multi-country orgs)
STALE_DAYS_DEFAULT = 30                    # from this age on: the GeoIP estimate describes today's holder
HTTP_TIMEOUT = 300

RDAP_BASE = {
    "ripencc": "https://rdap.db.ripe.net/ip/{ip}",
    "arin":    "https://rdap.arin.net/registry/ip/{ip}",
    "apnic":   "https://rdap.apnic.net/ip/{ip}",
    "lacnic":  "https://rdap.lacnic.net/rdap/ip/{ip}",
    "afrinic": "https://rdap.afrinic.net/rdap/ip/{ip}",
    "none":    "https://rdap.org/ip/{ip}",
}
# ARIN network objects carry no country field; there the country line of the
# registrant's address (country name) counts. LACNIC delivers the country structured.
NAME_TO_ISO = {
    "UNITED STATES": "US", "USA": "US", "UNITED STATES OF AMERICA": "US", "US": "US",
    "CANADA": "CA", "CA": "CA", "PUERTO RICO": "PR", "BAHAMAS": "BS", "BARBADOS": "BB",
    "JAMAICA": "JM", "BERMUDA": "BM", "CAYMAN ISLANDS": "KY", "VIRGIN ISLANDS, U.S.": "VI",
    "US VIRGIN ISLANDS": "VI", "ANTIGUA AND BARBUDA": "AG", "SAINT LUCIA": "LC", "GRENADA": "GD",
    "DOMINICA": "DM", "SAINT KITTS AND NEVIS": "KN", "TRINIDAD AND TOBAGO": "TT",
    "TURKS AND CAICOS ISLANDS": "TC", "ANGUILLA": "AI", "BRITISH VIRGIN ISLANDS": "VG",
    "VIRGIN ISLANDS, BRITISH": "VG", "MONTSERRAT": "MS", "GUAM": "GU", "AMERICAN SAMOA": "AS",
    "NORTHERN MARIANA ISLANDS": "MP", "SAINT VINCENT AND THE GRENADINES": "VC",
    "UNITED KINGDOM": "GB", "GERMANY": "DE", "NETHERLANDS": "NL", "FRANCE": "FR", "JAPAN": "JP",
    "AUSTRALIA": "AU", "SINGAPORE": "SG", "HONG KONG": "HK", "IRELAND": "IE", "SWITZERLAND": "CH",
    "SWEDEN": "SE", "INDIA": "IN", "BRAZIL": "BR", "MEXICO": "MX", "ISRAEL": "IL", "SPAIN": "ES",
    "ITALY": "IT", "BELGIUM": "BE", "LUXEMBOURG": "LU", "NEW ZEALAND": "NZ", "SOUTH AFRICA": "ZA",
    "KOREA, REPUBLIC OF": "KR", "TAIWAN": "TW", "TAIWAN, PROVINCE OF CHINA": "TW", "CHINA": "CN",
    "RUSSIAN FEDERATION": "RU", "VENEZUELA": "VE", "VENEZUELA, BOLIVARIAN REPUBLIC OF": "VE",
    "LITHUANIA": "LT", "COLOMBIA": "CO", "ECUADOR": "EC", "ROMANIA": "RO", "TURKEY": "TR",
    "TÜRKIYE": "TR", "INDONESIA": "ID", "ARGENTINA": "AR", "CHILE": "CL", "PERU": "PE",
    "PANAMA": "PA", "COSTA RICA": "CR", "GUATEMALA": "GT", "HONDURAS": "HN", "NICARAGUA": "NI",
    "EL SALVADOR": "SV", "DOMINICAN REPUBLIC": "DO", "URUGUAY": "UY", "PARAGUAY": "PY",
    "BOLIVIA": "BO", "BOLIVIA, PLURINATIONAL STATE OF": "BO", "PHILIPPINES": "PH", "THAILAND": "TH",
    "MALAYSIA": "MY", "VIET NAM": "VN", "VIETNAM": "VN", "POLAND": "PL", "AUSTRIA": "AT",
    "DENMARK": "DK", "NORWAY": "NO", "FINLAND": "FI", "PORTUGAL": "PT", "GREECE": "GR",
    "CZECHIA": "CZ", "CZECH REPUBLIC": "CZ", "HUNGARY": "HU", "UKRAINE": "UA", "NIGERIA": "NG",
    "KENYA": "KE", "EGYPT": "EG", "UNITED ARAB EMIRATES": "AE", "SAUDI ARABIA": "SA",
    "BAHRAIN": "BH", "BANGLADESH": "BD", "VIRGIN ISLANDS, U.S": "VI", "PAKISTAN": "PK",
    "SRI LANKA": "LK", "NEPAL": "NP", "CAMBODIA": "KH", "MONGOLIA": "MN", "QATAR": "QA",
    "KUWAIT": "KW", "OMAN": "OM", "JORDAN": "JO", "LEBANON": "LB", "MOROCCO": "MA",
    "TUNISIA": "TN", "ALGERIA": "DZ", "GHANA": "GH", "TANZANIA, UNITED REPUBLIC OF": "TZ",
    "UGANDA": "UG", "ETHIOPIA": "ET", "MAURITIUS": "MU", "CABO VERDE": "CV", "CAPE VERDE": "CV",
}


# --------------------------------------------------------------------------
# Delegation files: acquisition
# --------------------------------------------------------------------------
def snapshot_urls(date: str, latest: bool = False) -> dict[str, str]:
    """URLs of the five delegation files for YYYYMMDD (or the *-latest files)."""
    if latest:
        return {
            "ripencc": "https://ftp.ripe.net/pub/stats/ripencc/delegated-ripencc-extended-latest",
            "apnic":   "https://ftp.apnic.net/stats/apnic/delegated-apnic-extended-latest",
            "arin":    "https://ftp.arin.net/pub/stats/arin/delegated-arin-extended-latest",
            "lacnic":  "https://ftp.lacnic.net/pub/stats/lacnic/delegated-lacnic-extended-latest",
            "afrinic": "https://ftp.afrinic.net/pub/stats/afrinic/delegated-afrinic-extended-latest",
        }
    y = date[:4]
    this_year = dt.date.today().strftime("%Y") == y
    return {
        "ripencc": f"https://ftp.ripe.net/pub/stats/ripencc/{y}/delegated-ripencc-extended-{date}.bz2",
        "apnic":   f"https://ftp.apnic.net/stats/apnic/{y}/delegated-apnic-extended-{date}.gz",
        # ARIN: current year in the main folder (uncompressed), older years under archive/<year>/ as .gz
        "arin":    (f"https://ftp.arin.net/pub/stats/arin/delegated-arin-extended-{date}" if this_year
                    else f"https://ftp.arin.net/pub/stats/arin/archive/{y}/delegated-arin-extended-{date}.gz"),
        "lacnic":  f"https://ftp.lacnic.net/pub/stats/lacnic/archive/{y}/delegated-lacnic-extended-{date}",
        "afrinic": f"https://ftp.afrinic.net/pub/stats/afrinic/{y}/delegated-afrinic-extended-{date}",
    }


def snapshot_path(date: str, rir: str, base: Path = SNAPSHOT_DIR) -> Path:
    return base / date / f"delegated-{rir}-extended-{date}.gz"


def snapshot_complete(date: str, base: Path = SNAPSHOT_DIR) -> bool:
    return all(snapshot_path(date, r, base).exists() and snapshot_path(date, r, base).stat().st_size > 0
               for r in RIRS)


def fetch_snapshot(date: str, base: Path = SNAPSHOT_DIR, latest: bool = False, log=print) -> None:
    """Downloads the missing delegation files of a snapshot date (target always .gz)."""
    (base / date).mkdir(parents=True, exist_ok=True)
    for rir, url in snapshot_urls(date, latest).items():
        dst = snapshot_path(date, rir, base)
        if dst.exists() and dst.stat().st_size > 0:
            continue
        r = requests.get(url, headers={"User-Agent": UA}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        raw = r.content
        if url.endswith(".bz2"):
            raw = bz2.decompress(raw)
        elif url.endswith(".gz"):
            raw = gzip.decompress(raw)
        head = raw[:200].decode("utf-8", "replace").splitlines()[0] if raw else ""
        with gzip.open(dst, "wb") as fh:
            fh.write(raw)
        (base / date / f"delegated-{rir}-extended-{date}.source.txt").write_text(
            f"{url}\nfetched_at_utc={dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}\n"
            f"header={head}\n", encoding="utf-8")
        log(f"  RIR delegation {rir:8s} {date}: {len(raw)/1e6:5.1f} MB downloaded")


def ensure_snapshot(date: str, base: Path = SNAPSHOT_DIR, fetch: bool = True, log=print) -> bool:
    """True if the snapshot date is completely available locally (after download if needed)."""
    if snapshot_complete(date, base):
        return True
    if not fetch:
        return False
    try:
        fetch_snapshot(date, base, latest=(date == today_str()), log=log)
    except requests.RequestException as e:
        log(f"  RIR delegation {date}: download failed ({e})")
        return False
    return snapshot_complete(date, base)


def today_str() -> str:
    return dt.date.today().strftime("%Y%m%d")


def latest_local_snapshot(base: Path = SNAPSHOT_DIR) -> str | None:
    """Most recent complete snapshot date in the directory (or None)."""
    if not base.exists():
        return None
    dates = sorted(d.name for d in base.iterdir() if d.is_dir() and d.name.isdigit() and snapshot_complete(d.name, base))
    return dates[-1] if dates else None


def snapshot_date_for(ts: dt.datetime | dt.date, granularity: str = "month") -> str:
    """Snapshot date for a record timestamp.

    'month': first day of the timestamp's month (one download per month; the
    measured drift of the registry country is ~0.1-0.2 % per month, see module
    docstring). 'day': exact day."""
    d = ts.date() if isinstance(ts, dt.datetime) else ts
    if granularity == "day":
        return d.strftime("%Y%m%d")
    return d.strftime("%Y%m01")


# --------------------------------------------------------------------------
# Delegation files: evaluation
# --------------------------------------------------------------------------
class Delegations:
    """Sorted IPv4 intervals of a snapshot date -> (cc, registry, status).

    Line format: registry|cc|ipv4|start|count|date|status[|opaque-id].
    Comments (#) and summary lines are skipped."""

    def __init__(self, date: str, base: Path = SNAPSHOT_DIR):
        self.date = date
        starts, ends, vals = [], [], []
        self.headers: dict[str, str] = {}
        for rir in RIRS:
            f = snapshot_path(date, rir, base)
            if not f.exists():
                raise FileNotFoundError(f"delegation file missing: {f} — python maltego/registry.py fetch {date}")
            with gzip.open(f, "rt", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if not line or line[0] == "#":
                        continue
                    if rir not in self.headers:
                        self.headers[rir] = line.strip()
                    p = line.rstrip("\n").split("|")
                    if len(p) < 7 or p[2] != "ipv4" or p[5] == "summary":
                        continue
                    try:
                        start = int(ipaddress.IPv4Address(p[3]))
                        n = int(p[4])
                    except (ValueError, ipaddress.AddressValueError):
                        continue
                    starts.append(start); ends.append(start + n - 1)
                    vals.append((p[1].strip().upper(), p[0], p[6].strip().lower()))
        order = sorted(range(len(starts)), key=starts.__getitem__)
        self.starts = [starts[i] for i in order]
        self.ends = [ends[i] for i in order]
        self.vals = [vals[i] for i in order]

    def lookup(self, ip: str) -> tuple[str, str, str]:
        """(cc, registry, status); ('', '', 'none') if no block covers the address.
        With nested blocks (rare, legacy/ERX) the narrowest wins."""
        try:
            x = int(ipaddress.IPv4Address(ip))
        except (ValueError, ipaddress.AddressValueError):
            return ("", "", "none")
        i = bisect.bisect_right(self.starts, x) - 1
        best = None
        j = i
        while j >= 0 and self.starts[j] <= x and (i - j) < 64:
            if self.ends[j] >= x:
                size = self.ends[j] - self.starts[j]
                if best is None or size < best[0]:
                    best = (size, self.vals[j])
            j -= 1
        return best[1] if best else ("", "", "none")


_DELEG_CACHE: dict[str, Delegations] = {}


def delegations(date: str, base: Path = SNAPSHOT_DIR, fetch: bool = True, log=print) -> Delegations | None:
    """Delegations object for a snapshot date (cached process-wide); None if not obtainable."""
    key = f"{base}|{date}"
    if key not in _DELEG_CACHE:
        if not ensure_snapshot(date, base, fetch=fetch, log=log):
            return None
        _DELEG_CACHE[key] = Delegations(date, base)
    return _DELEG_CACHE[key]


def state_of(cc: str) -> str:
    """Country field -> state ('' if none: empty/EU/ZZ/AP)."""
    cc = (cc or "").strip().upper()
    return "" if cc in NO_STATE else cc


# --------------------------------------------------------------------------
# RDAP: most specific object, live
# --------------------------------------------------------------------------
def _registrant_adr(body: dict) -> tuple[str, str]:
    """(last line of the adr label, structured country component) of the registrant;
    fallback: first entity with an address."""
    for pref in (True, False):
        for e in body.get("entities", []) or []:
            if pref and "registrant" not in (e.get("roles") or []):
                continue
            for it in (e.get("vcardArray") or [None, []])[1]:
                if it and it[0] == "adr":
                    label = (it[1] or {}).get("label", "") or ""
                    last = label.strip().splitlines()[-1].strip() if label.strip() else ""
                    struct = it[3][-1].strip() if isinstance(it[3], list) and it[3] else ""
                    if last or struct:
                        return (last, struct)
    return ("", "")


def rdap_country_from_body(body: dict) -> tuple[str, str]:
    """(ISO country, origin of the field) from an RDAP network object."""
    cc = (body.get("country") or "").strip().upper()
    if cc:
        return cc, "network.country"
    last, struct = _registrant_adr(body)
    if struct and len(struct) == 2 and struct.isalpha():
        return struct.upper(), "registrant.adr.cc"
    name = last.upper().rstrip(".")
    if len(name) == 2 and name.isalpha():
        return name, "registrant.adr.last"
    if name in NAME_TO_ISO:
        return NAME_TO_ISO[name], "registrant.adr.name"
    return "", ("unmapped:" + last if last else "none")


def rdap_lookup(ip: str, rir: str | None = None, timeout: float = 20.0) -> dict:
    """Live RDAP of the responsible registry (without a hint: rdap.org bootstrap).
    Returns cc, cc_source, handle, type, name, rir, url, http_status, error."""
    rir = rir if rir in RDAP_BASE else "none"
    url = RDAP_BASE[rir].format(ip=ip)
    out = {"ip": ip, "rir": rir, "url": url, "cc": "", "cc_source": "", "handle": None,
           "type": None, "name": None, "http_status": None, "error": None}
    try:
        r = requests.get(url, headers={"User-Agent": UA, "Accept": "application/rdap+json"},
                         timeout=timeout, allow_redirects=True)
    except requests.RequestException as e:
        out["error"] = repr(e)
        return out
    out["http_status"] = r.status_code
    if r.status_code != 200:
        return out
    try:
        b = r.json()
    except ValueError:
        out["error"] = "no JSON"
        return out
    out["cc"], out["cc_source"] = rdap_country_from_body(b)
    out["handle"], out["type"], out["name"] = b.get("handle"), b.get("type"), b.get("name")
    if rir == "none":
        # rdap.org redirects; derive the target registry from the final URL
        host = r.url.split("/")[2] if "//" in r.url else ""
        out["rir"] = next((k for k in RIRS if k[:4] in host or (k == "ripencc" and "ripe" in host)), "none")
    return out


# --------------------------------------------------------------------------
# Frozen country-error rates per cell registry flag x S gate
# --------------------------------------------------------------------------
_RATES: dict | None = json.loads(RATES_PATH.read_text(encoding="utf-8")) if RATES_PATH.exists() else None


def registry_rates() -> dict | None:
    """Contents of registry_rates.json (None if the file is missing)."""
    return _RATES


def country_miss_rate(flag: bool, low_mass: bool, level: str = "whois") -> dict | None:
    """P(country vote wrong) for the cell (registry flag, S gate) on the
    reference population -- the country-level counterpart of geoloc.miss_probability.

    level 'whois' (RDAP object, primary) or 'delegation' (delegation file,
    offline fallback). Returns {'p_country_miss', 'n', 'level', 'cell',
    'population'} or None (no rates file / cell without a rate)."""
    if _RATES is None or level not in _RATES.get("levels", {}):
        return None
    cell = f"{'flag' if flag else 'noflag'}_{'lowmass' if low_mass else 'highmass'}"
    z = _RATES["levels"][level]["cells"].get(cell)
    if not z or z.get("p_country_miss") is None:
        return None
    return {"p_country_miss": float(z["p_country_miss"]), "n": int(z["n"]), "level": level,
            "cell": cell, "population": _RATES["population"]}


def registry_flag(ctx: dict, vote_country: str | None, country_mass: float | None = None) -> dict:
    """Registry flag of a case: registry country != country vote, at the best
    available level, plus the frozen country-error rate of its cell.

    Level (pre-specified, exp_registry_country): 'whois' if the RDAP object
    names a state (primary flag); otherwise 'delegation' = today's state of the
    delegation file (offline fallback). The country vote describes today's
    holder, so TODAY's registry state always counts (a re-delegation is warned
    about separately). low_mass = core mass < s_gate (a fallback vote has mass 0
    and counts as low, as in the experiment).

    Returns registry_flag (True/False/None without a vote or a registry country),
    registry_level, registry_flag_country (country of the chosen level),
    p_country_miss + p_country_miss_n (cell flag x gate, None without a rates file)."""
    out = {"registry_flag": None, "registry_level": None, "registry_flag_country": "",
           "p_country_miss": None, "p_country_miss_n": None, "p_country_miss_cell": None}
    if ctx.get("rdap_country"):
        out["registry_level"], out["registry_flag_country"] = "whois", ctx["rdap_country"]
    elif ctx.get("registry_country_today"):
        out["registry_level"], out["registry_flag_country"] = "delegation", ctx["registry_country_today"]
    if not vote_country or not out["registry_level"]:
        return out
    out["registry_flag"] = out["registry_flag_country"] != vote_country
    s_gate = float((_RATES or {}).get("s_gate", 0.5))
    low = (country_mass is None) or (float(country_mass) < s_gate)
    r = country_miss_rate(out["registry_flag"], low, out["registry_level"])
    if r is not None:
        out["p_country_miss"], out["p_country_miss_n"], out["p_country_miss_cell"] = r["p_country_miss"], r["n"], r["cell"]
    return out


# --------------------------------------------------------------------------
# Context + warnings (shared by batch and transform)
# --------------------------------------------------------------------------
def parse_timestamp(text: str | None) -> dt.datetime | None:
    """ISO 8601 (date or date+time, optional Z/offset), 'YYYY-MM-DD HH:MM:SS',
    'DD.MM.YYYY', epoch seconds/milliseconds. None if empty/unreadable."""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    if s.isdigit():
        v = int(s)
        if v > 10_000_000_000:          # milliseconds
            v //= 1000
        return dt.datetime.fromtimestamp(v, tz=dt.timezone.utc)
    s2 = s.replace("Z", "+00:00")
    for fmt in (None, "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%d.%m.%Y", "%Y%m%d"):
        try:
            d = dt.datetime.fromisoformat(s2) if fmt is None else dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d
    return None


def current_delegations(base: Path = SNAPSHOT_DIR, fetch: bool = True, max_age_days: int = 31,
                        log=print) -> Delegations | None:
    """Delegations for "today": the most recent local snapshot, provided it is at most
    max_age_days old (drift of ~0.1-0.2 %/month does not justify a daily 50-MB
    download); otherwise today's snapshot (download), otherwise the most recent local one."""
    loc = latest_local_snapshot(base)
    if loc is not None:
        age = (dt.date.today() - dt.datetime.strptime(loc, "%Y%m%d").date()).days
        if age <= max_age_days:
            return delegations(loc, base, fetch=False, log=log)
    D = delegations(today_str(), base, fetch=fetch, log=log)
    if D is None and loc is not None:
        D = delegations(loc, base, fetch=False, log=log)
    return D


def registry_context(ip: str, as_of: dt.datetime | None = None, *, rdap: bool = False,
                     granularity: str = "month", base: Path = SNAPSHOT_DIR, fetch: bool = True,
                     max_today_age_days: int = 31, log=print) -> dict:
    """Registry country for an IP: delegation today, delegation as of the record
    date (if as_of), optionally whois/RDAP. Network-free except for (a) missing
    delegation snapshots (download of the public bulk files) and (b) rdap=True."""
    ctx = {"as_of": as_of.isoformat(timespec="seconds") if as_of else None,
           "record_age_days": None, "registry_rir": "", "registry_status_asof": "",
           "registry_snapshot_asof": None, "registry_country_asof": "",
           "registry_snapshot_today": None, "registry_country_today": "",
           "registry_redelegated": None,
           "rdap_country": "", "rdap_handle": "", "rdap_type": "", "rdap_error": None}
    now = dt.datetime.now(dt.timezone.utc)
    if as_of is not None:
        ctx["record_age_days"] = max(0, (now - as_of).days)

    # "today": most recent local snapshot if at most max_today_age_days old, otherwise download
    today = today_str()
    D_now = current_delegations(base, fetch=fetch, max_age_days=max_today_age_days, log=log)
    if D_now is not None:
        cc, rir, status = D_now.lookup(ip)
        ctx["registry_snapshot_today"] = D_now.date
        ctx["registry_country_today"] = state_of(cc)
        ctx["registry_rir"] = rir or ""
        if as_of is None:
            ctx["registry_status_asof"] = status

    if as_of is not None:
        snap = snapshot_date_for(as_of, granularity)
        if snap >= today:
            snap = D_now.date if D_now is not None else today
        D_then = D_now if (D_now is not None and snap == D_now.date) else delegations(snap, base, fetch=fetch, log=log)
        if D_then is not None:
            cc, rir, status = D_then.lookup(ip)
            ctx["registry_snapshot_asof"] = D_then.date
            ctx["registry_country_asof"] = state_of(cc)
            ctx["registry_status_asof"] = status
            if not ctx["registry_rir"]:
                ctx["registry_rir"] = rir or ""
            if D_now is not None:
                ctx["registry_redelegated"] = (ctx["registry_country_asof"] != ctx["registry_country_today"]) \
                    or (status in ("available", "reserved", "none")) != \
                       (D_now.lookup(ip)[2] in ("available", "reserved", "none"))

    if rdap:
        r = rdap_lookup(ip, ctx["registry_rir"] or None)
        ctx["rdap_country"] = state_of(r["cc"])
        ctx["rdap_handle"] = r["handle"] or ""
        ctx["rdap_type"] = r["type"] or ""
        ctx["rdap_error"] = r["error"] or (None if r["http_status"] == 200 else f"HTTP {r['http_status']}")
        ctx["rdap_rir"] = r["rir"]
    return ctx


def registry_warnings(ctx: dict, vote_country: str | None, stale_days: int = STALE_DAYS_DEFAULT,
                      country_mass: float | None = None) -> list[str]:
    """Warnings from the registry context, in priority order: age, re-delegation,
    registry/vote contradiction (registry flag with frozen rate), missing registry
    statement. country_mass (core mass of the vote) selects the flag x gate cell."""
    w: list[str] = []
    age = ctx.get("record_age_days")
    if age is not None and age > stale_days:
        w.append(f"record is {age} days old — the GeoIP estimate and the whois describe the CURRENT "
                 f"holder of the address; for the record date only the registry country of the delegation is documentable"
                 + (f" ({ctx['registry_country_asof'] or 'unallocated'}, snapshot {ctx['registry_snapshot_asof']})"
                    if ctx.get("registry_snapshot_asof") else ""))
    if ctx.get("registry_redelegated"):
        w.append(f"address re-delegated since the record date: registry country {ctx['registry_country_asof'] or '--'} "
                 f"({ctx['registry_snapshot_asof']}) -> {ctx['registry_country_today'] or '--'} "
                 f"({ctx['registry_snapshot_today']}) — today's location is not usable for the record date")
    # Always check the contradiction against TODAY's registry state: the country vote
    # describes today's holder (a re-delegation is warned about above).
    # Primary level is the whois object, fallback the delegation (registry_flag).
    fl = registry_flag(ctx, vote_country, country_mass)
    deleg_today = ctx.get("registry_country_today") or ""
    if fl["registry_flag"]:
        lvl = "whois object" if fl["registry_level"] == "whois" else "RIR delegation"
        rate = ""
        if fl["p_country_miss"] is not None:
            L = _RATES["levels"][fl["registry_level"]]
            rate = (f" — on the reference population (n={L['n']:,}) the country vote is wrong under this flag in "
                    f"{L['lift']['p_miss_flag_pct']:.1f} % of cases instead of {L['lift']['p_miss_noflag_pct']:.1f} % "
                    f"(cell with S gate: {100 * fl['p_country_miss']:.1f} %, n={fl['p_country_miss_n']})")
        extra = ""
        if fl["registry_level"] == "whois" and deleg_today and deleg_today not in (vote_country, fl["registry_flag_country"]):
            extra = f"; RIR delegation names {deleg_today}"
        w.append(f"registry country {fl['registry_flag_country']} ({lvl}) contradicts the country vote {vote_country}"
                 f"{rate}{extra}; registration/hosting country and physical location can diverge")
    elif fl["registry_flag"] is False and fl["registry_level"] == "whois" and deleg_today and deleg_today != vote_country:
        w.append(f"RIR delegation names {deleg_today}, whois object and country vote {vote_country} agree "
                 f"(no registry flag: sub-allocation in a country other than the LIR's seat)")
    if ctx.get("registry_snapshot_today") and not ctx.get("registry_country_today"):
        w.append("registry lists no state for the address today (unallocated/reserved/EU)")
    if ctx.get("rdap_error"):
        w.append(f"whois/RDAP not retrievable ({ctx['rdap_error']})")
    return w


# --------------------------------------------------------------------------
def _main(argv: list[str]) -> int:
    if argv and argv[0] == "fetch":
        for d in argv[1:] or [today_str()]:
            ok = ensure_snapshot(d, fetch=True)
            print(f"{d}: {'complete' if ok else 'MISSING'}")
        return 0
    if not argv:
        print(__doc__.split("Standalone:")[1] if "Standalone:" in __doc__ else __doc__)
        return 2
    ip = argv[0]
    as_of = parse_timestamp(argv[argv.index("--as-of") + 1]) if "--as-of" in argv else None
    ctx = registry_context(ip, as_of, rdap="--rdap" in argv)
    print(json.dumps(ctx, indent=2, ensure_ascii=False))
    for w in registry_warnings(ctx, None):
        print("WARNING:", w)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
