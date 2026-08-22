"""Obtain local GeoIP LITE databases — with chain-of-custody evidence.

Downloads the local LITE DBs (genuine lines independent of the free web APIs) to
``data/db/`` and documents every download forensically via ``data/store`` (SHA-256
of the saved file + audit-ledger entry in the hash chain), analogous to
``fetch_anchors.py`` / ``fetch_sources.py``. The DB file itself is the exhibit
(``raw_file`` in the ledger → ``--verify`` of the existing tools also checks its hash).

Databases / intended ``lineage``:
  maxmind_geolite2   GeoLite2-City.mmdb           MaxMind      (account ID + license key)
  dbip_lite          dbip-city-lite-YYYY-MM.mmdb  DB-IP        (free, no login)
  ip2location_lite   IP2LOCATION-LITE-DB5.BIN     IP2Location  (free account; token or manual)

Credentials (do NOT check in) from the environment or from ``.env`` in the repo
root (template: ``.env.example``):
  MAXMIND_ACCOUNT_ID, MAXMIND_LICENSE_KEY, IP2LOCATION_TOKEN (optional)

Recorded per DB: source URL, server-attested ``Date``/``Last-Modified``,
SHA-256 of the saved file and — where the provider supplies it — the provider's
own SHA-256 (MaxMind) as an independent integrity confirmation.

Invocations:
  python data/fetch_geodbs.py --all            # all available DBs (tokens/keys required)
  python data/fetch_geodbs.py --maxmind        # GeoLite2-City only
  python data/fetch_geodbs.py --dbip           # DB-IP City Lite only (no login)
  python data/fetch_geodbs.py --dbip --month 2026-05   # DB-IP for a specific month
  python data/fetch_geodbs.py --ip2location    # IP2Location LITE DB5 (needs IP2LOCATION_TOKEN)
"""

from __future__ import annotations

import argparse
import gzip
import io
import sys
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import store  # noqa: E402

DB_DIR = store.BASE_DIR / "db"
TIMEOUT = 120
USER_AGENT = "robust-ipgeo/research (academic; contact via repo)"

MAXMIND_URL = "https://download.maxmind.com/app/geoip_download"          # permalink (license_key param)
MAXMIND_DL_URL = "https://download.maxmind.com/geoip/databases"          # current endpoint (basic auth)
DBIP_URL = "https://download.db-ip.com/free/dbip-city-lite-{month}.mmdb.gz"
IP2LOCATION_URL = "https://www.ip2location.com/download/"


# ---------- load .env (stdlib, without python-dotenv) ----------

def load_env() -> dict[str, str]:
    """Read the repo-root ``.env`` → {KEY: VALUE} (does not override a real environment)."""
    return store.load_env()


def _guard_payload(content: bytes, *, min_bytes: int, what: str) -> None:
    """Prevents an error message (e.g. 'NO PERMISSION') from being saved as a DB
    and thus written into the forensic ledger."""
    if len(content) < min_bytes:
        snippet = content[:120].decode("latin-1", "replace").strip()
        raise SystemExit(f"{what}: unexpectedly small response ({len(content):,} bytes) — "
                         f"probably an error message, not saved.\n  Content: {snippet!r}")


# ---------- provenance record for a saved DB file ----------

def _provenance_for_db(*, fetch_id, fetched_at, source, lineage, source_url,
                       db_path: Path, server_date, last_modified,
                       attested_sha256=None, extra=None) -> dict:
    rel = db_path.relative_to(store.BASE_DIR)
    entry = {
        "fetch_id": fetch_id,
        "fetched_at_utc": fetched_at,
        "kind": "geodb",
        "source": source,                 # source key (for SOURCES later)
        "lineage": lineage,               # documented data origin
        "source_url": source_url,
        "http_method": "GET",
        "tool": "data/fetch_geodbs.py",
        "tool_git_commit": store.git_commit(),
        "server_date": server_date,       # attested by the server
        "last_modified": last_modified,
        "db_file": str(rel),
        "db_size_bytes": db_path.stat().st_size,
        "raw_file": str(rel),             # the DB file IS the exhibit
        "sha256_raw": store.sha256_file(db_path),
    }
    if attested_sha256:
        entry["provider_sha256"] = attested_sha256   # independent provider confirmation
        entry["provider_sha256_ok"] = (attested_sha256 == entry["sha256_raw"])
    if extra:
        entry.update(extra)
    saved = store.append_provenance(entry)
    return saved


def _report(saved: dict, label: str) -> None:
    print(f"  {label:18s} → data/{saved['db_file']}")
    print(f"    {saved['db_size_bytes']:>12,} bytes  sha256={saved['sha256_raw'][:16]}…")
    if "provider_sha256_ok" in saved:
        mark = "OK" if saved["provider_sha256_ok"] else "!! MISMATCH"
        print(f"    Provider SHA-256: {mark}")
    print(f"    Server date: {saved.get('server_date')}  ledger={saved['entry_sha256'][:16]}…")


# ---------- MaxMind GeoLite2-City ----------

def _maxmind_get(env: dict, edition: str, suffix: str):
    """GET a MaxMind edition. Prefers the current endpoint with basic auth
    (account ID + key); without an account ID, falls back to the permalink
    method (license_key query parameter)."""
    account = env.get("MAXMIND_ACCOUNT_ID")
    key = env["MAXMIND_LICENSE_KEY"]
    headers = {"User-Agent": USER_AGENT}
    if account:
        url = f"{MAXMIND_DL_URL}/{edition}/download"
        return requests.get(url, params={"suffix": suffix}, headers=headers,
                            auth=(account, key), timeout=TIMEOUT)
    return requests.get(MAXMIND_URL, headers=headers, timeout=TIMEOUT,
                        params={"edition_id": edition, "license_key": key, "suffix": suffix})


def fetch_maxmind(env: dict, edition: str = "GeoLite2-City") -> dict:
    if not env.get("MAXMIND_LICENSE_KEY"):
        raise SystemExit("MAXMIND_LICENSE_KEY missing (set it in .env or the environment).")

    resp = _maxmind_get(env, edition, "tar.gz")
    if resp.status_code in (401, 403):
        server_msg = resp.text.strip()[:200]
        raise SystemExit(
            f"MaxMind refuses the download of {edition} (HTTP {resp.status_code}).\n"
            f"  Server response: {server_msg!r}\n"
            "  → This is a missing ACCOUNT PERMISSION, not a key typo.\n"
            "  GeoLite2 is a separate, free sign-up: register once at\n"
            "  https://www.maxmind.com/en/geolite2/signup and confirm the EULA.\n"
            "  Afterwards GeoLite2 appears under 'Download Databases' and\n"
            "  the same key works. (Check: does GeoLite2 show up there?)")
    resp.raise_for_status()
    blob = resp.content

    # provider SHA-256 (over the tar.gz) as an independent integrity confirmation
    sha_resp = _maxmind_get(env, edition, "tar.gz.sha256")
    attested_targz_sha = (sha_resp.text.split()[0].strip()
                          if sha_resp.ok and sha_resp.text.strip() else None)
    targz_sha = store.sha256_bytes(blob)
    if attested_targz_sha and attested_targz_sha != targz_sha:
        raise SystemExit(f"MaxMind SHA-256 does not match (download corrupt?): "
                         f"expected {attested_targz_sha[:16]}…, got {targz_sha[:16]}…")

    # extract the mmdb from the tar.gz (path: <edition>_YYYYMMDD/<edition>.mmdb)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    db_path = DB_DIR / f"{edition}.mmdb"
    build_label = None
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        member = next((m for m in tar.getmembers() if m.name.endswith(".mmdb")), None)
        if member is None:
            raise SystemExit("No .mmdb found in the MaxMind archive.")
        build_label = Path(member.name).parent.name   # e.g. GeoLite2-City_20260603
        fh = tar.extractfile(member)
        db_path.write_bytes(fh.read())

    fetch_id = store.new_fetch_id()
    saved = _provenance_for_db(
        fetch_id=fetch_id, fetched_at=store.utc_now_iso(),
        source="maxmind_geolite2", lineage="maxmind_geolite2",
        source_url=f"{MAXMIND_URL}?edition_id={edition}&suffix=tar.gz",
        db_path=db_path, server_date=resp.headers.get("Date"),
        last_modified=resp.headers.get("Last-Modified"),
        attested_sha256=None,
        extra={"edition": edition, "db_build_label": build_label,
               "provider_targz_sha256": attested_targz_sha,
               "provider_targz_sha256_ok": (attested_targz_sha == targz_sha
                                            if attested_targz_sha else None)})
    _report(saved, "MaxMind GeoLite2")
    if build_label:
        print(f"    DB build: {build_label}")
    return saved


# ---------- DB-IP City Lite (free, no login) ----------

def fetch_dbip(month: str | None = None) -> dict:
    month = month or datetime.now(timezone.utc).strftime("%Y-%m")
    url = DBIP_URL.format(month=month)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    if resp.status_code == 404:
        # monthly file possibly not published yet → try the previous month
        y, m = (int(x) for x in month.split("-"))
        prev = f"{y-1}-12" if m == 1 else f"{y}-{m-1:02d}"
        print(f"  DB-IP {month} not found (404) → trying {prev}")
        return fetch_dbip(prev)
    resp.raise_for_status()
    data = gzip.decompress(resp.content)
    _guard_payload(data, min_bytes=1_000_000, what="DB-IP City Lite")

    DB_DIR.mkdir(parents=True, exist_ok=True)
    db_path = DB_DIR / f"dbip-city-lite-{month}.mmdb"
    db_path.write_bytes(data)

    saved = _provenance_for_db(
        fetch_id=store.new_fetch_id(), fetched_at=store.utc_now_iso(),
        source="dbip_lite", lineage="dbip_lite", source_url=url,
        db_path=db_path, server_date=resp.headers.get("Date"),
        last_modified=resp.headers.get("Last-Modified"),
        extra={"db_month": month})
    _report(saved, "DB-IP City Lite")
    return saved


# ---------- IP2Location LITE DB5 (free account; token or manual) ----------

def fetch_ip2location(env: dict, file_code: str = "DB5LITEBIN") -> dict:
    # IMPORTANT: the FILE download needs the download token (account → download),
    # NOT the web-service API key (api.ip2location.io). The endpoint answers with
    # a 302 to a signed storage URL; requests follows the redirect automatically.
    token = env.get("IP2LOCATION_DOWNLOAD_TOKEN")
    if not token:
        raise SystemExit(
            "IP2LOCATION_DOWNLOAD_TOKEN missing. Set the download token (≠ .io API key)\n"
            "in .env, OR download the IP2LOCATION-LITE-DB5.BIN manually from\n"
            "ip2location.com and place it in data/db/ — then use `--register-existing`.")
    resp = requests.get(IP2LOCATION_URL, params={"token": token, "file": file_code},
                        headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    resp.raise_for_status()
    # IP2Location signals errors as short plain text (e.g. 'NO PERMISSION',
    # 'INVALID TOKEN', '5 TIMES') with HTTP 200 — intercept before saving.
    _guard_payload(resp.content, min_bytes=1_000_000, what="IP2Location LITE DB5")
    DB_DIR.mkdir(parents=True, exist_ok=True)
    db_path = DB_DIR / "IP2LOCATION-LITE-DB5.BIN"
    # the response is usually a ZIP containing the .BIN
    if resp.content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            name = next((n for n in zf.namelist() if n.upper().endswith(".BIN")), None)
            if name is None:
                raise SystemExit("No .BIN found in the IP2Location ZIP.")
            db_path.write_bytes(zf.read(name))
    else:
        db_path.write_bytes(resp.content)

    saved = _provenance_for_db(
        fetch_id=store.new_fetch_id(), fetched_at=store.utc_now_iso(),
        source="ip2location_lite", lineage="ip2location_lite",
        source_url=f"{IP2LOCATION_URL}?file={file_code}",
        db_path=db_path, server_date=resp.headers.get("Date"),
        last_modified=resp.headers.get("Last-Modified"))
    _report(saved, "IP2Location LITE")
    return saved


def register_existing(path: Path, source: str, lineage: str) -> dict:
    """Add an already manually placed DB file to the provenance chain."""
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    saved = _provenance_for_db(
        fetch_id=store.new_fetch_id(), fetched_at=store.utc_now_iso(),
        source=source, lineage=lineage, source_url="(placed manually)",
        db_path=path, server_date=None, last_modified=None,
        extra={"registered_existing": True})
    _report(saved, f"{source} (manual)")
    return saved


# ---------- CLI ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="Obtain LITE DBs (with provenance).")
    ap.add_argument("--all", action="store_true", help="fetch all available DBs")
    ap.add_argument("--maxmind", action="store_true", help="GeoLite2-City (MaxMind)")
    ap.add_argument("--dbip", action="store_true", help="DB-IP City Lite (no login)")
    ap.add_argument("--ip2location", action="store_true", help="IP2Location LITE DB5 (token)")
    ap.add_argument("--month", default=None, help="DB-IP: month YYYY-MM (default: current)")
    ap.add_argument("--register-existing", nargs=3, metavar=("PATH", "SOURCE", "LINEAGE"),
                    help="add an already placed file to the provenance chain")
    args = ap.parse_args()

    if args.register_existing:
        path, source, lineage = args.register_existing
        register_existing(Path(path), source, lineage)
        return

    env = load_env()
    do_mm = args.maxmind or args.all
    do_dbip = args.dbip or args.all
    do_ip2 = args.ip2location or args.all
    if not (do_mm or do_dbip or do_ip2):
        ap.error("nothing selected: --maxmind / --dbip / --ip2location / --all")

    print("LITE DB download (exhibit + hash-chain entry per DB):")
    if do_dbip:
        fetch_dbip(args.month)
    if do_mm:
        fetch_maxmind(env)
    if do_ip2:
        if env.get("IP2LOCATION_TOKEN"):
            fetch_ip2location(env)
        elif args.all:
            print("  IP2Location skipped (no IP2LOCATION_TOKEN) — fetch manually.")
        else:
            fetch_ip2location(env)   # raises with instructions

    print("\nCheck the chain:  python data/fetch_anchors.py --verify")


if __name__ == "__main__":
    main()
