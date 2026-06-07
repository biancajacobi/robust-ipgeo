"""Lokale GeoIP-LITE-Datenbanken beschaffen — mit Chain-of-Custody-Beleg.

Lädt die lokalen LITE-DBs (echte, von den freien Web-APIs unabhängige Linien) nach
``data/db/`` und dokumentiert jeden Download forensisch über ``data/store`` (SHA-256
der gespeicherten Datei + Audit-Ledger-Eintrag in der Hash-Kette), analog zu
``fetch_anchors.py`` / ``fetch_sources.py``. Die DB-Datei selbst ist das Beweisstück
(``raw_file`` im Ledger → ``--verify`` der bestehenden Werkzeuge prüft ihren Hash mit).

Datenbanken / geplante ``lineage``:
  maxmind_geolite2   GeoLite2-City.mmdb           MaxMind      (Account-ID + License-Key)
  dbip_lite          dbip-city-lite-YYYY-MM.mmdb  DB-IP        (frei, ohne Login)
  ip2location_lite   IP2LOCATION-LITE-DB5.BIN     IP2Location  (freier Account; Token oder manuell)

Zugangsdaten (NICHT einchecken) aus der Umgebung oder aus ``.env`` im Repo-Wurzel-
verzeichnis (Vorlage: ``.env.example``):
  MAXMIND_ACCOUNT_ID, MAXMIND_LICENSE_KEY, IP2LOCATION_TOKEN (optional)

Belegt wird je DB: Quelle-URL, vom Server attestiertes ``Date``/``Last-Modified``,
SHA-256 der gespeicherten Datei und — wo der Anbieter sie liefert — dessen eigener
SHA-256 (MaxMind) als unabhängige Integritätsbestätigung.

Aufrufe:
  python data/fetch_geodbs.py --all            # alle verfügbaren DBs (Token/Keys vorausgesetzt)
  python data/fetch_geodbs.py --maxmind        # nur GeoLite2-City
  python data/fetch_geodbs.py --dbip           # nur DB-IP City Lite (kein Login)
  python data/fetch_geodbs.py --dbip --month 2026-05   # DB-IP für einen bestimmten Monat
  python data/fetch_geodbs.py --ip2location    # IP2Location LITE DB5 (braucht IP2LOCATION_TOKEN)
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
USER_AGENT = "robust-geoip-reference/research (academic; contact via repo)"

MAXMIND_URL = "https://download.maxmind.com/app/geoip_download"          # Permalink (license_key-Param)
MAXMIND_DL_URL = "https://download.maxmind.com/geoip/databases"          # aktueller Endpoint (Basic-Auth)
DBIP_URL = "https://download.db-ip.com/free/dbip-city-lite-{month}.mmdb.gz"
IP2LOCATION_URL = "https://www.ip2location.com/download/"


# ---------- .env laden (stdlib, ohne python-dotenv) ----------

def load_env() -> dict[str, str]:
    """Repo-Wurzel-``.env`` einlesen → {KEY: VALUE} (überschreibt keine echte Umgebung)."""
    return store.load_env()


def _guard_payload(content: bytes, *, min_bytes: int, what: str) -> None:
    """Verhindert, dass eine Fehlermeldung (z. B. 'NO PERMISSION') als DB gespeichert
    und damit ins forensische Ledger geschrieben wird."""
    if len(content) < min_bytes:
        snippet = content[:120].decode("latin-1", "replace").strip()
        raise SystemExit(f"{what}: unerwartet kleine Antwort ({len(content):,} Bytes) — "
                         f"vermutlich Fehlermeldung, nicht gespeichert.\n  Inhalt: {snippet!r}")


# ---------- Provenance-Beleg für eine gespeicherte DB-Datei ----------

def _provenance_for_db(*, fetch_id, fetched_at, source, lineage, source_url,
                       db_path: Path, server_date, last_modified,
                       attested_sha256=None, extra=None) -> dict:
    rel = db_path.relative_to(store.BASE_DIR)
    entry = {
        "fetch_id": fetch_id,
        "fetched_at_utc": fetched_at,
        "kind": "geodb",
        "source": source,                 # Quellen-Key (für SOURCES später)
        "lineage": lineage,               # dokumentierte Datenherkunft
        "source_url": source_url,
        "http_method": "GET",
        "tool": "data/fetch_geodbs.py",
        "tool_git_commit": store.git_commit(),
        "server_date": server_date,       # vom Server attestiert
        "last_modified": last_modified,
        "db_file": str(rel),
        "db_size_bytes": db_path.stat().st_size,
        "raw_file": str(rel),             # die DB-Datei IST das Beweisstück
        "sha256_raw": store.sha256_file(db_path),
    }
    if attested_sha256:
        entry["provider_sha256"] = attested_sha256   # unabhängige Anbieter-Bestätigung
        entry["provider_sha256_ok"] = (attested_sha256 == entry["sha256_raw"])
    if extra:
        entry.update(extra)
    saved = store.append_provenance(entry)
    return saved


def _report(saved: dict, label: str) -> None:
    print(f"  {label:18s} → data/{saved['db_file']}")
    print(f"    {saved['db_size_bytes']:>12,} Bytes  sha256={saved['sha256_raw'][:16]}…")
    if "provider_sha256_ok" in saved:
        mark = "OK" if saved["provider_sha256_ok"] else "!! MISMATCH"
        print(f"    Anbieter-SHA-256: {mark}")
    print(f"    Server-Date: {saved.get('server_date')}  Ledger={saved['entry_sha256'][:16]}…")


# ---------- MaxMind GeoLite2-City ----------

def _maxmind_get(env: dict, edition: str, suffix: str):
    """GET einer MaxMind-Edition. Bevorzugt den aktuellen Endpoint mit Basic-Auth
    (Account-ID + Key); ohne Account-ID Fallback auf die Permalink-Methode
    (license_key-Query-Parameter)."""
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
        raise SystemExit("MAXMIND_LICENSE_KEY fehlt (in .env oder Umgebung setzen).")

    resp = _maxmind_get(env, edition, "tar.gz")
    if resp.status_code in (401, 403):
        server_msg = resp.text.strip()[:200]
        raise SystemExit(
            f"MaxMind verweigert den Download von {edition} (HTTP {resp.status_code}).\n"
            f"  Serverantwort: {server_msg!r}\n"
            "  → Das ist eine fehlende KONTO-BERECHTIGUNG, kein Key-Tippfehler.\n"
            "  GeoLite2 ist eine separate, kostenlose Anmeldung: einmalig unter\n"
            "  https://www.maxmind.com/en/geolite2/signup registrieren und die EULA\n"
            "  bestätigen. Danach erscheint GeoLite2 unter 'Download Databases' und\n"
            "  derselbe Key funktioniert. (Prüfen: taucht GeoLite2 dort auf?)")
    resp.raise_for_status()
    blob = resp.content

    # Anbieter-SHA-256 (über das tar.gz) als unabhängige Integritätsbestätigung
    sha_resp = _maxmind_get(env, edition, "tar.gz.sha256")
    attested_targz_sha = (sha_resp.text.split()[0].strip()
                          if sha_resp.ok and sha_resp.text.strip() else None)
    targz_sha = store.sha256_bytes(blob)
    if attested_targz_sha and attested_targz_sha != targz_sha:
        raise SystemExit(f"MaxMind-SHA-256 stimmt nicht (Download korrupt?): "
                         f"erwartet {attested_targz_sha[:16]}…, erhalten {targz_sha[:16]}…")

    # mmdb aus dem tar.gz extrahieren (Pfad: <Edition>_YYYYMMDD/<Edition>.mmdb)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    db_path = DB_DIR / f"{edition}.mmdb"
    build_label = None
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        member = next((m for m in tar.getmembers() if m.name.endswith(".mmdb")), None)
        if member is None:
            raise SystemExit("Kein .mmdb im MaxMind-Archiv gefunden.")
        build_label = Path(member.name).parent.name   # z. B. GeoLite2-City_20260603
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
        print(f"    DB-Stand: {build_label}")
    return saved


# ---------- DB-IP City Lite (frei, ohne Login) ----------

def fetch_dbip(month: str | None = None) -> dict:
    month = month or datetime.now(timezone.utc).strftime("%Y-%m")
    url = DBIP_URL.format(month=month)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    if resp.status_code == 404:
        # Monatsdatei evtl. noch nicht veröffentlicht → Vormonat versuchen
        y, m = (int(x) for x in month.split("-"))
        prev = f"{y-1}-12" if m == 1 else f"{y}-{m-1:02d}"
        print(f"  DB-IP {month} nicht gefunden (404) → versuche {prev}")
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


# ---------- IP2Location LITE DB5 (freier Account; Token oder manuell) ----------

def fetch_ip2location(env: dict, file_code: str = "DB5LITEBIN") -> dict:
    # WICHTIG: der DATEI-Download braucht den Download-Token (Account → Download),
    # NICHT den Web-Service-API-Key (api.ip2location.io). Der Endpoint antwortet mit
    # 302 auf eine signierte Storage-URL; requests folgt dem Redirect automatisch.
    token = env.get("IP2LOCATION_DOWNLOAD_TOKEN")
    if not token:
        raise SystemExit(
            "IP2LOCATION_DOWNLOAD_TOKEN fehlt. Download-Token (≠ .io-API-Key) in .env\n"
            "setzen, ODER die IP2LOCATION-LITE-DB5.BIN manuell von ip2location.com\n"
            "herunterladen und nach data/db/ legen — dann `--register-existing` nutzen.")
    resp = requests.get(IP2LOCATION_URL, params={"token": token, "file": file_code},
                        headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    resp.raise_for_status()
    # IP2Location signalisiert Fehler als kurzen Klartext (z. B. 'NO PERMISSION',
    # 'INVALID TOKEN', '5 TIMES') mit HTTP 200 — vor dem Speichern abfangen.
    _guard_payload(resp.content, min_bytes=1_000_000, what="IP2Location LITE DB5")
    DB_DIR.mkdir(parents=True, exist_ok=True)
    db_path = DB_DIR / "IP2LOCATION-LITE-DB5.BIN"
    # Antwort ist i. d. R. ein ZIP mit der .BIN
    if resp.content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            name = next((n for n in zf.namelist() if n.upper().endswith(".BIN")), None)
            if name is None:
                raise SystemExit("Keine .BIN im IP2Location-ZIP gefunden.")
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
    """Bereits manuell abgelegte DB-Datei in die Provenance-Kette aufnehmen."""
    if not path.exists():
        raise SystemExit(f"Datei nicht gefunden: {path}")
    saved = _provenance_for_db(
        fetch_id=store.new_fetch_id(), fetched_at=store.utc_now_iso(),
        source=source, lineage=lineage, source_url="(manuell abgelegt)",
        db_path=path, server_date=None, last_modified=None,
        extra={"registered_existing": True})
    _report(saved, f"{source} (manuell)")
    return saved


# ---------- CLI ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="LITE-DBs beschaffen (mit Provenance).")
    ap.add_argument("--all", action="store_true", help="alle verfügbaren DBs holen")
    ap.add_argument("--maxmind", action="store_true", help="GeoLite2-City (MaxMind)")
    ap.add_argument("--dbip", action="store_true", help="DB-IP City Lite (kein Login)")
    ap.add_argument("--ip2location", action="store_true", help="IP2Location LITE DB5 (Token)")
    ap.add_argument("--month", default=None, help="DB-IP: Monat YYYY-MM (Default: aktueller)")
    ap.add_argument("--register-existing", nargs=3, metavar=("PATH", "SOURCE", "LINEAGE"),
                    help="bereits abgelegte Datei in die Provenance-Kette aufnehmen")
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
        ap.error("nichts ausgewählt: --maxmind / --dbip / --ip2location / --all")

    print("LITE-DB-Download (Beweisstück + Hash-Ketten-Eintrag je DB):")
    if do_dbip:
        fetch_dbip(args.month)
    if do_mm:
        fetch_maxmind(env)
    if do_ip2:
        if env.get("IP2LOCATION_TOKEN"):
            fetch_ip2location(env)
        elif args.all:
            print("  IP2Location übersprungen (kein IP2LOCATION_TOKEN) — manuell holen.")
        else:
            fetch_ip2location(env)   # wirft mit Anleitung

    print("\nKette prüfen:  python data/fetch_anchors.py --verify")


if __name__ == "__main__":
    main()
