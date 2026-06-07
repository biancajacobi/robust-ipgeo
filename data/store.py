"""Text-basierter Provenance-Store (Chain of Custody) für Datenabrufe.

Ziel: jeden Abruf forensisch nachvollziehbar belegen — *wann*, *woher*, *was*
und *womit* abgerufen wurde, und ob die Daten seither unverändert sind.

Ablage (alles textbasiert, keine Binär-DB nötig):
  data/provenance.jsonl       append-only Audit-Ledger, EINE JSON-Zeile je Abruf;
                              tamper-evident über eine Hash-Kette (jede Zeile
                              referenziert den Hash der vorigen).  -> versioniert.
  data/cache/raw/<id>.json    Rohantwort der API, wortgetreu (das "Beweisstück");
                              -> gitignored (Größe), per SHA-256 im Ledger belegt.
  data/cache/<name>.csv       normalisierte Arbeitskopie für die Auswertung.

Verifikation: ``verify_chain()`` prüft die Hash-Kette des Ledgers und – wo die
Rohdatei noch vorliegt – deren SHA-256 gegen den im Ledger festgehaltenen Wert.
Stdlib-only (json, hashlib, csv, subprocess) — bewusst dependency-arm, damit der
forensische Kern unabhängig von Dritt-Bibliotheken nachvollziehbar bleibt.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
RAW_DIR = CACHE_DIR / "raw"

# Dataset-Switch: erlaubt einen separaten Lauf (z. B. RIPE-Atlas-Probes) gegen
# eigene Cache-/Ledger-Dateien, ohne die Anchor-Artefakte zu beruehren.
# GEOIP_DATASET=anchors (Default) -> unveraenderte Pfade; =probes -> *_probes-Varianten.
DATASET = (os.environ.get("GEOIP_DATASET") or "anchors").strip() or "anchors"
_SUFFIX = "" if DATASET == "anchors" else f"_{DATASET}"
GROUND_TRUTH_CSV = CACHE_DIR / f"{DATASET}.csv"            # anchors.csv | probes.csv
OBSERVATIONS_CSV = CACHE_DIR / f"observations{_SUFFIX}.csv"
SOURCES_RAW_CSV = CACHE_DIR / f"sources_raw{_SUFFIX}.jsonl"
PROVENANCE_FILE = BASE_DIR / f"provenance{_SUFFIX}.jsonl"

GENESIS = "0" * 64  # prev-Hash der allerersten Ledger-Zeile

ANCHOR_COLUMNS = ["anchor_id", "ip", "lat", "lon", "city", "country", "asn",
                  "hostname", "is_disabled"]

# Eine Beobachtung = Schätzung einer Quelle für eine IP (Spalten der Arbeitskopie).
# ``lineage`` = Datenherkunft der Quelle (für Linien-Unabhängigkeit / gewichtete
# Aggregation; gleiche lineage = korrelierte Quellen).
# Qualitäts-/Default-Felder:
#   accuracy_radius      MaxMind-Konfidenzradius in km (nur MaxMind; sonst leer) —
#                        stärkstes Qualitätssignal; trennt empirisch ~8 km von ~558 km.
#   db_version           Build der lokalen DB (Offline-Quellen): identifiziert, welcher
#                        mmdb/BIN-Stand diese Antwort erzeugt hat (Chain-of-Custody).
#   is_default_centroid  True = Hub-/Kapital-Default (Centroid-Detektion, Wächter B).
#   centroid_match       gematchter Hub/Mittelpunkt (Koordinate) oder leer.
OBSERVATION_COLUMNS = ["ip", "source", "lineage", "lat", "lon", "city", "country",
                       "accuracy_radius", "db_version", "is_default_centroid",
                       "centroid_match", "status", "fetched_at_utc"]


# ---------- Hash-/Zeit-Helfer ----------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(obj) -> bytes:
    """Deterministische JSON-Bytes (Schlüssel sortiert) für reproduzierbare Hashes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def sha256_json(obj) -> str:
    """SHA-256 über die kanonische JSON-Darstellung eines Objekts."""
    return sha256_bytes(_canonical(obj))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_fetch_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_commit() -> str:
    """Aktueller Git-Commit des Abruf-Werkzeugs (best effort)."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BASE_DIR,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------- lokale Zugangsdaten (.env, stdlib statt python-dotenv) ----------

# Bekannte Credential-Schlüssel der Download-/API-Quellen (NIE im Repo; in .env).
CRED_KEYS = ("MAXMIND_ACCOUNT_ID", "MAXMIND_LICENSE_KEY", "IP2LOCATION_TOKEN",
             "IP2LOCATION_DOWNLOAD_TOKEN", "IPINFO_TOKEN")


def load_env(keys: tuple[str, ...] = CRED_KEYS) -> dict[str, str]:
    """Zugangsdaten aus der Repo-Wurzel-``.env`` lesen → {KEY: VALUE}.

    Stdlib-only (kein python-dotenv). Echte Umgebungsvariablen haben Vorrang vor
    den .env-Werten — praktisch für CI/headless-Läufe ohne Datei.
    """
    import os
    env_path = BASE_DIR.parent / ".env"
    values: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip().strip('"').strip("'")
    for k in keys:
        if os.environ.get(k):
            values[k] = os.environ[k]
    return values


# ---------- Rohdaten ablegen ----------

def save_raw(fetch_id: str, raw_bytes: bytes, label: str = "response") -> tuple[Path, str]:
    """Rohantwort wortgetreu sichern. Rückgabe: (Pfad, SHA-256)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{label}_{fetch_id}.json"
    path.write_bytes(raw_bytes)
    return path, sha256_bytes(raw_bytes)


# ---------- Audit-Ledger (Hash-Kette) ----------

def last_provenance() -> dict | None:
    if not PROVENANCE_FILE.exists():
        return None
    last = None
    with open(PROVENANCE_FILE, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                last = json.loads(line)
    return last


def append_provenance(entry: dict) -> dict:
    """Abruf-Datensatz an das Ledger anhängen und in die Hash-Kette einbinden.

    Ergänzt automatisch ``prev_sha256`` (Hash der vorigen Zeile) und
    ``entry_sha256`` (Hash dieser Zeile inkl. prev) -> tamper-evident.
    """
    prev = last_provenance()
    entry = dict(entry)
    entry.pop("entry_sha256", None)
    entry["prev_sha256"] = prev["entry_sha256"] if prev else GENESIS
    entry["entry_sha256"] = sha256_json(entry)

    PROVENANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROVENANCE_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def verify_chain() -> list[dict]:
    """Ledger prüfen. Rückgabe: ein Befund-Dict je Zeile (ok + Problemliste)."""
    findings = []
    if not PROVENANCE_FILE.exists():
        return findings
    prev_hash = GENESIS
    with open(PROVENANCE_FILE, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            problems = []
            # 1) Hash-Kette
            if entry.get("prev_sha256") != prev_hash:
                problems.append("prev_sha256 passt nicht (Kette unterbrochen)")
            stored = entry.get("entry_sha256")
            recomputed = sha256_json({k: v for k, v in entry.items()
                                      if k != "entry_sha256"})
            if stored != recomputed:
                problems.append("entry_sha256 stimmt nicht (Zeile verändert)")
            # 2) Rohdatei-Integrität, falls noch vorhanden
            raw_rel = entry.get("raw_file")
            if raw_rel:
                raw_path = BASE_DIR / raw_rel
                if not raw_path.exists():
                    problems.append("Rohdatei fehlt (nur Ledger-Beleg vorhanden)")
                elif sha256_file(raw_path) != entry.get("sha256_raw"):
                    problems.append("sha256_raw != Rohdatei (Daten verändert)")
            findings.append({"line": n, "fetch_id": entry.get("fetch_id"),
                             "ok": not problems, "problems": problems})
            prev_hash = stored
    return findings


# ---------- normalisierte Arbeitskopie ----------

def save_anchors_csv(records: list[dict], path: str | Path | None = None) -> Path:
    path = Path(path) if path else GROUND_TRUTH_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ANCHOR_COLUMNS)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k) for k in ANCHOR_COLUMNS})
    return path


def load_anchors_csv(path: str | Path | None = None) -> list[dict]:
    path = Path(path) if path else GROUND_TRUTH_CSV
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def save_observations_csv(records: list[dict], path: str | Path | None = None) -> Path:
    path = Path(path) if path else OBSERVATIONS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OBSERVATION_COLUMNS)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k) for k in OBSERVATION_COLUMNS})
    return path


def load_observations_csv(path: str | Path | None = None) -> list[dict]:
    path = Path(path) if path else OBSERVATIONS_CSV
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
