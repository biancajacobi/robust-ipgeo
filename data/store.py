"""Text-based provenance store (chain of custody) for data fetches.

Goal: document every fetch in a forensically traceable way — *when*, *from
where*, *what* and *with what* was fetched, and whether the data has remained
unchanged since.

Storage layout (all text-based, no binary DB required):
  data/provenance.jsonl       append-only audit ledger, ONE JSON line per fetch;
                              tamper-evident via a hash chain (each line
                              references the hash of the previous one).  -> versioned.
  data/cache/raw/<id>.json    raw API response, verbatim (the "exhibit");
                              -> gitignored (size), attested via SHA-256 in the ledger.
  data/cache/<name>.csv       normalized working copy for the evaluation.

Verification: ``verify_chain()`` checks the ledger's hash chain and — where the
raw file is still present — its SHA-256 against the value recorded in the ledger.
Stdlib-only (json, hashlib, csv, subprocess) — deliberately dependency-light so
that the forensic core remains verifiable independently of third-party libraries.
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

# Dataset switch: allows a separate run (e.g. RIPE Atlas probes) against
# its own cache/ledger files without touching the anchor artifacts.
# GEOIP_DATASET=anchors (default) -> unchanged paths; =probes -> *_probes variants.
DATASET = (os.environ.get("GEOIP_DATASET") or "anchors").strip() or "anchors"
_SUFFIX = "" if DATASET == "anchors" else f"_{DATASET}"
GROUND_TRUTH_CSV = CACHE_DIR / f"{DATASET}.csv"            # anchors.csv | probes.csv
OBSERVATIONS_CSV = CACHE_DIR / f"observations{_SUFFIX}.csv"
SOURCES_RAW_CSV = CACHE_DIR / f"sources_raw{_SUFFIX}.jsonl"
PROVENANCE_FILE = BASE_DIR / f"provenance{_SUFFIX}.jsonl"

GENESIS = "0" * 64  # prev hash of the very first ledger line

ANCHOR_COLUMNS = ["anchor_id", "ip", "lat", "lon", "city", "country", "asn",
                  "hostname", "is_disabled"]

# One observation = one source's estimate for one IP (columns of the working copy).
# ``lineage`` = data origin of the source (for line independence / weighted
# aggregation; same lineage = correlated sources).
# Quality/default fields:
#   accuracy_radius      MaxMind confidence radius in km (MaxMind only; else empty) —
#                        strongest quality signal; empirically separates ~8 km from ~558 km.
#   db_version           build of the local DB (offline sources): identifies which
#                        mmdb/BIN snapshot produced this answer (chain of custody).
#   is_default_centroid  True = hub/capital default (centroid detection, guard B).
#   centroid_match       matched hub/centroid (coordinate) or empty.
OBSERVATION_COLUMNS = ["ip", "source", "lineage", "lat", "lon", "city", "country",
                       "accuracy_radius", "db_version", "is_default_centroid",
                       "centroid_match", "status", "fetched_at_utc"]


# ---------- hash/time helpers ----------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(obj) -> bytes:
    """Deterministic JSON bytes (sorted keys) for reproducible hashes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def sha256_json(obj) -> str:
    """SHA-256 over the canonical JSON representation of an object."""
    return sha256_bytes(_canonical(obj))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_fetch_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_commit() -> str:
    """Current git commit of the fetch tool (best effort)."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BASE_DIR,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------- local credentials (.env, stdlib instead of python-dotenv) ----------

# Known credential keys of the download/API sources (NEVER in the repo; in .env).
CRED_KEYS = ("MAXMIND_ACCOUNT_ID", "MAXMIND_LICENSE_KEY", "IP2LOCATION_TOKEN",
             "IP2LOCATION_DOWNLOAD_TOKEN", "IPINFO_TOKEN")


def load_env(keys: tuple[str, ...] = CRED_KEYS) -> dict[str, str]:
    """Read credentials from the repo-root ``.env`` → {KEY: VALUE}.

    Stdlib-only (no python-dotenv). Real environment variables take precedence
    over the .env values — convenient for CI/headless runs without a file.
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


# ---------- store raw data ----------

def save_raw(fetch_id: str, raw_bytes: bytes, label: str = "response") -> tuple[Path, str]:
    """Save the raw response verbatim. Returns: (path, SHA-256)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{label}_{fetch_id}.json"
    path.write_bytes(raw_bytes)
    return path, sha256_bytes(raw_bytes)


# ---------- audit ledger (hash chain) ----------

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
    """Append a fetch record to the ledger and link it into the hash chain.

    Automatically adds ``prev_sha256`` (hash of the previous line) and
    ``entry_sha256`` (hash of this line incl. prev) -> tamper-evident.
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
    """Check the ledger. Returns: one finding dict per line (ok + list of problems)."""
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
            # 1) hash chain
            if entry.get("prev_sha256") != prev_hash:
                problems.append("prev_sha256 does not match (chain broken)")
            stored = entry.get("entry_sha256")
            recomputed = sha256_json({k: v for k, v in entry.items()
                                      if k != "entry_sha256"})
            if stored != recomputed:
                problems.append("entry_sha256 does not match (line modified)")
            # 2) raw-file integrity, if still present
            raw_rel = entry.get("raw_file")
            if raw_rel:
                raw_path = BASE_DIR / raw_rel
                if not raw_path.exists():
                    problems.append("raw file missing (only ledger record present)")
                elif sha256_file(raw_path) != entry.get("sha256_raw"):
                    problems.append("sha256_raw != raw file (data modified)")
            findings.append({"line": n, "fetch_id": entry.get("fetch_id"),
                             "ok": not problems, "problems": problems})
            prev_hash = stored
    return findings


# ---------- normalized working copy ----------

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
