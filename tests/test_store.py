"""Tests for the provenance store: hashing + tamper-evident ledger chain."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import store


def test_sha256_bytes_known_vector():
    # well-known NIST test vector for "abc"
    assert store.sha256_bytes(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_sha256_json_is_key_order_independent():
    assert store.sha256_json({"a": 1, "b": 2}) == store.sha256_json({"b": 2, "a": 1})


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "BASE_DIR", tmp_path)
    monkeypatch.setattr(store, "PROVENANCE_FILE", tmp_path / "provenance.jsonl")


def test_chain_links_and_verifies(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    e1 = store.append_provenance({"fetch_id": "A"})
    e2 = store.append_provenance({"fetch_id": "B"})

    assert e1["prev_sha256"] == store.GENESIS
    assert e2["prev_sha256"] == e1["entry_sha256"]  # chain is linked

    findings = store.verify_chain()
    assert len(findings) == 2
    assert all(f["ok"] for f in findings)


def test_chain_detects_tampering(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    store.append_provenance({"fetch_id": "A", "payload": "original"})
    store.append_provenance({"fetch_id": "B", "payload": "second"})

    # modify the first line after the fact
    pf = store.PROVENANCE_FILE
    lines = pf.read_text(encoding="utf-8").splitlines()
    obj = json.loads(lines[0])
    obj["payload"] = "TAMPERED"
    lines[0] = json.dumps(obj, ensure_ascii=False)
    pf.write_text("\n".join(lines) + "\n", encoding="utf-8")

    findings = store.verify_chain()
    assert findings[0]["ok"] is False  # entry_sha256 exposes the manipulation
    assert any("entry_sha256" in p for p in findings[0]["problems"])


def test_verify_flags_missing_raw_file(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    store.append_provenance({
        "fetch_id": "A",
        "raw_file": "cache/raw/does_not_exist.json",
        "sha256_raw": "deadbeef",
    })
    findings = store.verify_chain()
    assert findings[0]["ok"] is False
    # matches the literal problem string emitted by store.verify_chain
    # ("raw file missing"; older builds emitted the German "Rohdatei fehlt")
    assert any(("raw file missing" in p) or ("Rohdatei fehlt" in p)
               for p in findings[0]["problems"])
