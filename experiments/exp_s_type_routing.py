"""Type derivation for the two-branch design: does the ip-api ``mobile`` flag carry?

Starting question (transform design): one estimator+calibration per known access
type and one for unknown -- can the type be derived from the APIs at all?
The operational view (exp_s_pooled_categories) showed that class knowledge only
pays off for cellular; this script measures whether a LIVE-available signal
reliably routes into that one branch.

Prior finding from the session diagnosis of 2026-08-01 (then only /tmp,
ephemeral -- this script casts it reproducibly):
  - keywords in org/ISP/whois strings: unusable (P 0.47 / R 0.46),
    large carriers serve fixed-line+cellular under one name.
  - ip-api ``mobile`` flag: P 0.92 / R 0.50 -- usable as a router.
  - routing value: one calibration +0.263 -> flag-based routing +0.287;
    oracle routing via RIPE tag only +0.280. The precise flag isolates the
    "clearly mobile" cases; the missed half behaves like fixed-line.
  - paired bootstrap: delta +0.024, 95% CI [+0.0005, +0.048] -- significant,
    but narrow.

Setup (identical to the operational view, only the calibration is routed):
  - model R7 (QW + in25/in50/in100 per source), trained ONLY on the anchors,
  - mixed population probes + probes_holdout + probes_mobile_ext (dedup.),
  - comparison of three calibration strategies, each Platt out-of-fold:
      (a) ONE layer without type knowledge   (operational number, reference)
      (b) TWO branches by ip-api flag        (available live!)
      (c) TWO branches by RIPE class         (oracle, unknown in operation)
  - paired bootstrap (B=10000) for delta (b)-(a).

The flags come from the ip-api batch API (fields=query,status,mobile; free tier: HTTP,
max. 100 IPs/request, 15 requests/min) and are written to
data/cache/ipapi_mobile_flags.json (like all raw IPs, the IPs stay in the
private data/cache; the eval/out artefact carries NO IP column).
CAUTION: the flag is a live signal -- ip-api may answer differently today than
on 2026-08-01; digit-exact reproduction of the prior findings is not guaranteed.

Usage:   python experiments/exp_s_type_routing.py
Output:  tables (stdout) + eval/out/s_type_routing.csv
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.exp_support_concentration import ece                    # noqa: E402
from experiments.exp_s_declared_risk import (                            # noqa: E402
    TAU, OUT, _mat, _fit_ridge, _pick_lambda, _sig,
)
from experiments.exp_s_pooled_categories import (                        # noqa: E402
    WIN, POOL, _frame, _case_ips, _brier, _platt_oof,
)

FLAGS_PATH = ROOT / "data" / "cache" / "ipapi_mobile_flags.json"
BATCH_URL = "http://ip-api.com/batch?fields=query,status,mobile"
B_BOOT = 10_000


# --------------------------------------------------------------------------- #
# ip-api mobile flag (batch API, cached)
# --------------------------------------------------------------------------- #
def fetch_flags(ips: list[str]) -> dict[str, bool]:
    cache: dict[str, bool] = (json.loads(FLAGS_PATH.read_text())
                              if FLAGS_PATH.exists() else {})
    missing = sorted({ip for ip in ips if ip not in cache})
    if missing:
        print(f"ip-api batch: {len(missing)} flags missing from cache, fetching live ...")
    for i in range(0, len(missing), 100):
        chunk = missing[i:i + 100]
        req = urllib.request.Request(
            BATCH_URL, data=json.dumps(chunk).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                for rec in json.loads(resp.read()):
                    if rec.get("status") == "success":
                        cache[rec["query"]] = bool(rec.get("mobile", False))
        except OSError as exc:
            print(f"  batch {i // 100 + 1}: {exc} -- missing ones count as unflagged")
        FLAGS_PATH.write_text(json.dumps(cache, sort_keys=True))
        if i + 100 < len(missing):
            time.sleep(4.5)                       # free tier: 15 batch requests/min
    return cache


# --------------------------------------------------------------------------- #
def _bss(ph: np.ndarray, y: np.ndarray) -> float:
    return 1 - _brier(ph, y) / float(np.mean((y.mean() - y) ** 2))


def _routed_platt(z: np.ndarray, y: np.ndarray, branch: np.ndarray) -> np.ndarray:
    """One separate Platt layer per branch, each out-of-fold within the branch."""
    ph = np.empty(len(y))
    for b in (True, False):
        m = branch == b
        ph[m] = _platt_oof(z[m], y[m])
    return ph


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    A = _frame("anchors")
    yA = (A.err > TAU).astype(int).values

    # Pool as in the operational view -- incl. deduplication over the IPs; the
    # IP list is additionally kept in memory here to attach the flags.
    parts, ip_parts, seen = [], [], set()
    for ds in POOL:
        T = _frame(ds).assign(origin=ds)
        ips = _case_ips(ds)
        if len(ips) != len(T):
            raise RuntimeError(f"{ds}: {len(ips)} cases, but {len(T)} frame rows")
        keep = [ip not in seen for ip in ips]
        seen.update(ips)
        parts.append(T[pd.Series(keep, index=T.index)].reset_index(drop=True))
        ip_parts += [ip for ip, k in zip(ips, keep) if k]
    P = pd.concat(parts, ignore_index=True)
    y = (P.err > TAU).astype(int).values

    flags_by_ip = fetch_flags(ip_parts)
    flag = np.array([flags_by_ip.get(ip, False) for ip in ip_parts])
    n_missing = sum(ip not in flags_by_ip for ip in ip_parts)
    if n_missing:
        print(f"({n_missing} IP(s) without ip-api answer -> unflagged = default branch)")

    # --- flag quality against the RIPE access-technology tags (cat from build_frame)
    truth = (P.cat == "cellular").values
    tp = int((flag & truth).sum())
    prec = tp / flag.sum() if flag.sum() else float("nan")
    rec = tp / truth.sum() if truth.sum() else float("nan")
    print("=" * 84)
    print("FLAG QUALITY: ip-api ``mobile`` vs RIPE access technology (cellular)")
    print("=" * 84)
    print(f"n={len(P)}  flagged={int(flag.sum())}  cellular={int(truth.sum())}  "
          f"Precision {prec:.2f}  Recall {rec:.2f}")
    rows.append({"section": "flag_quality", "variant": "ip_api_mobile",
                 "n": len(P), "metric": "precision_recall",
                 "value": round(prec, 3), "extra": f"recall={rec:.3f} tp={tp} "
                 f"flagged={int(flag.sum())} cellular={int(truth.sum())}"})

    # --- model: trained on anchors only (identical to the operational view)
    Xa, Xp = _mat(A, WIN), _mat(P, WIN)
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-9
    lam = _pick_lambda(Xa, yA, 0)
    w = _fit_ridge(np.column_stack([np.ones(len(A)), (Xa - mu) / sd]), yA, lam)
    z = np.column_stack([np.ones(len(P)), (Xp - mu) / sd]) @ w

    ph_one = _platt_oof(z, y)                       # (a) one calibration
    ph_flag = _routed_platt(z, y, flag)             # (b) routing via ip-api flag
    ph_orac = _routed_platt(z, y, truth)            # (c) oracle via RIPE tag

    print("\n" + "=" * 84)
    print("ROUTING VALUE (R7 anchor-trained, mixed pool, Platt per strategy)")
    print("=" * 84)
    for name, ph, note in (
            ("one calibration    ", ph_one, "operational number without type knowledge"),
            ("ip-api flag routing", ph_flag,
             f"base rate flagged {y[flag].mean():.3f} vs {y[~flag].mean():.3f}"),
            ("oracle (RIPE tag)  ", ph_orac, "NOT available in operation")):
        print(f"  {name}  BSS {_bss(ph, y):+.3f}  ECE {ece(ph, y):.3f}   ({note})")
        rows.append({"section": "routing", "variant": name.strip(), "n": len(P),
                     "metric": "bss_platt", "value": round(_bss(ph, y), 3),
                     "extra": f"ECE={ece(ph, y):.3f} {note}"})

    # --- paired bootstrap: delta flag routing minus one calibration
    rng = np.random.default_rng(0)
    deltas = np.empty(B_BOOT)
    for b in range(B_BOOT):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        ref = float(np.mean((yb.mean() - yb) ** 2))
        deltas[b] = (_brier(ph_one[idx], yb) - _brier(ph_flag[idx], yb)) / ref
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    frac = float((deltas > 0).mean())
    print("\npaired bootstrap (B={:,}): delta BSS {:+.3f}, 95% CI [{:+.4f}, {:+.4f}], "
          "{:.1%} of draws > 0".format(B_BOOT, float(deltas.mean()), lo, hi, frac))
    rows.append({"section": "bootstrap", "variant": "flag_vs_one", "n": len(P),
                 "metric": "delta_bss", "value": round(float(deltas.mean()), 4),
                 "extra": f"ci=[{lo:+.4f},{hi:+.4f}] frac_pos={frac:.3f} B={B_BOOT}"})

    print("\nREADING:")
    print("  The flag is precise but incomplete (recall ~0.5). For routing that is")
    print("  fine: the missed half behaves like fixed line and is well served by the")
    print("  default branch. The design answer stands: TWO branches (default +")
    print("  cellular via ip-api flag), not one estimator per type. For the paper,")
    print("  confirm the delta on a fresh holdout -- the CI is narrow.")

    pd.DataFrame(rows).to_csv(OUT / "s_type_routing.csv", index=False)
    print(f"\nCSV: {OUT}/s_type_routing.csv")


if __name__ == "__main__":
    main()
