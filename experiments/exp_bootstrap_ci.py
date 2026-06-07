"""T1 — Paired-Bootstrap-Konfidenzintervalle für die Aggregation L1·b.

Fuer jede Einzelquelle (und die
ungewichtete Baseline L0·a) wird die anchor-weise Median-Differenz

    Δ = median(d_{L1·b}) − median(d_{Quelle})

als Punktschätzer berichtet und über einen *gepaarten* Bootstrap (dieselben
neu gezogenen Anchor-Indizes für Zähler und Nenner) mit B = 10.000 Wiederholungen
und festem seed = 0 ein 95%-Perzentil-Konfidenzintervall bestimmt. Negative Δ
bedeuten, dass die Aggregation genauer ist.

Die Paarung erfolgt je Quelle über die Schnittmenge der Anchors, auf denen sowohl
L1·b als auch die Quelle einen Fehlerwert besitzen (relevant nur für MaxMind/geojs:
1.076 statt 1.077).

Aufruf:  python experiments/exp_bootstrap_ci.py
Ergebnis: Tabelle (stdout) + eval/out/t1_bootstrap_ci.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.pipeline import load_cases                       # noqa: E402
from eval.metrics import haversine_error                   # noqa: E402
from experiments.exp_t6_defaults import (                  # noqa: E402
    loo_pseudo_radii, estimate, EPS_GRID,
)

B = 10_000
SEED = 0
OUT = Path("eval/out"); OUT.mkdir(parents=True, exist_ok=True)

# Reihenfolge & Anzeigenamen der Einzelquellen
SOURCE_LABELS = [
    ("ip2location_lite", "IP2Location LITE"),
    ("ipinfo",           "ipinfo.io"),
    ("ipwho_is",         "ipwho.is"),
    ("ip_api",           "ip-api.com"),
    ("dbip_lite",        "DB-IP Lite"),
    ("maxmind_geolite2", "MaxMind GeoLite2 / geojs"),
    ("reallyfreegeoip",  "reallyfreegeoip.org"),
]


def paired_bootstrap(d_agg, d_ref, b=B, seed=SEED):
    """Δ = median(d_agg) − median(d_ref) plus gepaartes 95%-Perzentil-CI.

    d_agg, d_ref sind über denselben Anchor-Index ausgerichtet (gleiche Länge).
    """
    d_agg = np.asarray(d_agg, dtype=float)
    d_ref = np.asarray(d_ref, dtype=float)
    n = len(d_agg)
    delta = float(np.median(d_agg) - np.median(d_ref))
    rng = np.random.default_rng(seed)
    boot = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, n)            # gepaart: gleiche Indizes beidseitig
        boot[i] = np.median(d_agg[idx]) - np.median(d_ref[idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return delta, float(lo), float(hi)


def paired_bootstrap_stat(d_agg, d_ref, stat, b=B, seed=SEED):
    """Δ = stat(d_agg) − stat(d_ref) plus gepaartes 95%-Perzentil-CI fuer eine
    beliebige Statistik (z. B. Mittelwert oder Tail-Rate). Negative Δ = Aggregation besser.
    """
    d_agg = np.asarray(d_agg, dtype=float)
    d_ref = np.asarray(d_ref, dtype=float)
    n = len(d_agg)
    delta = float(stat(d_agg) - stat(d_ref))
    rng = np.random.default_rng(seed)
    boot = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, n)            # gepaart: gleiche Indizes beidseitig
        boot[i] = stat(d_agg[idx]) - stat(d_ref[idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return delta, float(lo), float(hi)


def main():
    cases = load_cases()
    loo = loo_pseudo_radii(cases)
    globals_ = [loo[s]["_global"] for s in loo]
    anchor_eps = min(EPS_GRID, key=lambda e: abs(e - float(np.median(globals_))))

    # Anchor-weise Fehler: Aggregation L1·b, ungewichtete Baseline L0·a, je Einzelquelle.
    agg_l1b, base_l0a = {}, {}
    src_err = {s: {} for s, _ in SOURCE_LABELS}
    for c in cases:
        ip = c["ip"]
        agg_l1b[ip] = haversine_error(estimate(c, "L1", "b", loo, eps=anchor_eps), c["truth"])
        base_l0a[ip] = haversine_error(estimate(c, "L0", "a", loo, eps=anchor_eps), c["truth"])
        for p in c["provenance"]:
            if p["source"] in src_err:
                src_err[p["source"]][ip] = haversine_error((p["lat"], p["lon"]), c["truth"])

    rows = []
    print("=" * 78)
    print(f"T1 — Paired-Bootstrap-CIs (B={B:,}, seed={SEED}) | Δ = median(L1·b) − median(Quelle)")
    print("=" * 78)
    print(f"{'Quelle/Referenz':28s} {'Med Quelle':>10s} {'Δ [km]':>8s}   95%-CI")

    def emit(label, ref_map):
        ips = sorted(set(agg_l1b) & set(ref_map))
        d_agg = [agg_l1b[ip] for ip in ips]
        d_ref = [ref_map[ip] for ip in ips]
        med_ref = float(np.median(d_ref))
        delta, lo, hi = paired_bootstrap(d_agg, d_ref)
        sig = "" if lo <= 0 <= hi else "  *"      # * = CI schließt 0 nicht ein
        print(f"{label:28s} {med_ref:10.2f} {delta:+8.2f}   [{lo:+.2f}, {hi:+.2f}]{sig} (n={len(ips)})")
        rows.append({"reference": label, "median_source_km": round(med_ref, 2),
                     "delta_km": round(delta, 2), "ci_lo_km": round(lo, 2),
                     "ci_hi_km": round(hi, 2), "n": len(ips)})

    for key, label in SOURCE_LABELS:
        emit(label, src_err[key])
    print("-" * 78)
    emit("naiv ungewichtet L0·a (Baseline)", base_l0a)

    print(f"\n  median(L1·b) gesamt = {np.median(list(agg_l1b.values())):.2f} km (n={len(agg_l1b)})")
    pd.DataFrame(rows).to_csv(OUT / "t1_bootstrap_ci.csv", index=False)
    print(f"  CSV: {OUT}/t1_bootstrap_ci.csv")

    # --- Mittel- und Tail-Raten-Differenzen gegen die im Mittel/Tail fuehrenden Quellen ---
    # (FF1: die Aggregation verliert dort ex post; hier inferenzstatistisch abgesichert)
    print("\n" + "=" * 78)
    print(f"T1 — Mittel- & Tail-Raten-Differenzen (L1·b − Quelle), gepaart, B={B:,}")
    print("=" * 78)
    print(f"{'Quelle':18s} {'Δ Mittel [km]':>22s}   {'Δ Tail-Rate [pp]':>22s}")
    mean_stat = lambda d: float(np.mean(d))
    tail_stat = lambda d: float(100.0 * np.mean(np.asarray(d) > 100.0))
    mt_rows = []
    for key, label in [("ip2location_lite", "IP2Location LITE"), ("ipinfo", "ipinfo.io")]:
        ips = sorted(set(agg_l1b) & set(src_err[key]))
        da = np.array([agg_l1b[ip] for ip in ips])
        dr = np.array([src_err[key][ip] for ip in ips])
        dm, lom, him = paired_bootstrap_stat(da, dr, mean_stat)
        dt, lot, hit = paired_bootstrap_stat(da, dr, tail_stat)
        sm = "" if lom <= 0 <= him else "  *"
        st = "" if lot <= 0 <= hit else "  *"
        print(f"{label:18s} {dm:+8.1f} [{lom:+7.1f},{him:+7.1f}]{sm}   "
              f"{dt:+6.1f} [{lot:+6.1f},{hit:+6.1f}]{st}")
        mt_rows.append({"reference": label, "delta_mean_km": round(dm, 1),
                        "mean_ci_lo": round(lom, 1), "mean_ci_hi": round(him, 1),
                        "delta_tail_pp": round(dt, 1), "tail_ci_lo": round(lot, 1),
                        "tail_ci_hi": round(hit, 1), "n": len(ips)})
    pd.DataFrame(mt_rows).to_csv(OUT / "t1_bootstrap_meantail_ci.csv", index=False)
    print("  (+ = Aggregation schlechter als Quelle; * = CI schließt 0 nicht ein)")
    print(f"  CSV: {OUT}/t1_bootstrap_meantail_ci.csv")


if __name__ == "__main__":
    main()
