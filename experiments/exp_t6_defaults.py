"""T6 — Robuste Aggregation unter strukturell korrelierten Hub-Defaults.

Zwei getrennte Beiträge, bewusst NICHT vermischt:

 (I) Punktschätzer-Vergleich: Linien-Skala × Aggregator-Variante × Schwierigkeits-Eimer
       Linien-Skala   L0 keine Linien | L1 Anbieter-Linien (geojs+rfg+GeoLite2=1)
                      | L2 + Methodik-Linie (DB-IP+IP2Location = WHOIS, ½ je Quelle)
       Variante       (a) naiv | (b) konfidenzgewichtet 1/(r+ε) | (c) centroid-gefiltert
                      | (d) b+c   — alle als gewichteter geometrischer Median (RFA)
       Eimer          easy (alle Quellen <50 km) | uneinig | hart (beste Quelle >50 km)
       Konfidenz r:   MaxMind = echter accuracy_radius; übrige Quellen = LEAVE-ONE-OUT
                      Median-Genauigkeit (Pseudo-Radius ohne die bewertete IP → kein
                      In-Sample-Snooping).
 (II) 2D-Konfidenz-Matrix (PARALLEL, kein Punktschätzer): Streuung × Hub-Flag.
       Dies ist das FRÜHERE Label (mediane Paardistanz × Hub). Das Headline-Konfidenz-
       maß dieses Projekts ist die linien-gewichtete Stütz-Konzentration S; ihre Herleitung,
       der out-of-fold-Vergleich gegen dieses Label und die Kalibrierung stehen in
       experiments/exp_support_concentration.py.

Aufruf:  python experiments/exp_t6_defaults.py
Ergebnis: Tabellen (stdout) + eval/out/t6_*.csv
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.pipeline import load_cases          # noqa: E402
from eval.metrics import haversine_error       # noqa: E402
from estimators.baselines import weighted_geometric_median  # noqa: E402

EPS_KM = 56.0                 # Fallback-Default für ε (≈ Q3 der „echte Stadt"-Fehler); die Headline
                              # nutzt jedoch den LOO-abgeleiteten anchor_eps (s. main()), nicht diesen Wert.
EPS_GRID = [10, 30, 50, 100]  # Sensitivität (Anhang)
LEVELS = ["L0", "L1", "L2"]
VARIANTS = ["a", "b", "c", "d"]
WHOIS_LINEAGES = ("dbip_lite", "ip2location_lite")
OUT = Path("eval/out"); OUT.mkdir(parents=True, exist_ok=True)


# ---------- Leave-one-out Pseudo-Radien (Track-Record je Quelle, ohne die IP) ----------

def loo_pseudo_radii(cases):
    src_err = defaultdict(dict)
    for c in cases:
        for p in c["provenance"]:
            src_err[p["source"]][c["ip"]] = haversine_error((p["lat"], p["lon"]), c["truth"])
    loo = {}
    for s, d in src_err.items():
        ips = list(d)
        errs = np.array([d[ip] for ip in ips], dtype=float)
        per_ip = {ip: float(np.median(np.delete(errs, i))) for i, ip in enumerate(ips)}
        per_ip["_global"] = float(np.median(errs))
        loo[s] = per_ip
    return loo


def point_radius(p, ip, loo):
    """MaxMind: echter accuracy_radius; sonst Leave-one-out-Pseudo-Radius der Quelle."""
    if p["source"] == "maxmind_geolite2" and p["accuracy_radius"] is not None:
        return p["accuracy_radius"]
    tab = loo.get(p["source"], {})
    return tab.get(ip, tab.get("_global", EPS_KM))


# ---------- Linien-Gewichte je Skala ----------

def line_weights_for(prov, level):
    if level == "L0":
        return np.ones(len(prov))
    lineages = [p["lineage"] for p in prov]
    if level == "L2":
        lineages = ["whois_family" if l in WHOIS_LINEAGES else l for l in lineages]
    elif level == "L2_cond":
        # BEDINGTE WHOIS-Halbierung (eingefroren, vor der Messung definiert): DB-IP und
        # IP2Location bilden NUR dann eine gemeinsame Linie, wenn auf DIESER IP BEIDE das
        # is_default_centroid-Flag tragen (gemeinsames Versagen); sonst zählen sie separat.
        d = next((p for p in prov if p["lineage"] == "dbip_lite"), None)
        i = next((p for p in prov if p["lineage"] == "ip2location_lite"), None)
        if d and i and d["is_default_centroid"] and i["is_default_centroid"]:
            lineages = ["whois_family" if l in WHOIS_LINEAGES else l for l in lineages]
    counts = Counter(lineages)
    return np.array([1.0 / counts[l] for l in lineages], dtype=float)


# ---------- Aggregator-Varianten ----------

def estimate(case, level, variant, loo, eps=EPS_KM, form="lin"):
    prov = case["provenance"]
    if variant in ("c", "d"):                       # Centroid-Filter
        kept = [p for p in prov if not p["is_default_centroid"]]
        prov = kept if kept else prov               # nie auf 0 Punkte filtern
    pts = np.array([[p["lat"], p["lon"]] for p in prov], dtype=float)
    w = line_weights_for(prov, level)
    if variant in ("b", "d"):                       # Konfidenz-Gewicht
        rad = np.array([point_radius(p, case["ip"], loo) for p in prov], dtype=float)
        w = w / (rad ** 2 + eps ** 2) if form == "quad" else w / (rad + eps)
    est = weighted_geometric_median(pts, w)
    return (float(est[0]), float(est[1]))


def bucket(case):
    errs = [haversine_error((p["lat"], p["lon"]), case["truth"]) for p in case["provenance"]]
    if min(errs) > 50:
        return "hart"
    if max(errs) < 50:
        return "easy"
    return "uneinig"


def _q(errs):
    a = np.array(errs, dtype=float)
    return np.median(a), np.percentile(a, 25), np.percentile(a, 75)


# ---------- (I) Punktschätzer-Vergleich ----------

def run_point_estimators(cases, loo, eps=EPS_KM):
    rows = []
    for c in cases:
        bkt = bucket(c)
        for level in LEVELS:
            for v in VARIANTS:
                err = haversine_error(estimate(c, level, v, loo, eps=eps), c["truth"])
                rows.append({"ip": c["ip"], "level": level, "variant": v,
                             "bucket": bkt, "error_km": err})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t6_point_estimators.csv", index=False)

    print("=" * 74)
    print("(I) PUNKTSCHÄTZER — Median-Fehler km [Q1–Q3], je Linien-Skala × Variante")
    print("=" * 74)
    print(f"{'':4s} " + "  ".join(f"{v:>14s}" for v in VARIANTS))
    for level in LEVELS:
        cells = []
        for v in VARIANTS:
            m, q1, q3 = _q(df[(df.level == level) & (df.variant == v)].error_km)
            cells.append(f"{m:5.0f}[{q1:3.0f}-{q3:4.0f}]")
        print(f"{level:4s} " + "  ".join(f"{c:>14s}" for c in cells))

    print("\nMittelwert-Fehler km (TAIL-sensitiv — hier zeigt sich der Default-Effekt):")
    for level in LEVELS:
        cells = [f"{df[(df.level==level)&(df.variant==v)].error_km.mean():6.0f}" for v in VARIANTS]
        print(f"{level:4s} " + "  ".join(f"{v}={c}" for v, c in zip(VARIANTS, cells)))
    print("Anteil Fehler > 100 km (%) (badly-wrong-Rate):")
    for level in LEVELS:
        cells = [f"{100*(df[(df.level==level)&(df.variant==v)].error_km>100).mean():5.1f}" for v in VARIANTS]
        print(f"{level:4s} " + "  ".join(f"{v}={c}" for v, c in zip(VARIANTS, cells)))

    print("\nPer Eimer (Median-Fehler km), Skala L2:")
    for bkt in ["easy", "uneinig", "hart"]:
        n = df[(df.level == "L2") & (df.variant == "a") & (df.bucket == bkt)].shape[0]
        cells = [f"{np.median(df[(df.level=='L2')&(df.variant==v)&(df.bucket==bkt)].error_km):5.0f}"
                 for v in VARIANTS]
        print(f"  {bkt:8s} (n={n:4d}):  " + "  ".join(f"{v}={c}" for v, c in zip(VARIANTS, cells)))

    print("\nBeitrag Linien-Skala (Variante b) — Median | Mittelwert | %>100km:")
    for level in LEVELS:
        sub = df[(df.level == level) & (df.variant == "b")].error_km
        print(f"  {level}: median={np.median(sub):4.1f}  mean={sub.mean():6.1f}  >100km={100*(sub>100).mean():4.1f}%")
    print("  → L0→L1→L2: Median tail-blind; Mittelwert/Tail zeigen den Bündelungs-Effekt")
    return df


# ---------- ε-Sensitivität (Anhang) ----------

def run_eps_grid(cases, loo, anchor_eps):
    """Eingefrorenes Grid ε×Form (Variante L1+b), alle 8 berichtet. Headline =
    linear, ε=anchor_eps (Median der LOO-Pseudo-Radien, GT-aggregations-unabhängig)."""
    print("\n" + "=" * 86)
    print(f"ε×FORM-GRID (L1+b) — eingefroren; Headline=linear ε={anchor_eps} (LOO-Radien-Median). "
          "Referenz ipinfo-Mittel=86.6")
    print("  je Zelle: Mittel | Median | %>100km | (easy/uneinig/hart Median)")
    print("=" * 86)
    grid_rows = []
    for form in ("lin", "quad"):
        for eps in EPS_GRID:
            rows = [(haversine_error(estimate(c, "L1", "b", loo, eps=eps, form=form), c["truth"]),
                     bucket(c)) for c in cases]
            e = np.array([r[0] for r in rows])
            by = {b: np.median([r[0] for r in rows if r[1] == b]) for b in ("easy", "uneinig", "hart")}
            hl = "  ◀ HEADLINE" if (form == "lin" and eps == anchor_eps) else ""
            print(f"  {form:4s} ε={eps:4d}: {e.mean():6.1f} | {np.median(e):4.1f} | "
                  f"{100*(e>100).mean():4.1f}% | ({by['easy']:.0f}/{by['uneinig']:.0f}/{by['hart']:.0f}){hl}")
            grid_rows.append({"form": form, "eps_km": eps, "mean_km": round(float(e.mean()), 1),
                              "median_km": round(float(np.median(e)), 1),
                              "tail_rate_pct": round(100 * float((e > 100).mean()), 1),
                              "hart_median_km": round(float(by["hart"]), 0),
                              "headline": form == "lin" and eps == anchor_eps})
    pd.DataFrame(grid_rows).to_csv(OUT / "t6_epsilon_grid.csv", index=False)
    print(f"  CSV: {OUT}/t6_epsilon_grid.csv")


def run_confidence_recall(cases, loo, anchor_eps):
    """Forensischer Verteidigungstest: Wie viele Aggregations-Misses (>100km) markiert
    die Konfidenz-Matrix als unsicher (hohe Streuung ODER Hub)?"""
    miss = flagged_miss = flagged_all = total = 0
    for c in cases:
        err = haversine_error(estimate(c, "L1", "b", loo, eps=anchor_eps), c["truth"])
        pts = c["points"]; n = len(pts)
        pd_ = [haversine_error(tuple(pts[i]), tuple(pts[j])) for i in range(n) for j in range(i + 1, n)]
        spread = float(np.median(pd_)) if pd_ else 0.0
        hub = np.mean([p["is_default_centroid"] for p in c["provenance"]]) >= 0.5
        uncertain = (spread >= 50) or hub
        total += 1
        flagged_all += uncertain
        if err > 100:
            miss += 1
            flagged_miss += uncertain
    print("\n" + "=" * 74)
    print("KONFIDENZ-RECALL auf Aggregations-Misses (L1+b, Fehler >100 km)")
    print("=" * 74)
    print(f"  Misses: {miss}/{total}  |  davon als unsicher geflaggt: {flagged_miss} "
          f"({100*flagged_miss/miss:.0f}% Recall)")
    print(f"  Geflaggt gesamt: {flagged_all}/{total} ({100*flagged_all/total:.0f}%)  "
          f"→ Präzision der Flags auf Misses: {100*flagged_miss/flagged_all:.0f}%")
    print("  → hoher Recall = die Fälle, in denen eine Einzelquelle gewänne, sind als unsicher kenntlich")


# ---------- (II) 2D-Konfidenz-Matrix (parallel) ----------

def run_confidence_matrix(cases, loo, eps):
    rows = []
    for c in cases:
        pts = c["points"]
        n = len(pts)
        pdists = [haversine_error(tuple(pts[i]), tuple(pts[j]))
                  for i in range(n) for j in range(i + 1, n)]
        spread = float(np.median(pdists)) if pdists else 0.0    # frueheres Label; Headline-Mass = S, s. exp_support_concentration.py
        hub_frac = np.mean([p["is_default_centroid"] for p in c["provenance"]])
        # Fehler des EINGESETZTEN Schätzers (L1+b) — konsistent mit Heatmap & Recall
        est_err = haversine_error(estimate(c, "L1", "b", loo, eps=eps), c["truth"])
        rows.append({"ip": c["ip"], "spread_km": spread, "hub": hub_frac >= 0.5,
                     "low_spread": spread < 50, "est_error_km": est_err})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t6_confidence_matrix.csv", index=False)

    print("\n" + "=" * 74)
    print("(II) 2D-KONFIDENZ-MATRIX (parallel) — Streuung × Hub-Flag")
    print("     je Feld: n | Median | Mittelwert | %>100km  (Tail entlarvt False-Consensus)")
    print("=" * 74)
    labels = {(True, False): "niedrig+kein_hub  → belastbar",
              (True, True): "niedrig+hub       → ROT (Sicherheits-Illusion)",
              (False, False): "hoch+kein_hub     → ehrliche Unsicherheit",
              (False, True): "hoch+hub          → Inkonsistenz"}
    for low in (True, False):
        for hub in (False, True):
            sub = df[(df.low_spread == low) & (df.hub == hub)].est_error_km
            med = np.median(sub) if len(sub) else float("nan")
            mean = sub.mean() if len(sub) else float("nan")
            tail = 100 * (sub > 100).mean() if len(sub) else float("nan")
            print(f"  {labels[(low,hub)]:46s} n={len(sub):4d} | {med:5.0f} | {mean:6.0f} | {tail:4.1f}%")
    illusion = df[(df.low_spread) & (df.hub)].est_error_km
    lo, hi = _bootstrap_tail_ci(illusion)
    print(f"\n  Bootstrap-95%-CI auf %>100km der „niedrig+hub\"-Zelle (n={len(illusion)}): "
          f"[{lo:.1f}, {hi:.1f}] %  (indikativ — kleine Zelle)")
    return df


def run_ff1_baseline(cases, loo, eps=EPS_KM):
    """FF1 in einer Zahl: schlägt die robuste Aggregation (L1+b) jede Einzelquelle?"""
    from collections import defaultdict
    src = defaultdict(list)
    for c in cases:
        for p in c["provenance"]:
            src[p["source"]].append(haversine_error((p["lat"], p["lon"]), c["truth"]))
    agg = [haversine_error(estimate(c, "L1", "b", loo, eps=eps), c["truth"]) for c in cases]
    print("\n" + "=" * 74)
    print("FF1 — robuste Aggregation (L1+b) vs. beste Einzelquelle  (Mittel | %>100km)")
    print("=" * 74)
    rows = [("AGG L1+b", np.mean(agg), 100 * np.mean(np.array(agg) > 100), len(agg))]
    for s, e in src.items():
        e = np.array(e)
        rows.append((s, e.mean(), 100 * (e > 100).mean(), len(e)))
    rows.sort(key=lambda r: r[1])
    for name, m, tail, n in rows:
        mark = "  <== Aggregation" if name == "AGG L1+b" else ""
        print(f"  {name:18s} n={n:4d}  mean={m:6.1f} km  >100km={tail:4.1f}%{mark}")
    best = min((r for r in rows if r[0] != "AGG L1+b"), key=lambda r: r[1])
    print(f"  → Aggregation {np.mean(agg):.1f} km vs. beste Einzelquelle {best[0]} {best[1]:.1f} km")


def run_sensitivity_lines(cases, loo, eps=EPS_KM):
    print("\n" + "=" * 74)
    print("Sensitivity — Linien-Definition (Variante b)  Mittel | %>100km")
    print("=" * 74)
    for level in ["L1", "L2", "L2_cond"]:
        e = np.array([haversine_error(estimate(c, level, "b", loo, eps=eps), c["truth"]) for c in cases])
        note = "  (bedingt: WHOIS-Linie nur wenn beide default)" if level == "L2_cond" else ""
        print(f"  {level:8s} mean={e.mean():6.1f}  >100km={100*(e>100).mean():4.1f}%{note}")


def _bootstrap_tail_ci(errs, thresh=100, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(errs, dtype=float)
    rates = [100 * (rng.choice(a, size=len(a), replace=True) > thresh).mean() for _ in range(n_boot)]
    return np.percentile(rates, [2.5, 97.5])


if __name__ == "__main__":
    cases = load_cases()
    print(f"Fälle: {len(cases)}  (Quellen je Fall im Schnitt "
          f"{np.mean([len(c['points']) for c in cases]):.1f})")
    loo = loo_pseudo_radii(cases)
    # GT-aggregations-unabhängiger ε-Anker (VOR dem Lauf festgelegt): nächster Grid-Wert
    # zum Median der LOO-Pseudo-Radien der Quellen.
    globals_ = [loo[s]["_global"] for s in loo]
    med_radius = float(np.median(globals_))
    anchor_eps = min(EPS_GRID, key=lambda e: abs(e - med_radius))
    print(f"Median LOO-Pseudo-Radius der Quellen = {med_radius:.1f} km → Headline-ε = {anchor_eps} km")

    run_point_estimators(cases, loo, anchor_eps)
    run_sensitivity_lines(cases, loo, anchor_eps)
    run_eps_grid(cases, loo, anchor_eps)
    run_ff1_baseline(cases, loo, anchor_eps)
    run_confidence_matrix(cases, loo, anchor_eps)
    run_confidence_recall(cases, loo, anchor_eps)
    print(f"\nCSVs: {OUT}/t6_point_estimators.csv, {OUT}/t6_confidence_matrix.csv")
