"""T6 — Robust aggregation under structurally correlated hub defaults.

Two separate contributions, deliberately NOT mixed:

 (I) Point-estimator comparison: line scale x aggregator variant x difficulty bucket
       Line scale   L0 no lines | L1 provider lines (geojs+rfg+GeoLite2=1)
                    | L2 + methodology line (DB-IP+IP2Location = WHOIS, 1/2 per source)
       Variant      (a) naive | (b) radius-weighted 1/(r+eps) | (c) centroid-filtered
                    | (d) b+c   — all as a weighted geometric median (RFA)
       Bucket       easy (all sources <50 km) | disagree | hard (best source >50 km)
       Radius r: MaxMind = true accuracy_radius; remaining sources = LEAVE-ONE-OUT
                    median accuracy (pseudo-radius without the evaluated IP -> no
                    in-sample snooping).
 (II) 2D predecessor-label matrix (PARALLEL, not a point estimator): spread x hub flag.
       This is the FORMER label (median pairwise distance x hub). The headline
       risk signal is the line-weighted support concentration S;
       its derivation, the out-of-fold comparison against this label, and the
       calibration are in experiments/exp_support_concentration.py.

Usage:   python experiments/exp_t6_defaults.py
Output:  tables (stdout) + eval/out/t6_*.csv
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

EPS_KM = 56.0                 # fallback default for eps (~ Q3 of the "true city" errors); the
                              # headline uses the LOO-derived anchor_eps (see main()), not this value.
EPS_GRID = [10, 30, 50, 100]  # sensitivity (appendix)
LEVELS = ["L0", "L1", "L2"]
VARIANTS = ["a", "b", "c", "d"]
WHOIS_LINEAGES = ("dbip_lite", "ip2location_lite")
OUT = Path("eval/out"); OUT.mkdir(parents=True, exist_ok=True)


# ---------- Leave-one-out pseudo-radii (track record per source, without the IP) ----------

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
    """MaxMind: true accuracy_radius; otherwise the source's leave-one-out pseudo-radius."""
    if p["source"] == "maxmind_geolite2" and p["accuracy_radius"] is not None:
        return p["accuracy_radius"]
    tab = loo.get(p["source"], {})
    return tab.get(ip, tab.get("_global", EPS_KM))


# ---------- Line weights per scale ----------

def line_weights_for(prov, level):
    if level == "L0":
        return np.ones(len(prov))
    lineages = [p["lineage"] for p in prov]
    if level == "L2":
        lineages = ["whois_family" if l in WHOIS_LINEAGES else l for l in lineages]
    elif level == "L2_cond":
        # CONDITIONAL WHOIS halving (frozen, defined before the measurement): DB-IP and
        # IP2Location form a shared line ONLY if on THIS IP BOTH carry the
        # is_default_centroid flag (joint failure); otherwise they count separately.
        d = next((p for p in prov if p["lineage"] == "dbip_lite"), None)
        i = next((p for p in prov if p["lineage"] == "ip2location_lite"), None)
        if d and i and d["is_default_centroid"] and i["is_default_centroid"]:
            lineages = ["whois_family" if l in WHOIS_LINEAGES else l for l in lineages]
    counts = Counter(lineages)
    return np.array([1.0 / counts[l] for l in lineages], dtype=float)


# ---------- Aggregator variants ----------

def estimate(case, level, variant, loo, eps=EPS_KM, form="lin"):
    prov = case["provenance"]
    if variant in ("c", "d"):                       # centroid filter
        kept = [p for p in prov if not p["is_default_centroid"]]
        prov = kept if kept else prov               # never filter down to 0 points
    pts = np.array([[p["lat"], p["lon"]] for p in prov], dtype=float)
    w = line_weights_for(prov, level)
    if variant in ("b", "d"):                       # radius weight
        rad = np.array([point_radius(p, case["ip"], loo) for p in prov], dtype=float)
        w = w / (rad ** 2 + eps ** 2) if form == "quad" else w / (rad + eps)
    est = weighted_geometric_median(pts, w)
    return (float(est[0]), float(est[1]))


def bucket(case):
    errs = [haversine_error((p["lat"], p["lon"]), case["truth"]) for p in case["provenance"]]
    if min(errs) > 50:
        return "hard"
    if max(errs) < 50:
        return "easy"
    return "disagree"


def _q(errs):
    a = np.array(errs, dtype=float)
    return np.median(a), np.percentile(a, 25), np.percentile(a, 75)


# ---------- (I) Point-estimator comparison ----------

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
    print("(I) POINT ESTIMATORS — median error km [Q1–Q3], per line scale × variant")
    print("=" * 74)
    print(f"{'':4s} " + "  ".join(f"{v:>14s}" for v in VARIANTS))
    for level in LEVELS:
        cells = []
        for v in VARIANTS:
            m, q1, q3 = _q(df[(df.level == level) & (df.variant == v)].error_km)
            cells.append(f"{m:5.0f}[{q1:3.0f}-{q3:4.0f}]")
        print(f"{level:4s} " + "  ".join(f"{c:>14s}" for c in cells))

    print("\nMean error km (TAIL-sensitive — this is where the default effect shows):")
    for level in LEVELS:
        cells = [f"{df[(df.level==level)&(df.variant==v)].error_km.mean():6.0f}" for v in VARIANTS]
        print(f"{level:4s} " + "  ".join(f"{v}={c}" for v, c in zip(VARIANTS, cells)))
    print("Share of errors > 100 km (%) (badly-wrong rate):")
    for level in LEVELS:
        cells = [f"{100*(df[(df.level==level)&(df.variant==v)].error_km>100).mean():5.1f}" for v in VARIANTS]
        print(f"{level:4s} " + "  ".join(f"{v}={c}" for v, c in zip(VARIANTS, cells)))

    print("\nPer bucket (median error km), scale L2:")
    for bkt in ["easy", "disagree", "hard"]:
        n = df[(df.level == "L2") & (df.variant == "a") & (df.bucket == bkt)].shape[0]
        cells = [f"{np.median(df[(df.level=='L2')&(df.variant==v)&(df.bucket==bkt)].error_km):5.0f}"
                 for v in VARIANTS]
        print(f"  {bkt:8s} (n={n:4d}):  " + "  ".join(f"{v}={c}" for v, c in zip(VARIANTS, cells)))

    print("\nContribution of the line scale (variant b) — median | mean | %>100km:")
    for level in LEVELS:
        sub = df[(df.level == level) & (df.variant == "b")].error_km
        print(f"  {level}: median={np.median(sub):4.1f}  mean={sub.mean():6.1f}  >100km={100*(sub>100).mean():4.1f}%")
    print("  → L0→L1→L2: median is tail-blind; mean/tail show the bundling effect")
    return df


# ---------- eps sensitivity (appendix) ----------

def run_eps_grid(cases, loo, anchor_eps):
    """Frozen grid eps x form (variant L1+b), all 8 reported. Headline =
    linear, eps=anchor_eps (median of the LOO pseudo-radii, GT-aggregation-independent)."""
    print("\n" + "=" * 86)
    print(f"ε×FORM GRID (L1+b) — frozen; headline=linear ε={anchor_eps} (median of LOO radii). "
          "Reference ipinfo mean=86.6")
    print("  per cell: mean | median | %>100km | (easy/disagree/hard median)")
    print("=" * 86)
    grid_rows = []
    for form in ("lin", "quad"):
        for eps in EPS_GRID:
            rows = [(haversine_error(estimate(c, "L1", "b", loo, eps=eps, form=form), c["truth"]),
                     bucket(c)) for c in cases]
            e = np.array([r[0] for r in rows])
            by = {b: np.median([r[0] for r in rows if r[1] == b]) for b in ("easy", "disagree", "hard")}
            hl = "  ◀ HEADLINE" if (form == "lin" and eps == anchor_eps) else ""
            print(f"  {form:4s} ε={eps:4d}: {e.mean():6.1f} | {np.median(e):4.1f} | "
                  f"{100*(e>100).mean():4.1f}% | ({by['easy']:.0f}/{by['disagree']:.0f}/{by['hard']:.0f}){hl}")
            grid_rows.append({"form": form, "eps_km": eps, "mean_km": round(float(e.mean()), 1),
                              "median_km": round(float(np.median(e)), 1),
                              "tail_rate_pct": round(100 * float((e > 100).mean()), 1),
                              "hard_median_km": round(float(by["hard"]), 0),
                              "headline": form == "lin" and eps == anchor_eps})
    pd.DataFrame(grid_rows).to_csv(OUT / "t6_epsilon_grid.csv", index=False)
    print(f"  CSV: {OUT}/t6_epsilon_grid.csv")


# NOTE: artifact/function name kept for stability; 'confidence' is the legacy name of the 2D predecessor label (paper terminology: predecessor label / risk signal).
def run_confidence_recall(cases, loo, anchor_eps):
    """Forensic defence test: how many aggregation misses (>100km) does the
    predecessor-label matrix mark as uncertain (high spread OR hub)?"""
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
    print("PREDECESSOR-LABEL RECALL on aggregation misses (L1+b, error >100 km)")
    print("=" * 74)
    print(f"  Misses: {miss}/{total}  |  of which flagged as uncertain: {flagged_miss} "
          f"({100*flagged_miss/miss:.0f}% recall)")
    print(f"  Flagged overall: {flagged_all}/{total} ({100*flagged_all/total:.0f}%)  "
          f"→ precision of the flags on misses: {100*flagged_miss/flagged_all:.0f}%")
    print("  → high recall = the cases in which a single source would win are marked as uncertain")


# ---------- (II) 2D predecessor-label matrix (parallel) ----------

# NOTE: artifact/function name kept for stability; 'confidence' is the legacy name of the 2D predecessor label (paper terminology: predecessor label / risk signal).
def run_confidence_matrix(cases, loo, eps):
    rows = []
    for c in cases:
        pts = c["points"]
        n = len(pts)
        pdists = [haversine_error(tuple(pts[i]), tuple(pts[j]))
                  for i in range(n) for j in range(i + 1, n)]
        spread = float(np.median(pdists)) if pdists else 0.0    # former label; headline measure = S, see exp_support_concentration.py
        hub_frac = np.mean([p["is_default_centroid"] for p in c["provenance"]])
        # error of the DEPLOYED estimator (L1+b) — consistent with heatmap & recall
        est_err = haversine_error(estimate(c, "L1", "b", loo, eps=eps), c["truth"])
        rows.append({"ip": c["ip"], "spread_km": spread, "hub": hub_frac >= 0.5,
                     "low_spread": spread < 50, "est_error_km": est_err})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t6_confidence_matrix.csv", index=False)

    print("\n" + "=" * 74)
    print("(II) 2D PREDECESSOR-LABEL MATRIX (parallel) — spread × hub flag")
    print("     per cell: n | median | mean | %>100km  (tail exposes false consensus)")
    print("=" * 74)
    labels = {(True, False): "low+no_hub   → reliable",
              (True, True): "low+hub      → RED (illusion of certainty)",
              (False, False): "high+no_hub  → honest uncertainty",
              (False, True): "high+hub     → inconsistency"}
    for low in (True, False):
        for hub in (False, True):
            sub = df[(df.low_spread == low) & (df.hub == hub)].est_error_km
            med = np.median(sub) if len(sub) else float("nan")
            mean = sub.mean() if len(sub) else float("nan")
            tail = 100 * (sub > 100).mean() if len(sub) else float("nan")
            print(f"  {labels[(low,hub)]:46s} n={len(sub):4d} | {med:5.0f} | {mean:6.0f} | {tail:4.1f}%")
    illusion = df[(df.low_spread) & (df.hub)].est_error_km
    lo, hi = _bootstrap_tail_ci(illusion)
    print(f"\n  Bootstrap 95% CI on %>100km of the 'low+hub' cell (n={len(illusion)}): "
          f"[{lo:.1f}, {hi:.1f}] %  (indicative — small cell)")
    return df


def run_rq1_baseline(cases, loo, eps=EPS_KM):
    """RQ1 in one number: does the robust aggregation (L1+b) beat every single source?"""
    from collections import defaultdict
    src = defaultdict(list)
    for c in cases:
        for p in c["provenance"]:
            src[p["source"]].append(haversine_error((p["lat"], p["lon"]), c["truth"]))
    agg = [haversine_error(estimate(c, "L1", "b", loo, eps=eps), c["truth"]) for c in cases]
    print("\n" + "=" * 74)
    print("RQ1 — robust aggregation (L1+b) vs. best single source  (mean | %>100km)")
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
    print(f"  → aggregation {np.mean(agg):.1f} km vs. best single source {best[0]} {best[1]:.1f} km")


def run_sensitivity_lines(cases, loo, eps=EPS_KM):
    print("\n" + "=" * 74)
    print("Sensitivity — line definition (variant b)  mean | %>100km")
    print("=" * 74)
    for level in ["L1", "L2", "L2_cond"]:
        e = np.array([haversine_error(estimate(c, level, "b", loo, eps=eps), c["truth"]) for c in cases])
        note = "  (conditional: WHOIS line only if both are default)" if level == "L2_cond" else ""
        print(f"  {level:8s} mean={e.mean():6.1f}  >100km={100*(e>100).mean():4.1f}%{note}")


def _bootstrap_tail_ci(errs, thresh=100, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(errs, dtype=float)
    rates = [100 * (rng.choice(a, size=len(a), replace=True) > thresh).mean() for _ in range(n_boot)]
    return np.percentile(rates, [2.5, 97.5])


if __name__ == "__main__":
    cases = load_cases()
    print(f"Cases: {len(cases)}  (sources per case on average "
          f"{np.mean([len(c['points']) for c in cases]):.1f})")
    loo = loo_pseudo_radii(cases)
    # GT-aggregation-independent eps anchor (fixed BEFORE the run): the grid value
    # closest to the median of the sources' LOO pseudo-radii.
    globals_ = [loo[s]["_global"] for s in loo]
    med_radius = float(np.median(globals_))
    anchor_eps = min(EPS_GRID, key=lambda e: abs(e - med_radius))
    print(f"Median LOO pseudo-radius of the sources = {med_radius:.1f} km → headline ε = {anchor_eps} km")

    run_point_estimators(cases, loo, anchor_eps)
    run_sensitivity_lines(cases, loo, anchor_eps)
    run_eps_grid(cases, loo, anchor_eps)
    run_rq1_baseline(cases, loo, anchor_eps)
    run_confidence_matrix(cases, loo, anchor_eps)
    run_confidence_recall(cases, loo, anchor_eps)
    print(f"\nCSVs: {OUT}/t6_point_estimators.csv, {OUT}/t6_confidence_matrix.csv")
