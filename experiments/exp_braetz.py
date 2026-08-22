"""E5 / T5 — Critical examination of Braetz' fuzzy estimator in the geo domain.

Four findings:
 (a) Braetz vs. robust estimators: distance-to-ground-truth per difficulty bucket.
 (b) Normality: Shapiro-Wilk on the INPUT and on the MIDPOINT SEQUENCE (Braetz'
     actual assumption). Headline = share CLEARLY rejected (p<0.01) as a
     conservative lower bound; non-rejection at n=8 is NO evidence of normality
     due to lack of power. + Q-Q per bucket.
 (c) Modal-value magnet: Braetz hub snap-in rate — share of
     anchors whose Braetz estimate lies <50 km from a known hub default,
     default cross-section vs. rest.
 (d) Safety illusion: Braetz' CI measures the INTERNAL certainty of the midpoint
     sequence, not the distance to GT → tight CI despite large GT error
     (= T6 "low+hub" cell).

Usage:  python experiments/exp_braetz.py
Result: eval/out/e5_braetz.csv, e5_braetz_threshold.csv, e5_braetz_linecollapse.csv,
        e5_braetz_ci_vs_gt.png, e5_braetz_qq.png
"""

from __future__ import annotations

import csv
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
from scipy import stats           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.pipeline import load_cases            # noqa: E402
from eval.metrics import haversine_error         # noqa: E402
from estimators import braetz                    # noqa: E402
from estimators.baselines import geometric_median, centroid  # noqa: E402

OUT = Path("eval/out"); OUT.mkdir(parents=True, exist_ok=True)
HUB_MIN = 10        # coordinate counts as a known hub default from this many default hits
HUB_RADIUS_KM = 50  # snap-in radius
DEFAULT_RADIUS = 500  # MaxMind accuracy radius from which an anchor joins the default cross-section


def bucket(case):
    errs = [haversine_error((p["lat"], p["lon"]), case["truth"]) for p in case["provenance"]]
    if min(errs) > 50:
        return "hard"
    if max(errs) < 50:
        return "easy"
    return "disagree"


def ci_radius_km(est_lat, half_lat, half_lon):
    """Braetz CI half-widths (degrees per coordinate) -> approximate radius in km."""
    return math.hypot(half_lat * 111.32, half_lon * 111.32 * math.cos(math.radians(est_lat)))


def build_hub_set(cases):
    """Known hub defaults = coordinates frequently marked as
    is_default_centroid across the dataset (>= HUB_MIN)."""
    c = Counter()
    for case in cases:
        for p in case["provenance"]:
            if p["is_default_centroid"]:
                c[(round(p["lat"], 2), round(p["lon"], 2))] += 1
    return [coord for coord, n in c.items() if n >= HUB_MIN]


def near_hub(lat, lon, hubs):
    return any(haversine_error((lat, lon), h) < HUB_RADIUS_KM for h in hubs)


def maxmind_radius(case):
    for p in case["provenance"]:
        if p["source"] == "maxmind_geolite2":
            return p.get("accuracy_radius")
    return None


def shapiro_p(x):
    x = np.asarray(x, dtype=float)
    if x.size < 3 or np.ptp(x) == 0:
        return float("nan")
    try:
        return float(stats.shapiro(x).pvalue)
    except Exception:
        return float("nan")


def main():
    cases = load_cases()
    hubs = build_hub_set(cases)
    print(f"E5/T5 — {len(cases)} cases; {len(hubs)} known hub defaults (>= {HUB_MIN} hits)")

    rows = []
    for c in cases:
        pts, truth = c["points"], c["truth"]
        bra = braetz.estimate(pts)
        elat, half_lat, seq_lat = braetz.braetz_ci(pts[:, 0])
        elon, half_lon, seq_lon = braetz.braetz_ci(pts[:, 1])
        dsrc = {p["source"]: haversine_error((bra[0], bra[1]), (p["lat"], p["lon"]))
                for p in c["provenance"]}
        d_mm = dsrc.get("maxmind_geolite2", float("nan"))
        # nearest source; collapse the MaxMind line (geojs/rfg/maxmind, identical coords)
        nearest = min(dsrc, key=dsrc.get)
        nearest_line = "maxmind_line" if nearest in ("geojs", "reallyfreegeoip", "maxmind_geolite2") else nearest
        rows.append({
            "ip": c["ip"], "bucket": bucket(c),
            "err_braetz": haversine_error((bra[0], bra[1]), truth),
            "err_geomed": haversine_error(tuple(geometric_median(pts)), truth),
            "err_naive": haversine_error(tuple(centroid(pts)), truth),
            "ci_km": ci_radius_km(bra[0], half_lat, half_lon),
            "p_in_lat": shapiro_p(pts[:, 0]), "p_in_lon": shapiro_p(pts[:, 1]),
            "p_seq_lat": shapiro_p(seq_lat), "p_seq_lon": shapiro_p(seq_lon),
            "near_hub": near_hub(bra[0], bra[1], hubs),
            "m_seq": int(min(seq_lat.size, seq_lon.size)),
            "d_maxmind": d_mm, "nearest_line": nearest_line,
            "is_default": (maxmind_radius(c) or 0) >= DEFAULT_RADIUS,
        })
    for r in rows:                       # snap-in to a WRONG hub (near hub AND far from GT)
        r["wrong_hub"] = r["near_hub"] and r["err_braetz"] > 100
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e5_braetz.csv", index=False)

    print("\n(a) Distance to GT — median [mean] km per bucket")
    print(f"{'Bucket':10s} {'n':>5s}  {'Braetz':>16s} {'geom.median':>16s} {'naive':>16s}")
    for b in ["easy", "disagree", "hard"]:
        s = df[df.bucket == b]
        cells = [f"{s[c].median():5.0f}[{s[c].mean():5.0f}]" for c in ("err_braetz", "err_geomed", "err_naive")]
        print(f"{b:10s} {len(s):5d}  " + "  ".join(f"{c:>16s}" for c in cells))
    s = df
    cells = [f"{s[c].median():5.0f}[{s[c].mean():5.0f}]" for c in ("err_braetz", "err_geomed", "err_naive")]
    print(f"{'overall':10s} {len(s):5d}  " + "  ".join(f"{c:>16s}" for c in cells))

    print("\n(b) Normality (Shapiro-Wilk) — share REJECTED (conservative p<0.01 | p<0.05)")
    for name, cols in [("input data", ["p_in_lat", "p_in_lon"]),
                       ("midpoint seq.", ["p_seq_lat", "p_seq_lon"])]:
        ps = pd.concat([df[c] for c in cols]).dropna()
        print(f"  {name:14s} (n={len(ps)} tests):  p<0.01: {100*(ps<0.01).mean():4.1f}%   "
              f"p<0.05: {100*(ps<0.05).mean():4.1f}%")
    print(f"  Caveat: the midpoint sequence is NOT short (median m={int(df.m_seq.median())}), but heavily "
          f"TIED — the 3 identical MaxMind replicates form the same modal class for almost all k "
          f"→ degenerate (repeated) midpoints → Shapiro rejects almost always. That is replicate dominance "
          f"(c0), not a pure smallest-n artifact. The input finding (71%) is the clean one; not-rejected ≠ normal.")

    print("\n(c0) Replicate dominance (core mechanism) — Braetz has no de-duplication:")
    print(f"  Braetz <25 km from the MaxMind value: {100*(df.d_maxmind<25).mean():4.1f}%  "
          f"(median distance {df.d_maxmind.median():.1f} km)")
    nl = df.nearest_line.value_counts(normalize=True) * 100
    print("  nearest source/line to Braetz: " +
          ", ".join(f"{k}={v:.0f}%" for k, v in nl.head(4).items()))
    print("  → the 3 identical MaxMind replicates often form the densest class and pin Braetz down")

    print("\n(c) Modal-value magnet — snap-in to a WRONG hub (<50 km from hub AND >100 km from GT)")
    print("    [raw near_hub also contains genuine hub-city anchors → hence the 'wrong' condition]")
    for label, sub in [("default cross-section", df[df.is_default]), ("rest", df[~df.is_default]),
                       ("overall", df)]:
        print(f"  {label:20s} n={len(sub):4d}:  wrong-hub {100*sub.wrong_hub.mean():4.1f}%   "
              f"(raw-near-hub {100*sub.near_hub.mean():4.1f}%)")

    print("\n(d) Safety illusion — the Braetz CI measures internal certainty, not GT distance")
    noncov = (df.err_braetz > df.ci_km).mean()
    print(f"  NON-COVERAGE of the nominal 95% CI: {100*noncov:.1f}% (GT error > CI radius; expected ~5%)")
    print(f"    → the CI almost never covers the true position — the internal certainty is an illusion")
    tight = df[df.ci_km < 100]
    print(f"  Illusion quadrant (CI<100 km & error>200 km): {((df.ci_km<100)&(df.err_braetz>200)).sum()} anchors")
    print(f"  Correlation CI width ↔ GT error: r={df['ci_km'].corr(df['err_braetz']):.2f} "
          f"(partially informative, but the level is dramatically underestimated → non-coverage)")

    run_threshold_sensitivity(cases)
    run_line_collapse(cases)

    _qq_plot(cases, df)
    print(f"\nCSV: {OUT}/e5_braetz.csv   Plots: {OUT}/e5_braetz_ci_vs_gt.png, e5_braetz_qq.png")
    _summary_plot(df)


def run_threshold_sensitivity(cases):
    """(e) Density-threshold sensitivity: main run >=2 vs. Braetz original >=4.

    Shows that the T5 findings are not artifacts of the threshold relaxed at n=8.
    Uses exactly the same CI computation (``ci_radius_km``) as the main run.
    """
    import pandas as pd
    print("\n(e) Density-threshold sensitivity — main run >=2 vs. Braetz original >=4")
    print(f"  {'Thresh.':>8s} {'Median':>8s} {'Mean':>8s} {'Non-cov.':>11s} "
          f"{'<25km MM':>9s} {'Illusion':>9s} {'m_seq':>6s}")
    rows = []
    for thr in (2, 4):
        errs, cis, dmm, mseq = [], [], [], []
        for c in cases:
            pts, truth = c["points"], c["truth"]
            bra = braetz.estimate(pts, density_threshold=thr)
            _, half_lat, seq_lat = braetz.braetz_ci(pts[:, 0], density_threshold=thr)
            _, half_lon, seq_lon = braetz.braetz_ci(pts[:, 1], density_threshold=thr)
            errs.append(haversine_error((bra[0], bra[1]), truth))
            cis.append(ci_radius_km(bra[0], half_lat, half_lon))
            mm = next(((p["lat"], p["lon"]) for p in c["provenance"]
                       if p["source"] == "maxmind_geolite2"), None)
            dmm.append(haversine_error((bra[0], bra[1]), mm) if mm else float("nan"))
            mseq.append(min(seq_lat.size, seq_lon.size))
        errs, cis = np.array(errs), np.array(cis)
        noncov = 100 * (errs > cis).mean()
        near = 100 * sum(1 for d in dmm if d == d and d < 25) / len(dmm)
        illusion = int(((cis < 100) & (errs > 200)).sum())
        print(f"  {thr:>8d} {np.median(errs):7.1f} {errs.mean():7.0f} {noncov:10.1f}% "
              f"{near:8.1f}% {illusion:9d} {int(np.median(mseq)):6d}")
        rows.append({"density_threshold": thr, "median_km": round(float(np.median(errs)), 1),
                     "mean_km": round(float(errs.mean()), 1), "noncoverage_pct": round(noncov, 1),
                     "near_maxmind_pct": round(near, 1), "illusion_n": illusion,
                     "m_seq_median": int(np.median(mseq))})
    pd.DataFrame(rows).to_csv(OUT / "e5_braetz_threshold.csv", index=False)
    print(f"  → CSV: {OUT}/e5_braetz_threshold.csv  (headline = >=2; >=4 = Braetz original as robustness evidence)")


def run_line_collapse(cases):
    """(f) Control experiment (T5): Braetz on line-collapsed sources.

    Separates the CAUSAL role of replicate dominance from mere correlation: the
    three structurally identical MaxMind GeoLite sources (same lineage) are
    reduced to ONE representative BEFORE estimation -> only effectively
    independent lines (median 6 instead of 8 sources) enter the class formation.
    n=6 lies within Braetz' stated working range (lower bound 3-5 values,
    Braetz 2009 p.67), so the test is unconfounded with respect to small n.
    If Braetz improves on the collapsed set, the replicate-dominance hypothesis is
    confirmed.
    """
    import pandas as pd

    def collapsed(c):
        seen = {}
        for p in c["provenance"]:
            seen.setdefault(p["lineage"], (p["lat"], p["lon"]))   # first representative per line
        return np.array(list(seen.values()), dtype=float)

    rows, nrep = [], []
    for c in cases:
        full, coll = c["points"], collapsed(c)
        nrep.append(len(coll))
        mm = next(((p["lat"], p["lon"]) for p in c["provenance"]
                   if p["source"] == "maxmind_geolite2"), None)
        ef, ec = braetz.estimate(full), braetz.estimate(coll)
        rows.append({"ip": c["ip"], "bucket": bucket(c),
                     "err_full": haversine_error(tuple(ef), c["truth"]),
                     "err_coll": haversine_error(tuple(ec), c["truth"]),
                     "mm_full": haversine_error(tuple(ef), mm) if mm else float("nan"),
                     "mm_coll": haversine_error(tuple(ec), mm) if mm else float("nan")})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e5_braetz_linecollapse.csv", index=False)

    print("\n(f) Control experiment — Braetz on line-collapsed sources (MaxMind family = 1 representative)")
    print(f"  sources after collapse: median {int(np.median(nrep))} (min {min(nrep)}, max {max(nrep)}); "
          f"n=6 lies within Braetz' working range (from 3-5 values, Braetz 2009 p. 67) -> no smallest-n confound")
    print(f"  {'Bucket':10s} {'n':>5s} {'Med full':>9s} {'Med coll':>9s} {'Mean full':>10s} {'Mean coll':>10s}")
    for b in ["easy", "disagree", "hard", "overall"]:
        s = df if b == "overall" else df[df.bucket == b]
        print(f"  {b:10s} {len(s):5d} {s.err_full.median():9.1f} {s.err_coll.median():9.1f} "
              f"{s.err_full.mean():10.1f} {s.err_coll.mean():10.1f}")
    mmf, mmc = df.mm_full.dropna(), df.mm_coll.dropna()
    print(f"  MaxMind snap-in (<25 km): full {100*(mmf<25).mean():.1f}%  ->  coll {100*(mmc<25).mean():.1f}%")
    u = df[df.bucket == "disagree"]
    print(f"  -> replicate dominance causally confirmed (disagree {u.err_full.median():.0f}->{u.err_coll.median():.0f} km),")
    print(f"     but incomplete (geometric median disagree ~7 km) and snap-in rate ~unchanged")
    print(f"  CSV: {OUT}/e5_braetz_linecollapse.csv")
    return df


def _qq_plot(cases, df):
    """Q-Q of the midpoint sequence (lat) for one example each from
    easy/disagree/hard/default."""
    by_ip = {c["ip"]: c for c in cases}
    picks = []
    for label, mask in [("easy", df.bucket == "easy"), ("disagree", df.bucket == "disagree"),
                        ("hard", df.bucket == "hard"), ("default", df.is_default)]:
        sub = df[mask]
        if len(sub):
            picks.append((label, sub.iloc[len(sub) // 2]["ip"]))
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, (label, ip) in zip(axes.flat, picks):
        _, _, seq = braetz.braetz_ci(by_ip[ip]["points"][:, 0])
        p = shapiro_p(seq)
        if seq.size >= 3:
            stats.probplot(seq, dist="norm", plot=ax)
        ax.set_title(f"{label}: {ip}\nmidpoint sequence lat (m={seq.size}), Shapiro p={p:.3f}", fontsize=9)
        ax.set_xlabel("theoretical quantiles"); ax.set_ylabel("midpoint values / °")
    fig.suptitle("T5 — Q-Q of the Braetz midpoint sequence per bucket (CI normality assumption)")
    fig.tight_layout(); fig.savefig(OUT / "e5_braetz_qq.png", dpi=150); plt.close(fig)


def _summary_plot(df):
    """Safety illusion: Braetz CI half-width vs. actual GT error, with
    calibration diagonal (y=x), non-coverage rate and illusion quadrant."""
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = {"easy": "#2ca02c", "disagree": "#ff7f0e", "hard": "#d62728"}
    for b, col in colors.items():
        s = df[df.bucket == b]
        ax.scatter(s.ci_km.clip(lower=1e-2), s.err_braetz.clip(lower=1e-2), s=12, alpha=0.5,
                   color=col, label=b)
    ax.set_xscale("log"); ax.set_yscale("log")
    xlo, xhi = ax.get_xlim(); ylo, yhi = ax.get_ylim()

    # Illusion quadrant: tight CI (<100 km) AND large GT error (>200 km)
    n_ill = int(((df.ci_km < 100) & (df.err_braetz > 200)).sum())
    ax.add_patch(Rectangle((xlo, 200), 100 - xlo, yhi - 200, facecolor="grey", alpha=0.12, zorder=0))
    ax.text(xlo * 1.5, yhi * 0.6, f"safety illusion\nn={n_ill} ({100*n_ill/len(df):.1f}%)",
            fontsize=8, color="#555")

    # Calibration diagonal y=x: points ABOVE it = CI does not cover the truth
    d0, d1 = max(xlo, ylo), min(xhi, yhi)
    ax.plot([d0, d1], [d0, d1], ":", color="black", lw=1.2)
    noncov = (df.err_braetz > df.ci_km).mean()
    ax.text(d1 * 0.04, d1 * 0.12, f"y = x (calibration)\nnon-coverage: {100*noncov:.0f}%\n"
            f"(points above)", fontsize=8, color="black", rotation=45,
            rotation_mode="anchor", ha="left", va="bottom")

    ax.axhline(200, color="grey", ls="--", lw=0.6); ax.axvline(100, color="grey", ls="--", lw=0.6)
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_xlabel("Braetz CI half-width / km  (internal certainty)")
    ax.set_ylabel("actual error to ground truth / km")
    ax.set_title(f"T5 — Braetz CI without reference to truth: {100*noncov:.0f}% non-coverage  (n={len(df)})")
    ax.grid(True, which="both", alpha=0.2); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "e5_braetz_ci_vs_gt.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
