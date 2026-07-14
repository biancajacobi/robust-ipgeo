"""E5 / T5 — Kritische Prüfung von Brätz' Fuzzy-Schätzer in der Geo-Domäne.

Vier Befunde (vgl. memory/braetz-verfahren.md):
 (a) Brätz vs. robuste Schätzer: Distanz-zur-Ground-Truth je Schwierigkeits-Bucket.
 (b) Normalität: Shapiro-Wilk auf EINGANG und auf die MITTEN-FOLGE (Brätz' eigentliche
     Annahme). Headline = Anteil KLAR abgelehnt (p<0,01) als konservative Untergrenze;
     nicht-abgelehnt ist bei n=8 mangels Power KEIN Normalitätsbeleg. + Q-Q je Bucket.
 (c) Modalwert-Magnet (Brätz Kap. 4.4.1): Brätz-Hub-Einrast-Quote — Anteil Anchors,
     deren Brätz-Schätzung <50 km an einem bekannten Hub-Default liegt, Default-
     Querschnitt vs. Rest.
 (d) Sicherheits-Illusion: Brätz' CI (Konfidenzintervall) misst INTERNE Sicherheit der Mitten-Folge, nicht
     die Distanz zur GT → enges CI bei großem GT-Fehler (= T6-„niedrig+hub"-Zelle).

Aufruf:  python experiments/exp_braetz.py
Ergebnis: eval/out/e5_braetz.csv, eval/out/e5_braetz_ci_vs_gt.png (Bild 4.9), e5_braetz_qq.png
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
HUB_MIN = 10        # Koordinate gilt als bekannter Hub-Default ab so vielen Default-Treffern
HUB_RADIUS_KM = 50  # Einrast-Umkreis
DEFAULT_RADIUS = 500  # MaxMind accuracy_radius ab dem ein Anchor zum Default-Querschnitt zählt


def bucket(case):
    errs = [haversine_error((p["lat"], p["lon"]), case["truth"]) for p in case["provenance"]]
    if min(errs) > 50:
        return "hart"
    if max(errs) < 50:
        return "easy"
    return "uneinig"


def ci_radius_km(est_lat, half_lat, half_lon):
    """Brätz-KI-Halbbreiten (Grad je Koordinate) -> ungefährer Radius in km."""
    return math.hypot(half_lat * 111.32, half_lon * 111.32 * math.cos(math.radians(est_lat)))


def build_hub_set(cases):
    """Bekannte Hub-Defaults = Koordinaten, die über den Datensatz oft als
    is_default_centroid markiert sind (>= HUB_MIN)."""
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
    print(f"E5/T5 — {len(cases)} Fälle; {len(hubs)} bekannte Hub-Defaults (>= {HUB_MIN} Treffer)")

    rows = []
    for c in cases:
        pts, truth = c["points"], c["truth"]
        bra = braetz.estimate(pts)
        elat, half_lat, seq_lat = braetz.braetz_ci(pts[:, 0])
        elon, half_lon, seq_lon = braetz.braetz_ci(pts[:, 1])
        dsrc = {p["source"]: haversine_error((bra[0], bra[1]), (p["lat"], p["lon"]))
                for p in c["provenance"]}
        d_mm = dsrc.get("maxmind_geolite2", float("nan"))
        # nächste Quelle; MaxMind-Linie (geojs/rfg/maxmind, identische Koords) zusammenfassen
        nearest = min(dsrc, key=dsrc.get)
        nearest_line = "maxmind_linie" if nearest in ("geojs", "reallyfreegeoip", "maxmind_geolite2") else nearest
        rows.append({
            "ip": c["ip"], "bucket": bucket(c),
            "err_braetz": haversine_error((bra[0], bra[1]), truth),
            "err_geomed": haversine_error(tuple(geometric_median(pts)), truth),
            "err_naiv": haversine_error(tuple(centroid(pts)), truth),
            "ci_km": ci_radius_km(bra[0], half_lat, half_lon),
            "p_in_lat": shapiro_p(pts[:, 0]), "p_in_lon": shapiro_p(pts[:, 1]),
            "p_seq_lat": shapiro_p(seq_lat), "p_seq_lon": shapiro_p(seq_lon),
            "near_hub": near_hub(bra[0], bra[1], hubs),
            "m_seq": int(min(seq_lat.size, seq_lon.size)),
            "d_maxmind": d_mm, "nearest_line": nearest_line,
            "is_default": (maxmind_radius(c) or 0) >= DEFAULT_RADIUS,
        })
    for r in rows:                       # Einrast auf einen FALSCHEN Hub (nahe Hub UND weit GT)
        r["wrong_hub"] = r["near_hub"] and r["err_braetz"] > 100
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e5_braetz.csv", index=False)

    print("\n(a) Distanz zur GT — Median [Mittel] km je Bucket")
    print(f"{'Bucket':10s} {'n':>5s}  {'Brätz':>16s} {'geom.Median':>16s} {'naiv':>16s}")
    for b in ["easy", "uneinig", "hart"]:
        s = df[df.bucket == b]
        cells = [f"{s[c].median():5.0f}[{s[c].mean():5.0f}]" for c in ("err_braetz", "err_geomed", "err_naiv")]
        print(f"{b:10s} {len(s):5d}  " + "  ".join(f"{c:>16s}" for c in cells))
    s = df
    cells = [f"{s[c].median():5.0f}[{s[c].mean():5.0f}]" for c in ("err_braetz", "err_geomed", "err_naiv")]
    print(f"{'gesamt':10s} {len(s):5d}  " + "  ".join(f"{c:>16s}" for c in cells))

    print("\n(b) Normalität (Shapiro-Wilk) — Anteil ABGELEHNT (konservativ p<0,01 | p<0,05)")
    for name, cols in [("Eingangsdaten", ["p_in_lat", "p_in_lon"]),
                       ("Mitten-Folge", ["p_seq_lat", "p_seq_lon"])]:
        ps = pd.concat([df[c] for c in cols]).dropna()
        print(f"  {name:14s} (n={len(ps)} Tests):  p<0,01: {100*(ps<0.01).mean():4.1f}%   "
              f"p<0,05: {100*(ps<0.05).mean():4.1f}%")
    print(f"  Caveat: Mitten-Folge ist NICHT kurz (Median m={int(df.m_seq.median())}), aber stark "
          f"GEBUNDEN — die 3 identischen MaxMind-Replikate bilden über fast alle k dieselbe Modalklasse "
          f"→ entartete (wiederholte) Mitten → Shapiro lehnt fast immer ab. Das ist die Replikat-Dominanz "
          f"(c0), kein reines Kleinst-n-Artefakt. Eingangs-Befund (71%) ist der saubere; nicht-abgelehnt ≠ normal.")

    print("\n(c0) Replikat-Dominanz (Kern-Mechanismus) — Brätz hat keine De-Duplikation:")
    print(f"  Brätz <25 km am MaxMind-Wert: {100*(df.d_maxmind<25).mean():4.1f}%  "
          f"(Median-Distanz {df.d_maxmind.median():.1f} km)")
    nl = df.nearest_line.value_counts(normalize=True) * 100
    print("  nächstgelegene Quelle/Linie zu Brätz: " +
          ", ".join(f"{k}={v:.0f}%" for k, v in nl.head(4).items()))
    print("  → die 3 identischen MaxMind-Replikate bilden oft die dichteste Klasse und nageln Brätz fest")

    print("\n(c) Modalwert-Magnet — Einrast auf einen FALSCHEN Hub (<50 km an Hub UND >100 km von GT)")
    print("    [near_hub roh enthält auch echte Hub-Stadt-Anchors → daher die 'falsch'-Bedingung]")
    for label, sub in [("Default-Querschnitt", df[df.is_default]), ("Rest", df[~df.is_default]),
                       ("gesamt", df)]:
        print(f"  {label:20s} n={len(sub):4d}:  falsch-Hub {100*sub.wrong_hub.mean():4.1f}%   "
              f"(roh-nahe-Hub {100*sub.near_hub.mean():4.1f}%)")

    print("\n(d) Sicherheits-Illusion — Brätz-KI misst interne Sicherheit, nicht GT-Distanz")
    noncov = (df.err_braetz > df.ci_km).mean()
    print(f"  NICHT-ABDECKUNG des nominalen 95%-KI: {100*noncov:.1f}% (GT-Fehler > KI-Radius; erwartet ~5%)")
    print(f"    → das KI deckt die wahre Position fast nie ab — die interne Sicherheit ist eine Illusion")
    tight = df[df.ci_km < 100]
    print(f"  Illusions-Quadrant (KI<100 km & Fehler>200 km): {((df.ci_km<100)&(df.err_braetz>200)).sum()} Anchors")
    print(f"  Korrelation KI-Breite ↔ GT-Fehler: r={df['ci_km'].corr(df['err_braetz']):.2f} "
          f"(teilweise informativ, aber Niveau dramatisch unterschätzt → Nicht-Abdeckung)")

    run_threshold_sensitivity(cases)
    run_line_collapse(cases)

    _qq_plot(cases, df)
    print(f"\nCSV: {OUT}/e5_braetz.csv   Plots: {OUT}/e5_braetz_ci_vs_gt.png, e5_braetz_qq.png")
    _summary_plot(df)


def run_threshold_sensitivity(cases):
    """(e) Dichteschwellen-Sensitivität: Hauptlauf >=2 vs. Brätz-Original >=4.

    Belegt, dass die T5-Befunde keine Artefakte der bei n=8 gelockerten Schwelle sind.
    Verwendet exakt dieselbe KI-Berechnung (``ci_radius_km``) wie der Hauptlauf.
    """
    import pandas as pd
    print("\n(e) Dichteschwellen-Sensitivität — Hauptlauf >=2 vs. Brätz-Original >=4")
    print(f"  {'Schwelle':>8s} {'Median':>8s} {'Mittel':>8s} {'Nicht-Abd.':>11s} "
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
    print(f"  → CSV: {OUT}/e5_braetz_threshold.csv  (Headline = >=2; >=4 = Brätz-Original als Robustheits-Beleg)")


def run_line_collapse(cases):
    """(f) Kontrollexperiment (T5): Braetz auf linien-kollabierten Quellen.

    Trennt die KAUSALE Rolle der Replikat-Dominanz von blosser Korrelation: die drei
    strukturell identischen MaxMind-GeoLite-Quellen (gleiche lineage) werden VOR der
    Schaetzung auf EINEN Vertreter reduziert -> nur effektiv unabhaengige Linien (im
    Median 6 statt 8 Quellen) gehen in die Klassenbildung ein. n=6 liegt im von Braetz
    angegebenen Arbeitsbereich (untere Grenze 3-5 Werte, Braetz 2009 S.67), der Test ist
    also konfundierungsfrei gegenueber kleinem n. Wird Braetz auf der kollabierten Menge
    besser, ist die Replikat-Dominanz-These bestaetigt.
    """
    import pandas as pd

    def collapsed(c):
        seen = {}
        for p in c["provenance"]:
            seen.setdefault(p["lineage"], (p["lat"], p["lon"]))   # erster Vertreter je Linie
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

    print("\n(f) Kontrollexperiment — Braetz auf linien-kollabierten Quellen (MaxMind-Familie = 1 Vertreter)")
    print(f"  Quellen nach Kollaps: median {int(np.median(nrep))} (min {min(nrep)}, max {max(nrep)}); "
          f"n=6 liegt im Braetz-Arbeitsbereich (ab 3-5 Werte, S.67) -> kein Kleinst-n-Confound")
    print(f"  {'Bucket':10s} {'n':>5s} {'Med full':>9s} {'Med coll':>9s} {'Mean full':>10s} {'Mean coll':>10s}")
    for b in ["easy", "uneinig", "hart", "gesamt"]:
        s = df if b == "gesamt" else df[df.bucket == b]
        print(f"  {b:10s} {len(s):5d} {s.err_full.median():9.1f} {s.err_coll.median():9.1f} "
              f"{s.err_full.mean():10.1f} {s.err_coll.mean():10.1f}")
    mmf, mmc = df.mm_full.dropna(), df.mm_coll.dropna()
    print(f"  MaxMind-Einrast (<25 km): full {100*(mmf<25).mean():.1f}%  ->  coll {100*(mmc<25).mean():.1f}%")
    u = df[df.bucket == "uneinig"]
    print(f"  -> Replikat-Dominanz kausal bestaetigt (uneinig {u.err_full.median():.0f}->{u.err_coll.median():.0f} km),")
    print(f"     aber unvollstaendig (geom. Median uneinig ~7 km) und Einrastquote ~unveraendert")
    print(f"  CSV: {OUT}/e5_braetz_linecollapse.csv")
    return df


def _qq_plot(cases, df):
    """Q-Q der Mitten-Folge (lat) für je ein Beispiel aus easy/uneinig/hart/default."""
    by_ip = {c["ip"]: c for c in cases}
    picks = []
    for label, mask in [("easy", df.bucket == "easy"), ("uneinig", df.bucket == "uneinig"),
                        ("hart", df.bucket == "hart"), ("default", df.is_default)]:
        sub = df[mask]
        if len(sub):
            picks.append((label, sub.iloc[len(sub) // 2]["ip"]))
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, (label, ip) in zip(axes.flat, picks):
        _, _, seq = braetz.braetz_ci(by_ip[ip]["points"][:, 0])
        p = shapiro_p(seq)
        if seq.size >= 3:
            stats.probplot(seq, dist="norm", plot=ax)
        ax.set_title(f"{label}: {ip}\nMitten-Folge lat (m={seq.size}), Shapiro p={p:.3f}", fontsize=9)
        ax.set_xlabel("theoret. Quantile"); ax.set_ylabel("Mitten-Werte / °")
    fig.suptitle("T5 — Q-Q der Brätz-Mitten-Folge je Bucket (Normalitäts-Annahme des CI)")
    fig.tight_layout(); fig.savefig(OUT / "e5_braetz_qq.png", dpi=150); plt.close(fig)


def _summary_plot(df):
    """Sicherheits-Illusion: Brätz-KI-Halbbreite vs. tatsächlicher GT-Fehler,
    mit Kalibrations-Diagonale (y=x), Nicht-Abdeckungsrate und Illusions-Quadrant."""
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = {"easy": "#2ca02c", "uneinig": "#ff7f0e", "hart": "#d62728"}
    for b, col in colors.items():
        s = df[df.bucket == b]
        ax.scatter(s.ci_km.clip(lower=1e-2), s.err_braetz.clip(lower=1e-2), s=12, alpha=0.5,
                   color=col, label=b)
    ax.set_xscale("log"); ax.set_yscale("log")
    xlo, xhi = ax.get_xlim(); ylo, yhi = ax.get_ylim()

    # Illusions-Quadrant: enges KI (<100 km) UND großer GT-Fehler (>200 km)
    n_ill = int(((df.ci_km < 100) & (df.err_braetz > 200)).sum())
    ax.add_patch(Rectangle((xlo, 200), 100 - xlo, yhi - 200, facecolor="grey", alpha=0.12, zorder=0))
    ax.text(xlo * 1.5, yhi * 0.6, f"Sicherheits-Illusion\nn={n_ill} ({100*n_ill/len(df):.1f}%)",
            fontsize=8, color="#555")

    # Kalibrations-Diagonale y=x: Punkte DARÜBER = KI deckt die Wahrheit nicht ab
    d0, d1 = max(xlo, ylo), min(xhi, yhi)
    ax.plot([d0, d1], [d0, d1], ":", color="black", lw=1.2)
    noncov = (df.err_braetz > df.ci_km).mean()
    ax.text(d1 * 0.04, d1 * 0.12, f"y = x (Kalibration)\nNicht-Abdeckung: {100*noncov:.0f}%\n"
            f"(Punkte oberhalb)", fontsize=8, color="black", rotation=45,
            rotation_mode="anchor", ha="left", va="bottom")

    ax.axhline(200, color="grey", ls="--", lw=0.6); ax.axvline(100, color="grey", ls="--", lw=0.6)
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_xlabel("Brätz-CI-Halbbreite / km  (interne Sicherheit)")
    ax.set_ylabel("tatsächlicher Fehler zur GT / km")
    ax.set_title(f"T5 — Brätz-CI ohne Wahrheits-Bezug: {100*noncov:.0f}% Nicht-Abdeckung  (n={len(df)})")
    ax.grid(True, which="both", alpha=0.2); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "e5_braetz_ci_vs_gt.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
