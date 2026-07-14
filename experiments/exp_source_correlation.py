"""Quellen-(Un)Abhängigkeit: paarweise Übereinstimmung der Geo-Quellen.

Belegt empirisch, wie viele *effektive unabhängige Linien* hinter den nominell
mehreren Quellen stehen (zentral für FF2/FF3: robuste Aggregation setzt
unabhängige ehrliche Quellen voraus). Rechnet die paarweise Median-Distanz auf
der **gemeinsamen Schnittmenge** der einbezogenen Quellen (gleiche IP-Basis für
alle Paare → kein Teilmengen-Artefakt). Quellen mit zu geringer Abdeckung
(z. B. rate-limitiertes ipapi_co) werden ausgeschlossen und benannt.

Output: eval/out/source_correlation.{csv,png} (Median-Distanz-Matrix + Heatmap)
        + eval/out/source_family_drift.csv (Intra-Familien-Drift, §3.2.4:
          geojs bit-identisch zu MaxMind; reallyfreegeoip gleiche Abstammung
          mit abweichendem Datenstand — Richtung aus den Daten nicht bestimmbar).

    python experiments/exp_source_correlation.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import store  # noqa: E402
from eval import report  # noqa: E402
from eval.metrics import haversine  # noqa: E402

MIN_COVERAGE = 0.20  # Quelle muss für >= 20 % der IPs einen Erfolg liefern
FAMILY_REF = "maxmind_geolite2"
FAMILY_OTHERS = ("geojs", "reallyfreegeoip")


def _family_drift(pts, out_dir):
    """Intra-Familien-Statistik: identisch-Quoten, Divergenz-Anteile und
    Fehlervergleich auf dem divergenten Teil (>100 km) gegen die Ground Truth."""
    import pandas as pd
    anchors = {a["ip"]: (float(a["lat"]), float(a["lon"]))
               for a in store.load_anchors_csv()
               if a.get("lat") and a.get("lon")}
    rows = []
    for other in FAMILY_OTHERS:
        ips = sorted(set(pts[FAMILY_REF]) & set(pts[other]) & set(anchors))
        d = np.array([haversine(*pts[FAMILY_REF][ip], *pts[other][ip])
                      for ip in ips])
        div = [ip for ip, x in zip(ips, d) if x > 100]
        row = {"paar": f"{FAMILY_REF}~{other}", "n": len(ips),
               "identisch_pct": round(100 * float((d < 0.01).mean()), 1),
               "unter_0_1km_pct": round(100 * float((d < 0.1).mean()), 1),
               "ueber_1km_pct": round(100 * float((d > 1).mean()), 1),
               "ueber_100km_pct": round(100 * float((d > 100).mean()), 1),
               "median_paardistanz_km": round(float(np.median(d)), 3)}
        if div:
            e_ref = np.array([haversine(*pts[FAMILY_REF][ip], *anchors[ip])
                              for ip in div])
            e_oth = np.array([haversine(*pts[other][ip], *anchors[ip])
                              for ip in div])
            row.update(div_err_median_maxmind_km=round(float(np.median(e_ref)), 1),
                       div_err_median_other_km=round(float(np.median(e_oth)), 1),
                       div_other_schlechter_pct=round(100 * float((e_oth > e_ref).mean()), 0))
        rows.append(row)
        print(f"{FAMILY_REF} ~ {other}: identisch {row['identisch_pct']} %  "
              f">100 km {row['ueber_100km_pct']} %")
    pd.DataFrame(rows).to_csv(out_dir / "source_family_drift.csv", index=False)
    print(f"CSV: {out_dir}/source_family_drift.csv")


def _by_source(observations):
    """{source: {ip: (lat, lon)}} für erfolgreiche Beobachtungen + lineage-Map."""
    pts: dict[str, dict] = {}
    lineage: dict[str, str] = {}
    for o in observations:
        if o.get("status") != "success":
            continue
        try:
            lat, lon = float(o["lat"]), float(o["lon"])
        except (TypeError, ValueError):
            continue
        if lat == 0.0 and lon == 0.0:  # "Null Island" = Unbekannt, nicht vergleichbar
            continue
        pts.setdefault(o["source"], {})[o["ip"]] = (lat, lon)
        lineage[o["source"]] = o.get("lineage", "unknown")
    return pts, lineage


def run() -> None:
    obs = store.load_observations_csv()
    n_ips = len({o["ip"] for o in obs})
    pts, lineage = _by_source(obs)

    included = sorted(s for s in pts if len(pts[s]) >= MIN_COVERAGE * n_ips)
    excluded = sorted(set(pts) - set(included))
    if len(included) < 2:
        print("Zu wenige abgedeckte Quellen für eine Korrelationsmatrix.")
        return

    common = set.intersection(*(set(pts[s]) for s in included))
    print(f"{n_ips} IPs gesamt | einbezogen: {', '.join(included)}")
    if excluded:
        print(f"ausgeschlossen (<{MIN_COVERAGE:.0%} Abdeckung): "
              + ", ".join(f"{s} ({len(pts[s])})" for s in excluded))
    print(f"gemeinsame Schnittmenge: {len(common)} IPs\n")

    # paarweise Median-Distanz auf der gemeinsamen Schnittmenge
    import pandas as pd
    mat = pd.DataFrame(np.zeros((len(included), len(included))),
                       index=included, columns=included)
    for a, b in itertools.combinations(included, 2):
        d = np.array([haversine(*pts[a][ip], *pts[b][ip]) for ip in common])
        mat.loc[a, b] = mat.loc[b, a] = float(np.median(d))

    report.OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = report.OUT_DIR / "source_correlation.csv"
    mat.round(2).to_csv(csv_path)
    _family_drift(pts, report.OUT_DIR)
    png_path = _heatmap(mat, included, lineage, len(common))

    print("Paarweise Median-Distanz [km] (gemeinsame Schnittmenge):")
    print(mat.round(1).to_string())
    # effektive Linien: Quellen gleicher lineage zusammenfassen
    lines: dict[str, list] = {}
    for s in included:
        lines.setdefault(lineage[s], []).append(s)
    print("\nLinien (lineage -> Quellen):")
    for lin, members in lines.items():
        print(f"  {lin:22} {', '.join(members)}")
    print(f"\n=> {len(included)} Quellen, aber nur {len(lines)} deklarierte Linien.")
    print(f"\nMatrix: {csv_path}\nHeatmap: {png_path}")


def _heatmap(mat, labels, lineage, n_common):
    plt = report._plt()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(mat.to_numpy(), cmap="viridis_r")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([f"{s}\n[{lineage[s]}]" for s in labels], fontsize=7, rotation=40, ha="right")
    ax.set_yticklabels([f"{s}" for s in labels], fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{mat.iloc[i, j]:.0f}", ha="center", va="center",
                    color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="Median-Distanz [km]")
    ax.set_title(f"Quellen-Übereinstimmung (n={n_common} gemeinsame IPs)\n"
                 "kleine Distanz = korreliert (nahe derselben Linie)")
    fig.tight_layout()
    path = report.OUT_DIR / "source_correlation.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    run()
