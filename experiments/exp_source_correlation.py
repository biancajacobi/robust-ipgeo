"""Source (in)dependence: pairwise agreement of the geo sources.

Documents empirically how many *effective independent lines* stand behind the
nominally multiple sources (central to RQ2/RQ3: robust aggregation presupposes
independent honest sources). Computes the pairwise median distance on the
**common intersection** of the included sources (same IP basis for all pairs →
no subset artifact). Sources with too little coverage (e.g. rate-limited
ipapi_co) are excluded and named.

Output: eval/out/source_correlation.{csv,png} (median distance matrix + heatmap)
        + eval/out/source_family_drift.csv (intra-family drift, see the
          accompanying paper: geojs bit-identical to MaxMind; reallyfreegeoip
          same ancestry with a diverging data snapshot — direction not
          determinable from the data).

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

MIN_COVERAGE = 0.20  # source must deliver a success for >= 20% of the IPs
FAMILY_REF = "maxmind_geolite2"
FAMILY_OTHERS = ("geojs", "reallyfreegeoip")


def _family_drift(pts, out_dir):
    """Intra-family statistics: identical rates, divergence shares and
    error comparison on the divergent part (>100 km) against the ground truth."""
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
        row = {"pair": f"{FAMILY_REF}~{other}", "n": len(ips),
               "identical_pct": round(100 * float((d < 0.01).mean()), 1),
               "under_0_1km_pct": round(100 * float((d < 0.1).mean()), 1),
               "over_1km_pct": round(100 * float((d > 1).mean()), 1),
               "over_100km_pct": round(100 * float((d > 100).mean()), 1),
               "median_pair_distance_km": round(float(np.median(d)), 3)}
        if div:
            e_ref = np.array([haversine(*pts[FAMILY_REF][ip], *anchors[ip])
                              for ip in div])
            e_oth = np.array([haversine(*pts[other][ip], *anchors[ip])
                              for ip in div])
            row.update(div_err_median_maxmind_km=round(float(np.median(e_ref)), 1),
                       div_err_median_other_km=round(float(np.median(e_oth)), 1),
                       div_other_worse_pct=round(100 * float((e_oth > e_ref).mean()), 0))
        rows.append(row)
        print(f"{FAMILY_REF} ~ {other}: identical {row['identical_pct']} %  "
              f">100 km {row['over_100km_pct']} %")
    pd.DataFrame(rows).to_csv(out_dir / "source_family_drift.csv", index=False)
    print(f"CSV: {out_dir}/source_family_drift.csv")


def _by_source(observations):
    """{source: {ip: (lat, lon)}} for successful observations + lineage map."""
    pts: dict[str, dict] = {}
    lineage: dict[str, str] = {}
    for o in observations:
        if o.get("status") != "success":
            continue
        try:
            lat, lon = float(o["lat"]), float(o["lon"])
        except (TypeError, ValueError):
            continue
        if lat == 0.0 and lon == 0.0:  # "Null Island" = unknown, not comparable
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
        print("Too few covered sources for a correlation matrix.")
        return

    common = set.intersection(*(set(pts[s]) for s in included))
    print(f"{n_ips} IPs total | included: {', '.join(included)}")
    if excluded:
        print(f"excluded (<{MIN_COVERAGE:.0%} coverage): "
              + ", ".join(f"{s} ({len(pts[s])})" for s in excluded))
    print(f"common intersection: {len(common)} IPs\n")

    # pairwise median distance on the common intersection
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

    print("Pairwise median distance [km] (common intersection):")
    print(mat.round(1).to_string())
    # effective lines: merge sources of the same lineage
    lines: dict[str, list] = {}
    for s in included:
        lines.setdefault(lineage[s], []).append(s)
    print("\nLines (lineage -> sources):")
    for lin, members in lines.items():
        print(f"  {lin:22} {', '.join(members)}")
    print(f"\n=> {len(included)} sources, but only {len(lines)} declared lines.")
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
    fig.colorbar(im, ax=ax, label="median distance [km]")
    ax.set_title(f"Source agreement (n={n_common} common IPs)\n"
                 "small distance = correlated (close to the same line)")
    fig.tight_layout()
    path = report.OUT_DIR / "source_correlation.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    run()
