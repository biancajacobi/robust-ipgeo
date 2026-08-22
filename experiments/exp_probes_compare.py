"""Comparison of anchors vs. probes on an identical pipeline (directional stress test).

Loads BOTH datasets explicitly (no env switch needed). The SINGLE-SOURCE rows
are restricted to the sources present in both (fair comparison); the
aggregations run on each dataset's FULL source set (anchor headline incl. the
ipapi_co residual hits, see comment in main). Reports the same core metrics
per dataset:
single-source accuracy, robust aggregation L1*b, difficulty bucket, Braetz,
default rate. Writes eval/out_probes/probes_vs_anchors.csv.

CAVEAT: probe coordinates are self-reported + privacy-rounded -> noisier ground
truth. Higher probe errors mix (a) harder IPs and (b) GT noise.

Invocation: python experiments/exp_probes_compare.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data import store                                  # noqa: E402
from eval.pipeline import load_cases                    # noqa: E402
from eval.metrics import haversine_error                 # noqa: E402
from estimators.baselines import geometric_median, centroid  # noqa: E402
from estimators import braetz                            # noqa: E402
import experiments.exp_t6_defaults as T6                 # noqa: E402

CACHE = store.CACHE_DIR
OUT = ROOT / "eval" / "out_probes"; OUT.mkdir(parents=True, exist_ok=True)


def cases_from(truth_csv, obs_csv):
    return load_cases(anchors=store.load_anchors_csv(CACHE / truth_csv),
                      observations=store.load_observations_csv(CACHE / obs_csv))


def restrict(cases, sources):
    """Restrict cases to a subset of sources (fair source cut)."""
    out = []
    for c in cases:
        prov = [p for p in c["provenance"] if p["source"] in sources]
        if not prov:
            continue
        d = dict(c); d["provenance"] = prov
        d["points"] = np.array([[p["lat"], p["lon"]] for p in prov], float)
        d["sources"] = [p["source"] for p in prov]
        out.append(d)
    return out


def metrics(cases, label):
    loo = T6.loo_pseudo_radii(cases)
    eps = min(T6.EPS_GRID, key=lambda e: abs(e - float(np.median([loo[s]["_global"] for s in loo]))))
    # Individual sources
    src = {}
    for c in cases:
        for p in c["provenance"]:
            src.setdefault(p["source"], []).append(haversine_error((p["lat"], p["lon"]), c["truth"]))
    # Aggregations + Braetz + buckets
    agg, gm, bra, buckets = [], [], [], {"easy": 0, "disagree": 0, "hard": 0}
    defrate = []
    for c in cases:
        agg.append(haversine_error(T6.estimate(c, "L1", "b", loo, eps=eps), c["truth"]))
        gm.append(haversine_error(tuple(geometric_median(c["points"])), c["truth"]))
        bra.append(haversine_error(tuple(braetz.estimate(c["points"])), c["truth"]))
        buckets[T6.bucket(c)] += 1
        defrate.append(np.mean([p["is_default_centroid"] for p in c["provenance"]]) >= 0.5)
    n = len(cases)

    def stat(e):
        e = np.asarray(e, float)
        return dict(median=float(np.median(e)), mean=float(e.mean()),
                    tail=float(100 * (e > 100).mean()), n=len(e))

    rows = []
    for s, e in sorted(src.items(), key=lambda kv: np.median(kv[1])):
        rows.append({"dataset": label, "name": s, **stat(e)})
    rows.append({"dataset": label, "name": "AGG L1*b", **stat(agg)})
    rows.append({"dataset": label, "name": "geom_median (L0)", **stat(gm)})
    rows.append({"dataset": label, "name": "braetz", **stat(bra)})
    meta = {"dataset": label, "n_cases": n, "eps_headline": eps,
            "bucket_easy": buckets["easy"], "bucket_disagree": buckets["disagree"],
            "bucket_hard": buckets["hard"], "majority_default_pct": round(100 * np.mean(defrate), 1)}
    return rows, meta


def load_tags():
    m, p = {}, CACHE / "probes_tags.csv"
    if p.exists():
        for r in csv.DictReader(open(p, encoding="utf-8")):
            m[r["ip"]] = set(t for t in r["tags"].split(";") if t)
    return m


def classify(tags):
    if {"datacentre", "datacenter"} & tags:
        return "datacentre"
    if "mobile" in tags:
        return "mobile"
    if {"home", "nat"} & tags:
        return "home/nat"
    return "other"


def _agg_errors(cases):
    """L1*b aggregation error per case (dataset-specific LOO/eps)."""
    loo = T6.loo_pseudo_radii(cases)
    eps = min(T6.EPS_GRID, key=lambda e: abs(e - float(np.median([loo[s]["_global"] for s in loo]))))
    return np.array([haversine_error(T6.estimate(c, "L1", "b", loo, eps=eps), c["truth"]) for c in cases])


def stratify_by_tag(prb_cases, mob_cases=None):
    """Break down probes by connection class (home/nat, mobile, datacentre).

    Partly controls the GT-noise confound: within the probe set, GT quality is
    comparable, so class differences primarily reflect the localizability of the
    IP class (residential/NAT vs. data center). ``mob_cases``: optional,
    deliberately sampled mobile dataset (its own bucket, larger n)."""
    tags = load_tags()
    loo = T6.loo_pseudo_radii(prb_cases)
    eps = min(T6.EPS_GRID, key=lambda e: abs(e - float(np.median([loo[s]["_global"] for s in loo]))))
    groups = {}
    for c in prb_cases:
        cls = classify(tags.get(c["ip"], set()))
        err = haversine_error(T6.estimate(c, "L1", "b", loo, eps=eps), c["truth"])
        groups.setdefault(cls, []).append(err)
    print("\n" + "=" * 78)
    print("PROBE STRATIFICATION by connection class (aggregation L1*b)")
    print(f"  {'Class':22s} {'n':>4s} {'median':>7s} {'mean':>9s} {'tail%':>6s}")
    rows = []

    def emit(label, e):
        e = np.asarray(e, float)
        print(f"  {label:22s} {len(e):4d} {np.median(e):7.1f} {e.mean():9.1f} {100*(e>100).mean():6.1f}")
        rows.append({"class": label, "n": len(e), "median": round(float(np.median(e)), 1),
                     "mean": round(float(e.mean()), 1), "tail": round(100*float((e > 100).mean()), 1)})

    for cls in ["datacentre", "other", "home/nat"]:
        if cls in groups:
            emit(cls, groups[cls])
    if "mobile" in groups:
        emit("mobile (sample)", groups["mobile"])
    if mob_cases:
        emit("mobile (targeted)", _agg_errors(mob_cases))
    pd.DataFrame(rows).to_csv(OUT / "probes_by_tag.csv", index=False)
    print(f"  CSV: {OUT}/probes_by_tag.csv")
    print("  Note: 'mobile' = RIPE tag (permanently installed 4G routers, NOT roaming phones);")
    print("        Nabi's 179-207 km apply to carrier-CGNAT cellular -> structurally not captured here.")


def main():
    anc = cases_from("anchors.csv", "observations.csv")
    prb = cases_from("probes.csv", "observations_probes.csv")
    common = sorted(set(p["source"] for c in anc for p in c["provenance"])
                    & set(p["source"] for c in prb for p in c["provenance"]))
    print(f"Common sources ({len(common)}): {', '.join(common)}")
    print(f"Anchors: {len(anc)} cases | Probes: {len(prb)} cases\n")

    # Metrics on the FULL cases per dataset (anchor aggregation = headline
    # configuration, incl. residual ipapi_co hits; probes have no ipapi_co anyway). The
    # single-source rows are filtered to the eight common sources.
    (ar, am), (pr, pm) = metrics(anc, "anchors"), metrics(prb, "probes")

    df = pd.DataFrame(ar + pr)
    df.to_csv(OUT / "probes_vs_anchors.csv", index=False)

    aggregators = ("AGG L1*b", "geom_median (L0)", "braetz")
    print("=" * 78)
    print(f"ACCURACY per source/aggregator  (single sources: the {len(common)} common ones)")
    print(f"{'':22s} {'Anchors med/mean/tail':>26s}   {'Probes med/mean/tail':>26s}")
    names = [r["name"] for r in ar if r["name"] in common or r["name"] in aggregators]
    ad = {r["name"]: r for r in ar}; pd_ = {r["name"]: r for r in pr}
    for nm in names:
        a, p = ad[nm], pd_.get(nm, {})
        print(f"  {nm:20s} {a['median']:6.1f} /{a['mean']:7.1f} /{a['tail']:5.1f}%   "
              f"{p.get('median',float('nan')):6.1f} /{p.get('mean',float('nan')):7.1f} /{p.get('tail',float('nan')):5.1f}%")
    print("\nDifficulty bucket (share) and default rate:")
    for m in (am, pm):
        tot = m["n_cases"]
        print(f"  {m['dataset']:8s} n={tot:4d}  easy {100*m['bucket_easy']/tot:4.1f}%  "
              f"disagree {100*m['bucket_disagree']/tot:4.1f}%  hard {100*m['bucket_hard']/tot:4.1f}%  "
              f"| majority default {m['majority_default_pct']:.1f}%  | headline eps {m['eps_headline']}")
    print(f"\nCSV: {OUT}/probes_vs_anchors.csv")

    mob = None
    if (CACHE / "probes_mobile.csv").exists() and (CACHE / "observations_probes_mobile.csv").exists():
        mob = cases_from("probes_mobile.csv", "observations_probes_mobile.csv")
    stratify_by_tag(prb, mob)
    print("\nNote: probe GT is self-reported/rounded -> directional stress test, not a clean validation.")


if __name__ == "__main__":
    main()
