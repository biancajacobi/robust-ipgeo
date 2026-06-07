"""Vergleich Anchors vs. Probes auf identischer Pipeline (direktionaler Stresstest).

Laedt BEIDE Datensaetze explizit (kein Env-Switch noetig), beschraenkt auf die in
beiden vorhandenen Quellen (fairer Vergleich) und berichtet je Datensatz dieselben
Kern-Metriken: Einzelquellen-Genauigkeit, robuste Aggregation L1*b, Schwierigkeits-
Eimer, Braetz, Default-Rate. Schreibt eval/out_probes/probes_vs_anchors.csv.

VORBEHALT: Probe-Koordinaten sind selbstgemeldet + privacy-gerundet -> rauschigere
Ground Truth. Hoehere Probe-Fehler mischen (a) schwerere IPs und (b) GT-Rauschen.

Aufruf:  python experiments/exp_probes_compare.py
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
    """Faelle auf eine Quellen-Teilmenge beschneiden (fairer Quellen-Schnitt)."""
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
    # Einzelquellen
    src = {}
    for c in cases:
        for p in c["provenance"]:
            src.setdefault(p["source"], []).append(haversine_error((p["lat"], p["lon"]), c["truth"]))
    # Aggregationen + Braetz + Buckets
    agg, gm, bra, buckets = [], [], [], {"easy": 0, "uneinig": 0, "hart": 0}
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
            "bucket_easy": buckets["easy"], "bucket_uneinig": buckets["uneinig"],
            "bucket_hart": buckets["hart"], "majority_default_pct": round(100 * np.mean(defrate), 1)}
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
    """Aggregationsfehler L1*b je Fall (eigene LOO/eps pro Datensatz)."""
    loo = T6.loo_pseudo_radii(cases)
    eps = min(T6.EPS_GRID, key=lambda e: abs(e - float(np.median([loo[s]["_global"] for s in loo]))))
    return np.array([haversine_error(T6.estimate(c, "L1", "b", loo, eps=eps), c["truth"]) for c in cases])


def stratify_by_tag(prb_cases, mob_cases=None):
    """Probes nach Anschluss-Klasse (home/nat, mobile, datacentre) aufschlüsseln.

    Kontrolliert teils den GT-Rausch-Confound: innerhalb des Probe-Sets ist die
    GT-Qualität vergleichbar, sodass Klassenunterschiede primär die Lokalisierbarkeit
    der IP-Klasse widerspiegeln (residentiell/NAT vs. Rechenzentrum). ``mob_cases``:
    optionaler, gezielt gezogener Mobile-Datensatz (eigener Bucket, groesseres n)."""
    tags = load_tags()
    loo = T6.loo_pseudo_radii(prb_cases)
    eps = min(T6.EPS_GRID, key=lambda e: abs(e - float(np.median([loo[s]["_global"] for s in loo]))))
    groups = {}
    for c in prb_cases:
        cls = classify(tags.get(c["ip"], set()))
        err = haversine_error(T6.estimate(c, "L1", "b", loo, eps=eps), c["truth"])
        groups.setdefault(cls, []).append(err)
    print("\n" + "=" * 78)
    print("PROBE-STRATIFIZIERUNG nach Anschluss-Klasse (Aggregation L1*b)")
    print(f"  {'Klasse':22s} {'n':>4s} {'median':>7s} {'mean':>9s} {'tail%':>6s}")
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
        emit("mobile (Sample)", groups["mobile"])
    if mob_cases:
        emit("mobile (gezielt)", _agg_errors(mob_cases))
    pd.DataFrame(rows).to_csv(OUT / "probes_by_tag.csv", index=False)
    print(f"  CSV: {OUT}/probes_by_tag.csv")
    print("  Hinweis: 'mobile' = RIPE-Tag (fest installierte 4G-Router, KEINE roamenden Handys);")
    print("           Nabis 179-207 km gelten fuer Carrier-CGNAT-Mobilfunk -> hier strukturell nicht erfasst.")


def main():
    anc = cases_from("anchors.csv", "observations.csv")
    prb = cases_from("probes.csv", "observations_probes.csv")
    common = sorted(set(p["source"] for c in anc for p in c["provenance"])
                    & set(p["source"] for c in prb for p in c["provenance"]))
    print(f"Gemeinsame Quellen ({len(common)}): {', '.join(common)}")
    print(f"Anchors: {len(anc)} Faelle | Probes: {len(prb)} Faelle\n")

    # Metriken auf den VOLLEN Cases je Datensatz (Anchor-Aggregation = Headline aus
    # der Hauptauswertung, inkl. ipapi_co-Resthits; Probes haben ohnehin kein ipapi_co). Die
    # Einzelquellen-Zeilen werden auf die gemeinsamen acht Quellen gefiltert.
    (ar, am), (pr, pm) = metrics(anc, "anchors"), metrics(prb, "probes")

    df = pd.DataFrame(ar + pr)
    df.to_csv(OUT / "probes_vs_anchors.csv", index=False)

    aggregators = ("AGG L1*b", "geom_median (L0)", "braetz")
    print("=" * 78)
    print(f"GENAUIGKEIT je Quelle/Aggregator  (Einzelquellen: gemeinsame {len(common)})")
    print(f"{'':22s} {'Anchors med/mean/tail':>26s}   {'Probes med/mean/tail':>26s}")
    names = [r["name"] for r in ar if r["name"] in common or r["name"] in aggregators]
    ad = {r["name"]: r for r in ar}; pd_ = {r["name"]: r for r in pr}
    for nm in names:
        a, p = ad[nm], pd_.get(nm, {})
        print(f"  {nm:20s} {a['median']:6.1f} /{a['mean']:7.1f} /{a['tail']:5.1f}%   "
              f"{p.get('median',float('nan')):6.1f} /{p.get('mean',float('nan')):7.1f} /{p.get('tail',float('nan')):5.1f}%")
    print("\nSchwierigkeits-Eimer (Anteil) und Default-Rate:")
    for m in (am, pm):
        tot = m["n_cases"]
        print(f"  {m['dataset']:8s} n={tot:4d}  easy {100*m['bucket_easy']/tot:4.1f}%  "
              f"uneinig {100*m['bucket_uneinig']/tot:4.1f}%  hart {100*m['bucket_hart']/tot:4.1f}%  "
              f"| Mehrheits-Default {m['majority_default_pct']:.1f}%  | Headline-eps {m['eps_headline']}")
    print(f"\nCSV: {OUT}/probes_vs_anchors.csv")

    mob = None
    if (CACHE / "probes_mobile.csv").exists() and (CACHE / "observations_probes_mobile.csv").exists():
        mob = cases_from("probes_mobile.csv", "observations_probes_mobile.csv")
    stratify_by_tag(prb, mob)
    print("\nHinweis: Probe-GT ist selbstgemeldet/gerundet -> direktionaler Stresstest, keine saubere Validierung.")


if __name__ == "__main__":
    main()
