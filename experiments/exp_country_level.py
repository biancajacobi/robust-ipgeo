#!/usr/bin/env python3
"""Country level: hit rate "estimated country == ground-truth country" of the
aggregation L1·b versus every single source, across all populations.

Motivation (application chapter, 2026-09-22): instrument choice (domestic
request / European Production Order / mutual legal assistance) and the
residence test of Art. 8(2) Regulation (EU) 2023/1543 are COUNTRY decisions.
The paper so far reported km errors and the 100-km tail rate only; a country
hit rate was missing. It is the figure rows 1–2 of tab:scenarios must rest on.

How does an aggregated coordinate become a country? There is no offline
reverse geocoding in the repo (no shapely/geopandas, no boundary data, and
experiments must not fetch anything). Hence – as for S – from the countries of
the contributing sources, pre-specified:
  (a) "core vote" (MAIN COLUMN): line-weighted country mass of the sources
      within the frozen core radius R_CORE=50 km around the L1·b estimate
      (the same mass whose total is S); highest mass wins, ties alphabetical.
      If no source lies in the core, the country of the nearest source counts
      (fallback, reported).
  (b) "line vote" (control): line-collapsed majority vote over ALL responding
      sources with a country label, weight 1/|line among voters| (like
      city_vote_collapsed without the city part).
Sources without a country label abstain; with no label at all the case counts
as a miss (conservative).

Calibration frozen (anchors: LOO radii + ε), sources per population as in
T8/T9 (pool: SEVEN, giga: EIGHT). Δ = hit rate(aggregation) − hit rate(source)
in pp on the paired IPs (source responded AND carries a label), paired iid
bootstrap (B=10,000, seed 0) + ASN-cluster bootstrap (giga additionally school
clusters). In addition: hit rate by S threshold (core mass ≥ 0.5 vs. < 0.5) –
carries the "gate" statement of the application chapter.

Addendum (review pass 2026-09-22): per population Fisher's exact test miss x
stratum (does the gate carry?) with gate recall, and a miss profile (share of
country misses with a source majority >= 75 %/unanimous on the wrong country,
top truth>vote pairs) — measures the hypothesis "registration/hosting country
instead of physical location".

Coding (review pass 2026-09-22, "coding question"): the unit that matters for
the instrument choice is the STATE a legal instrument is addressed to, not the
ISO 3166-1 code. Ground truth AND source labels are therefore normalised via
STATE_OF in the main coding "state" (clear cases only: integral parts or
dependent territories without their own mutual-legal-assistance capacity, e.g.
Guam→US, Réunion→FR). Deliberately NOT normalised: Hong Kong/Macau (separate
MLA jurisdictions — China↔Hong Kong thus counts as a genuine error), Kosovo
(XK, disputed — counts as an error), British Crown dependencies/overseas
territories, AW/CW/SX, FO/GL (separate legal orders). As a control the same
evaluation additionally runs raw ("iso", without normalisation; headline/gate/
miss profile only, without the single-source bootstrap) — both are in the CSV
(column coding).

Usage:  GEOIP_DATASET=anchors python experiments/exp_country_level.py
Result: table (stdout) + eval/out/country_level.csv (aggregated, IP-free)
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data import store                                     # noqa: E402
from eval.pipeline import load_cases                       # noqa: E402
from eval.metrics import haversine_error                   # noqa: E402
import experiments.exp_t6_defaults as T6                   # noqa: E402
from experiments.exp_radius_transfer import eps_for        # noqa: E402
from experiments.exp_pool_transfer import restrict, SEVEN  # noqa: E402
from experiments.exp_giga_transfer import EIGHT            # noqa: E402
from experiments.exp_cluster_bootstrap import make_groups, cluster_boot  # noqa: E402

CACHE = ROOT / "data" / "cache"
OUT = ROOT / "eval" / "out"
OUT.mkdir(parents=True, exist_ok=True)
R_CORE = 50.0          # frozen core radius (as for S)
S_GATE = 0.5           # pre-specified gate threshold for the stratification
B, SEED = 10_000, 0

# (label, dataset, observations file, source restriction)
POPS = [
    ("anchors",          "anchors",          "observations.csv",                  None),
    ("probes",           "probes",           "observations_probes.csv",           None),
    ("probes_holdout",   "probes_holdout",   "observations_probes_holdout.csv",   None),
    ("probes_mobile_ext", "probes_mobile_ext", "observations_probes_mobile_ext.csv", None),
    ("probes_pool",      "probes_pool",      "observations_probes_pool.csv",      SEVEN),
    ("giga",             "giga",             "observations_giga.csv",             EIGHT),
]
SOURCE_ORDER = ["ip_api", "ipinfo", "ipwho_is", "ipapi_co", "reallyfreegeoip",
                "maxmind_geolite2", "geojs", "dbip_lite", "ip2location_lite"]

# territory code -> state a legal instrument is addressed to (clear cases only).
STATE_OF = {
    # US outlying territories
    "GU": "US", "PR": "US", "VI": "US", "AS": "US", "MP": "US", "UM": "US",
    # French overseas territories (departements/collectivites d'outre-mer)
    "RE": "FR", "GP": "FR", "MQ": "FR", "GF": "FR", "YT": "FR", "PM": "FR",
    "NC": "FR", "PF": "FR", "WF": "FR", "BL": "FR", "MF": "FR", "TF": "FR",
    # Norway, Finland, Australia, Caribbean Netherlands (special municipalities)
    "SJ": "NO", "BV": "NO", "AX": "FI",
    "CX": "AU", "CC": "AU", "NF": "AU", "HM": "AU",
    "BQ": "NL",
}
# Not normalised (deliberately): HK, MO, XK, GG, JE, IM, GI, BM, KY, VG, FK, AW, CW, SX, FO, GL.
CODINGS = ("state", "iso")     # main coding first
_NORMALIZE = True              # set in main() per coding


def city_lut_from(obs_csv: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """(ip, source) -> (country, city). Own variant with keep_default_na=False:
    pandas otherwise reads the ISO code "NA" (Namibia) as a missing value — in
    the first run this produced ten spurious misses on Giga (all Namibia pairs)."""
    df = pd.read_csv(obs_csv, dtype=str, keep_default_na=False)
    return {(str(r["ip"]), str(r["source"])): (str(r.get("country") or "").strip(),
                                               str(r.get("city") or "").strip())
            for _, r in df.iterrows()}


def _cc(x: str) -> str:
    x = (x or "").strip().upper()
    if x in ("", "NAN", "NONE"):
        return ""
    return STATE_OF.get(x, x) if _NORMALIZE else x


def country_votes(case, lut, loo, eps):
    """Returns (core_country, core_mass, fallback_used, line_vote_country)."""
    prov = case["provenance"]
    labels = [_cc(lut.get((case["ip"], p["source"]), ("", ""))[0]) for p in prov]
    pts = np.array([[p["lat"], p["lon"]] for p in prov], float)
    w = np.asarray(T6.line_weights_for(prov, "L1"), float)
    w = w / w.sum()
    est = T6.estimate(case, "L1", "b", loo, eps=eps)
    d = np.array([haversine_error(tuple(p), est) for p in pts])

    # (a) core vote
    core = [(labels[i], w[i]) for i in range(len(prov)) if d[i] < R_CORE and labels[i]]
    fallback = False
    if core:
        mass = Counter()
        for c, wi in core:
            mass[c] += wi
        win = sorted(mass.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        core_cc, core_mass = win[0], float(win[1])
    else:
        fallback = True
        order = np.argsort(d)
        core_cc = next((labels[i] for i in order if labels[i]), "")
        core_mass = 0.0

    # (b) line vote (collapsed): weight 1/|line among voters|
    voters = [(labels[i], prov[i]["lineage"]) for i in range(len(prov)) if labels[i]]
    line_cc = ""
    if voters:
        per_line = Counter(l for _, l in voters)
        mass = Counter()
        for c, l in voters:
            mass[c] += 1.0 / per_line[l]
        line_cc = sorted(mass.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return core_cc, core_mass, fallback, line_cc


def hit_rate(h) -> float:
    return float(100.0 * np.mean(h)) if len(h) else float("nan")


def paired_delta(h_agg, h_ref, groups_list):
    """Δ in pp (agg − ref) + iid CI + cluster CIs per grouping."""
    h_agg = np.asarray(h_agg, float)
    h_ref = np.asarray(h_ref, float)
    n = len(h_agg)
    delta = 100.0 * (h_agg.mean() - h_ref.mean())
    rng = np.random.default_rng(SEED)
    boot = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        boot[i] = 100.0 * (h_agg[idx].mean() - h_ref[idx].mean())
    out = {"delta_pp": delta,
           "iid_lo": float(np.percentile(boot, 2.5)),
           "iid_hi": float(np.percentile(boot, 97.5))}
    for gname, labels in groups_list:
        groups = make_groups(labels)
        cb = cluster_boot(lambda idx: 100.0 * (h_agg[idx].mean() - h_ref[idx].mean()),
                          groups, B)
        out[f"{gname}_lo"] = float(np.percentile(cb, 2.5))
        out[f"{gname}_hi"] = float(np.percentile(cb, 97.5))
        out[f"{gname}_clusters"] = len(groups)
    return out


def main() -> None:
    if os.environ.get("GEOIP_DATASET", "anchors") != "anchors":
        sys.exit("exp_country_level: please run with GEOIP_DATASET=anchors "
                 "(anchor calibration is frozen; populations are loaded explicitly).")

    anchors9 = load_cases()
    loo_a = T6.loo_pseudo_radii(anchors9)
    eps_a = eps_for(loo_a)
    print(f"Anchor calibration frozen: eps={eps_a} km; R_CORE={R_CORE} km; "
          f"S gate={S_GATE}; B={B}")

    rows = []
    for coding in CODINGS:
        global _NORMALIZE
        _NORMALIZE = (coding == "state")
        print("\n" + "#" * 78)
        print(f"# coding: {coding}" + (" (territory codes normalised to the state, see STATE_OF)"
                                        if _NORMALIZE else " (raw, ISO 3166-1; control)"))
        print("#" * 78)
        start = len(rows)
        for label, ds, obs_name, srcs in POPS:
            if ds == "anchors":
                cases = anchors9
            else:
                cases = load_cases(store.load_anchors_csv(CACHE / f"{ds}.csv"),
                                   store.load_observations_csv(CACHE / obs_name))
            if srcs is not None:
                cases = restrict(cases, srcs)
                for c in cases:   # align the source list with the restriction (baselines = aggregation sources)
                    c["sources"] = [p["source"] for p in c["provenance"]]
            lut = city_lut_from(CACHE / obs_name)
            truth = np.array([_cc(c["country"]) for c in cases])
            asn = np.array([str(c.get("asn") or c["ip"]) for c in cases])
            groups_list = [("asn", asn)]
            if ds == "giga":
                school_of = {}
                for r in store.load_anchors_csv(CACHE / "giga.csv"):
                    school_of[r["ip"]] = r.get("anchor_id") or r["ip"]
                groups_list.append(("school", np.array([school_of.get(c["ip"], c["ip"])
                                                         for c in cases])))

            core_cc, core_mass, fb, line_cc, maj = [], [], [], [], []
            for c in cases:
                a, m, f, l = country_votes(c, lut, loo_a, eps_a)
                core_cc.append(a); core_mass.append(m); fb.append(f); line_cc.append(l)
                labs = [_cc(lut.get((c['ip'], p['source']), ('', ''))[0]) for p in c['provenance']]
                labs = [x for x in labs if x]
                maj.append(max(Counter(labs).values()) / len(labs) if labs else 0.0)
            core_cc = np.array(core_cc); core_mass = np.array(core_mass)
            fb = np.array(fb); line_cc = np.array(line_cc)
            maj = np.array(maj)
            h_core = (core_cc == truth) & (core_cc != "")
            h_line = (line_cc == truth) & (line_cc != "")

            print("\n" + "=" * 78)
            print(f"{label} (n={len(cases)}, sources={'all' if srcs is None else len(srcs)}, "
                  f"nearest-source fallback {int(fb.sum())} = {100*fb.mean():.1f} %, "
                  f"no label at all {int((core_cc == '').sum())})")
            print("=" * 78)
            print(f"  {'aggregation L1·b core vote':30s} n={len(cases):5d} | country hit "
                  f"{hit_rate(h_core):5.1f} %")
            print(f"  {'aggregation line vote':30s} n={len(cases):5d} | country hit "
                  f"{hit_rate(h_line):5.1f} %")
            rows.append({"population": label, "estimator": "agg_core_vote", "n": len(cases),
                         "n_paired": len(cases), "hit_pct": round(hit_rate(h_core), 2),
                         "fallback_pct": round(100 * fb.mean(), 2)})
            rows.append({"population": label, "estimator": "agg_line_vote", "n": len(cases),
                         "n_paired": len(cases), "hit_pct": round(hit_rate(h_line), 2)})

            # S-gate stratification (core mass = S share of the winning country)
            for name, mask in (("S_ge_gate", core_mass >= S_GATE),
                               ("S_lt_gate", core_mass < S_GATE)):
                if mask.sum():
                    print(f"  {'  core vote | ' + name:30s} n={int(mask.sum()):5d} | country hit "
                          f"{hit_rate(h_core[mask]):5.1f} %")
                    rows.append({"population": label, "estimator": f"agg_core_vote_{name}",
                                 "n": int(mask.sum()), "n_paired": int(mask.sum()),
                                 "hit_pct": round(hit_rate(h_core[mask]), 2)})

            # gate test: Fisher's exact test miss x stratum (review objection: "hint, not evidence")
            lo = core_mass < S_GATE
            a = int((~h_core & lo).sum()); b = int((h_core & lo).sum())
            c_ = int((~h_core & ~lo).sum()); d = int((h_core & ~lo).sum())
            odds, pval = fisher_exact([[a, b], [c_, d]])
            n_miss = a + c_
            recall = 100.0 * a / n_miss if n_miss else float("nan")
            print(f"  Gate (mass<{S_GATE}): misses {a}/{a+b} low vs. {c_}/{c_+d} high | "
                  f"Fisher p={pval:.2e}, OR={odds:.1f} | gate recall of country misses {recall:.0f} %")
            rows.append({"population": label, "estimator": "gate_fisher", "n": len(cases),
                         "misses_total": n_miss, "misses_low": a, "n_low": a + b,
                         "misses_high": c_, "n_high": c_ + d, "fisher_p": float(pval),
                         "odds_ratio": (None if not np.isfinite(odds) else round(float(odds), 2)),
                         "gate_recall_pct": (None if not n_miss else round(recall, 1))})
            # miss profile: unanimous/majority consensus on the WRONG country?
            miss_idx = np.where(~h_core)[0]
            unanimous = int((maj[miss_idx] >= 0.999).sum())
            maj75 = int((maj[miss_idx] >= 0.75).sum())
            pairs = Counter(f"{truth[i]}>{core_cc[i] or '--'}" for i in miss_idx)
            top = ";".join(f"{k}:{v}" for k, v in pairs.most_common(6))
            print(f"  miss profile: {len(miss_idx)} misses, of which source majority>=75 % on another country "
                  f"{maj75}, unanimous {unanimous} | top truth>vote: {top}")
            rows.append({"population": label, "estimator": "miss_profile", "n": len(cases),
                         "misses_total": len(miss_idx), "miss_majority75": maj75,
                         "miss_unanimous": unanimous, "top_pairs": top})

            # single sources (paired) — main coding only (bootstrap is expensive)
            present = {s for c in cases for s in c["sources"]}
            for s in ([s for s in SOURCE_ORDER if s in present] if _NORMALIZE else []):
                idx, h_src = [], []
                for i, c in enumerate(cases):
                    if s in c["sources"]:
                        cc = _cc(lut.get((c["ip"], s), ("", ""))[0])
                        if cc:
                            idx.append(i); h_src.append(cc == truth[i])
                if len(idx) < 30:
                    continue
                idx = np.array(idx); h_src = np.array(h_src, float)
                res = paired_delta(h_core[idx].astype(float), h_src,
                                   [(g, lab[idx]) for g, lab in groups_list])
                sig_iid = not (res["iid_lo"] <= 0 <= res["iid_hi"])
                sig_asn = not (res["asn_lo"] <= 0 <= res["asn_hi"])
                extra = ""
                if "school_lo" in res:
                    extra = (f"  school [{res['school_lo']:+5.1f}, {res['school_hi']:+5.1f}]"
                             f"{'*' if not (res['school_lo'] <= 0 <= res['school_hi']) else ' '}")
                print(f"  {s:30s} n={len(idx):5d} | country hit {hit_rate(h_src):5.1f} % | "
                      f"Δ {res['delta_pp']:+5.1f} pp  iid [{res['iid_lo']:+5.1f}, {res['iid_hi']:+5.1f}]"
                      f"{'*' if sig_iid else ' '}  ASN [{res['asn_lo']:+5.1f}, {res['asn_hi']:+5.1f}]"
                      f"{'*' if sig_asn else ' '}{extra}")
                row = {"population": label, "estimator": s, "n": len(cases),
                       "n_paired": len(idx), "hit_pct": round(hit_rate(h_src), 2),
                       "agg_hit_pct_paired": round(hit_rate(h_core[idx]), 2)}
                row.update({k: (round(v, 2) if isinstance(v, float) else v)
                            for k, v in res.items()})
                row["sig_iid"] = sig_iid
                row["sig_asn"] = sig_asn
                rows.append(row)

        for r in rows[start:]:
            r["coding"] = coding

    df = pd.DataFrame(rows)
    df = df[["coding"] + [c for c in df.columns if c != "coding"]]
    df.to_csv(OUT / "country_level.csv", index=False)
    print(f"\nCSV: {OUT / 'country_level.csv'}")


if __name__ == "__main__":
    main()
