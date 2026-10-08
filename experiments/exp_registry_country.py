#!/usr/bin/env python3
"""Registry-country baseline for T10: is the "simple lookup" -- the country of
the organisation to which the registry delegated the address block -- as good
at country level as the aggregation over third-party databases?

Question (2026-09-27): choosing the legal instrument and the residence test
under Art. 8(2) Regulation (EU) 2023/1543 both require the country. The RIR
delegation is authoritative for the ADDRESSEE of an order (the provider's
seat). Whether it also hits the country of the USER is an empirical question --
RIPE states explicitly that the country field carries no geographic meaning.
T10 showed unanimous-but-wrong misses on the "registration/hosting country";
here we measure whether the registry is wrong in exactly those places as well
(same mechanism) or not.

Data: RIR Statistics Exchange Format files (delegated-*-extended) of all five
registries, evaluated OFFLINE (data/cache/rir/<YYYYMMDD>/, via
data/fetch_rir.py; no IP leaves the machine). Per population the snapshot from
the query day of the GeoIP observations (paired with T10), plus the most recent
snapshot (drift control: "does anything change?"). Blocks with status
available/reserved, or with an empty/"EU"/"ZZ" country field, yield no state
country -> counted as a registry miss (conservative, reported separately).

Two registry levels:
  (1) "Delegated" -- delegation file: country of the organisation to which the
      registry delegated the block (LIR/ISP seat). Offline, dated.
  (2) "Whois" -- RDAP, most specific object (what `whois <ip>` shows: the
      sub-allocation/assignment with its own country field; for ARIN the
      country line of the registrant address, for LACNIC the structured
      address). Cached via data/fetch_rdap_country.py; ONLY for the RIPE Atlas
      populations (Giga school IPs are not sent to further services -> n/a there).

Evaluation per population (coding "state" as in T10, STATE_OF normalisation on
both sides; "iso" raw as a control):
  1. Hit rate registry country vs. ground truth; paired delta aggregation
     (L1.b core vote, frozen anchor calibration) - registry in pp,
     iid + ASN-cluster bootstrap (Giga additionally school cluster), B=10,000.
  2. Cross-tabulation Agg x Registry (both right / agg only / registry only /
     both wrong) -> complementarity.
  3. Mechanism test: on agg misses -- does the (wrong) agg vote agree with the
     registry country? (Yes = the databases followed the registration.)
  4. "Reading the registry against the estimate" (claim of the application
     chapter): flag = registry country != agg vote. Recall of the agg misses,
     precision, flag rate; comparison with the S gate (core mass < 0.5) and the
     combination (flag OR gate).
  5. Drift: registry country on the query day vs. most recent snapshot -- share
     of changed countries per population, hit rate today.
  6. Hit rate per registry (ripencc/arin/apnic/lacnic/afrinic) -- both levels.
  7. Decomposition of the agg country errors (addendum 2026-09-27, "common mode"):
     (i) registry-following = vote == registry country, both wrong -> irreducible
     floor (by construction invisible to S AND to the flag), as % of all addresses;
     (ii) vote != registry, registry wrong as well; (iii) vote != registry, registry
     right. (ii)+(iii) = what the flag can catch.
  8. Lift of the flag: P(miss | registry!=vote) vs. P(miss | registry==vote) as a
     risk ratio, iid + ASN-cluster bootstrap CI (Giga additionally school).
  9. Flag x S gate (core mass < 0.5): four cells with n and miss rate -- basis of
     the frozen rate file maltego/registry_rates.json (maltego/make_registry_rates.py).

PRE-SPECIFIED (2026-09-27, before the paper numbers): primary flag = whois-object
country (estimator registry_rdap) != core vote; delegation = offline fallback;
coding "state"; reference population for the rates = probes_pool (anchors are
hosting-heavy); Giga without whois.

Usage:   GEOIP_DATASET=anchors python experiments/exp_registry_country.py
Output:  table (stdout) + eval/out/registry_country.csv (aggregated, IP-free)
"""
from __future__ import annotations

import bisect
import gzip
import ipaddress
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data import store                                     # noqa: E402
from eval.pipeline import load_cases                       # noqa: E402
import experiments.exp_t6_defaults as T6                   # noqa: E402
from experiments.exp_radius_transfer import eps_for        # noqa: E402
from experiments.exp_pool_transfer import restrict         # noqa: E402
from experiments.exp_cluster_bootstrap import make_groups, cluster_boot  # noqa: E402
import experiments.exp_country_level as CL                 # noqa: E402
from experiments.exp_country_level import (                # noqa: E402
    POPS, CACHE, OUT, S_GATE, city_lut_from, country_votes, paired_delta, hit_rate)

RIR_DIR = ROOT / "data" / "cache" / "rir"
RIRS = ("ripencc", "arin", "apnic", "lacnic", "afrinic")
# query day of the GeoIP observations per population (see data/fetch_rir.py)
POP_DATE = {"anchors": "20260604", "probes": "20260607", "probes_holdout": "20260801",
            "probes_mobile_ext": "20260801", "probes_pool": "20260820", "giga": "20260821"}
NO_STATE = {"", "EU", "ZZ", "AP"}     # no state that a legal instrument could be addressed to


sys.path.insert(0, str(ROOT / "maltego"))
from registry import Delegations, NAME_TO_ISO  # noqa: E402  (single implementation: maltego/registry.py)

RDAP_CACHE = ROOT / "data" / "cache" / "rdap_country_cache.jsonl"
_UNMAPPED: Counter = Counter()


def load_rdap() -> dict[str, tuple[str, str, str]]:
    """ip -> (cc_raw, type, registry) from the RDAP cache (most specific object;
    data/fetch_rdap_country.py). Country field of the network object, else the
    structured registrant address (LACNIC), else the country name of the address
    (ARIN) via NAME_TO_ISO."""
    import json
    out = {}
    if not RDAP_CACHE.exists():
        return out
    for line in open(RDAP_CACHE, encoding="utf-8"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("http_status") != 200:
            continue
        cc = (d.get("country") or "").strip().upper()
        if not cc:
            cc = (d.get("registrant_adr_cc") or "").strip().upper()
        if not cc:
            name = (d.get("registrant_adr_last") or "").strip().upper().rstrip(".")
            if len(name) == 2 and name.isalpha():
                cc = name
            elif name in NAME_TO_ISO:
                cc = NAME_TO_ISO[name]
            elif name:
                _UNMAPPED[name] += 1
        out[d["ip"]] = (cc, d.get("type") or "", d.get("rir") or "")
    return out


def state_cc(cc: str) -> str:
    """Country field -> state (STATE_OF normalisation as in T10), or '' if no state."""
    cc = (cc or "").strip().upper()
    if cc in NO_STATE:
        return ""
    return CL._cc(cc)


def confusion(h_a, h_r) -> dict:
    h_a = np.asarray(h_a, bool); h_r = np.asarray(h_r, bool)
    return {"both_hit": int((h_a & h_r).sum()), "agg_only": int((h_a & ~h_r).sum()),
            "reg_only": int((~h_a & h_r).sum()), "both_miss": int((~h_a & ~h_r).sum())}


def flag_stats(flag, miss) -> dict:
    flag = np.asarray(flag, bool); miss = np.asarray(miss, bool)
    caught = int((flag & miss).sum())
    return {"flag_n": int(flag.sum()), "flag_pct": round(100 * flag.mean(), 2),
            "misses": int(miss.sum()), "caught": caught,
            "recall_pct": (round(100 * caught / miss.sum(), 1) if miss.sum() else None),
            "precision_pct": (round(100 * caught / flag.sum(), 1) if flag.sum() else None)}


def risk_ratio(flag, miss, groups_list) -> dict:
    """P(miss|flag), P(miss|no flag), RR + iid and cluster-bootstrap CI (B, SEED as in T1).
    Degenerate draws (no flagged cases / no misses without flag) are discarded."""
    flag = np.asarray(flag, bool); miss = np.asarray(miss, bool)
    n = len(flag)

    def stat(idx):
        f = flag[idx]; m = miss[idx]
        if f.sum() == 0 or (~f).sum() == 0:
            return None
        p1 = m[f].mean(); p0 = m[~f].mean()
        if p0 == 0:
            return None
        return p1 / p0
    p1 = float(miss[flag].mean()) if flag.sum() else float("nan")
    p0 = float(miss[~flag].mean()) if (~flag).sum() else float("nan")
    out = {"p_miss_flag_pct": round(100 * p1, 3), "p_miss_noflag_pct": round(100 * p0, 3),
           "risk_ratio": (round(p1 / p0, 2) if p0 and np.isfinite(p1) else None)}
    if not (miss.sum() and flag.sum()):
        return out
    rng = np.random.default_rng(CL.SEED)
    vals = []
    for _ in range(CL.B):
        v = stat(rng.integers(0, n, n))
        if v is not None:
            vals.append(v)
    if vals:
        out["rr_iid_lo"] = round(float(np.percentile(vals, 2.5)), 2)
        out["rr_iid_hi"] = round(float(np.percentile(vals, 97.5)), 2)
    for gname, labels in groups_list:
        cb = cluster_boot(stat, make_groups(labels), CL.B)
        if len(cb):
            out[f"rr_{gname}_lo"] = round(float(np.percentile(cb, 2.5)), 2)
            out[f"rr_{gname}_hi"] = round(float(np.percentile(cb, 97.5)), 2)
    return out


def report_layer(tag, est, snapshot, reg_cc, h_agg, agg_cc, agg_mass, truth, groups_list,
                 coding, label, rows, extra=None) -> None:
    """Prints and records hit rate, paired delta, cross-tabulation, mechanism,
    flags and top pairs of one registry level against the aggregation."""
    n = len(truth)
    h_reg = (reg_cc == truth) & (reg_cc != "")
    miss = ~h_agg
    print(f"  {tag:30s} country hits {hit_rate(h_reg):5.1f} %  (no state country: {int((reg_cc == '').sum())})")
    res = paired_delta(h_agg.astype(float), h_reg.astype(float), groups_list)
    sig_iid = not (res["iid_lo"] <= 0 <= res["iid_hi"])
    sig_asn = not (res["asn_lo"] <= 0 <= res["asn_hi"])
    sch = ""
    if "school_lo" in res:
        sch = (f"  school [{res['school_lo']:+5.1f}, {res['school_hi']:+5.1f}]"
               f"{'*' if not (res['school_lo'] <= 0 <= res['school_hi']) else ' '}")
    print(f"    delta Agg - Registry {res['delta_pp']:+5.1f} pp  iid [{res['iid_lo']:+5.1f}, {res['iid_hi']:+5.1f}]"
          f"{'*' if sig_iid else ' '}  ASN [{res['asn_lo']:+5.1f}, {res['asn_hi']:+5.1f}]{'*' if sig_asn else ' '}{sch}")
    conf = confusion(h_agg, h_reg)
    print(f"    cross-tab: both right {conf['both_hit']}, agg only {conf['agg_only']}, "
          f"registry only {conf['reg_only']}, both wrong {conf['both_miss']}")
    same_as_reg = int((miss & (agg_cc == reg_cc) & (agg_cc != "")).sum())
    reg_right_there = int((miss & h_reg).sum())
    print(f"    mechanism: {int(miss.sum())} agg misses, of which agg vote == registry country "
          f"{same_as_reg}; registry right there {reg_right_there}")
    flag_reg = (reg_cc != agg_cc)
    flag_gate = agg_mass < S_GATE
    f_r = flag_stats(flag_reg, miss); f_g = flag_stats(flag_gate, miss); f_c = flag_stats(flag_reg | flag_gate, miss)
    for nm, f in (("flag registry!=vote", f_r), ("gate core mass<0.5", f_g), ("flag OR gate", f_c)):
        print(f"    {nm:22s} flagged {f['flag_n']:5d} ({f['flag_pct']:5.2f} %) | misses caught "
              f"{f['caught']:3d}/{f['misses']:3d} = recall {f['recall_pct']} % | precision {f['precision_pct']} %")
    # decomposition of the agg misses (common mode) + lift + flag x gate
    cm = miss & ~flag_reg                       # vote == registry (both wrong)
    dev_wrong = miss & flag_reg & ~h_reg        # vote != registry, registry wrong as well
    dev_right = miss & flag_reg & h_reg         # vote != registry, registry right
    rr = risk_ratio(flag_reg, miss, groups_list)
    print(f"    common mode: registry-following {int(cm.sum())} ({100*cm.mean():.2f} % of all addresses, floor) | "
          f"deviating & registry wrong {int(dev_wrong.sum())} | deviating & registry right {int(dev_right.sum())}")
    print(f"    lift: P(miss|flag) {rr['p_miss_flag_pct']:.2f} % vs. P(miss|no flag) {rr['p_miss_noflag_pct']:.2f} % "
          f"-> RR {rr.get('risk_ratio')}  iid [{rr.get('rr_iid_lo')}, {rr.get('rr_iid_hi')}]  "
          f"ASN [{rr.get('rr_asn_lo')}, {rr.get('rr_asn_hi')}]"
          + (f"  school [{rr.get('rr_school_lo')}, {rr.get('rr_school_hi')}]" if 'rr_school_lo' in rr else ""))
    cells = {}
    for fname, fm in (("flag", flag_reg), ("noflag", ~flag_reg)):
        for gname, gm in (("lowmass", flag_gate), ("highmass", ~flag_gate)):
            m = fm & gm
            cells[f"{fname}_{gname}_n"] = int(m.sum())
            cells[f"{fname}_{gname}_miss_pct"] = round(100 * miss[m].mean(), 3) if m.sum() else None
    print("    flag x gate (n / miss rate): " + "  ".join(
        f"{k[:-2]}={cells[k]}/{cells[k[:-2] + '_miss_pct']}%" for k in cells if k.endswith("_n")))
    pairs = Counter(f"{truth[i]}>{reg_cc[i] or '--'}" for i in np.where(~h_reg)[0])
    top = ";".join(f"{k}:{v}" for k, v in pairs.most_common(6))
    print(f"    registry misses, top truth>registry: {top}")
    rows.append({"coding": coding, "population": label, "estimator": f"{est}__common_mode", "n": n,
                 "agg_miss_n": int(miss.sum()), "cm_registry_following_n": int(cm.sum()),
                 "cm_registry_following_pct_of_n": round(100 * cm.mean(), 3),
                 "dev_registry_wrong_n": int(dev_wrong.sum()), "dev_registry_right_n": int(dev_right.sum())})
    rows.append({"coding": coding, "population": label, "estimator": f"{est}__flag_lift", "n": n,
                 "flag_n": int(flag_reg.sum()), **rr})
    rows.append({"coding": coding, "population": label, "estimator": f"{est}__flag_x_gate", "n": n, **cells})
    row = {"coding": coding, "population": label, "estimator": est, "n": n, "snapshot": snapshot,
           "hit_pct": round(hit_rate(h_reg), 2), "no_state_n": int((reg_cc == "").sum()),
           "agg_hit_pct_paired": round(hit_rate(h_agg), 2), **conf,
           "agg_miss_n": int(miss.sum()), "agg_miss_vote_eq_registry": same_as_reg,
           "agg_miss_registry_right": reg_right_there, "top_pairs": top}
    row.update({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()})
    row["sig_iid"] = sig_iid; row["sig_asn"] = sig_asn
    if extra:
        row.update(extra)
    rows.append(row)
    for nm, f in (("flag_ne_vote", f_r), ("gate_low_mass", f_g), ("flag_or_gate", f_c)):
        rows.append({"coding": coding, "population": label, "estimator": f"{est}__{nm}", "n": n, **f})
    return h_reg


def main() -> None:
    if os.environ.get("GEOIP_DATASET", "anchors") != "anchors":
        sys.exit("exp_registry_country: please run with GEOIP_DATASET=anchors.")

    dates_needed = sorted(set(POP_DATE.values()))
    latest = sorted(d.name for d in RIR_DIR.iterdir() if d.is_dir() and d.name.isdigit())[-1]
    print(f"RIR snapshots: {', '.join(dates_needed)}; most recent = {latest}")
    deleg = {d: Delegations(d) for d in dates_needed + ([latest] if latest not in dates_needed else [])}
    for d, D in deleg.items():
        print(f"  {d}: {len(D.starts):,} IPv4 blocks | " +
              " ".join(f"{r}:{D.headers.get(r, '?').split('|')[2] if '|' in D.headers.get(r, '') else '?'}"
                       for r in RIRS))
    D_latest = deleg[latest]

    anchors9 = load_cases()
    loo_a = T6.loo_pseudo_radii(anchors9)
    eps_a = eps_for(loo_a)
    print(f"Anchor calibration frozen: eps={eps_a} km; R_CORE={CL.R_CORE} km; "
          f"S gate={S_GATE}; B={CL.B}")

    rdap = load_rdap()
    print(f"RDAP cache (most specific object): {len(rdap):,} IPs"
          + (f"; unmapped country names: {_UNMAPPED.most_common(8)}" if _UNMAPPED else ""))

    rows = []
    for coding in CL.CODINGS:
        CL._NORMALIZE = (coding == "state")
        print("\n" + "#" * 78)
        print(f"# coding: {coding}")
        print("#" * 78)
        for label, ds, obs_name, srcs in POPS:
            cases = anchors9 if ds == "anchors" else load_cases(
                store.load_anchors_csv(CACHE / f"{ds}.csv"),
                store.load_observations_csv(CACHE / obs_name))
            if srcs is not None:
                cases = restrict(cases, srcs)
                for c in cases:
                    c["sources"] = [p["source"] for p in c["provenance"]]
            lut = city_lut_from(CACHE / obs_name)
            truth = np.array([CL._cc(c["country"]) for c in cases])
            asn = np.array([str(c.get("asn") or c["ip"]) for c in cases])
            groups_list = [("asn", asn)]
            if ds == "giga":
                school_of = {r["ip"]: (r.get("anchor_id") or r["ip"])
                             for r in store.load_anchors_csv(CACHE / "giga.csv")}
                groups_list.append(("school", np.array([school_of.get(c["ip"], c["ip"]) for c in cases])))

            # aggregation (as in T10)
            agg_cc, agg_mass = [], []
            for c in cases:
                a, m, _, _ = country_votes(c, lut, loo_a, eps_a)
                agg_cc.append(a); agg_mass.append(m)
            agg_cc = np.array(agg_cc); agg_mass = np.array(agg_mass)
            h_agg = (agg_cc == truth) & (agg_cc != "")

            # registry on the query day + today
            D = deleg[POP_DATE[label]]
            reg = [D.lookup(c["ip"]) for c in cases]
            reg_cc = np.array([state_cc(r[0]) for r in reg])
            reg_rir = np.array([r[1] or "none" for r in reg])
            reg_status = np.array([r[2] for r in reg])
            reg_now = [D_latest.lookup(c["ip"]) for c in cases]
            reg_now_cc = np.array([state_cc(r[0]) for r in reg_now])
            h_reg_now = (reg_now_cc == truth) & (reg_now_cc != "")
            changed = int((reg_now_cc != reg_cc).sum())

            n = len(cases)
            print("\n" + "=" * 78)
            print(f"{label} (n={n}, delegation snapshot {D.date}, today {latest}; "
                  f"status: {dict(Counter(reg_status.tolist()))})")
            print("=" * 78)
            print(f"  {'Aggregation L1.b core vote':30s} country hits {hit_rate(h_agg):5.1f} %")
            rows.append({"coding": coding, "population": label, "estimator": "agg_core_vote", "n": n,
                         "hit_pct": round(hit_rate(h_agg), 2)})

            # level 1: delegated (delegation file, query day)
            h_reg = report_layer("Registry delegated (query day)", "registry_delegated", D.date, reg_cc,
                                 h_agg, agg_cc, agg_mass, truth, groups_list, coding, label, rows)
            print(f"  {'Registry delegated (today)':30s} country hits {hit_rate(h_reg_now):5.1f} %  "
                  f"| country changed since query day: {changed} ({100*changed/n:.2f} %)")
            rows.append({"coding": coding, "population": label, "estimator": "registry_delegated_latest",
                         "n": n, "snapshot": latest, "hit_pct": round(hit_rate(h_reg_now), 2),
                         "changed_n": changed, "changed_pct": round(100 * changed / n, 3)})
            per_rir = []
            for r in RIRS:
                m = reg_rir == r
                if m.sum():
                    per_rir.append(f"{r}:{int(m.sum())}/{hit_rate(h_reg[m]):.1f}%/agg {hit_rate(h_agg[m]):.1f}%")
                    rows.append({"coding": coding, "population": label, "estimator": f"registry_delegated_{r}",
                                 "n": int(m.sum()), "hit_pct": round(hit_rate(h_reg[m]), 2),
                                 "agg_hit_pct_paired": round(hit_rate(h_agg[m]), 2)})
            print(f"    per registry (n/delegated hits/agg hits): {'  '.join(per_rir)}")

            # level 2: whois/RDAP most specific object (only where cached)
            have = [c["ip"] in rdap for c in cases]
            cov = 100.0 * np.mean(have)
            if cov >= 95.0:
                rd = [rdap.get(c["ip"], ("", "", "")) for c in cases]
                rdap_cc = np.array([state_cc(r[0]) for r in rd])
                rdap_type = np.array([r[1].upper() for r in rd])
                h_rd = report_layer(f"Registry whois/RDAP ({cov:.1f} % covered)", "registry_rdap", "rdap_cache",
                                    rdap_cc, h_agg, agg_cc, agg_mass, truth, groups_list, coding, label, rows,
                                    extra={"coverage_pct": round(cov, 2)})
                rdap_rir = np.array([(r[2] or "none") for r in rd])
                per = []
                for r_ in RIRS:
                    m = rdap_rir == r_
                    if m.sum():
                        per.append(f"{r_}:{int(m.sum())}/{hit_rate(h_rd[m]):.1f}%/agg {hit_rate(h_agg[m]):.1f}%")
                        rows.append({"coding": coding, "population": label, "estimator": f"registry_rdap_{r_}",
                                     "n": int(m.sum()), "hit_pct": round(hit_rate(h_rd[m]), 2),
                                     "agg_hit_pct_paired": round(hit_rate(h_agg[m]), 2)})
                print(f"    per registry (n/whois hits/agg hits): {'  '.join(per)}")
                # finer than the delegation? (object type) and delta between the levels
                finer = rdap_cc != reg_cc
                print(f"    whois != delegated for {int(finer.sum())} IPs; of which whois right "
                      f"{int((finer & h_rd).sum())}, delegated right {int((finer & h_reg).sum())} | "
                      f"object types: {dict(Counter(rdap_type.tolist()).most_common(6))}")
                rows.append({"coding": coding, "population": label, "estimator": "rdap_vs_delegated", "n": n,
                             "differs_n": int(finer.sum()), "rdap_right_n": int((finer & h_rd).sum()),
                             "delegated_right_n": int((finer & h_reg).sum())})
            else:
                print(f"  {'Registry whois/RDAP':30s} n/a (cache coverage {cov:.1f} %"
                      + ("; Giga deliberately not queried" if ds == "giga" else "") + ")")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "registry_country.csv", index=False)
    print(f"\nCSV: {OUT / 'registry_country.csv'}")


if __name__ == "__main__":
    main()
