#!/usr/bin/env python3
"""Giga mechanism diagnosis: WHY are the Global-South errors so large?

Backs the mechanism claims of the T9 section with numbers instead of
plausibility (natural follow-up question: "CGNAT etc. — do we have
evidence for that?").

Four pieces of evidence per country (n >= 30 pairs):
  1. TOP ASNs + share (from giga.csv): operator concentration.
  2. SHARED IPs before filter 4 (IP at >1 school within the measurement
     window, reconstructed from the raw measurement cache): direct indicator
     of shared NAT/CGNAT egress. FINDING: high only in Fiji (~33 %), low in
     South Africa (~5 %) — CGNAT egress is NOT the main driver of the ZA
     errors.
  3. LOCATION CONCENTRATION of the estimates: unique source coordinates
     (0.01-degree grid) vs. unique school locations. FINDING: sources place
     entire operator blocks onto a few hub cities (ZA: ~20 estimate
     locations for 86 school locations; UZ: 3-7 for 256). The error is the
     distance subscriber<->hub — it scales with country size and operator
     centralization, not with the access technology alone (UZ = fixed-line
     incumbent shows the same pattern; AL = small country also has few
     estimate locations, but small errors). This is the block-geolocation
     mechanic from Nabi et al. (arXiv:2605.21937); CGNAT prevalence in
     mobile networks: Richter et al., IMC 2016.
  4. REFERENCE ERROR of the measurement's own GeoIP estimate
     (ClientInfo.Lat/Lon from the NDT client): sees the same errors -> not a
     pipeline artifact. (NOT an independent line — same DB families; only a
     diagnostic reference.)

Optional (--fetch-flags): ip-api access-type flags (mobile/hosting) per IP,
45 req/min, dedicated cache data/cache/giga_mobile_flags.json (independent
of the pipeline cache; flag quality per exp_s_type_routing: P 0.92 / R 0.50
-> the mobile share is a LOWER BOUND; fixed-wireless providers such as Rain
(ZA) are reported as "fixed" but are radio access).

Additional table FLAG x COUNTRY (giga_mechanism_flags.csv, frozen errors):
FINDING: pooled, mobile looks dramatically
worse (237 vs. 71 km median) — WITHIN-COUNTRY it flips, however (ZA:
mobile 406 < fixed 566 < hosting 663; KZ: mobile 167 < fixed 243). The
pooled gap is country composition; the driver is the block granularity of
entire national address spaces — the access technology is correlated, not
causal.

Usage:  python experiments/exp_giga_mechanism.py [--fetch-flags]
Result: table (stdout) + eval/out/giga_mechanism.csv
Data: ODbL — Source: Giga (UNICEF) and contributors.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.fetch_giga import extract_ip                      # noqa: E402

CACHE = Path("data/cache")
OUT = Path("eval/out")
OUT.mkdir(parents=True, exist_ok=True)
FLAGS_CACHE = CACHE / "giga_mobile_flags.json"
MIN_N = 30
GRID = 2  # rounding of the location grid to 0.01 degrees


def hav(a, b):
    R = 6371.0088
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = math.radians(b[0] - a[0]), math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def fetch_flags(ips: list[str]) -> dict[str, str]:
    """ip-api access type per IP (mobile/hosting/fixed), with persistent cache."""
    cache: dict[str, str] = {}
    if FLAGS_CACHE.exists():
        cache = json.loads(FLAGS_CACHE.read_text(encoding="utf-8"))
    todo = [ip for ip in ips if ip not in cache]
    if todo:
        print(f"  ip-api flags: {len(todo)} IPs pending (~{len(todo)*1.4/60:.0f} min at 45/min) …")
    for i, ip in enumerate(todo, 1):
        url = f"http://ip-api.com/json/{ip}?fields=status,mobile,hosting"
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                d = json.load(r)
            if d.get("status") == "success":
                cache[ip] = ("mobile" if d.get("mobile")
                             else "hosting" if d.get("hosting") else "fixed")
        except OSError:
            pass
        if i % 200 == 0:
            FLAGS_CACHE.write_text(json.dumps(cache, indent=0), encoding="utf-8")
            print(f"    … {i}/{len(todo)}")
        time.sleep(1.4)
    FLAGS_CACHE.write_text(json.dumps(cache, indent=0), encoding="utf-8")
    return cache


def main() -> None:
    do_flags = "--fetch-flags" in sys.argv

    giga = list(csv.DictReader(open(CACHE / "giga.csv", encoding="utf-8")))
    by_cc = defaultdict(list)
    for r in giga:
        by_cc[r["country"]].append(r)
    countries = sorted(c for c, rs in by_cc.items() if len(rs) >= MIN_N)

    # --- Evidence 2: shared IPs before filter 4 (from the raw measurement cache) ---
    ip_schools, ips_by_cc = defaultdict(set), defaultdict(set)
    raw = CACHE / f"giga_measurements_raw_giga.jsonl"
    with open(raw, encoding="utf-8") as fh:
        for line in fh:
            for m in json.loads(line)["data"]:
                ip, _ = extract_ip((m.get("ClientInfo") or {}).get("Hostname"))
                gid = m.get("giga_id_school")
                if ip and gid:
                    ip_schools[ip].add(gid)
                    ips_by_cc[m.get("country_code")].add(ip)

    # --- Evidence 3/4: estimate locations + ClientInfo reference ---
    obs = defaultdict(dict)
    with open(CACHE / "observations_giga.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["status"] == "success" and r["lat"]:
                obs[r["source"]][r["ip"]] = (round(float(r["lat"]), GRID),
                                             round(float(r["lon"]), GRID))
    ci_geo = {}
    with open(raw, encoding="utf-8") as fh:
        for line in fh:
            for m in json.loads(line)["data"]:
                ci = m.get("ClientInfo") or {}
                hn = ci.get("Hostname")
                if hn and ci.get("Latitude") is not None:
                    ci_geo.setdefault(hn, (ci["Latitude"], ci["Longitude"]))

    flags = fetch_flags([r["ip"] for r in giga]) if do_flags else (
        json.loads(FLAGS_CACHE.read_text(encoding="utf-8")) if FLAGS_CACHE.exists() else {})

    rows = []
    for cc in countries:
        rs = by_cc[cc]
        n = len(rs)
        asns = Counter(r["asn"] for r in rs)
        top_asn, top_n = asns.most_common(1)[0]
        pre_ips = ips_by_cc.get(cc, set())
        shared = sum(1 for ip in pre_ips if len(ip_schools[ip]) > 1)
        school_locs = {(round(float(r["lat"]), GRID), round(float(r["lon"]), GRID)) for r in rs}
        est_locs = {obs["maxmind_geolite2"][r["ip"]] for r in rs
                    if r["ip"] in obs["maxmind_geolite2"]}
        ci_err = sorted(hav((float(r["lat"]), float(r["lon"])), ci_geo[r["hostname"]])
                        for r in rs if r["hostname"] in ci_geo)
        fl = Counter(flags.get(r["ip"]) for r in rs if r["ip"] in flags)
        n_fl = sum(fl.values())
        row = {
            "country": cc, "n_pairs": n,
            "top_asn": f"AS{top_asn}", "top_asn_share_pct": round(100 * top_n / n, 1),
            "shared_ip_pct_prefilter": round(100 * shared / len(pre_ips), 1) if pre_ips else None,
            "est_locations_maxmind": len(est_locs), "school_locations": len(school_locs),
            "clientinfo_ref_median_km": round(ci_err[len(ci_err) // 2], 1) if ci_err else None,
            "mobile_flag_pct": round(100 * fl.get("mobile", 0) / n_fl, 1) if n_fl else None,
            "hosting_flag_pct": round(100 * fl.get("hosting", 0) / n_fl, 1) if n_fl else None,
        }
        rows.append(row)
        print(f"{cc}: n={n:4d} | top {row['top_asn']} {row['top_asn_share_pct']}% | "
              f"shared(pre-F4) {row['shared_ip_pct_prefilter']}% | "
              f"est. locations {row['est_locations_maxmind']} vs. school locations {row['school_locations']} | "
              f"ClientInfo ref {row['clientinfo_ref_median_km']} km | "
              f"mobile {row['mobile_flag_pct']}% / hosting {row['hosting_flag_pct']}%")

    pd.DataFrame(rows).to_csv(OUT / "giga_mechanism.csv", index=False)
    print(f"\nCSV: {OUT / 'giga_mechanism.csv'}")

    # --- Flag x country: frozen errors per access type WITHIN countries ---
    if flags:
        import numpy as np
        from data import store as _store                       # noqa: PLC0415
        from eval.pipeline import load_cases                   # noqa: PLC0415
        import experiments.exp_t6_defaults as T6               # noqa: PLC0415
        from experiments.exp_radius_transfer import eps_for    # noqa: PLC0415
        from experiments.exp_pool_transfer import (            # noqa: PLC0415
            restrict, errors, SEVEN,
        )
        anchors = load_cases()
        loo_a = T6.loo_pseudo_radii(anchors)
        cases = restrict(load_cases(
            _store.load_anchors_csv(CACHE / "giga.csv"),
            _store.load_observations_csv(CACHE / "observations_giga.csv")),
            SEVEN + ("ipwho_is",))
        e = errors(cases, loo_a, eps_for(loo_a))
        cc_arr = np.array([c["country"] for c in cases])
        fl_arr = np.array([flags.get(c["ip"], "unknown") for c in cases])
        frows = []
        print("\nFlag x country (frozen, n >= 15):")
        for land in countries + ["ALL"]:
            for f in ("mobile", "fixed", "hosting"):
                m = (fl_arr == f) if land == "ALL" else ((cc_arr == land) & (fl_arr == f))
                if m.sum() < 15:
                    continue
                frows.append({"country": land, "flag": f, "n": int(m.sum()),
                              "median_km": round(float(np.median(e[m])), 1),
                              "tail_pct": round(float(100 * np.mean(e[m] > 100)), 1)})
                print(f"  {land:<5} {f:<8} n={m.sum():5d} | "
                      f"median {np.median(e[m]):7.1f} | tail {100*np.mean(e[m]>100):5.1f} %")
        pd.DataFrame(frows).to_csv(OUT / "giga_mechanism_flags.csv", index=False)
        print(f"CSV: {OUT / 'giga_mechanism_flags.csv'}")
        print("Reading: the pooled mobile gap is country composition; "
              "within-country it flips/shrinks (ZA: mobile < fixed < hosting).")
    print("Reading: error ~ distance subscriber<->block hub (location concentration), "
          "not access technology alone;\nCGNAT egress (shared IPs) dominant only in Fiji; "
          "mobile share = lower bound (flag recall 0.50).")
    print("Data: ODbL — Source: Giga (UNICEF) and contributors.")


if __name__ == "__main__":
    main()
