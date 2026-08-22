#!/usr/bin/env python3
"""Giga transfer: L1·b + S on UNICEF Giga school measurements (end users, Global South).

The dataset (data/fetch_giga.py; nabi filter chain, PTR-verified IP
extraction from client hostnames) is the END-USER validation of the
transfer: 2,127 (IP, school) pairs, 19 countries, school GPS as
GeoIP-INDEPENDENT ground truth (government data — no circularity risk as
with system-auto-geoip probes, but its own GT uncertainty: access point ≠
school location, coordinates from ministries of education).

Sources: the EIGHT fully collected ones (7 pool sources + ipwho.is via
pro quota) = 6 effective lines; ipapi.co remains Cloudflare-blocked.
reallyfreegeoip coverage is reported (server-side residual error set);
missing rfg responses leave the MaxMind line intact via
maxmind_geolite2+geojs.

Design (pre-specified, analogous to exp_pool_transfer 2026-08-20):
  1. FROZEN AS MAIN COLUMN: anchor calibration (LOO radii + ε) applied
     unchanged; fresh refitting only as a control.
  2. SOURCE CONFOUND CONTROL: anchor reference restricted to the same
     8 sources.
  3. SELECTION BIAS PROBE: stratification by IP extraction method
     (method-literal = no PTR present vs. method-rdns = PTR-encoded,
     back-verified). Similar errors in both strata = the ISP selection
     of the rDNS extraction does not carry the results.
  4. CHURN PROBE: stratification by PAIR AGE (last Giga measurement ->
     earliest source fetch; data/cache/giga_pair_times.csv). If the
     error does not grow with age, DHCP churn does not bite within the
     window.
  5. ASN CLUSTER BOOTSTRAP for the core numbers (cluster = ASN,
     B = 10,000): the rDNS selection concentrates the dataset on a few
     providers — iid CIs would be dishonest here.

S transfer: as in exp_pool_transfer (anchor logit raw + Platt OOF on Giga).

Data: ODbL — source: Giga (UNICEF) and contributors.

Usage:  python experiments/exp_giga_transfer.py
Result: tables (stdout) + eval/out/giga_transfer.csv
        + eval/out/giga_transfer_strata.csv
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import store                                     # noqa: E402
from eval.pipeline import load_cases                       # noqa: E402
import experiments.exp_t6_defaults as T6                   # noqa: E402
from experiments.exp_radius_transfer import eps_for        # noqa: E402
from experiments.exp_pool_transfer import (                # noqa: E402
    restrict, errors, s_values, stats, fit_logit, apply_logit, platt_oof,
    TAU, SEVEN,
)
from experiments.exp_support_concentration import bss, ece, K as KFOLD  # noqa: E402

CACHE = Path("data/cache")
OUT = Path("eval/out")
OUT.mkdir(parents=True, exist_ok=True)
EIGHT = SEVEN + ("ipwho_is",)
B, SEED = 10_000, 0


def _parse_ts(s: str):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, AttributeError):
        return None


def load_giga_tags() -> dict[str, set]:
    tags: dict[str, set] = {}
    with open(CACHE / "giga_tags.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            tags[r["ip"]] = set(r["tags"].split(";"))
    return tags


def load_pair_ages() -> dict[str, float]:
    """IP -> pair age in days (last measurement -> earliest source fetch)."""
    times_path = CACHE / "giga_pair_times.csv"
    if not times_path.exists():
        return {}
    fetch_by_ip: dict[str, datetime] = {}
    with open(CACHE / "observations_giga.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ts = _parse_ts(r.get("fetched_at_utc", ""))
            if ts:
                prev = fetch_by_ip.get(r["ip"])
                fetch_by_ip[r["ip"]] = min(prev, ts) if prev else ts
    ages = {}
    with open(times_path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            last, fetch = _parse_ts(r.get("last_created_at", "")), fetch_by_ip.get(r["ip"])
            if last and fetch:
                ages[r["ip"]] = (fetch - last).total_seconds() / 86400.0
    return ages


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


def main() -> None:
    # ---- Anchor calibration (radii from 9 sources, fit on the 8 Giga sources) ----
    anchors9 = load_cases()
    loo_a = T6.loo_pseudo_radii(anchors9)
    eps_a = eps_for(loo_a)
    anchors8 = restrict(anchors9, EIGHT)
    e_a8 = errors(anchors8, loo_a, eps_a)
    S_a8 = s_values(anchors8, loo_a, eps_a)
    y_a8 = (e_a8 > TAU).astype(float)
    mu_a, sd_a, w_a = fit_logit(S_a8, y_a8)
    print(f"Anchor calibration frozen: eps={eps_a} km; "
          f"anchors on 8 sources: {stats(e_a8)}")

    # ---- Giga population ----
    cases = restrict(load_cases(
        store.load_anchors_csv(CACHE / "giga.csv"),
        store.load_observations_csv(CACHE / "observations_giga.csv")), EIGHT)
    n_rfg = sum(any(p["source"] == "reallyfreegeoip" for p in c["provenance"])
                for c in cases)
    loo_g = T6.loo_pseudo_radii(cases)
    eps_g = eps_for(loo_g)
    e_froz = errors(cases, loo_a, eps_a)
    e_fresh = errors(cases, loo_g, eps_g)
    asn = np.array([str(c.get("asn")) for c in cases])

    rows, strata_rows = [], []
    print("\n" + "=" * 78)
    print(f"giga (n={len(cases)}, rfg coverage {n_rfg}/{len(cases)} "
          f"= {100*n_rfg/len(cases):.1f} %)")
    print("=" * 78)
    for name, e in [("frozen", e_froz), ("fresh", e_fresh)]:
        s = stats(e)
        print(f"  L1·b {name:12s}: median {s['median_km']:6.2f} | "
              f"mean {s['mean_km']:6.1f} | tail>100 {s['tail_pct']:4.1f} %")
        rows.append({"population": "giga", "config": name, **s,
                     "rfg_coverage_pct": round(100*n_rfg/len(cases), 1)})

    # ---- S transfer ----
    S_g = s_values(cases, loo_a, eps_a)
    y_g = (e_froz > TAU).astype(float)
    ph_raw = apply_logit(S_g, mu_a, sd_a, w_a)
    ph_platt = platt_oof(S_g, y_g)
    print(f"  S transfer: raw BSS {bss(ph_raw, y_g):+.3f} (ECE {ece(ph_raw, y_g):.3f})"
          f" | Platt OOF BSS {bss(ph_platt, y_g):+.3f} (ECE {ece(ph_platt, y_g):.3f})"
          f" | miss rate {y_g.mean():.3f}")
    rows.append({"population": "giga", "config": "S_transfer",
                 "bss_raw": round(float(bss(ph_raw, y_g)), 3),
                 "bss_platt_oof": round(float(bss(ph_platt, y_g)), 3),
                 "ece_raw": round(float(ece(ph_raw, y_g)), 3),
                 "ece_platt_oof": round(float(ece(ph_platt, y_g)), 3),
                 "missrate": round(float(y_g.mean()), 3), "n": len(cases)})

    # ---- Platt OOF group-sensitive control: randomly
    # stratified folds can place pairs of the same school / the same ASN in
    # both train and test; the refit has only 2 parameters, but the effect is
    # measured here rather than asserted (group CV by school and by ASN).
    school_of: dict[str, str] = {}
    with open(CACHE / "giga.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            school_of[r["ip"]] = r["anchor_id"]
    sch = np.array([school_of.get(c["ip"], c["ip"]) for c in cases])

    def platt_oof_grouped(S, y, groups):
        rng_g = np.random.default_rng(SEED)
        ug = rng_g.permutation(np.unique(groups))
        fold_of_g = {g: i % KFOLD for i, g in enumerate(ug)}
        F = np.array([fold_of_g[g] for g in groups])
        ph = np.empty(len(y))
        for fi in range(KFOLD):
            tr, te = F != fi, F == fi
            mu, sd, w = fit_logit(S[tr], y[tr])
            ph[te] = apply_logit(S[te], mu, sd, w)
        return ph

    # Group bootstrap CI: point estimates around +0.05 alone do not separate
    # from zero. Draw groups with replacement (each drawn group = its own
    # cluster) and repeat the full grouped OOF Platt computation per
    # replicate; separate RNG so all existing random sequences (fold
    # assignment, ASN cboot) stay unchanged.
    B_GRP = 2_000

    def platt_group_bss_ci(S, y, groups):
        rng_b = np.random.default_rng(SEED)
        ug = np.unique(groups)
        idx_of_g = {g: np.where(groups == g)[0] for g in ug}
        out = np.empty(B_GRP)
        for b in range(B_GRP):
            pick = rng_b.choice(ug, size=len(ug), replace=True)
            idx = [idx_of_g[g] for g in pick]
            Sb = np.concatenate([S[i] for i in idx])
            yb = np.concatenate([y[i] for i in idx])
            gb = np.concatenate([np.full(len(i), k) for k, i in enumerate(idx)])
            out[b] = bss(platt_oof_grouped(Sb, yb, gb), yb)
        return np.percentile(out, [2.5, 97.5])

    for gname, grp in (("asn", asn), ("school", sch)):
        ph_grp = platt_oof_grouped(S_g, y_g, grp)
        print(f"  Platt OOF {gname}-grouped: BSS {bss(ph_grp, y_g):+.3f} "
              f"(ECE {ece(ph_grp, y_g):.3f})")
        rows.append({"population": "giga",
                     "config": f"S_transfer_platt_group_{gname}",
                     "bss_platt_oof": round(float(bss(ph_grp, y_g)), 3),
                     "ece_platt_oof": round(float(ece(ph_grp, y_g)), 3),
                     "n": len(cases)})
        lo, hi = platt_group_bss_ci(S_g, y_g, grp)
        print(f"    group bootstrap 95% CI (B={B_GRP}, cluster={gname}): "
              f"BSS [{lo:+.3f}, {hi:+.3f}]")
        rows.append({"population": "giga",
                     "config": f"S_transfer_platt_group_{gname}_ci",
                     "bss_platt_oof": round(float(bss(ph_grp, y_g)), 3),
                     "ci_lo": round(float(lo), 3), "ci_hi": round(float(hi), 3),
                     "B": B_GRP, "n": len(cases)})

    # ---- ASN cluster bootstrap of the core numbers ----
    counts = pd.Series(asn).value_counts()
    print(f"\n  ASN structure: {len(counts)} ASNs | median "
          f"{int(counts.median())} pair(s)/ASN | top ASN {counts.iloc[0]} "
          f"({100*counts.iloc[0]/len(cases):.1f} %) | "
          f"top 10 {100*counts.iloc[:10].sum()/len(cases):.1f} %")
    rng = np.random.default_rng(SEED)
    uasn = np.unique(asn)
    idx_of = {a: np.where(asn == a)[0] for a in uasn}

    def cboot(stat):
        out = np.empty(B)
        for b in range(B):
            pick = rng.choice(uasn, size=len(uasn), replace=True)
            idx = np.concatenate([idx_of[a] for a in pick])
            out[b] = stat(idx)
        return np.percentile(out, [2.5, 97.5])

    for name, stat, point in [
            ("median_km", lambda i: np.median(e_froz[i]), float(np.median(e_froz))),
            ("tail_pct", lambda i: 100 * np.mean(e_froz[i] > TAU),
             float(100 * np.mean(e_froz > TAU))),
            ("delta_median_frozen_minus_fresh_km",
             lambda i: np.median(e_froz[i]) - np.median(e_fresh[i]),
             float(np.median(e_froz) - np.median(e_fresh)))]:
        lo, hi = cboot(stat)
        print(f"  {name:38s} {point:8.2f}  ASN 95% CI [{lo:+.2f}, {hi:+.2f}]")
        rows.append({"population": "giga", "config": f"asn_ci_{name}",
                     "point": round(point, 3), "ci_lo": round(float(lo), 3),
                     "ci_hi": round(float(hi), 3), "n": len(cases)})

    # ---- Stratum A: extraction method (selection bias probe) ----
    tags = load_giga_tags()
    meth = np.array(["literal" if "method-literal" in tags.get(c["ip"], set())
                     else "rdns" for c in cases])
    print("\n  Stratification by extraction method (frozen):")
    for mname in ("literal", "rdns"):
        m = meth == mname
        s = stats(e_froz[m])
        print(f"    method-{mname:8s}: n={s['n']:5d} | median {s['median_km']:6.2f} "
              f"| mean {s['mean_km']:6.1f} | tail {s['tail_pct']:4.1f} %")
        strata_rows.append({"stratum": f"method-{mname}", **s})
    m_lit = meth == "literal"
    lo, hi = cboot(lambda i: np.median(e_froz[i][m_lit[i]])
                   - np.median(e_froz[i][~m_lit[i]]))
    gap = float(np.median(e_froz[m_lit]) - np.median(e_froz[~m_lit]))
    print(f"    Median difference literal−rdns: {gap:+.2f} km  "
          f"ASN 95% CI [{lo:+.2f}, {hi:+.2f}]")
    strata_rows.append({"stratum": "gap_literal_minus_rdns_median_km",
                        "median_km": round(gap, 3), "ci_lo": round(float(lo), 3),
                        "ci_hi": round(float(hi), 3)})

    # ---- Stratum B: pair age (churn probe) ----
    ages_by_ip = load_pair_ages()
    if ages_by_ip:
        age = np.array([ages_by_ip.get(c["ip"], np.nan) for c in cases])
        ok = ~np.isnan(age)
        rho = spearman(age[ok], e_froz[ok])
        med_age = float(np.median(age[ok]))
        young = ok & (age <= med_age)
        old = ok & (age > med_age)
        print(f"\n  Stratification by pair age (median split at {med_age:.1f} d, "
              f"n={int(ok.sum())}; Spearman rho error~age = {rho:+.3f}):")
        for sname, m in [("young", young), ("old", old)]:
            s = stats(e_froz[m])
            print(f"    {sname:6s} (≤/>{med_age:.1f} d): n={s['n']:5d} | "
                  f"median {s['median_km']:6.2f} | mean {s['mean_km']:6.1f} "
                  f"| tail {s['tail_pct']:4.1f} %")
            strata_rows.append({"stratum": f"age-{sname}", **s})
        lo, hi = cboot(lambda i: np.median(e_froz[i][old[i]])
                       - np.median(e_froz[i][young[i]]))
        gap = float(np.median(e_froz[old]) - np.median(e_froz[young]))
        print(f"    Median difference old−young: {gap:+.2f} km  "
              f"ASN 95% CI [{lo:+.2f}, {hi:+.2f}]")
        strata_rows.append({"stratum": "gap_old_minus_young_median_km",
                            "median_km": round(gap, 3),
                            "ci_lo": round(float(lo), 3),
                            "ci_hi": round(float(hi), 3),
                            "spearman_rho": round(rho, 3)})
    else:
        print("\n  (Pair age skipped: giga_pair_times.csv missing — "
              "rerun data/fetch_giga.py)")

    # ---- Stratum C: countries (n >= 30) ----
    cc = np.array([str(c.get("country")) for c in cases])
    print("\n  Countries (frozen, n>=30):")
    for country in sorted(set(cc)):
        m = cc == country
        if m.sum() < 30:
            continue
        s = stats(e_froz[m])
        print(f"    {country}: n={s['n']:5d} | median {s['median_km']:6.2f} "
              f"| mean {s['mean_km']:6.1f} | tail {s['tail_pct']:4.1f} %")
        strata_rows.append({"stratum": f"country-{country}", **s})

    pd.DataFrame(rows).to_csv(OUT / "giga_transfer.csv", index=False)
    pd.DataFrame(strata_rows).to_csv(OUT / "giga_transfer_strata.csv", index=False)
    print(f"\nCSV: {OUT}/giga_transfer.csv + giga_transfer_strata.csv")
    print("Data: ODbL — source: Giga (UNICEF) and contributors.")


if __name__ == "__main__":
    main()
