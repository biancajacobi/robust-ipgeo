"""Cluster bootstrap for the central confidence intervals.

The previous bootstrap CIs treat the cases as iid across the anchors.
In reality, anchors clump in networks: the 1,077 anchors spread over
805 ASNs (largest cluster 25, 18 clusters with >= 5 anchors), the probe
pools analogously. Shared infrastructure within an ASN makes errors
correlated; iid CIs then tend to be too narrow.

Correction here: cluster bootstrap -- instead of individual cases, whole
clusters are drawn with replacement (all cases of a drawn cluster come along
together). Three rows are reported per statistic:

  iid          reproduction of the published CI (identical code path/seed),
  Cluster=ASN  primary correction (ASN from the anchor/probe metadata,
               fully populated -- the /24 fallback from the plan is dropped),
  Cluster=Country deliberately coarser clumping as a stress test (112 countries).

Covered CIs (the "central" ones):
  1. T1 delta median  L1*b vs. 7 sources + L0*a, incl. Bonferroni
     [tab:t1-bootstrap, exp_bootstrap_ci]
  2. T1 mean/tail-rate deltas vs. IP2Location/ipinfo (RQ1 safeguard)
     [exp_bootstrap_ci]
  3. Delta BSS S minus 2D predecessor label, OOF on the anchors
     [CI [+0.073, +0.201], exp_support_concentration]
  4. Delta BSS flag routing minus single calibration, mixed probe pool
     [CI [+0.0005, +0.048] -- narrow; exp_s_type_routing]

Methodological note: the cluster bootstrap runs CONDITIONAL on the once-fitted
(OOF) predictions -- it corrects the inference (CI width), not the fitting.
Cluster-aware fold construction is the subject of the nested-CV experiment
(separate control), not of this script.

Usage:  python experiments/exp_cluster_bootstrap.py
Result: tables (stdout) + eval/out/cluster_bootstrap_ci.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.pipeline import load_cases                        # noqa: E402
from eval.metrics import haversine_error                    # noqa: E402
from experiments.exp_t6_defaults import (                   # noqa: E402
    loo_pseudo_radii, estimate, EPS_GRID,
)
from experiments.exp_bootstrap_ci import (                  # noqa: E402
    B, SEED, BONF_ALPHA, SOURCE_LABELS, paired_bootstrap, paired_bootstrap_stat,
)
from experiments.exp_support_concentration import (         # noqa: E402
    build_frame, oof_predict, TAU, R_HEADLINE,
)
from experiments.exp_s_declared_risk import (               # noqa: E402
    _mat, _fit_ridge, _pick_lambda,
)
from experiments.exp_s_pooled_categories import (           # noqa: E402
    WIN, POOL, _frame, _case_ips, _brier, _platt_oof,
)
from experiments.exp_s_type_routing import fetch_flags, _routed_platt  # noqa: E402

OUT = ROOT / "eval" / "out"
B_BSS = 2_000          # as in exp_support_concentration (section 3)
rows: list[dict] = []


# --------------------------------------------------------------------------- #
# Cluster-bootstrap core
# --------------------------------------------------------------------------- #
def make_groups(labels) -> list[np.ndarray]:
    """Indices per cluster (order deterministic via sorted labels)."""
    labels = np.asarray([str(x) for x in labels])
    return [np.where(labels == u)[0] for u in np.unique(labels)]


def cluster_boot(stat, groups, b, seed=SEED):
    """Cluster bootstrap: draw G clusters with replacement, evaluate the
    statistic on the pooled case indices. stat(idx) may return None
    (degenerate draw, discarded -- analogous to the iid counterpart)."""
    rng = np.random.default_rng(seed)
    G = len(groups)
    vals = []
    for _ in range(b):
        gi = rng.integers(0, G, G)
        v = stat(np.concatenate([groups[g] for g in gi]))
        if v is not None:
            vals.append(v)
    return np.asarray(vals)


def emit(section, reference, scheme, delta, boot, n, n_cl, bonf=False):
    lo, hi = np.percentile(boot, [2.5, 97.5])
    row = {"section": section, "reference": reference, "scheme": scheme,
           "n": n, "n_clusters": n_cl, "delta": round(float(delta), 4),
           "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
           "sig": bool(not (lo <= 0 <= hi)),
           "frac_gt0": round(float(np.mean(boot > 0)), 3)}
    line = (f"    {scheme:14s} [{lo:+9.4f}, {hi:+9.4f}]"
            f"{' *' if row['sig'] else '  '}")
    if bonf:
        lo_b, hi_b = np.percentile(boot, [100 * BONF_ALPHA / 2,
                                          100 * (1 - BONF_ALPHA / 2)])
        row.update({"bonf_lo": round(float(lo_b), 4),
                    "bonf_hi": round(float(hi_b), 4),
                    "bonf_sig": bool(not (lo_b <= 0 <= hi_b))})
        line += (f"  Bonf[{lo_b:+9.4f}, {hi_b:+9.4f}]"
                 f"{' *' if row['bonf_sig'] else '  '}")
    line += f"  (Cluster={n_cl})" if scheme != "iid" else f"  (n={n})"
    print(line)
    rows.append(row)


def check_repro(label, got_lo, got_hi, exp_lo, exp_hi, dec=4):
    ok = (round(got_lo, dec), round(got_hi, dec)) == (exp_lo, exp_hi)
    msg = "OK" if ok else (f"MISMATCH! expected [{exp_lo}, {exp_hi}], "
                           f"got [{got_lo:.4f}, {got_hi:.4f}]")
    print(f"    Reproduction {label}: {msg}")
    return ok


# --------------------------------------------------------------------------- #
# Sections 1+2: T1 deltas (anchors)
# --------------------------------------------------------------------------- #
def t1_sections(cases, loo, eps):
    agg_l1b, base_l0a, meta = {}, {}, {}
    src_err = {s: {} for s, _ in SOURCE_LABELS}
    for c in cases:
        ip = c["ip"]
        agg_l1b[ip] = haversine_error(estimate(c, "L1", "b", loo, eps=eps), c["truth"])
        base_l0a[ip] = haversine_error(estimate(c, "L0", "a", loo, eps=eps), c["truth"])
        meta[ip] = (c.get("asn"), c.get("country"))
        for p in c["provenance"]:
            if p["source"] in src_err:
                src_err[p["source"]][ip] = haversine_error((p["lat"], p["lon"]), c["truth"])

    print("=" * 84)
    print(f"1) T1 delta median (L1*b - reference), B={B:,}, seed={SEED}"
          f"  |  * = CI excludes 0")
    print("=" * 84)

    def one(label, ref_map):
        ips = sorted(set(agg_l1b) & set(ref_map))
        da = np.array([agg_l1b[ip] for ip in ips])
        dr = np.array([ref_map[ip] for ip in ips])
        delta, lo, hi, lo_b, hi_b = paired_bootstrap(da, dr)
        print(f"  {label}  Delta {delta:+.2f} km")
        row = {"section": "t1_delta_median", "reference": label, "scheme": "iid",
               "n": len(ips), "n_clusters": len(ips), "delta": round(delta, 4),
               "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
               "sig": bool(not (lo <= 0 <= hi)),
               "bonf_lo": round(lo_b, 4), "bonf_hi": round(hi_b, 4),
               "bonf_sig": bool(not (lo_b <= 0 <= hi_b))}
        rows.append(row)
        print(f"    {'iid':14s} [{lo:+9.4f}, {hi:+9.4f}]"
              f"{' *' if row['sig'] else '  '}  Bonf[{lo_b:+9.4f}, {hi_b:+9.4f}]"
              f"{' *' if row['bonf_sig'] else '  '}  (n={len(ips)})")
        stat = lambda idx: float(np.median(da[idx]) - np.median(dr[idx]))
        for scheme, key in (("cluster_asn", 0), ("cluster_country", 1)):
            groups = make_groups([meta[ip][key] for ip in ips])
            emit("t1_delta_median", label, scheme, delta,
                 cluster_boot(stat, groups, B), len(ips), len(groups), bonf=True)

    for key, label in SOURCE_LABELS:
        one(label, src_err[key])
    one("naive unweighted L0*a (baseline)", base_l0a)

    # Section 2: mean & tail-rate deltas (RQ1)
    print("\n" + "=" * 84)
    print(f"2) T1 mean/tail-rate deltas (L1*b - source), B={B:,}  |  + = aggregation worse")
    print("=" * 84)
    mean_stat = lambda d: float(np.mean(d))
    tail_stat = lambda d: float(100.0 * np.mean(np.asarray(d) > 100.0))
    for key, label in [("ip2location_lite", "IP2Location LITE"), ("ipinfo", "ipinfo.io")]:
        ips = sorted(set(agg_l1b) & set(src_err[key]))
        da = np.array([agg_l1b[ip] for ip in ips])
        dr = np.array([src_err[key][ip] for ip in ips])
        for sname, sfun in (("mean [km]", mean_stat), ("tail rate [pp]", tail_stat)):
            delta, lo, hi = paired_bootstrap_stat(da, dr, sfun)
            ref = f"{label} / {sname}"
            print(f"  {ref}  Delta {delta:+.1f}")
            row = {"section": "t1_meantail", "reference": ref, "scheme": "iid",
                   "n": len(ips), "n_clusters": len(ips), "delta": round(delta, 4),
                   "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                   "sig": bool(not (lo <= 0 <= hi))}
            rows.append(row)
            print(f"    {'iid':14s} [{lo:+9.4f}, {hi:+9.4f}]{' *' if row['sig'] else '  '}  (n={len(ips)})")
            stat = lambda idx, f=sfun: float(f(da[idx]) - f(dr[idx]))
            for scheme, mkey in (("cluster_asn", 0), ("cluster_country", 1)):
                groups = make_groups([meta[ip][mkey] for ip in ips])
                emit("t1_meantail", ref, scheme, delta,
                     cluster_boot(stat, groups, B), len(ips), len(groups))

    return meta


# --------------------------------------------------------------------------- #
# Section 3: delta BSS S vs. predecessor label (anchors, OOF)
# --------------------------------------------------------------------------- #
def dbss_s_section(cases, loo, eps, meta):
    df = build_frame(cases, loo, eps)
    y = (df.err > TAU).astype(int).values
    win = f"core_w_{R_HEADLINE}"
    ph_w = oof_predict(df, [win], y)
    ph_m = oof_predict(df, ["med", "hub"], y)

    def dbss(idx):
        yy = y[idx]
        bb = yy.mean()
        if bb <= 0 or bb >= 1:
            return None
        base = bb * (1 - bb)
        return float((1 - np.mean((ph_w[idx] - yy) ** 2) / base)
                     - (1 - np.mean((ph_m[idx] - yy) ** 2) / base))

    print("\n" + "=" * 84)
    print(f"3) Delta BSS S - predecessor label (anchors, OOF conditional), B={B_BSS:,}")
    print("=" * 84)
    # iid reproduction: identical draw logic as exp_support_concentration
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(B_BSS):
        v = dbss(rng.integers(0, len(y), len(y)))
        if v is not None:
            diffs.append(v)
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    check_repro("CI [+0.073, +0.201]", round(lo, 3), round(hi, 3), 0.073, 0.201, dec=3)
    delta = float(np.mean(diffs))
    print(f"  Delta-BSS {delta:+.3f}")
    rows.append({"section": "dbss_s_vs_label", "reference": "S - (med+hub)",
                 "scheme": "iid", "n": len(y), "n_clusters": len(y),
                 "delta": round(delta, 4), "ci_lo": round(float(lo), 4),
                 "ci_hi": round(float(hi), 4), "sig": bool(not (lo <= 0 <= hi)),
                 "frac_gt0": round(float(np.mean(diffs > 0)), 3)})
    print(f"    {'iid':14s} [{lo:+9.4f}, {hi:+9.4f}]{' *' if not (lo <= 0 <= hi) else '  '}  (n={len(y)})")
    for scheme, key in (("cluster_asn", 0), ("cluster_country", 1)):
        groups = make_groups([meta[ip][key] for ip in df.ip])
        emit("dbss_s_vs_label", "S - (med+hub)", scheme, delta,
             cluster_boot(dbss, groups, B_BSS), len(y), len(groups))


# --------------------------------------------------------------------------- #
# Section 4: delta BSS flag routing (mixed probe pool)
# --------------------------------------------------------------------------- #
def dbss_routing_section():
    A = _frame("anchors")
    yA = (A.err > TAU).astype(int).values
    parts, ip_parts, seen = [], [], set()
    for ds in POOL:
        T = _frame(ds).assign(origin=ds)
        ips = _case_ips(ds)
        if len(ips) != len(T):
            raise RuntimeError(f"{ds}: {len(ips)} cases, but {len(T)} frame rows")
        keep = [ip not in seen for ip in ips]
        seen.update(ips)
        parts.append(T[pd.Series(keep, index=T.index)].reset_index(drop=True))
        ip_parts += [ip for ip, k in zip(ips, keep) if k]
    P = pd.concat(parts, ignore_index=True)
    y = (P.err > TAU).astype(int).values
    flags_map = fetch_flags(ip_parts)          # load once, not per IP (cache-file read)
    flag = np.array([flags_map.get(ip, False) for ip in ip_parts])

    Xa, Xp = _mat(A, WIN), _mat(P, WIN)
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-9
    w = _fit_ridge(np.column_stack([np.ones(len(A)), (Xa - mu) / sd]), yA,
                   _pick_lambda(Xa, yA, 0))
    z = np.column_stack([np.ones(len(P)), (Xp - mu) / sd]) @ w
    ph_one, ph_flag = _platt_oof(z, y), _routed_platt(z, y, flag)

    # ASN/country per pool IP from the probe metadata
    meta = {}
    for ds in POOL:
        for r in pd.read_csv(ROOT / "data" / "cache" / f"{ds}.csv").itertuples():
            meta[r.ip] = (r.asn, r.country)
    missing = [ip for ip in ip_parts if ip not in meta]
    if missing:
        raise RuntimeError(f"{len(missing)} pool IP(s) without metadata")

    def dbss(idx):
        yb = y[idx]
        bb = yb.mean()
        if bb <= 0 or bb >= 1:
            return None
        ref = float(np.mean((bb - yb) ** 2))
        return float((_brier(ph_one[idx], yb) - _brier(ph_flag[idx], yb)) / ref)

    print("\n" + "=" * 84)
    print(f"4) Delta BSS flag routing - single calibration (pool, conditional), B={B:,}")
    print("=" * 84)
    # iid reproduction: identical draw logic as exp_s_type_routing
    rng = np.random.default_rng(0)
    deltas = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        ref = float(np.mean((yb.mean() - yb) ** 2))
        deltas[b] = (_brier(ph_one[idx], yb) - _brier(ph_flag[idx], yb)) / ref
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    check_repro("CI [+0.0005, +0.048]", round(lo, 4), round(hi, 4), 0.0005, 0.048)
    delta = float(deltas.mean())
    print(f"  Delta-BSS {delta:+.3f}")
    rows.append({"section": "dbss_routing", "reference": "flag routing - single calib.",
                 "scheme": "iid", "n": len(y), "n_clusters": len(y),
                 "delta": round(delta, 4), "ci_lo": round(float(lo), 4),
                 "ci_hi": round(float(hi), 4), "sig": bool(not (lo <= 0 <= hi)),
                 "frac_gt0": round(float(np.mean(deltas > 0)), 3)})
    print(f"    {'iid':14s} [{lo:+9.4f}, {hi:+9.4f}]{' *' if not (lo <= 0 <= hi) else '  '}  (n={len(y)})")
    for scheme, key in (("cluster_asn", 0), ("cluster_country", 1)):
        groups = make_groups([meta[ip][key] for ip in ip_parts])
        emit("dbss_routing", "flag routing - single calib.", scheme, delta,
             cluster_boot(dbss, groups, B), len(y), len(groups))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    loo = loo_pseudo_radii(cases)
    eps = min(EPS_GRID, key=lambda e: abs(e - float(np.median(
        [loo[s]["_global"] for s in loo]))))

    meta = t1_sections(cases, loo, eps)
    dbss_s_section(cases, loo, eps, meta)
    dbss_routing_section()

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "cluster_bootstrap_ci.csv", index=False)

    sig = out.sig.fillna(False).astype(bool)
    # Rows without a Bonferroni column count as "sig" (= never appear as a flip)
    bsig = out.get("bonf_sig", pd.Series(True, index=out.index)).fillna(True).astype(bool)
    iid_sig = set(out[(out.scheme == "iid") & sig].reference)
    flips = out[(out.scheme != "iid") & ~sig & out.reference.isin(iid_sig)]
    iid_bsig = set(out[(out.scheme == "iid") & bsig & out.bonf_lo.notna()].reference)
    bflips = out[(out.scheme != "iid") & ~bsig & out.reference.isin(iid_bsig)]
    print("\nREADING:")
    print("  Cluster CIs are expectedly wider than iid; what matters is whether a")
    print("  statement that is significant under iid flips.")
    print("  Flipped (95% CI):  " + ("none" if flips.empty else "; ".join(
        f"{r.reference} [{r.scheme}]" for r in flips.itertuples())))
    print("  Flipped (Bonf CI): " + ("none" if bflips.empty else "; ".join(
        f"{r.reference} [{r.scheme}]" for r in bflips.itertuples())))
    print("  Conditional on fitted OOF predictions; cluster-aware folds ->")
    print("  nested-CV experiment.")
    print(f"\nCSV: {OUT}/cluster_bootstrap_ci.csv")


if __name__ == "__main__":
    main()
