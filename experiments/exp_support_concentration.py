"""Concentration-measure selection, fully OUT-OF-FOLD: dispersion vs. concentration measures.

Complements exp_spread_measure.py (which only computes in-sample AP) with the
honest out-of-fold evaluation using the same 10-fold stratified logic as
exp_label_calibration.py. Each candidate measure is evaluated as a continuous
forecast for the event "aggregation error > tau km" (L1+b) via an OOF logit:
Brier Skill Score (BSS), Average Precision (AP), Expected Calibration Error (ECE).

Main finding (support concentration): the LINE-WEIGHTED core share S -- share of
the lineage-collapsed source mass within r km of the L1+b estimator -- triples
the OUT-OF-FOLD calibration quality over the median pairwise-distance label
(BSS +0.068 -> +0.210, r=50) and subsumes the hub flag (core_w+hub changes nothing).

Discipline (anti-tuning): r is NOT optimised. The best radius moves with the
miss threshold tau (scale matching, see scale_matching_grid) -- picking the best
scorer would implicitly tune to the evaluation threshold. Headline radius r=50 km
is the pre-specified city-scale constant (identical to the low_spread/bucket limit).

Triage operating point (matched flag quota, 41% of the predecessor label): S sits
at the threshold inside a large tie group (identical S value); the exact quota is
only reachable by splitting that group. Order-independent: recall on the misses
72% -> ~91% unbiased (band 86-94% depending on tie resolution); strict threshold:
86% recall at only 29% flag quota. The operating points of the first evaluation
(92% recall at 41% quota, precisely 91.5%/26.7%; quota 20.1% at recall >= 72.1%,
precision 43.3%) are reproduced deterministically as declared audit rows (rounded
quota resp. target recall, tie resolution by data order). Details:
triage_matched_quote().

Usage:   python experiments/exp_support_concentration.py
Output:  tables (stdout) + eval/out/support_concentration_oof.csv
         + eval/out/support_concentration_triage.csv (operating-point comparison)
         + eval/out/support_concentration.png (BSS ladder + reliability diagram)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.pipeline import load_cases            # noqa: E402
from eval.metrics import haversine_error         # noqa: E402
from eval import report                          # noqa: E402  (_plt + OUT_DIR)
import experiments.exp_t6_defaults as T6         # noqa: E402

OUT = report.OUT_DIR
RADII = [25, 50, 75, 100]
TAU = 100.0
R_HEADLINE = 50           # pre-specified (city scale), NOT optimised
K, SEED = 10, 0


# --------------------------------------------------------------------------- #
# Feature construction (per anchor)
# --------------------------------------------------------------------------- #
def build_frame(cases, loo, eps):
    rows = []
    for c in cases:
        prov = c["provenance"]
        pts = np.array([[p["lat"], p["lon"]] for p in prov], float)
        n = len(pts)
        w = np.asarray(T6.line_weights_for(prov, "L1"), float)
        w = w / w.sum()
        est = np.array(T6.estimate(c, "L1", "b", loo, eps=eps))
        d = np.array([haversine_error(tuple(p), tuple(est)) for p in pts])          # distance source->estimator
        pdd = (np.array([haversine_error(tuple(pts[i]), tuple(pts[j]))
                         for i in range(n) for j in range(i + 1, n)])
               if n > 1 else np.array([0.0]))
        r = {
            "ip": c["ip"], "nlines": n,
            "hub": float(np.mean([p["is_default_centroid"] for p in prov]) >= 0.5),
            "med": float(np.median(pdd)), "mean": float(pdd.mean()),
            "max_pw": float(pdd.max()), "q75": float(np.percentile(pdd, 75)),
            "q90": float(np.percentile(pdd, 90)),
            "gap": float(np.max(np.diff(np.sort(d))) if n > 1 else 0.0),
            "mean_d": float(d.mean()), "max_d": float(d.max()),
            "err": haversine_error(tuple(est), c["truth"]),
        }
        for R in RADII:
            r[f"core_u_{R}"] = float((d < R).mean())          # core share, unweighted
            r[f"core_w_{R}"] = float(w[d < R].sum())          # core share, LINE-WEIGHTED (= S)
            r[f"kde_w_{R}"] = float(np.sum(w * np.exp(-(d / R) ** 2)))
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# OOF logit + metrics (dependency-free, identical to exp_label_calibration)
# --------------------------------------------------------------------------- #
def folds(y, seed=SEED):
    rng = np.random.default_rng(seed)
    f = np.empty(len(y), int)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        f[idx] = np.arange(len(idx)) % K
    return f


def _fit(X, y, it=200):
    w = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        W = p * (1 - p) + 1e-9
        w -= np.linalg.solve((X * W[:, None]).T @ X + 1e-6 * np.eye(X.shape[1]),
                             X.T @ (p - y) + 1e-6 * w)
    return w


def oof_predict(df, cols, y, seed=SEED):
    F = folds(y, seed)
    ph = np.empty(len(y))
    raw = np.column_stack([np.log1p(df[c].values) if df[c].max() > 1.5 else df[c].values for c in cols])
    for fi in range(K):
        tr, te = F != fi, F == fi
        mu, sd = raw[tr].mean(0), raw[tr].std(0) + 1e-9
        Xtr = np.column_stack([np.ones(tr.sum()), (raw[tr] - mu) / sd])
        Xte = np.column_stack([np.ones(te.sum()), (raw[te] - mu) / sd])
        ph[te] = 1.0 / (1.0 + np.exp(-Xte @ _fit(Xtr, y[tr])))
    return ph


def bss(ph, y):
    b = y.mean()
    return 1 - np.mean((ph - y) ** 2) / (b * (1 - b))


def avg_prec(ph, y):
    o = np.argsort(-ph)
    yy = y[o]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    P = tp / (tp + fp)
    R = np.concatenate([[0.0], tp / yy.sum()])
    return float(np.sum((R[1:] - R[:-1]) * P))


def ece(ph, y, nb=5):
    e = np.unique(np.quantile(ph, np.linspace(0, 1, nb + 1)))
    b = np.clip(np.digitize(ph, e[1:-1]), 0, len(e) - 2)
    out = 0.0
    for k in range(len(e) - 1):
        m = b == k
        if m.sum():
            out += m.sum() / len(y) * abs(ph[m].mean() - y[m].mean())
    return out


# --------------------------------------------------------------------------- #
# Triage comparison at the predecessor label's operating point (matched flag quota)
# --------------------------------------------------------------------------- #
def triage_matched_quote(df, y, win):
    """Binary triage comparison: S against the 2D predecessor label (spread x hub).

    The flag quota is EXPLICITLY taken over from the predecessor label (comparison
    design: same operating point, who marks more misses?). S is heavily tied at the
    threshold (tie group with identical S value); an exact quota is only reachable
    by splitting that group, and which part is taken is methodologically
    undetermined. Therefore reported, order-independently:
      (a) the unbiased recall under proportional tie resolution,
      (b) the band of the two honest thresholds (S < t resp. S <= t),
      (c) the fold-internal control (threshold as train quantile, recall on test).
    """
    S = df[win].values
    old = (df.med.values >= 50) | df.hub.values.astype(bool)
    n_flag, miss = int(old.sum()), int(y.sum())
    q = n_flag / len(y)
    t = float(np.sort(S)[n_flag - 1])
    lo, hi, tie = S < t, S <= t, S == t
    k = n_flag - int(lo.sum())                       # to be taken from the tie group
    tp_lo, tp_hi, tie_m = int(y[lo].sum()), int(y[hi].sum()), int(y[tie].sum())
    exp_tp = tp_lo + tie_m * k / int(tie.sum())      # proportional tie resolution

    # Audit reconstruction of the first evaluation's operating points
    # (deterministic): (1) rounded 41% quota as an exact count k = round(0.41*n),
    # tie resolution by data order (stable sort)
    # -> recall 91.5%, precision 26.7%; (2) first point of the S ranking with
    # recall >= 72.1% (rounded predecessor-label recall, cf. prec_at_recall in
    # exp_spread_measure.py) -> quota 20.1%, precision 43.3%. Both are concrete
    # tie resolutions, not a methodological statement -- that is carried by the
    # unbiased row and the band above.
    o = np.argsort(S, kind="stable")
    k41 = int(round(0.41 * len(y)))
    ex = np.zeros(len(y), bool)
    ex[o[:k41]] = True
    tp_ex = int(y[ex].sum())
    tgt = round(float(old[y == 1].mean()), 3)
    o72 = np.argsort(S)          # default argsort = tie order of prec_at_recall
    tp_seq = np.cumsum(y[o72])
    k72 = int(np.argmax(tp_seq / miss >= tgt)) + 1
    tp72 = int(tp_seq[k72 - 1])

    # fold-internal control: threshold from the training folds only (strict, S < t_fold)
    F = folds(y)
    fl = np.zeros(len(y), bool)
    for fi in range(K):
        tr, te = F != fi, F == fi
        fl[te] = S[te] < np.quantile(S[tr], old[tr].mean())

    print("\nTRIAGE at matched flag quota (operating point of the predecessor label):")
    print(f"  predecessor label:    quota {q:.1%}, recall {old[y == 1].mean():.1%}, "
          f"precision {y[old].mean():.1%}")
    print(f"  S tie group at threshold t={t:.4f}: {int(tie.sum())} anchors "
          f"({tie_m} misses), of which {k} to flag -> exact quota only via splitting")
    print(f"  S strict    (S < t):  quota {lo.mean():.1%}, recall {tp_lo / miss:.1%}, "
          f"precision {tp_lo / int(lo.sum()):.1%}")
    print(f"  S inclusive (S <= t): quota {hi.mean():.1%}, recall {tp_hi / miss:.1%}, "
          f"precision {tp_hi / int(hi.sum()):.1%}")
    print(f"  S unbiased at quota {q:.1%} (proportional tie resolution): "
          f"recall {exp_tp / miss:.1%}, precision {exp_tp / n_flag:.1%}")
    print(f"  S audit 1 (quota rounded 41 % = {k41} flags, ties by data order; "
          f"first evaluation): recall {tp_ex / miss:.1%}, precision {tp_ex / k41:.1%}")
    print(f"  S audit 2 (first ranking point with recall >= {tgt:.1%}; first evaluation): "
          f"quota {k72 / len(y):.1%}, recall {tp72 / miss:.1%}, precision {tp72 / k72:.1%}")
    print(f"  fold-internal threshold (train quantile): quota {fl.mean():.1%}, "
          f"recall {fl[y == 1].mean():.1%}, precision {y[fl].mean():.1%}")

    rows = [
        {"flag_rule": "predecessor label (spread>=50 OR hub)", "flag_quote": round(q, 3),
         "recall_misses": round(float(old[y == 1].mean()), 3), "precision": round(float(y[old].mean()), 3)},
        {"flag_rule": f"S < {t:.4f} (strict)", "flag_quote": round(float(lo.mean()), 3),
         "recall_misses": round(tp_lo / miss, 3), "precision": round(tp_lo / int(lo.sum()), 3)},
        {"flag_rule": f"S <= {t:.4f} (inclusive)", "flag_quote": round(float(hi.mean()), 3),
         "recall_misses": round(tp_hi / miss, 3), "precision": round(tp_hi / int(hi.sum()), 3)},
        {"flag_rule": "S, matched quota, proportional tie resolution (expected value)",
         "flag_quote": round(q, 3), "recall_misses": round(exp_tp / miss, 3),
         "precision": round(exp_tp / n_flag, 3)},
        {"flag_rule": "S, quota rounded 41 %, tie resolution by data order (audit; first evaluation)",
         "flag_quote": round(k41 / len(y), 3), "recall_misses": round(tp_ex / miss, 3),
         "precision": round(tp_ex / k41, 3)},
        {"flag_rule": "S, first ranking point with recall >= 72.1 % (audit; first evaluation)",
         "flag_quote": round(k72 / len(y), 3), "recall_misses": round(tp72 / miss, 3),
         "precision": round(tp72 / k72, 3)},
        {"flag_rule": "S < train quantile (fold-internal, 10-fold)", "flag_quote": round(float(fl.mean()), 3),
         "recall_misses": round(float(fl[y == 1].mean()), 3), "precision": round(float(y[fl].mean()), 3)},
    ]
    pd.DataFrame(rows).to_csv(OUT / "support_concentration_triage.csv", index=False)
    print(f"  CSV: {OUT}/support_concentration_triage.csv")


# --------------------------------------------------------------------------- #
# Figure: BSS ladder (left) + OOF reliability of the winner (right)
# --------------------------------------------------------------------------- #
def plot(ladder, ph_win, y, name="support_concentration"):
    plt = report._plt()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

    # (left) OOF BSS per measure
    labels = [l[0] for l in ladder]
    vals = [l[1] for l in ladder]
    colors = ["tab:gray"] * len(vals)
    colors[-1] = "tab:green"   # highlight the winner
    ypos = np.arange(len(vals))
    axL.barh(ypos, vals, color=colors)
    axL.set_yticks(ypos)
    axL.set_yticklabels(labels, fontsize=8)
    axL.invert_yaxis()
    for i, v in enumerate(vals):
        axL.text(v + 0.004, i, f"{v:+.3f}", va="center", fontsize=8)
    axL.set_xlabel("OOF Brier Skill Score (higher = better)")
    axL.set_title("concentration measures out-of-fold\n(event: L1+b error > 100 km)")
    axL.axvline(0, color="k", lw=0.8)
    axL.grid(True, axis="x", alpha=0.3)

    # (right) reliability diagram of the winner (OOF, quantile bins)
    edges = np.unique(np.quantile(ph_win, np.linspace(0, 1, 6)))
    idx = np.clip(np.digitize(ph_win, edges[1:-1]), 0, len(edges) - 2)
    fx, oy, sz = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum():
            fx.append(ph_win[m].mean()); oy.append(y[m].mean()); sz.append(m.sum())
    hi = max(max(fx), max(oy)) * 1.1
    axR.plot([0, hi], [0, hi], "k--", alpha=0.5, label="perfectly calibrated")
    axR.plot(fx, oy, "o-", color="tab:green", label="support concentration $S$ (OOF)")
    axR.axhline(y.mean(), color="tab:red", lw=0.8, ls=":", label=f"base rate {y.mean():.2f}")
    axR.set_xlim(0, hi); axR.set_ylim(0, hi)
    axR.set_xlabel("predicted miss probability")
    axR.set_ylabel("observed miss rate (out-of-fold)")
    axR.set_title(f"reliability: line-weighted support concentration r={R_HEADLINE}")
    axR.grid(True, alpha=0.3); axR.legend(fontsize=8)

    fig.tight_layout()
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def run():
    cases = load_cases()
    loo = T6.loo_pseudo_radii(cases)
    eps = min(T6.EPS_GRID, key=lambda e: abs(e - float(np.median([loo[s]["_global"] for s in loo]))))
    df = build_frame(cases, loo, eps)
    y = (df.err > TAU).astype(int).values
    win = f"core_w_{R_HEADLINE}"

    print("=" * 80)
    print(f"OOF CONCENTRATION-MEASURE COMPARISON  (event: L1+b error > {TAU:.0f} km, {K}-fold OOF)")
    print(f"n={len(df)}  misses={int(y.sum())}  base rate={y.mean():.3f}  "
          f"(sources per case {df.nlines.min():.0f}-{df.nlines.max():.0f}, "
          f"single-source cases={int((df.nlines < 2).sum())}; source-, not line-based)")
    print("=" * 80)

    candidates = [
        ("median pairwise (old)", ["med"]),
        ("median + hub (OLD LABEL)", ["med", "hub"]),
        ("mean pairwise", ["mean"]),
        ("max pairwise", ["max_pw"]),
        ("q75 pairwise", ["q75"]),
        ("q90 pairwise", ["q90"]),
        ("largest gap (gap)", ["gap"]),
        ("mean dist->estimator", ["mean_d"]),
        ("max dist->estimator", ["max_d"]),
        (f"core share unw. r={R_HEADLINE}", [f"core_u_{R_HEADLINE}"]),
        (f"support conc. WEIGHTED r={R_HEADLINE} (S)", [win]),
        (f"KDE conc. wgt. r={R_HEADLINE}", [f"kde_w_{R_HEADLINE}"]),
        (f"S + hub", [win, "hub"]),
    ]
    out_rows, ph_cache = [], {}
    print(f"\n{'measure':40s} {'OOF-BSS':>8s} {'OOF-AP':>7s} {'OOF-ECE':>8s}")
    for label, cols in candidates:
        ph = oof_predict(df, cols, y)
        ph_cache[label] = ph
        b, a, e = bss(ph, y), avg_prec(ph, y), ece(ph, y)
        print(f"{label:40s} {b:+8.3f} {a:7.3f} {e:8.3f}")
        out_rows.append({"measure": label, "oof_bss": round(b, 3), "oof_ap": round(a, 3), "oof_ece": round(e, 3)})

    print("\nradius robustness & lineage control (core share, OOF-BSS):")
    print(f"  {'r':>5s}  {'unweighted':>12s}  {'WEIGHTED':>10s}")
    for R in RADII:
        bu = bss(oof_predict(df, [f"core_u_{R}"], y), y)
        bw = bss(oof_predict(df, [f"core_w_{R}"], y), y)
        print(f"  {R:5d}  {bu:+12.3f}  {bw:+10.3f}")
        out_rows.append({"measure": f"core r={R} unw", "oof_bss": round(bu, 3), "oof_ap": "", "oof_ece": ""})
        out_rows.append({"measure": f"core r={R} WGT", "oof_bss": round(bw, 3), "oof_ap": "", "oof_ece": ""})

    # Seed robustness (winner vs old label)
    print("\nseed robustness (OOF-BSS over 5 fold seeds):")
    for label, cols in [("old label (med+hub)", ["med", "hub"]), (f"S (core_w_{R_HEADLINE})", [win])]:
        v = [bss(oof_predict(df, cols, y, seed=s), y) for s in range(5)]
        print(f"  {label:24s} mean={np.mean(v):+.3f}  [{min(v):+.3f}, {max(v):+.3f}]")

    # Bootstrap CI: winner vs old label
    ph_w, ph_m = ph_cache[f"support conc. WEIGHTED r={R_HEADLINE} (S)"], ph_cache["median + hub (OLD LABEL)"]
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y)); yy = y[idx]; bb = yy.mean()
        if bb <= 0 or bb >= 1:
            continue
        base = bb * (1 - bb)
        diffs.append((1 - np.mean((ph_w[idx] - yy) ** 2) / base) - (1 - np.mean((ph_m[idx] - yy) ** 2) / base))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\nbootstrap (2000x): BSS difference S minus OLD LABEL = {np.mean(diffs):+.3f}  "
          f"95% CI [{lo:+.3f}, {hi:+.3f}]  (>0 => genuinely better)")
    out_rows.append({"measure": "BOOTSTRAP diff S-(med+hub)", "oof_bss": round(float(np.mean(diffs)), 3),
                     "oof_ap": f"CI[{lo:+.3f},{hi:+.3f}]", "oof_ece": ""})

    # Murphy decomposition of the OOF forecast (Brier = reliability - resolution +
    # uncertainty), quantile-binned like ece() -- review addition 2026-08-18:
    # separates whether the BSS advantage stems from calibration (small
    # reliability) or discrimination (large resolution).
    def murphy_binned(ph, yy, nb=5):
        e = np.unique(np.quantile(ph, np.linspace(0, 1, nb + 1)))
        b = np.clip(np.digitize(ph, e[1:-1]), 0, len(e) - 2)
        base = yy.mean()
        rel = res = 0.0
        for k in range(len(e) - 1):
            m = b == k
            if m.sum():
                rel += m.mean() * (ph[m].mean() - yy[m].mean()) ** 2
                res += m.mean() * (yy[m].mean() - base) ** 2
        return rel, res, float(base * (1 - base))
    print("\nMurphy decomposition (quantile-binned, 5 bins): reliability / resolution / uncertainty")
    for label, ph in (("S (core_w_50)", ph_w), ("old label (med+hub)", ph_m)):
        rel, res, unc = murphy_binned(ph, y)
        print(f"  {label:24s} {rel:.4f} / {res:.4f} / {unc:.4f}   "
              f"(Brier {np.mean((ph - y) ** 2):.4f})")
        out_rows.append({"measure": f"MURPHY {label}",
                         "oof_bss": round(rel, 4), "oof_ap": round(res, 4),
                         "oof_ece": round(unc, 4)})

    # Scale-matching grid (anti-tuning evidence)
    print("\nscale matching (OOF-BSS, wgt. core share): does the best r move with tau? (=> fix r in advance)")
    print(f"  {'tau\\r':9s}" + "".join(f"r={R:<6d}" for R in RADII))
    for tau in (50, 100, 200):
        yt = (df.err > tau).astype(int).values
        vals = [bss(oof_predict(df, [f"core_w_{R}"], yt), yt) for R in RADII]
        star = [" "] * len(RADII); star[int(np.argmax(vals))] = "*"
        print(f"  tau={tau:<4d}  " + "".join(f"{v:+.3f}{s} " for v, s in zip(vals, star)))

    pd.DataFrame(out_rows).to_csv(OUT / "support_concentration_oof.csv", index=False)

    # Triage operating-point comparison (matched flag quota, ties explicit)
    triage_matched_quote(df, y, win)

    # Figure
    ladder = [("median pairwise", bss(ph_cache["median pairwise (old)"], y)),
              ("median + hub (old label)", bss(ph_cache["median + hub (OLD LABEL)"], y)),
              ("mean dist->estimator", bss(ph_cache["mean dist->estimator"], y)),
              (f"core share unw. r={R_HEADLINE}", bss(ph_cache[f"core share unw. r={R_HEADLINE}"], y)),
              (f"support conc. WEIGHTED r={R_HEADLINE} (S)", bss(ph_w, y))]
    fig_path = plot(ladder, ph_w, y)
    print(f"\nCSV: {OUT}/support_concentration_oof.csv")
    print(f"PNG: {fig_path}")


if __name__ == "__main__":
    run()
