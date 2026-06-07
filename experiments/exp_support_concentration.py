"""Konfidenz-Maß-Auswahl, vollständig OUT-OF-FOLD: Dispersions- vs. Konzentrationsmaße.

Ergänzt exp_spread_measure.py (das nur in-sample AP rechnet) um die ehrliche
out-of-fold-Bewertung mit derselben 10-fach-stratifizierten Logik wie
exp_label_calibration.py. Jedes Kandidatenmaß wird als kontinuierlicher Forecast
fuer das Ereignis "Aggregations-Fehler > tau km" (L1+b) via OOF-Logit bewertet:
Brier Skill Score (BSS), Average Precision (AP), Expected Calibration Error (ECE).

Hauptbefund (Stütz-Konzentration): der LINIEN-GEWICHTETE Kernanteil S -- Anteil der
lineage-kollabierten Quellen-Masse innerhalb r km vom L1+b-Schaetzer -- verdreifacht
die OUT-OF-FOLD-Kalibrierungsguete gegenueber dem median-Paardistanz-Label
(BSS +0,068 -> +0,210, r=50) und subsumiert den Hub-Flag (core_w+hub aendert nichts).

Disziplin (Anti-Tuning): r wird NICHT optimiert. Der beste Radius wandert mit der
Miss-Schwelle tau (Skalen-Matching, s. scale_matching_grid) -- den bestpunktenden zu
waehlen waere implizites Tunen auf die Bewertungsschwelle. Headline-Radius r=50 km ist
die vorab spezifizierte City-Skala-Konstante (identisch low_spread/Bucket-Grenze).

Aufruf:  python experiments/exp_support_concentration.py
Ergebnis: Tabellen (stdout) + eval/out/support_concentration_oof.csv
          + eval/out/support_concentration.png (BSS-Leiter + Reliability-Diagramm)
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
R_HEADLINE = 50           # vorab spezifiziert (City-Skala), NICHT optimiert
K, SEED = 10, 0


# --------------------------------------------------------------------------- #
# Merkmals-Konstruktion (pro Anchor)
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
        d = np.array([haversine_error(tuple(p), tuple(est)) for p in pts])          # Distanz Quelle->Schaetzer
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
            r[f"core_u_{R}"] = float((d < R).mean())          # Kernanteil ungewichtet
            r[f"core_w_{R}"] = float(w[d < R].sum())          # Kernanteil LINIEN-GEWICHTET (= S)
            r[f"kde_w_{R}"] = float(np.sum(w * np.exp(-(d / R) ** 2)))
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# OOF-Logit + Metriken (dependency-frei, identisch zu exp_label_calibration)
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
# Abbildung: BSS-Leiter (links) + OOF-Reliability des Siegers (rechts)
# --------------------------------------------------------------------------- #
def plot(ladder, ph_win, y, name="support_concentration"):
    plt = report._plt()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

    # (links) OOF-BSS je Maß
    labels = [l[0] for l in ladder]
    vals = [l[1] for l in ladder]
    colors = ["tab:gray"] * len(vals)
    colors[-1] = "tab:green"   # Sieger hervorheben
    ypos = np.arange(len(vals))
    axL.barh(ypos, vals, color=colors)
    axL.set_yticks(ypos)
    axL.set_yticklabels(labels, fontsize=8)
    axL.invert_yaxis()
    for i, v in enumerate(vals):
        axL.text(v + 0.004, i, f"{v:+.3f}", va="center", fontsize=8)
    axL.set_xlabel("OOF Brier Skill Score (höher = besser)")
    axL.set_title("Konfidenzmaße out-of-fold\n(Ereignis: L1+b-Fehler > 100 km)")
    axL.axvline(0, color="k", lw=0.8)
    axL.grid(True, axis="x", alpha=0.3)

    # (rechts) Reliability-Diagramm des Siegers (OOF, Quantil-Bins)
    edges = np.unique(np.quantile(ph_win, np.linspace(0, 1, 6)))
    idx = np.clip(np.digitize(ph_win, edges[1:-1]), 0, len(edges) - 2)
    fx, oy, sz = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum():
            fx.append(ph_win[m].mean()); oy.append(y[m].mean()); sz.append(m.sum())
    hi = max(max(fx), max(oy)) * 1.1
    axR.plot([0, hi], [0, hi], "k--", alpha=0.5, label="perfekt kalibriert")
    axR.plot(fx, oy, "o-", color="tab:green", label="Stütz-Konzentration $S$ (OOF)")
    axR.axhline(y.mean(), color="tab:red", lw=0.8, ls=":", label=f"Basisrate {y.mean():.2f}")
    axR.set_xlim(0, hi); axR.set_ylim(0, hi)
    axR.set_xlabel("vorhergesagte Miss-Wahrscheinlichkeit")
    axR.set_ylabel("beobachtete Miss-Rate (out-of-fold)")
    axR.set_title(f"Reliability: linien-gew. Stütz-Konzentration r={R_HEADLINE}")
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
    print(f"OOF-KONFIDENZMASS-VERGLEICH  (Ereignis: L1+b-Fehler > {TAU:.0f} km, {K}-fach OOF)")
    print(f"n={len(df)}  Misses={int(y.sum())}  Basisrate={y.mean():.3f}  "
          f"(nlines {df.nlines.min()}-{df.nlines.max()}, Einzelquell-Faelle={int((df.nlines < 2).sum())})")
    print("=" * 80)

    candidates = [
        ("median pairwise (alt)", ["med"]),
        ("median + hub (ALTES LABEL)", ["med", "hub"]),
        ("mean pairwise", ["mean"]),
        ("max pairwise", ["max_pw"]),
        ("q75 pairwise", ["q75"]),
        ("q90 pairwise", ["q90"]),
        ("groesste Luecke (gap)", ["gap"]),
        ("mean dist->Schaetzer", ["mean_d"]),
        ("max dist->Schaetzer", ["max_d"]),
        (f"Kernanteil ungew. r={R_HEADLINE}", [f"core_u_{R_HEADLINE}"]),
        (f"Stuetz-Konz. GEWICHTET r={R_HEADLINE} (S)", [win]),
        (f"KDE-Konz. gew. r={R_HEADLINE}", [f"kde_w_{R_HEADLINE}"]),
        (f"S + hub", [win, "hub"]),
    ]
    out_rows, ph_cache = [], {}
    print(f"\n{'Maß':40s} {'OOF-BSS':>8s} {'OOF-AP':>7s} {'OOF-ECE':>8s}")
    for label, cols in candidates:
        ph = oof_predict(df, cols, y)
        ph_cache[label] = ph
        b, a, e = bss(ph, y), avg_prec(ph, y), ece(ph, y)
        print(f"{label:40s} {b:+8.3f} {a:7.3f} {e:8.3f}")
        out_rows.append({"measure": label, "oof_bss": round(b, 3), "oof_ap": round(a, 3), "oof_ece": round(e, 3)})

    print("\nRadius-Robustheit & Lineage-Kontrolle (Kernanteil, OOF-BSS):")
    print(f"  {'r':>5s}  {'ungewichtet':>12s}  {'GEWICHTET':>10s}")
    for R in RADII:
        bu = bss(oof_predict(df, [f"core_u_{R}"], y), y)
        bw = bss(oof_predict(df, [f"core_w_{R}"], y), y)
        print(f"  {R:5d}  {bu:+12.3f}  {bw:+10.3f}")
        out_rows.append({"measure": f"core r={R} ungew", "oof_bss": round(bu, 3), "oof_ap": "", "oof_ece": ""})
        out_rows.append({"measure": f"core r={R} GEW", "oof_bss": round(bw, 3), "oof_ap": "", "oof_ece": ""})

    # Seed-Robustheit (Sieger vs altes Label)
    print("\nSeed-Robustheit (OOF-BSS über 5 Fold-Seeds):")
    for label, cols in [("altes Label (med+hub)", ["med", "hub"]), (f"S (core_w_{R_HEADLINE})", [win])]:
        v = [bss(oof_predict(df, cols, y, seed=s), y) for s in range(5)]
        print(f"  {label:24s} mean={np.mean(v):+.3f}  [{min(v):+.3f}, {max(v):+.3f}]")

    # Bootstrap-CI: Sieger vs altes Label
    ph_w, ph_m = ph_cache[f"Stuetz-Konz. GEWICHTET r={R_HEADLINE} (S)"], ph_cache["median + hub (ALTES LABEL)"]
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y)); yy = y[idx]; bb = yy.mean()
        if bb <= 0 or bb >= 1:
            continue
        base = bb * (1 - bb)
        diffs.append((1 - np.mean((ph_w[idx] - yy) ** 2) / base) - (1 - np.mean((ph_m[idx] - yy) ** 2) / base))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\nBootstrap (2000x): BSS-Differenz S minus ALTES LABEL = {np.mean(diffs):+.3f}  "
          f"95%-CI [{lo:+.3f}, {hi:+.3f}]  (>0 => real besser)")
    out_rows.append({"measure": "BOOTSTRAP diff S-(med+hub)", "oof_bss": round(float(np.mean(diffs)), 3),
                     "oof_ap": f"CI[{lo:+.3f},{hi:+.3f}]", "oof_ece": ""})

    # Skalen-Matching-Gitter (Anti-Tuning-Nachweis)
    print("\nSkalen-Matching (OOF-BSS, gew. Kernanteil): wandert bestes r mit tau? (=> r vorab fixieren)")
    print(f"  {'tau\\\\r':9s}" + "".join(f"r={R:<6d}" for R in RADII))
    for tau in (50, 100, 200):
        yt = (df.err > tau).astype(int).values
        vals = [bss(oof_predict(df, [f"core_w_{R}"], yt), yt) for R in RADII]
        star = [" "] * len(RADII); star[int(np.argmax(vals))] = "*"
        print(f"  tau={tau:<4d}  " + "".join(f"{v:+.3f}{s} " for v, s in zip(vals, star)))

    pd.DataFrame(out_rows).to_csv(OUT / "support_concentration_oof.csv", index=False)

    # Abbildung
    ladder = [("median pairwise", bss(ph_cache["median pairwise (alt)"], y)),
              ("median + hub (altes Label)", bss(ph_cache["median + hub (ALTES LABEL)"], y)),
              ("mean dist->Schätzer", bss(ph_cache["mean dist->Schaetzer"], y)),
              (f"Kernanteil ungew. r={R_HEADLINE}", bss(ph_cache[f"Kernanteil ungew. r={R_HEADLINE}"], y)),
              (f"Stütz-Konz. GEW. r={R_HEADLINE} (S)", bss(ph_w, y))]
    fig_path = plot(ladder, ph_w, y)
    print(f"\nCSV: {OUT}/support_concentration_oof.csv")
    print(f"PNG: {fig_path}")


if __name__ == "__main__":
    run()
