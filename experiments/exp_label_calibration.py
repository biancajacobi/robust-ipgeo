"""Kalibrierungsanalyse des 2D-Konfidenzlabels (Streuung x Hub-Flag, FF4).

Kernfrage: Ist das Label ein *kalibrierter* Risiko-Indikator für Aggregations-
Misses -- oder „nur" ein diskriminierender (aber unkalibrierter) Triage-Flag?

Ereignis:  Y = 1{Haversine-Fehler > tau km}, tau = 100 (Recall-Definition),
           tau = 25 als Sensitivitaet.
Zwei Sichten, beide OUT-OF-FOLD (10-fach stratifiziert, seed=0 -> kein In-Sample-Zirkel):

  (A) Quadranten-Sicht (label-treu): Forecast = Tail-Rate des Quadranten.
      Misst Diskriminierung (Brier Skill Score, Resolution). Die Reliability ist
      hier ~0 *per Konstruktion* (Forecast = Gruppenmittel) -> kein echter
      Kalibrationstest, aber die Risikostaffelung + Wilson-CIs sind aussagekraeftig.

  (B) Logistisches Surrogat (echter Kalibrationstest): kontinuierlicher Forecast
      p_hat aus den zwei Label-Achsen [log(1+Streuung), Hub] via IRLS-Logit.
      Dieser Forecast KANN miskalibriert sein -> das Reliability-Diagramm ist
      informativ. Surrogat der Achsen, nicht des diskreten Labels selbst.

Liest eval/out/t6_confidence_matrix.csv. Schreibt nach eval/out/:
  label_calibration_quadrants_tau{100,25}.csv
  label_calibration_bins_tau{100,25}.csv
  label_calibration.png  (Reliability-Diagramm, tau=100)

    python experiments/exp_label_calibration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import report  # noqa: E402  (_plt + OUT_DIR)

OUT = report.OUT_DIR
SEED = 0
K = 10


# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #
def stratified_folds(y: np.ndarray, k: int = K, seed: int = SEED) -> np.ndarray:
    """Fold-Index je Beobachtung; jede Klasse wird gleichmaessig auf k Folds verteilt."""
    rng = np.random.default_rng(seed)
    folds = np.empty(len(y), dtype=int)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % k
    return folds


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson-Score-Konfidenzintervall fuer einen Anteil k/n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def murphy(forecast: np.ndarray, y: np.ndarray) -> dict:
    """Brier-Score + Skill + ECE + Murphy-Zerlegung fuer stueckweise-konstante Forecasts.

    Brier = Reliability - Resolution + Uncertainty (Gruppierung nach distinktem Forecast-Wert).
    """
    base = float(y.mean())
    brier = float(np.mean((forecast - y) ** 2))
    rel = res = ece = 0.0
    for f in np.unique(forecast):
        m = forecast == f
        n = int(m.sum())
        obs = float(y[m].mean())
        rel += n / len(y) * (f - obs) ** 2
        res += n / len(y) * (obs - base) ** 2
        ece += n / len(y) * abs(f - obs)
    unc = base * (1 - base)
    bss = 1 - brier / unc if unc > 0 else float("nan")
    return {"base": base, "brier": brier, "bss": bss, "ece": ece,
            "reliability": rel, "resolution": res, "uncertainty": unc}


# --------------------------------------------------------------------------- #
# (A) Quadranten-Sicht
# --------------------------------------------------------------------------- #
def quadrant_calibration(df: pd.DataFrame, tau: float) -> tuple[dict, pd.DataFrame]:
    y = (df.est_error_km.values > tau).astype(int)
    quad = list(zip(df.low_spread.values, df.hub.values))
    folds = stratified_folds(y, K, SEED)

    p_hat = np.empty(len(y))
    for f in range(K):
        tr, te = folds != f, folds == f
        global_rate = y[tr].mean()
        rate = {}
        for q in set(quad):
            m = np.array([qi == q for qi in quad]) & tr
            rate[q] = y[m].mean() if m.sum() else global_rate
        for i in np.where(te)[0]:
            p_hat[i] = rate[quad[i]]

    metrics = murphy(p_hat, y)
    rows = []
    names = {(True, False): "belastbar", (True, True): "Sicherheits-Illusion",
             (False, False): "ehrliche Unsicherheit", (False, True): "Inkonsistenz"}
    for q in [(True, False), (True, True), (False, False), (False, True)]:
        m = np.array([qi == q for qi in quad])
        n = int(m.sum())
        lo, hi = wilson(int(y[m].sum()), n)
        rows.append({"quadrant": names[q], "low_spread": q[0], "hub": q[1], "n": n,
                     "forecast": float(p_hat[m].mean()) if n else np.nan,
                     "observed": float(y[m].mean()) if n else np.nan,
                     "ci_lo": lo, "ci_hi": hi})
    return metrics, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# (B) Logistisches Surrogat (IRLS, dependency-frei)
# --------------------------------------------------------------------------- #
def _logreg_fit(X: np.ndarray, y: np.ndarray, iters: int = 100, ridge: float = 1e-6) -> np.ndarray:
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        W = p * (1 - p) + 1e-9
        grad = X.T @ (p - y) + ridge * w
        H = (X * W[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w


def _design(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.log1p(df.spread_km.values), df.hub.values.astype(float)])


def logistic_calibration(df: pd.DataFrame, tau: float, n_bins: int = 5
                         ) -> tuple[dict, pd.DataFrame, np.ndarray]:
    y = (df.est_error_km.values > tau).astype(int)
    Xraw = _design(df)
    folds = stratified_folds(y, K, SEED)

    p_hat = np.empty(len(y))
    for f in range(K):
        tr, te = folds != f, folds == f
        mu, sd = Xraw[tr].mean(0), Xraw[tr].std(0) + 1e-9
        Xtr = np.column_stack([np.ones(tr.sum()), (Xraw[tr] - mu) / sd])
        Xte = np.column_stack([np.ones(te.sum()), (Xraw[te] - mu) / sd])
        w = _logreg_fit(Xtr, y[tr])
        p_hat[te] = 1.0 / (1.0 + np.exp(-Xte @ w))

    base = float(y.mean())
    brier = float(np.mean((p_hat - y) ** 2))
    bss = 1 - brier / (base * (1 - base))

    # Quantil-Bins (robust bei seltenem Ereignis); ECE gewichtet
    edges = np.unique(np.quantile(p_hat, np.linspace(0, 1, n_bins + 1)))
    bin_idx = np.clip(np.digitize(p_hat, edges[1:-1]), 0, len(edges) - 2)
    rows, ece = [], 0.0
    for b in range(len(edges) - 1):
        m = bin_idx == b
        n = int(m.sum())
        if n == 0:
            continue
        fbar, obs = float(p_hat[m].mean()), float(y[m].mean())
        lo, hi = wilson(int(y[m].sum()), n)
        ece += n / len(y) * abs(fbar - obs)
        rows.append({"bin": b, "n": n, "forecast": fbar, "observed": obs,
                     "ci_lo": lo, "ci_hi": hi})
    metrics = {"base": base, "brier": brier, "bss": bss, "ece": ece}
    return metrics, pd.DataFrame(rows), p_hat


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def plot_reliability(bins: pd.DataFrame, quad: pd.DataFrame, tau: float) -> Path:
    plt = report._plt()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    hi = max(bins.ci_hi.max(), quad.ci_hi.max(), bins.forecast.max()) * 1.05

    ax.plot([0, hi], [0, hi], "k--", alpha=0.5, label="perfekt kalibriert")
    ax.errorbar(bins.forecast, bins.observed,
                yerr=[bins.observed - bins.ci_lo, bins.ci_hi - bins.observed],
                fmt="o-", capsize=3, label="logistisches Surrogat (Bins)")
    ax.scatter(quad.forecast, quad.observed, s=20 + quad.n / 3.0, marker="s",
               color="tab:red", zorder=5, label="Quadranten (label-treu)")
    for _, r in quad.iterrows():
        ax.annotate(r["quadrant"], (r.forecast, r.observed), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")

    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel(f"Vorhergesagte Miss-Wahrscheinlichkeit (Fehler > {tau:.0f} km)")
    ax.set_ylabel("Beobachtete Miss-Rate (out-of-fold)")
    ax.set_title("Reliability-Diagramm: 2D-Konfidenzlabel")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "label_calibration.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
def run() -> None:
    csv = OUT / "t6_confidence_matrix.csv"
    if not csv.exists():
        print(f"Fehlt: {csv} — erst: python experiments/exp_t6_defaults.py")
        return
    df = pd.read_csv(csv)

    for tau in (100, 25):
        qm, qtab = quadrant_calibration(df, tau)
        lm, ltab, _ = logistic_calibration(df, tau)
        print("\n" + "=" * 74)
        print(f"KALIBRIERUNG Konfidenzlabel  |  tau={tau} km  |  Basisrate={qm['base']:.3f}  (n={len(df)})")
        print("=" * 74)
        print(f"(A) Quadranten-Sicht (Diskriminierung): "
              f"Brier={qm['brier']:.4f}  BSS={qm['bss']:+.3f}  "
              f"Resolution={qm['resolution']:.4f}  Reliability={qm['reliability']:.4f}")
        print(qtab.round(3).to_string(index=False))
        print(f"\n(B) Logistisches Surrogat (Kalibrierung): "
              f"Brier={lm['brier']:.4f}  BSS={lm['bss']:+.3f}  ECE={lm['ece']:.3f}")
        print(ltab.round(3).to_string(index=False))
        qtab.to_csv(OUT / f"label_calibration_quadrants_tau{tau}.csv", index=False)
        ltab.to_csv(OUT / f"label_calibration_bins_tau{tau}.csv", index=False)

    # Reliability-Diagramm fuer die Headline-Schwelle tau=100
    qm100, qtab100 = quadrant_calibration(df, 100)
    lm100, ltab100, _ = logistic_calibration(df, 100)
    path = plot_reliability(ltab100, qtab100, 100)
    print(f"\nAbbildung: {path}")


if __name__ == "__main__":
    run()
