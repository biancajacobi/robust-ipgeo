"""T6-Visualisierungen: ECDF (FF1), Konfidenz-Heatmap (mit Recall), Forest (2-spaltig).

Erzeugt drei Abbildungen nach eval/out/. Konventionen (durchgängig):
  - s/w-tauglich: zusätzlich zur Farbe Linien-/Marker-Typ-Varianz.
  - Achsen in km, Einheit ausgeschrieben („Fehler / km").
  - n-Werte im Titel jeder Abbildung (gegen Stichprobengrößen-Nachfragen).
  - Variante (a/b/c/d) = Farbe; Linien-Skala (L0/L1/L2) = Marker/Position.
Headline-Konfiguration: linear, ε = Median der LOO-Pseudo-Radien (Grid-Wert).

Aufruf:  python experiments/plot_t6.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_t6_defaults as t6           # noqa: E402
from eval.pipeline import load_cases    # noqa: E402
from eval.metrics import haversine_error  # noqa: E402

OUT = Path("eval/out"); OUT.mkdir(parents=True, exist_ok=True)
VAR_COLOR = {"a": "#444444", "b": "#1f77b4", "c": "#2ca02c", "d": "#d62728"}
VAR_LABEL = {"a": "(a) naiv", "b": "(b) konfidenzgew.", "c": "(c) gefiltert", "d": "(d) b+c"}
LVL_MARKER = {"L0": "o", "L1": "s", "L2": "^"}


def _ecdf(ax, errs, label, color, ls, marker):
    x = np.sort(np.asarray(errs, dtype=float))
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post", label=label, color=color, linestyle=ls, linewidth=1.8)
    # wenige Marker zur s/w-Unterscheidung
    idx = np.linspace(0, len(x) - 1, 8).astype(int)
    ax.plot(x[idx], y[idx], marker, color=color, linestyle="none", markersize=5)


def fig_ecdf(cases, loo, eps):
    e_naive = [haversine_error(t6.estimate(c, "L0", "a", loo), c["truth"]) for c in cases]
    e_best = [haversine_error(t6.estimate(c, "L1", "b", loo, eps=eps), c["truth"]) for c in cases]
    e_ipinfo = [haversine_error((p["lat"], p["lon"]), c["truth"])
                for c in cases for p in c["provenance"] if p["source"] == "ipinfo"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _ecdf(ax, e_naive, "L0+(a) naiv (Basislinie)", "#444444", ":", "o")
    _ecdf(ax, e_best, "L1+(b) beste Aggregation", "#1f77b4", "-", "s")
    _ecdf(ax, e_ipinfo, "ipinfo (beste Einzelquelle, ex post)", "#d62728", "--", "^")
    ax.axvline(100, color="grey", linewidth=1, alpha=0.7)
    ax.text(105, 0.05, "Tail-Schwelle 100 km", rotation=90, fontsize=8, color="grey", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("Fehler / km (log)")
    ax.set_ylabel("Anteil Fälle ≤ x  (ECDF)")
    ax.set_title(f"FF1 — Fehler-ECDF: Aggregation vs. beste Einzelquelle  (n = {len(cases)} Anchors)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "t6_ecdf.png", dpi=150)
    plt.close(fig)


def fig_heatmap(cases, loo, eps):
    rows = []
    miss = fmiss = fall = 0
    for c in cases:
        pts = c["points"]; n = len(pts)
        pd_ = [haversine_error(tuple(pts[i]), tuple(pts[j])) for i in range(n) for j in range(i + 1, n)]
        spread = float(np.median(pd_)) if pd_ else 0.0
        hub = np.mean([p["is_default_centroid"] for p in c["provenance"]]) >= 0.5
        err = haversine_error(t6.estimate(c, "L1", "b", loo, eps=eps), c["truth"])
        rows.append((spread < 50, hub, err))
        unc = (spread >= 50) or hub
        fall += unc
        if err > 100:
            miss += 1; fmiss += unc
    rows = np.array(rows, dtype=object)

    grid_tail = np.zeros((2, 2)); cell_txt = [["", ""], ["", ""]]
    for r, low in enumerate([True, False]):           # Zeile 0=niedrige Streuung
        for col, hub in enumerate([False, True]):       # Spalte 0=kein Hub
            errs = np.array([e for (lo, hu, e) in rows if lo == low and hu == hub], dtype=float)
            tail = 100 * (errs > 100).mean() if len(errs) else 0
            grid_tail[r, col] = tail
            cell_txt[r][col] = f"n={len(errs)}\nMed {np.median(errs):.0f} km\n>100km: {tail:.0f}%"

    fig, (ax, axt) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [3, 2]})
    im = ax.imshow(grid_tail, cmap="RdYlGn_r", vmin=0, vmax=45, aspect="auto")
    ax.set_xticks([0, 1], ["kein Hub", "Hub"])
    ax.set_yticks([0, 1], ["niedrige\nStreuung", "hohe\nStreuung"])
    quad = [["belastbar", "Sicherheits-Illusion"], ["ehrliche Unsicherheit", "Inkonsistenz"]]
    for r in range(2):
        for col in range(2):
            ax.text(col, r - 0.28, quad[r][col], ha="center", va="center", fontsize=9, fontweight="bold")
            ax.text(col, r + 0.12, cell_txt[r][col], ha="center", va="center", fontsize=8)
    ax.set_title(f"Konfidenz-Matrix: Streuung × Hub-Flag  (n = {len(cases)})")
    fig.colorbar(im, ax=ax, label="Anteil Fehler > 100 km / %", fraction=0.046)

    axt.axis("off")
    axt.text(0.0, 0.5,
             "Forensischer Hauptbefund\n(Recall der Konfidenz-Flags\nauf Aggregations-Misses):\n\n"
             f"• Misses (>100 km):  n = {miss}\n"
             f"• davon als unsicher\n  geflaggt:  {100*fmiss/miss:.0f} %  (Recall)\n"
             f"• Gesamt-Flag-Quote:  {100*fall/len(cases):.0f} %\n"
             f"• Präzision auf Misses:  {100*fmiss/fall:.0f} %\n\n"
             "→ scheitert „laut\", nicht „leise\"",
             fontsize=9, va="center", family="monospace",
             bbox=dict(boxstyle="round", facecolor="#f3f3f3", edgecolor="grey"))
    fig.tight_layout()
    fig.savefig(OUT / "t6_confidence_heatmap.png", dpi=150)
    plt.close(fig)


def fig_forest(cases, loo, eps):
    combos = [(lvl, v) for lvl in t6.LEVELS for v in t6.VARIANTS]
    means, tails, labels, colors, markers = [], [], [], [], []
    for lvl, v in combos:
        errs = np.array([haversine_error(t6.estimate(c, lvl, v, loo, eps=eps), c["truth"])
                         for c in cases])
        means.append(errs.mean()); tails.append(100 * (errs > 100).mean())
        labels.append(f"{lvl} · {v}"); colors.append(VAR_COLOR[v]); markers.append(LVL_MARKER[lvl])
    y = np.arange(len(combos))[::-1]   # oben = L0·a

    fig, (axm, axt) = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    for i in range(len(combos)):
        axm.plot(means[i], y[i], markers[i], color=colors[i], markersize=9)
        axt.plot(tails[i], y[i], markers[i], color=colors[i], markersize=9)
    axm.axvline(86.6, color="#d62728", linestyle="--", linewidth=1, label="ipinfo (Einzelquelle)")
    axm.set_yticks(y, labels, fontsize=8)
    axm.set_xlabel("Mittlerer Fehler / km"); axt.set_xlabel("Anteil Fehler > 100 km / %")
    axm.set_title("Tail-sensitiv: Mittelwert"); axt.set_title("Badly-wrong-Rate")
    for a in (axm, axt):
        a.grid(True, axis="x", alpha=0.25)
    # Legenden: Farbe=Variante, Marker=Linie
    from matplotlib.lines import Line2D
    var_leg = [Line2D([], [], marker="o", color=VAR_COLOR[v], linestyle="none", label=VAR_LABEL[v])
               for v in t6.VARIANTS]
    lvl_leg = [Line2D([], [], marker=LVL_MARKER[l], color="grey", linestyle="none", label=l)
               for l in t6.LEVELS]
    axm.legend(handles=var_leg + [Line2D([], [], color="#d62728", linestyle="--", label="ipinfo")],
               fontsize=7, loc="lower right")
    axt.legend(handles=lvl_leg, fontsize=7, loc="lower right", title="Linien-Skala")
    fig.suptitle(f"T6 — Linien-Skala × Aggregator-Variante  (n = {len(cases)} Anchors, Headline ε={eps} km, linear)")
    fig.tight_layout()
    fig.savefig(OUT / "t6_forest.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    cases = load_cases()
    loo = t6.loo_pseudo_radii(cases)
    eps = min(t6.EPS_GRID, key=lambda e: abs(e - np.median([loo[s]["_global"] for s in loo])))
    print(f"Headline-ε = {eps} km. Erzeuge 3 Abbildungen …")
    fig_ecdf(cases, loo, eps)
    fig_heatmap(cases, loo, eps)
    fig_forest(cases, loo, eps)
    print(f"Gespeichert: {OUT}/t6_ecdf.png, t6_confidence_heatmap.png, t6_forest.png")
