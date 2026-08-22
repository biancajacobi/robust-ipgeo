#!/usr/bin/env python3
"""Population dashboard as a figure: anchors · probes · pool · Giga.

Population-internal view (the suite run freshly per population,
t6_point_estimators.csv of the respective out directories, L1·b): median
error (log) and tail rate >100 km side by side. The frozen transfer numbers
(pool/Giga) deviate only marginally (freeze delta n.s. resp. small, see
exp_pool_transfer/exp_giga_transfer) — the figure shows the regime shift,
not the transfer deltas.

Usage:  python experiments/plot_populations.py
Result: eval/out/populations_dashboard.png (missing populations are skipped
        and reported)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POPS = [
    ("Anchors\n(n=1,077)", "out"),
    ("Probes 500", "out_probes"),
    ("Pool 8,896", "out_probes_pool"),
    ("Giga 2,127", "out_giga"),
]
BLUE = "#1f77b4"


def main() -> None:
    rows = []
    for label, d in POPS:
        # Pool SPECIAL CASE: the pool suite's t6_point_estimators runs on the
        # COMPLETED observation set (sixth line via the paid ipwho.is quota
        # = the T8 addendum, not pre-specified). The figure must show the
        # pre-specified five-line primary analysis (figure-text consistency
        # with T8) -> fresh in-population number from exp_pool_transfer.
        if d == "out_probes_pool":
            p5 = ROOT / "eval" / "out" / "pool_transfer.csv"
            if not p5.exists():
                print("(skipped, run exp_pool_transfer first)")
                continue
            t5 = pd.read_csv(p5)
            r5 = t5[(t5.population == "probes_pool")
                    & (t5.config.isin(["frisch", "fresh"]))].iloc[0]
            rows.append((label, float(r5.median_km), float(r5.tail_pct)))
            continue
        p = ROOT / "eval" / d / "t6_point_estimators.csv"
        if not p.exists():
            print(f"(skipped, not yet computed: eval/{d})")
            continue
        t = pd.read_csv(p)
        e = t[(t.level == "L1") & (t.variant == "b")].error_km
        rows.append((label, float(e.median()), float(100 * (e > 100).mean())))
    if len(rows) < 2:
        sys.exit("too few populations computed — run the suite(s) first")

    labels = [r[0] for r in rows][::-1]
    med = [r[1] for r in rows][::-1]
    tail = [r[2] for r in rows][::-1]
    y = range(len(rows))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
    ax1.hlines(y, 1, med, color="0.85", lw=1.5, zorder=1)
    ax1.plot(med, y, "s", color=BLUE, markersize=8, zorder=3)
    for yi, v in zip(y, med):
        ax1.annotate(f"{v:.1f}", (v, yi), textcoords="offset points",
                     xytext=(7, 6), fontsize=8, color="0.25")
    ax1.axvline(100, color="grey", lw=1, alpha=0.7, zorder=2)
    ax1.text(112, len(rows) - 0.8, "tail threshold\n100 km", fontsize=7,
             color="grey", va="top")
    ax1.set_xscale("log")
    ax1.set_xlim(1, 400)
    ax1.set_yticks(list(y))
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel("median error L1·b (km, log)", fontsize=9)
    ax1.grid(True, axis="x", alpha=0.3)

    ax2.hlines(y, 0, tail, color="0.85", lw=1.5, zorder=1)
    ax2.plot(tail, y, "s", color=BLUE, markersize=8, zorder=3)
    for yi, v in zip(y, tail):
        ax2.annotate(f"{v:.1f}", (v, yi), textcoords="offset points",
                     xytext=(7, 6), fontsize=8, color="0.25")
    ax2.set_xlim(0, 65)
    ax2.set_xlabel("tail > 100 km (%)", fontsize=9)
    ax2.grid(True, axis="x", alpha=0.3)

    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = ROOT / "eval" / "out" / "populations_dashboard.png"
    fig.savefig(out, dpi=130)
    print(f"figure: {out}")
    for (label, m, t_) in rows:
        print(f"  {label.replace(chr(10), ' '):20s} median {m:6.1f} km | tail {t_:4.1f} %")


if __name__ == "__main__":
    main()
