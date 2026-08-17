#!/usr/bin/env python3
"""§4.2 KR260 hardware closure figure (single-column, vertical 2-panel).
  (a) load-invariance: gather latency tail (CCDF, log y), idle vs +CPU load. The median is fixed, so both
      start together; only the load tail extends (max/median 1.05 -> 1.22).
  (b) input-invariance: gather median latency across five pathological scene distributions at fixed N_POINTS.
      Bounded under the 53.2 ms WCET; the degenerate single-pillar scene runs faster, never slower.
Style matches Fig 4 (Arial bold, rich palette). -> docs/figs/ip_determinism.{png,pdf,svg}
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

R = "experiments/results/kr260"
TEAL = "#0FA08C"; AMBER = "#EBA13A"; RED = "#E1483B"; INK = "#15202B"; GRAYTX = "#56616C"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.weight": "bold", "svg.fonttype": "none",
    "font.size": 18,
    "axes.labelsize": 19.5, "axes.labelweight": "bold", "axes.labelcolor": INK,
    "axes.edgecolor": "#33414D", "axes.linewidth": 1.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "text.color": INK, "xtick.color": "#33414D", "ytick.color": "#33414D",
    "xtick.labelsize": 17, "ytick.labelsize": 17,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 5.5, "ytick.major.size": 5.5,
    "xtick.major.width": 1.5, "ytick.major.width": 1.5,
    "legend.frameon": False, "legend.fontsize": 16,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

def letter(ax, s, dx=-0.22):
    ax.text(dx, 1.04, s, transform=ax.transAxes, fontsize=26, fontweight="bold",
            va="bottom", ha="left", color=INK)

def load(fn):
    return np.load(os.path.join(R, fn)).astype(float).ravel()

fig = plt.figure(figsize=(9.4, 4.3))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.34,
                      left=0.085, right=0.99, bottom=0.205, top=0.93)
axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

# ---- (a) load-invariance: tail CCDF (log y), normalized by median ----
for label, fn, color in [("idle", "dualip/gather_wcet_idle.npy", TEAL),
                         ("+CPU load", "dualip/gather_wcet_load.npy", RED)]:
    x = load(fn); xs = np.sort(x / np.median(x))
    ccdf = np.clip(1.0 - np.arange(xs.size) / xs.size, 3e-4, 1)
    axA.semilogy(xs, ccdf, color=color, lw=3.4, solid_capstyle="round", label=label, zorder=4)
axA.set_xlim(1.0, 1.25); axA.set_ylim(3e-4, 1)
axA.set_xticks([1.0, 1.05, 1.10, 1.15, 1.20, 1.25])
axA.set_xlabel("latency / median"); axA.set_ylabel("frac. slower than $x$")
axA.legend(loc="upper right", fontsize=15, handlelength=1.5, labelspacing=0.35, borderpad=0.4)
axA.grid(axis="y", color="#ECEFF2", lw=0.9, which="both")
letter(axA, "A")

# ---- (b) input-invariance: median latency across 5 scene distributions ----
SCENES = ["uniform", "scatter", "hotcell", "zipf", "single"]
meds = [np.median(load("gather_inv/gather_inv_%s.npy" % s)) for s in SCENES]
wcet = max(load("gather_inv/gather_inv_%s.npy" % s).max() for s in SCENES)
cols = [TEAL if s != "single" else AMBER for s in SCENES]
xs = np.arange(len(SCENES))
bars = axB.bar(xs, meds, width=0.66, color=cols, edgecolor="white", linewidth=1.1, zorder=3)
for b, m in zip(bars, meds):
    axB.text(b.get_x() + b.get_width()/2, m - 2.4, "%.1f" % m, ha="center", va="top",
             fontsize=15.5, color="white", fontweight="bold", zorder=5)
axB.axhline(wcet, color=RED, ls=(0, (4, 3)), lw=2.1, zorder=2)
axB.text(len(SCENES) / 2 - 0.5, wcet + 2.8, "WCET bound  %.1f ms" % wcet, ha="center", va="bottom",
         fontsize=15, color=RED, fontweight="bold")
axB.set_xticks(xs); axB.set_xticklabels(SCENES, fontsize=15, rotation=20, ha="right", rotation_mode="anchor")
axB.set_ylim(0, wcet + 10); axB.set_yticks([0, 20, 40])
axB.set_ylabel("gather latency (ms)")
letter(axB, "B")

out = "docs/figs/ip_determinism"
os.makedirs("docs/figs", exist_ok=True)
fig.savefig(out + ".png", dpi=440, bbox_inches="tight", pad_inches=0.06)
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.06)
fig.savefig(out + ".svg", bbox_inches="tight", pad_inches=0.06)
print("wrote", out, "| scene medians:", [round(m, 1) for m in meds], "| WCET", round(wcet, 1))
