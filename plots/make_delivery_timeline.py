#!/usr/bin/env python3
"""The delivery-timeline figure (R2): raw inference latency vs delivered-output latency, CCDF,
one axis, from the 100k-frame enforced run (e5_delivery_D100.npz). The raw tail runs past the
budget to 132.5 ms; the delivered curve terminates at deadline + enforcement envelope. Writes
paper/figs/fig_delivery_timeline.{png,pdf}.
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

TEAL = "#0FA08C"; RED = "#E1483B"; INK = "#15202B"
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.weight": "bold", "svg.fonttype": "none", "font.size": 21,
    "axes.labelsize": 21, "axes.labelweight": "bold", "axes.labelcolor": INK,
    "axes.edgecolor": "#33414D", "axes.linewidth": 1.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "text.color": INK, "xtick.color": "#33414D", "ytick.color": "#33414D",
    "xtick.labelsize": 19, "ytick.labelsize": 19,
    "legend.frameon": False, "legend.fontsize": 15,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

z = np.load("experiments/results/kr260/e1/e5_delivery_D100.npz")
raw, dlv = z["raw"], z["delivered"]
D = 100.0

def ccdf(x):
    xs = np.sort(x)
    p = 1.0 - np.arange(1, len(xs) + 1) / len(xs)
    return xs, np.maximum(p, 1.0 / len(xs))

fig, ax = plt.subplots(figsize=(4.8, 3.85))
fig.subplots_adjust(left=0.15, right=0.97, bottom=0.155, top=0.96)
xs, p = ccdf(raw)
ax.plot(xs, p, color=RED, lw=3.2, label="raw inference (unenforced)")
xs, p = ccdf(dlv)
ax.plot(xs, p, color=TEAL, lw=3.6, label="delivered output (enforced)")
ax.axvline(D, color=INK, lw=2.0, ls=(0, (2, 2)))
ax.text(D + 0.7, 3e-4, "100 ms deadline", fontsize=14, fontweight="bold", rotation=90, va="bottom")
ax.set_yscale("log")
ax.set_xlim(94, 136)
ax.set_ylim(1e-5, 1)
ax.set_xlabel("Per-frame latency (ms)")
ax.set_ylabel("CCDF  P(latency > x)")
ax.legend(loc="upper right")
os.makedirs("paper/figs", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig("paper/figs/fig_delivery_timeline.%s" % ext, dpi=300, bbox_inches="tight", pad_inches=0.05)
print("raw max %.2f | delivered max %.2f | fallbacks %d" % (raw.max(), dlv.max(), int(z["stale"].sum())))
