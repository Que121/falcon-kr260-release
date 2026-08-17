#!/usr/bin/env python3
"""Absolute-deadline companion to the relative-margin payoff figure.
Miss rate of the deployed six-camera KR260 pipeline against an absolute per-frame deadline in ms,
with the 100 ms perception-budget marker. Also reconciles the margin bookkeeping: the 8% zero-miss
margin belongs to the DPU probe series; the six-camera pipeline needs 14% idle and 2% dedicated.
Writes traces/deadline_absolute.csv and figs/fig_deadline_absolute.{png,pdf}.
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

R = "traces"
TEAL = "#0FA08C"; BLUE = "#2F66C4"; AMBER = "#EBA13A"; RED = "#E1483B"
INK = "#15202B"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.weight": "bold", "svg.fonttype": "none",
    "font.size": 21,
    "axes.labelsize": 21, "axes.labelweight": "bold", "axes.labelcolor": INK,
    "axes.edgecolor": "#33414D", "axes.linewidth": 1.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "text.color": INK, "xtick.color": "#33414D", "ytick.color": "#33414D",
    "xtick.labelsize": 19, "ytick.labelsize": 19,
    "legend.frameon": False, "legend.fontsize": 15,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

SERIES = [
    ("probe seed 1", "kr260/e1/kr260_6cam_dedicated_100k.npy", "#0B5FA5", "-", 4.2),
    ("probe seed 2", "kr260/e1/kr260_6cam_dedicated_100k_s2.npy", "#4C8FD6", "-", 3.2),
    ("probe seed 3", "kr260/e1/kr260_6cam_dedicated_100k_s3.npy", "#8FBBE8", "-", 3.2),
    ("6-cam probe (dedic.)", "kr260/kr260_6cam_taskset_a2.npy", TEAL, "-", 3.0),
    ("pipeline (taskset)",   "kr260/kr260_6cam_taskset.npy",    BLUE, "-", 2.6),
    ("pipeline (idle host)", "kr260/kr260_6cam_idle.npy",       AMBER, (0, (4, 2)), 2.6),
]

deadlines = np.linspace(90, 172, 901)
fig, ax = plt.subplots(figsize=(4.8, 3.85))
fig.subplots_adjust(left=0.145, right=0.975, bottom=0.155, top=0.965)

rows = ["series,n,p50_ms,p99_ms,max_ms,miss_pct_100ms,miss_pct_110ms,margin_zero_miss"]
for label, f, color, ls, lw in SERIES:
    x = np.load(os.path.join(R, f)).astype(float).ravel()
    mr = np.array([np.mean(x > d) for d in deadlines]) * 100
    ax.plot(deadlines, np.clip(mr, 8e-3, 100), color=color, ls=ls, lw=lw,
            solid_capstyle="round", label=label)
    rows.append("%s,%d,%.2f,%.2f,%.2f,%.3f,%.3f,%.4f" % (
        label, len(x), np.median(x), np.percentile(x, 99), x.max(),
        100 * np.mean(x > 100.0), 100 * np.mean(x > 110.0), x.max() / np.median(x)))

ax.axvline(100.0, color=RED, lw=2.2, ls=(0, (2, 2)), zorder=1)
ax.text(101.3, 1.0, "100 ms budget", color=RED, fontsize=13,
        fontweight="bold", rotation=90, va="center")
ax.set_yscale("log")
ax.set_ylim(8e-3, 100)
ax.set_xlim(90, 172)
ax.set_xlabel("absolute deadline (ms)")
ax.set_ylabel("Deadline-miss rate (%)")
ax.legend(loc="upper right", labelspacing=0.35, handlelength=1.5, borderpad=0.25, handletextpad=0.5)

os.makedirs("paper/figs", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig("figs/fig_deadline_absolute.%s" % ext, dpi=300, bbox_inches="tight", pad_inches=0.05)
open(os.path.join(R, "deadline_absolute.csv"), "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
