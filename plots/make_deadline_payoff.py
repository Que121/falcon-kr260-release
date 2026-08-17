#!/usr/bin/env python3
"""Contribution 3 -- the hard-deadline payoff: deadline-miss rate vs. WCET margin, per platform.
Set the hard deadline at a fixed relative margin d over each platform's own median (T = d*p50) and
measure the miss rate P(latency > T). KR260 (fixed-compute DPU) reaches zero misses at a small margin;
tailed GPU / Orin do not. Nature style (Arial bold, palette matched to Figs 4-5), no title, legend off
the curves. Writes experiments/results/deadline_payoff.csv and docs/figs/fig_deadline_payoff.{png,pdf,svg}.
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = "experiments/results"
TEAL = "#0FA08C"; BLUE = "#2F66C4"; AMBER = "#EBA13A"; RED = "#E1483B"; PLUM = "#8A52A8"
INK = "#15202B"; GRAYTX = "#56616C"

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
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 6, "ytick.major.size": 6,
    "xtick.major.width": 1.5, "ytick.major.width": 1.5,
    "legend.frameon": False, "legend.fontsize": 13,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

# (label, file, color, linestyle, lw, zorder)
SERIES = [
    ("KR260 probe (dedic.)", "kr260/e1/kr260_6cam_dedicated_100k.npy",  TEAL,  "-",          4.2, 7),
    ("KR260 DPU (idle)",    "kr260/kr260_long_idle_100k.npy",             TEAL,  (0, (4, 2)),  2.6, 6),
    ("Orin 15W (deployed)",       "orin/occfpga_orin_idle_15W_locked.npy",      AMBER, "-",          2.6, 4),
    ("Orin (thermal)",            "orin/occfpga_orin_sustained_15W_locked.npy", AMBER, (0, (4, 2)),  2.6, 4),
    ("Orin (+ co-tenant)",        "orin/occfpga_orin_cotenant_15W_locked.npy",  PLUM,  "-",          2.4, 3),
    ("GPU (loaded)",  "occfpga_gpu_long_loaded.npy",                RED,   "-",          3.6, 5),
]

def load(f):
    p = os.path.join(R, f); return np.load(p).astype(float).ravel() if os.path.exists(p) else None
def miss_rate(x, d): return float(np.mean(x > d * np.median(x)))

margins = np.linspace(1.0, 1.6, 241)
rows, handles = [], []
fig, ax = plt.subplots(figsize=(4.8, 3.85))
fig.subplots_adjust(left=0.135, right=0.975, bottom=0.155, top=0.965)

for label, f, color, ls, lw, z in SERIES:
    x = load(f)
    if x is None:
        print("  [skip missing]", f); continue
    mr = np.array([miss_rate(x, d) for d in margins]) * 100
    ax.plot(margins, np.clip(mr, 8e-3, 100), color=color, ls=ls, lw=lw, zorder=z, solid_capstyle="round")
    handles.append(Line2D([0], [0], color=color, ls=ls, lw=3.2, label=label))
    rows.append((label, x))

# 10% WCET-margin reference
ax.axvspan(1.0, 1.10, color=TEAL, alpha=0.06, zorder=0)
ax.axvline(1.10, color=GRAYTX, ls=":", lw=1.5, zorder=1)
ax.text(1.043, 0.0115, "10% margin", fontsize=14, color=GRAYTX, fontweight="bold", ha="center")

ax.set_yscale("log"); ax.set_ylim(8e-3, 2000); ax.set_xlim(1.0, 1.30)
ax.set_xticks([1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30])
ax.set_xlabel("hard deadline / platform p50")
ax.set_ylabel("deadline-miss rate (%)")
ax.grid(True, which="major", color="#E9ECEF", lw=1.0)
ax.grid(True, which="minor", color="#F4F6F7", lw=0.7)
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.005),
          ncol=2, fontsize=13.5, labelspacing=0.3, columnspacing=1.0,
          handlelength=1.4, borderpad=0.35, handletextpad=0.45,
          frameon=True, facecolor="white", framealpha=1.0, edgecolor="none")

# table + csv (unchanged)
DCOLS = [1.05, 1.10, 1.20, 1.40]
csv = ["platform,p50_ms,max_over_p50,cv_pct," + ",".join("miss_pct_d%.2f" % d for d in DCOLS) + ",margin_zero_miss"]
for label, x in rows:
    p50 = np.median(x); mz = x.max() / p50; cv = 100 * x.std() / x.mean()
    msr = [100 * miss_rate(x, d) for d in DCOLS]
    csv.append("%s,%.4f,%.4f,%.4f,%s,%.4f" % (label, p50, x.max()/p50, cv, ",".join("%.4f" % m for m in msr), mz))
open(os.path.join(R, "deadline_payoff.csv"), "w").write("\n".join(csv) + "\n")

out = "docs/figs/fig_deadline_payoff"
os.makedirs("docs/figs", exist_ok=True)
fig.savefig(out + ".png", dpi=420, bbox_inches="tight", pad_inches=0.06)
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.06)
fig.savefig(out + ".svg", bbox_inches="tight", pad_inches=0.06)
print("wrote", out, "| series:", [r[0] for r in rows])
for label, x in rows:
    print("  %-26s miss@1.10=%.3f%%  zero@%.3f" % (label, 100*miss_rate(x, 1.10), x.max()/np.median(x)))
