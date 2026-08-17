#!/usr/bin/env python
"""Figure: the board's 16.37 is NOT an INT8 ceiling. Every faithful DPU realization effect barely moves
the sim (23.13 -> 22.61); the board sits ~6 pts below the faithful-INT8 ceiling = recoverable deployment."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

labels = ["FP32", "idealized\nINT8",
          "+act\nsaturation", "+weight\nsaturation", "+BN-fold", "+static\nfix-points\n(DPU-faithful)",
          "On-board\n(deployed\nxmodels)"]
vals   = [25.0, 23.13, 23.70, 23.73, 22.87, 22.61, 16.37]
cols   = ["#4c4c4c", "#2a9d8f", "#2a9d8f", "#2a9d8f", "#2a9d8f", "#1d6f63", "#c1121f"]

fig, ax = plt.subplots(figsize=(10.6, 4.8))
x = range(len(vals))
ax.bar(x, vals, color=cols, width=0.66, zorder=3)
for xi, v in zip(x, vals):
    ax.text(xi, v + 0.25, "%.2f" % v, ha="center", va="bottom", fontsize=10.5, fontweight="bold")

# faithful-INT8 ceiling guide line
ax.axhline(22.61, ls="--", color="#1d6f63", lw=1.3, zorder=1)
ax.text(0.05, 22.9, "faithful-INT8 ceiling 22.61", color="#1d6f63", fontsize=8.6, fontweight="bold")

# the recoverable gap arrow (faithful ceiling -> board)
ax.add_patch(FancyArrowPatch((6, 22.61), (6, 16.37 + 0.2), arrowstyle="<->", color="#c1121f",
                             lw=2.2, mutation_scale=14, zorder=4))
ax.text(5.55, 19.4, "≈6.2 pt\nRECOVERABLE\ndeployment gap\n(not a ceiling)", ha="right", va="center",
        fontsize=9.2, color="#c1121f", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#c1121f", alpha=0.95))

ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8.4)
ax.set_ylabel("mIoU (Occ3D-nuScenes, 256-frame val subset)", fontsize=9.5)
ax.set_ylim(0, 27)
ax.set_title("Why the board (16.37) is below the sim: it is NOT the INT8 math.\n"
             "Every real DPU effect (saturation, BN-fold, static fix-points) costs ~0.5 pt total;\n"
             "the board sits ~6 pt under the faithful-INT8 ceiling = recoverable Vitis deployment.",
             fontsize=10.0)
ax.grid(axis="y", ls=":", alpha=0.5, zorder=0)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig("docs/figs/buildB/why_gap.%s" % ext, dpi=160, bbox_inches="tight")
print("saved docs/figs/buildB/why_gap.{png,pdf}")
