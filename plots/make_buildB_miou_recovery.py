#!/usr/bin/env python3
"""Build-B accuracy recovery: on-board DPU-INT8 mIoU as the per-tensor-pow2 quantization is fixed.
16-frame Occ3D val subset (first 16, val order). Shows the fix-point-clamp recovering the BEV stage
and isolates the image stage as the remaining (QAT-addressed) drag."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "experiments/results/buildB"
# measured on-board mIoU (16-frame subset)
bars = [
    ("FP32\nreference", 23.83, "#444"),
    ("both stages\nPTQ (start)", 4.77, "#b03a2e"),
    ("BEV fix-point-clamp\n+ image PTQ", 12.47, "#e08e0b"),
    ("BEV-only clamp\n(FP32 vt in)", 19.97, "#2e86c1"),
]
fig, ax = plt.subplots(figsize=(8.6, 4.4))
xs = range(len(bars)); vals = [b[1] for b in bars]; cols = [b[2] for b in bars]
b = ax.bar(xs, vals, color=cols, width=0.62)
ax.set_xticks(list(xs)); ax.set_xticklabels([b[0] for b in bars], fontsize=9)
ax.set_ylabel("Occ3D-nuScenes mIoU (16-frame val subset)")
ax.set_ylim(0, 26)
for r, v in zip(b, vals):
    ax.text(r.get_x()+r.get_width()/2, v+0.4, "%.2f" % v, ha="center", fontsize=10, fontweight="bold")
ax.axhline(23.83, ls="--", lw=0.8, color="#444", alpha=0.5)
ax.annotate("BEV stage matched\n(84% of FP32) via\nfix-point clamp", xy=(3, 19.97), xytext=(2.4, 7.5),
            fontsize=8.5, ha="center", arrowprops=dict(arrowstyle="->", color="#2e86c1"))
ax.annotate("image PTQ depth is\nthe remaining drag\n(QAT in progress)", xy=(2, 12.47), xytext=(1.15, 17.5),
            fontsize=8.5, ha="center", arrowprops=dict(arrowstyle="->", color="#e08e0b"))
ax.set_title("Build-B: recovering on-board DPU-INT8 occupancy mIoU\n(per-tensor power-of-2 8-bit; clamp via fix-point override)", fontsize=11)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join("docs/figs/buildB", "miou_recovery." + ext), dpi=140, bbox_inches="tight")
print("saved docs/figs/buildB/miou_recovery.{png,pdf}")
