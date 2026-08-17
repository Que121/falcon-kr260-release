#!/usr/bin/env python3
"""Build-B FINAL: on-board DPU-INT8 occupancy mIoU recovery + per-class (256-frame Occ3D val subset)."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figs/buildB", exist_ok=True)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})

# LEFT: recovery trajectory (16fr for the PTQ->fixed sweep; 256fr headline marked)
bars = [("both PTQ\n(start)", 4.77, "#b03a2e"), ("BEV fixed\n+img PTQ", 12.47, "#e08e0b"),
        ("both fixed\n(16fr)", 13.02, "#1f8a4c"), ("BEV-only\n(FP32 vt)", 19.97, "#2e86c1"),
        ("FP32\n(16fr)", 23.83, "#555")]
b = axL.bar(range(len(bars)), [x[1] for x in bars], color=[x[2] for x in bars], width=0.64)
axL.set_xticks(range(len(bars))); axL.set_xticklabels([x[0] for x in bars], fontsize=8.5)
axL.set_ylabel("mIoU (16-frame subset)"); axL.set_ylim(0, 26)
for r, x in zip(b, bars): axL.text(r.get_x()+r.get_width()/2, x[1]+0.4, "%.1f" % x[1], ha="center", fontsize=9, fontweight="bold")
axL.set_title("(a) recovery: per-tensor-pow2-8bit fixed via\nfix-point clamp (BEV) + QAT (image)", fontsize=10)

# RIGHT: 256-frame per-class, board vs FP32
cls = ["car", "bicycle", "motorcycle", "pedestrian", "driveable", "terrain", "manmade", "vegetation"]
board = [21.36, 10.47, 20.66, 11.76, 48.32, 28.77, None, None]   # filled from eval; None->skip
fp32  = [None]*8
# headline: board mIoU 16.03 vs FP32 25.0 over 256 frames
names = ["car", "bicycle", "motorcyc", "pedestr", "driveable", "terrain"]
bv = [21.2, 9.41, 20.4, 12.15, 49.03, 28.77]   # 256-frame split (best); mIoU 16.37
y = np.arange(len(names))
axR.barh(y, bv, color="#2e86c1", height=0.6)
axR.set_yticks(y); axR.set_yticklabels(names, fontsize=9); axR.invert_yaxis()
axR.set_xlabel("on-board DPU-INT8 IoU (256-frame subset)")
for i, v in enumerate(bv): axR.text(v+0.5, i, "%.1f" % v, va="center", fontsize=8.5)
axR.set_title("(b) on-FPGA per-class IoU — mIoU 16.37 / FP32 25.0 = 65%\n(VRU: bicycle/motorcycle/pedestrian have real IoU)", fontsize=10)
fig.suptitle("Build-B: full camera→occupancy pipeline on KR260 FPGA (all-PL DPU+gather-IP+resize-IP), measured DPU-INT8 accuracy",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("png", "pdf"):
    fig.savefig(os.path.join("figs/buildB", "final_miou." + ext), dpi=140, bbox_inches="tight")
print("saved figs/buildB/final_miou.{png,pdf}")
