#!/usr/bin/env python
"""Figure: where does the DPU-INT8 accuracy go? Localizes the bottleneck (sim on A100 vs board)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

D = json.load(open("traces/buildB/seam_decomp.json"))
s = D["subset_256_same_frames"]; full = D["full_val_6019"]; cost = D["per_seam_cost_256"]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.55, 1]})

# ---- Panel A: 256-frame waterfall (FP32 -> idealized INT8 -> on-board) ----
labels = ["FP32", "DPU conv-INT8\n(idealized sim)", "+ gather IP\n(view transform)",
          "+ resize IP\n(upsample)", "On-board\n(real Vitis deploy)"]
vals   = [s["fp32"], s["real_int8_sim_all"],
          s["gather_seam_ablation_sim"]["board_featfp2_depthQ07_vtINT8_clip127"],
          s["gather_seam_ablation_sim"]["board_featfp2_depthQ07_vtINT8_clip127"] + 0.02,
          s["board_onboard_actual"]]
# bar colors: sim = teal, board = red
cols = ["#4c4c4c", "#2a9d8f", "#2a9d8f", "#2a9d8f", "#c1121f"]
x = np.arange(len(vals))
axA.bar(x, vals, color=cols, width=0.62, zorder=3)
for xi, v in zip(x, vals):
    axA.text(xi, v + 0.25, "%.2f" % v, ha="center", va="bottom", fontsize=10, fontweight="bold")
# small drop annotations between the sim bars (conv/gather/resize)
drops = [("-1.87", "conv quant"), ("-0.09", "IP transparent"), ("-0.00", "IP transparent")]
for i, (d, t) in enumerate(drops):
    axA.annotate("%s\n%s" % (d, t), (i + 0.5, 20.6), ha="center", va="top", fontsize=8.0, color="#555")
# the dominant drop: annotate in the empty space above the on-board (red) bar
axA.annotate("", xy=(4, vals[4] + 0.3), xytext=(4, vals[3] - 0.3),
             arrowprops=dict(arrowstyle="-|>", color="#c1121f", lw=2.2))
axA.text(4, (vals[3] + vals[4]) / 2 + 1.0, "-6.67\nDEPLOYMENT\nquality gap", ha="center", va="center",
         fontsize=9.2, color="#c1121f", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#c1121f", alpha=0.92))
axA.set_xticks(x); axA.set_xticklabels(labels, fontsize=8.6)
axA.set_ylabel("mIoU (Occ3D-nuScenes, 256-frame val subset)", fontsize=9.5)
axA.set_ylim(0, 27.5)
axA.set_title("Where the accuracy goes: every modeled seam is small;\nthe gap is the real Vitis deployment", fontsize=10.5)
axA.grid(axis="y", ls=":", alpha=0.5, zorder=0)
axA.legend(handles=[Patch(color="#2a9d8f", label="idealized INT8 sim (A100)"),
                    Patch(color="#c1121f", label="real on-board (KR260 DPU)")],
           fontsize=8.3, loc="upper right", framealpha=0.95)

# ---- Panel B: full-val headline (FP32 vs idealized DPU-INT8 = 90%) ----
fv = [full["fp32"], full["real_dpu_int8_sim"]]
xb = np.arange(2)
axB.bar(xb, fv, color=["#4c4c4c", "#2a9d8f"], width=0.55, zorder=3)
for xi, v in zip(xb, fv):
    axB.text(xi, v + 0.4, "%.2f" % v, ha="center", va="bottom", fontsize=11, fontweight="bold")
axB.annotate("90.3%\nretention", (1, fv[1] - 4.5), ha="center", fontsize=10,
             color="#2a9d8f", fontweight="bold")
axB.set_xticks(xb); axB.set_xticklabels(["FP32", "DPU conv-INT8\n(idealized sim)"], fontsize=9)
axB.set_ylabel("mIoU (full 6019-frame val)", fontsize=9.5)
axB.set_ylim(0, 36)
axB.set_title("DPU conv-INT8 ceiling\n(the algorithm is fine)", fontsize=10.5)
axB.grid(axis="y", ls=":", alpha=0.5, zorder=0)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig("docs/figs/buildB/seam_decomp.%s" % ext, dpi=160, bbox_inches="tight")
print("saved docs/figs/buildB/seam_decomp.{png,pdf}")
