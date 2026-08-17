#!/usr/bin/env python3
"""Qualitative occupancy comparison across configurations on the SAME frame: Ground Truth vs FP32
prediction vs INT8 algorithm (GPU per-tensor-pow2 sim) vs on-board INT8 (KR260). BEV top-down
(camera-mask region, Occ3D-nuScenes colormap, VRUs circled). Shows the on-board INT8 deployment tracks
the INT8 algorithm and the FP32 reference. -> figs/occ_compare.{pdf,png}
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style as S; S.apply()
import numpy as np, matplotlib.pyplot as plt

FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 254
FREE, VRU = 17, (2, 6, 7)
cmp = np.load("traces/buildB/occ_cmp/occ_cmp_%04d.npz" % FRAME)
gt, mask, fp32 = cmp["gt"], cmp["gt_mask"].astype(bool), cmp["fp32"]
int8_path = "traces/buildB/occ_cmp/occ_int8_%04d.npy" % FRAME
int8 = np.load(int8_path) if os.path.exists(int8_path) else None   # INT8 algorithm (GPU sim); None until dumped
board = np.load("traces/buildB/fullrun/board_argmax_full.npy")[FRAME]

CMAP255 = np.array([
    [0,0,0],[255,120,50],[255,192,203],[255,255,0],[0,150,245],[0,255,255],[255,127,0],[255,0,0],
    [255,240,150],[135,60,0],[160,32,240],[255,0,255],[139,137,137],[75,0,75],[150,240,80],
    [230,230,250],[0,175,0],[255,255,255]], float) / 255.0
names = ["others","barrier","bicycle","bus","car","constr","motorcyc","pedestrian","cone","trailer",
         "truck","driveable","other-flat","sidewalk","terrain","manmade","vegetation","free"]

def bev(occ):
    o = np.where(mask, occ, FREE)                     # camera-evaluated region only
    cls = np.full(o.shape[:2], FREE, int)
    for m in [(o >= 11) & (o <= 16), (o >= 1) & (o <= 10)]:   # ground first, objects override
        ok = m.any(2); hz = m.shape[2] - 1 - m[:, :, ::-1].argmax(2)
        ii, jj = np.where(ok); cls[ii, jj] = o[ii, jj, hz[ii, jj]]
    return CMAP255[cls].transpose(1, 0, 2), np.isin(o, VRU).any(2)

panels = [("Ground truth", gt), ("FP32 prediction", fp32)]
if int8 is not None:
    panels.append(("INT8 algorithm (GPU)", int8))
panels.append(("on-board INT8 (KR260)", board))
fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.7))
for ax, (title, occ) in zip(axes, panels):
    img, vru_xy = bev(occ)
    ax.imshow(img, origin="lower", extent=[-40, 40, -40, 40], interpolation="nearest")
    vy, vx = np.where(vru_xy.T)
    ax.scatter((vx - 100) * 0.4, (vy - 100) * 0.4, s=12, facecolors="none", edgecolors="#FF0000",
               linewidths=0.7, alpha=0.9)
    ax.scatter([0], [0], marker="*", s=150, color="black", edgecolor="white", linewidth=0.8, zorder=5)
    ax.set_title(title, loc="center", fontsize=15, fontweight="semibold", pad=8)
    ax.set_xlabel("x (m)", fontsize=12)
    ax.tick_params(labelsize=10)
axes[0].set_ylabel("y (m)", fontsize=12)
axes[-1].scatter([], [], s=12, facecolors="none", edgecolors="#FF0000", label="VRU (ped/bike/moto)")
axes[-1].scatter([], [], marker="*", s=120, color="black", label="ego")
axes[-1].legend(loc="upper right", fontsize=9.5, framealpha=0.85)

_u = set(np.unique(gt)) | set(np.unique(fp32)) | set(np.unique(board)) | (set(np.unique(int8)) if int8 is not None else set())
present = sorted(_u)
present = [c for c in present if c != FREE]
from matplotlib.patches import Patch
handles = [Patch(facecolor=CMAP255[c], edgecolor="#999", label=names[c]) for c in present]
# reserve a wider bottom band so the enlarged class legend clears the x-axis tick labels
plt.tight_layout(rect=[0, 0.19, 1, 1])
fig.legend(handles=handles, loc="lower center", ncol=min(9, len(handles)), fontsize=13,
           bbox_to_anchor=(0.5, 0.004), frameon=False, handlelength=1.4, columnspacing=1.5)
os.makedirs("docs/figs", exist_ok=True)
plt.savefig("figs/occ_compare.png", bbox_inches="tight", dpi=400)
plt.savefig("figs/occ_compare.pdf", bbox_inches="tight")
print("wrote figs/occ_compare.{png,pdf} | frame %d | classes %s" % (FRAME, present))
