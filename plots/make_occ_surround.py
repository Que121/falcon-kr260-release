#!/usr/bin/env python3
"""FlashOcc-style surround figure: the 6 input cameras (2x3, surround layout) -> the on-board INT8
occupancy prediction on the KR260 (3D voxels + BEV top-down), Occ3D-nuScenes colormap. Mirrors the
official FlashOcc/SurroundOcc visualization (surround cams + dense voxel occ). -> figs/occ_surround.{pdf,png}
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style as S; S.apply()
import numpy as np, matplotlib.pyplot as plt
import matplotlib.image as mpimg

FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 254
FREE, VRU = 17, (2, 6, 7)
CAMD = "traces/buildB/occ_cmp/cams_%04d" % FRAME
OCC = np.load("traces/buildB/fullrun/board_argmax_full.npy")[FRAME]

CMAP255 = np.array([
    [0,0,0],[255,120,50],[255,192,203],[255,255,0],[0,150,245],[0,255,255],[255,127,0],[255,0,0],
    [255,240,150],[135,60,0],[160,32,240],[255,0,255],[139,137,137],[75,0,75],[150,240,80],
    [230,230,250],[0,175,0],[255,255,255]], float) / 255.0
names = ["others","barrier","bicycle","bus","car","constr","motorcyc","pedestrian","cone","trailer",
         "truck","driveable","other-flat","sidewalk","terrain","manmade","vegetation","free"]

fig = plt.figure(figsize=(13.5, 9.2))
gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 2.05], hspace=0.12, wspace=0.04)

# ---- 6 surround cameras (2x3 surround layout) ----
LAYOUT = [["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
          ["CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT"]]
for r in range(2):
    for c in range(3):
        ax = fig.add_subplot(gs[r, c]); cam = LAYOUT[r][c]
        f = os.path.join(CAMD, cam + ".jpg")
        if os.path.exists(f): ax.imshow(mpimg.imread(f))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(cam.replace("CAM_", "").replace("_", "-").title(), fontsize=9, pad=2, color=S.SUBTLE)
        for s in ax.spines.values(): s.set_edgecolor("#CCC")

# ---- 3D occupancy (bottom-left, 2 cols) ----
ax3 = fig.add_subplot(gs[2, 0:2], projection="3d")
xs, ys, zs = np.where(OCC != FREE); cls = OCC[xs, ys, zs]
obj = cls <= 10; gnd = ~obj
rng = np.random.default_rng(0)
kg = np.zeros(gnd.sum(), bool); kg[rng.choice(gnd.sum(), min(45000, gnd.sum()), replace=False)] = True
sel = obj.copy(); sel[np.where(gnd)[0]] = kg
xs, ys, zs, cls = xs[sel], ys[sel], zs[sel], cls[sel]
vm = np.isin(cls, VRU)
ax3.scatter((xs[~vm]-100)*0.4, (ys[~vm]-100)*0.4, zs[~vm]*0.4, c=CMAP255[cls[~vm]], s=5, marker="s",
            depthshade=False, edgecolors="none", alpha=0.85)
ax3.scatter((xs[vm]-100)*0.4, (ys[vm]-100)*0.4, zs[vm]*0.4, c=CMAP255[cls[vm]], s=18, marker="s",
            depthshade=False, edgecolors="none")
ax3.set_box_aspect((1, 1, 0.3)); ax3.view_init(elev=33, azim=-58)
ax3.set_xlabel("x (m)", labelpad=-3); ax3.set_ylabel("y (m)", labelpad=-3)
ax3.set_zticks([0, 5, 10, 15]); ax3.grid(False)
ax3.set_title("on-board INT8 occupancy — 3D voxels", fontsize=10.5, fontweight="semibold", y=0.97)

# ---- BEV (bottom-right) ----
axb = fig.add_subplot(gs[2, 2])
cls2 = np.full(OCC.shape[:2], FREE, int)
for m in [(OCC >= 11) & (OCC <= 16), (OCC >= 1) & (OCC <= 10)]:
    ok = m.any(2); hz = m.shape[2]-1-m[:, :, ::-1].argmax(2); ii, jj = np.where(ok)
    cls2[ii, jj] = OCC[ii, jj, hz[ii, jj]]
axb.imshow(CMAP255[cls2].transpose(1, 0, 2), origin="lower", extent=[-40, 40, -40, 40], interpolation="nearest")
vy, vx = np.where(np.isin(OCC, VRU).any(2).T)
axb.scatter((vx-100)*0.4, (vy-100)*0.4, s=8, facecolors="none", edgecolors="#FF0000", linewidths=0.6, alpha=0.9)
axb.scatter([0], [0], marker="*", s=130, color="black", edgecolor="white", linewidth=0.8, zorder=5)
axb.set_xlabel("x (m)"); axb.set_ylabel("y (m)")
axb.set_title("BEV top-down", fontsize=10.5, fontweight="semibold")

present = [c for c in np.unique(OCC) if c != FREE]
from matplotlib.patches import Patch
handles = [Patch(facecolor=CMAP255[c], edgecolor="#999", label=names[c]) for c in present]
fig.legend(handles=handles, loc="lower center", ncol=min(9, len(handles)), fontsize=8,
           bbox_to_anchor=(0.5, -0.02), frameon=False)
fig.suptitle("FlashOcc on the KR260: 6 surround cameras → on-board INT8 occupancy (Occ3D-nuScenes, frame %d)"
             % FRAME, fontsize=13, fontweight="bold", y=0.995, color=S.INK)
plt.tight_layout(rect=[0, 0.02, 1, 0.98])
os.makedirs("docs/figs", exist_ok=True)
plt.savefig("figs/occ_surround.png", bbox_inches="tight")
plt.savefig("figs/occ_surround.pdf", bbox_inches="tight")
print("wrote figs/occ_surround.{png,pdf} | frame %d" % FRAME)
