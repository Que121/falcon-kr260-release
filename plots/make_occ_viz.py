#!/usr/bin/env python3
"""Qualitative occupancy visualization of the on-board INT8 prediction (KR260): BEV top-down + 3D voxels,
Occ3D-nuScenes 18-class colormap, VRU (pedestrian/bicycle/motorcycle) emphasized. Frame from the on-board
full-pipeline output board_argmax_full.npy (200x200x16, ego-centred). -> docs/figs/occ_bev_3d.{pdf,png}
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style as S; S.apply()
import numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 254
OCC = np.load("experiments/results/buildB/fullrun/board_argmax_full.npy")[FRAME]  # (200,200,16)
FREE = 17
VRU = (2, 6, 7)

# Occ3D-nuScenes 18-class colormap (RGB 0-255)
CMAP255 = np.array([
    [  0,   0,   0],  # 0 others
    [255, 120,  50],  # 1 barrier
    [255, 192, 203],  # 2 bicycle (VRU)
    [255, 255,   0],  # 3 bus
    [  0, 150, 245],  # 4 car
    [  0, 255, 255],  # 5 construction veh
    [255, 127,   0],  # 6 motorcycle (VRU)
    [255,   0,   0],  # 7 pedestrian (VRU)
    [255, 240, 150],  # 8 traffic cone
    [135,  60,   0],  # 9 trailer
    [160,  32, 240],  # 10 truck
    [255,   0, 255],  # 11 driveable surface
    [139, 137, 137],  # 12 other flat
    [ 75,   0,  75],  # 13 sidewalk
    [150, 240,  80],  # 14 terrain
    [230, 230, 250],  # 15 manmade
    [  0, 175,   0],  # 16 vegetation
    [255, 255, 255],  # 17 free
], float) / 255.0

fig = plt.figure(figsize=(11, 4.6))

# ---- (a) BEV top-down: topmost non-free class per (x,y) ----
axL = fig.add_subplot(1, 2, 1)
def top_class(mask):                                  # highest-z class where mask true, per (x,y)
    any_ = mask.any(2)
    hz = mask.shape[2] - 1 - mask[:, :, ::-1].argmax(2)
    return any_, hz
bev_cls = np.full(OCC.shape[:2], FREE, int)
# ground/structure first (driveable/sidewalk/terrain/manmade/vegetation), then objects override on top
for mask in [(OCC >= 11) & (OCC <= 16), (OCC >= 1) & (OCC <= 10)]:
    ok, hz = top_class(mask); ii, jj = np.where(ok)
    bev_cls[ii, jj] = OCC[ii, jj, hz[ii, jj]]
img = CMAP255[bev_cls].transpose(1, 0, 2)            # (Y,X,3) for display
axL.imshow(img, origin="lower", extent=[-40, 40, -40, 40], interpolation="nearest")
# VRU emphasis: outline cells containing any VRU voxel
vru_xy = np.isin(OCC, VRU).any(2)
vy, vx = np.where(vru_xy.T)
axL.scatter((vx - 100) * 0.4, (vy - 100) * 0.4, s=10, facecolors="none", edgecolors="#FF0000",
            linewidths=0.6, alpha=0.9, label="VRU (ped/bike/moto)")
axL.scatter([0], [0], marker="*", s=160, color="black", edgecolor="white", linewidth=0.8, zorder=5, label="ego")
axL.set_xlabel("x (m)"); axL.set_ylabel("y (m)"); axL.set_title("(a)  BEV top-down (on-board INT8)", loc="left")
axL.legend(loc="upper right", fontsize=8, framealpha=0.85, facecolor="white")

# ---- (b) 3D voxels: scatter occupied voxels, iso view ----
axR = fig.add_subplot(1, 2, 2, projection="3d")
xs, ys, zs = np.where(OCC != FREE)
cls = OCC[xs, ys, zs]
# keep ALL objects/VRU (1-10); subsample ground/structure (11-16) for render speed
obj = cls <= 10
gnd = ~obj
rng = np.random.default_rng(0)
keep_g = np.zeros(gnd.sum(), bool); idx = rng.choice(gnd.sum(), min(45000, gnd.sum()), replace=False); keep_g[idx] = True
sel = obj.copy(); sel[np.where(gnd)[0]] = keep_g
xs, ys, zs, cls = xs[sel], ys[sel], zs[sel], cls[sel]
order = np.argsort(np.isin(cls, VRU))                # draw VRU last (on top)
xs, ys, zs, cls = xs[order], ys[order], zs[order], cls[order]
vmask = np.isin(cls, VRU)
axR.scatter((xs[~vmask] - 100) * 0.4, (ys[~vmask] - 100) * 0.4, zs[~vmask] * 0.4, c=CMAP255[cls[~vmask]],
            s=5, marker="s", depthshade=False, edgecolors="none", alpha=0.85)
axR.scatter((xs[vmask] - 100) * 0.4, (ys[vmask] - 100) * 0.4, zs[vmask] * 0.4, c=CMAP255[cls[vmask]],
            s=16, marker="s", depthshade=False, edgecolors="none")   # VRU bigger, on top
axR.set_box_aspect((1, 1, 0.32))
axR.view_init(elev=32, azim=-58)
axR.set_xlabel("x (m)", labelpad=-2); axR.set_ylabel("y (m)", labelpad=-2); axR.set_zlabel("z", labelpad=-6)
axR.set_zticks([0, 5, 10, 15]); axR.set_title("(b)  3D occupancy (on-board INT8)", loc="left")
axR.grid(False)
try: axR.set_facecolor("white")
except Exception: pass

# shared class legend (the classes present)
from matplotlib.patches import Patch
names = ["others","barrier","bicycle","bus","car","constr","motorcyc","pedestrian","cone","trailer",
         "truck","driveable","other-flat","sidewalk","terrain","manmade","vegetation","free"]
present = [c for c in np.unique(OCC) if c != FREE]
handles = [Patch(facecolor=CMAP255[c], edgecolor="#999", label=names[c]) for c in present]
fig.legend(handles=handles, loc="lower center", ncol=min(9, len(handles)), fontsize=7.5,
           bbox_to_anchor=(0.5, -0.06), frameon=False)

fig.suptitle("On-board INT8 occupancy prediction on the KR260 (Occ3D-nuScenes, frame %d)" % FRAME,
             fontsize=12.5, fontweight="bold", y=1.02, color=S.INK)
plt.tight_layout()
os.makedirs("docs/figs", exist_ok=True)
plt.savefig("docs/figs/occ_bev_3d.png", bbox_inches="tight")
plt.savefig("docs/figs/occ_bev_3d.pdf", bbox_inches="tight")
print("wrote docs/figs/occ_bev_3d.{png,pdf} | frame %d | classes present %s" % (FRAME, present))
