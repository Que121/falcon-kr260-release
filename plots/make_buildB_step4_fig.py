#!/usr/bin/env python3
"""Build-B step 4 viz: FULL image->occ pipeline running end-to-end ON the KR260 (all-PL).

The pipeline (every conv on B4096 DPU, every upsample on HLS resize IP, gather on HLS gather IP,
NO numpy/host compute except softmax+predicter which are off-DPU by design):
  image(6 cams) -> DPU image xmodel -> depth_net -> gather IP -> vt_out
                -> BEV (3 DPU conv + 2 resize-IP upsamples) -> predicter -> argmax occ
Top-down BEV occupancy (height-collapsed: any non-free voxel in column) FP32 vs on-board.
This documents the pipeline RUNS end-to-end on PL (Build-B headline). Accuracy at PTQ; fast_finetune/
QAT pending (DPU per-tensor pow2 INT8 of the heavy-tailed BEV encoder loses fidelity -- measured).
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

R = "traces/buildB"
ref = np.load(os.path.join(R, "frame_0000.npz"))["occ"].astype(np.int64)      # (200,200,16) FP32
brd = np.load(os.path.join(R, "step4_occ_fpref.npy")).astype(np.int64)        # on-board (FP32 vt in)
full = np.load(os.path.join(R, "step4_occ_ptq.npy")).astype(np.int64)         # full PTQ pipeline (image-PTQ vt)

def density(o):                                    # (200,200,16) -> occupied-voxel count per column (0..16)
    return (o != 17).sum(-1)

def voxel_geom_iou(a, b):                           # voxel-level: occupied = non-free, over all 640k voxels
    A = a != 17; B = b != 17
    return float((A & B).sum() / ((A | B).sum() + 1e-9))
def voxel_agree(a, b): return float((a == b).mean())

fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
vmax = max(density(ref).max(), density(brd).max(), density(full).max())
for a, o, t in zip(ax, [ref, brd, full],
                   ["FP32 reference\n(FlashOcc occ_head)",
                    "on-board BEV->occ\n(FP32 vt in) — isolates BEV-INT8",
                    "FULL on-board pipeline\nimage->DPU->gather->BEV->occ"]):
    im = a.imshow(density(o), cmap="viridis", vmin=0, vmax=vmax, origin="lower", interpolation="nearest")
    a.set_title(t, fontsize=9.5); a.axis("off")
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="occupied voxels / column")
fig.suptitle("Build-B Step 4 — full camera→occupancy pipeline on KR260 FPGA (all-PL).  Voxel geom-IoU vs FP32: "
             "BEV-int8 %.2f / full %.2f ; voxel-agree %.2f / %.2f  (PTQ; fast_finetune pending)"
             % (voxel_geom_iou(ref, brd), voxel_geom_iou(ref, full), voxel_agree(ref, brd), voxel_agree(ref, full)),
             fontsize=9.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
for ext in ("png", "pdf"):
    fig.savefig(os.path.join("figs/buildB", "step4_full_pipeline." + ext), dpi=140, bbox_inches="tight")
print("voxel geom-IoU: BEV-int8(FP32 vt) %.3f | full PTQ %.3f" % (voxel_geom_iou(ref, brd), voxel_geom_iou(ref, full)))
print("voxel agree: BEV-int8 %.3f | full PTQ %.3f" % (voxel_agree(ref, brd), voxel_agree(ref, full)))
print("saved figs/buildB/step4_full_pipeline.{png,pdf}")
