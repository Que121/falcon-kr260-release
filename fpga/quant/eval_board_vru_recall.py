#!/usr/bin/env python
"""B4 -- on-board VRU recall + delivered-recall safety payoff.
Extends eval_board_miou.py: instead of mIoU, computes per-VRU-class RECALL of the on-board INT8
occupancy vs Occ3D-nuScenes GT (camera-masked), then the *delivered* recall = recall * (1 - deadline
miss rate) per platform -- the in-time VRU-safety number C3 actually wants.

board_argmax (N,200,200,16) uint8 in val order (shuffle=False => index i = val sample i = GT i).
Run from the FlashOCC repo, env fbocc, on HPC (GT lives there):
  python eval_board_vru_recall.py <board_argmax_full.npy> [N]
"""
import os, sys, importlib, numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset

BD = sys.argv[1]
N  = int(sys.argv[2]) if len(sys.argv) > 2 else 6019

cfg = Config.fromfile("projects/configs/flashocc/flashocc-r50.py")
if getattr(cfg, "plugin", False):
    importlib.import_module(".".join(cfg.plugin_dir.rstrip("/").split("/")))
dc = cfg.data.test if hasattr(cfg.data, "test") else cfg.data.val
dc.test_mode = True
ds = build_dataset(dc)

board = np.load(BD).astype(np.uint8)                 # (N,200,200,16), val order
M = min(N, board.shape[0], len(ds.data_infos))
print("VRU recall over %d frames" % M, flush=True)

VRU = {2: "bicycle", 6: "motorcycle", 7: "pedestrian"}
FREE = 17
tp = {c: 0 for c in VRU}; fn = {c: 0 for c in VRU}; occ_tp = {c: 0 for c in VRU}

def gt_path(info):
    for k in ("occ_gt_path", "occ_path"):
        p = info.get(k)
        if p:
            return p if p.endswith(".npz") else os.path.join(p, "labels.npz")
    raise KeyError("no occ GT path; info keys=%s" % list(info)[:20])

for i in range(M):
    z = np.load(gt_path(ds.data_infos[i]))
    sem = z["semantics"]; mask = z["mask_camera"].astype(bool)   # (200,200,16)
    pr = board[i]
    for c in VRU:
        gtc = (sem == c) & mask
        tp[c]     += int(((pr == c)    & gtc).sum())
        fn[c]     += int(((pr != c)    & gtc).sum())
        occ_tp[c] += int(((pr != FREE) & gtc).sum())              # looser: predicted occupied (any class)
    if (i + 1) % 500 == 0:
        print("  ...%d" % (i + 1), flush=True)

print("\n=== on-board VRU recall (camera-masked, full val) ===", flush=True)
sum_tp = sum_occ = sum_den = 0
for c in VRU:
    den = max(tp[c] + fn[c], 1)
    sum_tp += tp[c]; sum_occ += occ_tp[c]; sum_den += den
    print("  %-11s recall(class)=%.4f  recall(occupied)=%.4f  N_gt=%d"
          % (VRU[c], tp[c] / den, occ_tp[c] / den, den), flush=True)
R_cls = sum_tp / max(sum_den, 1); R_occ = sum_occ / max(sum_den, 1)
print("  %-11s recall(class)=%.4f  recall(occupied)=%.4f" % ("VRU-mean", R_cls, R_occ), flush=True)

# deadline-miss at the 10% margin (from deadline_payoff.csv): a missed frame has NO in-time occupancy
MISS = {"KR260 DPU (dedicated)": 0.000, "Workstation GPU (loaded)": 0.045, "Jetson Orin (thermal)": 0.005}
print("\n=== delivered VRU recall = recall * (1 - miss@10%-margin) ===", flush=True)
for plat, m in MISS.items():
    print("  %-26s class=%.4f  occupied=%.4f  (miss=%.1f%%)"
          % (plat, R_cls * (1 - m), R_occ * (1 - m), 100 * m), flush=True)
print("EVAL_DONE", flush=True)
