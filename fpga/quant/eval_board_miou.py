#!/usr/bin/env python
"""Real Occ3D-nuScenes mIoU for ON-BOARD occupancy (FlashOcc Metric_mIoU vs GT + camera mask).

board_argmax (N,200,200,16) uint8 in val order (shuffle=False => index i = val sample i = GT i) ->
dataset.evaluate(occ_results, metric=['mIoU']). Also evaluates the dumped FP32 occ (frames' 'occ')
as the in-pipeline FP32 baseline, so retention = board_mIoU / fp32_mIoU is apples-to-apples.

  python eval_board_miou.py <board_argmax.npy> <N> [fp_io_dir]
"""
import os, sys, importlib, numpy as np, torch
from mmcv import Config
from mmdet3d.datasets import build_dataset

BD = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
FPDIR = sys.argv[3] if len(sys.argv) > 3 else ""

cfg = Config.fromfile("projects/configs/flashocc/flashocc-r50.py")
if getattr(cfg, "plugin", False):
    importlib.import_module(".".join(cfg.plugin_dir.rstrip("/").split("/")))
data_cfg = cfg.data.test if hasattr(cfg.data, "test") else cfg.data.val
data_cfg.test_mode = True
dataset = build_dataset(data_cfg)

board = np.load(BD).astype(np.uint8)            # (N,200,200,16)
M = min(N, board.shape[0])
print("evaluating BOARD occ over %d frames" % M, flush=True)
res = dataset.evaluate([board[i] for i in range(M)], metric=["mIoU"])
print("BOARD mIoU result:", {k: (float(v) if np.isscalar(v) else v) for k, v in (res.items() if isinstance(res, dict) else [])}, flush=True)

if FPDIR:
    fp = []
    for i in range(M):
        f = os.path.join(FPDIR, "frame_%04d.npz" % i)
        fp.append(np.load(f)["occ"].astype(np.uint8))
    print("evaluating FP32 (dumped occ) baseline over %d frames" % M, flush=True)
    res2 = dataset.evaluate(fp, metric=["mIoU"])
    print("FP32 baseline mIoU result:", {k: (float(v) if np.isscalar(v) else v) for k, v in (res2.items() if isinstance(res2, dict) else [])}, flush=True)
print("EVAL_DONE", flush=True)
