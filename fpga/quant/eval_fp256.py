#!/usr/bin/env python
"""FP32 baseline mIoU over the first N val frames (dumped occ), for a less-noisy reference than 16fr.
  python eval_fp256.py <io_dir> <N>"""
import os, sys, importlib, numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset
IO = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 256
cfg = Config.fromfile("projects/configs/flashocc/flashocc-r50.py")
if getattr(cfg, "plugin", False):
    importlib.import_module(".".join(cfg.plugin_dir.rstrip("/").split("/")))
dc = cfg.data.test if hasattr(cfg.data, "test") else cfg.data.val
dc.test_mode = True
ds = build_dataset(dc)
occ = [np.load(os.path.join(IO, "frame_%04d.npz" % i))["occ"].astype(np.uint8) for i in range(N)]
print("FP32 baseline over %d frames" % N, flush=True)
res = ds.evaluate(occ, metric=["mIoU"])
print("done", flush=True)
