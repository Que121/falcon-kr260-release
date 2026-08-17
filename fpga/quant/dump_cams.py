#!/usr/bin/env python
"""Copy the 6 surround-camera JPEGs for given val frame(s) to /scratch for the FlashOcc-style figure.
Run from the FlashOCC dir (fbocc env).  python dump_cams.py 254 [306 ...]"""
import os, sys, importlib, shutil
from mmcv import Config
from mmdet3d.datasets import build_dataset

frames = [int(x) for x in sys.argv[1:]] or [254]
cfg = Config.fromfile("projects/configs/flashocc/flashocc-r50.py")
if getattr(cfg, "plugin", False):
    importlib.import_module(".".join(cfg.plugin_dir.rstrip("/").split("/")))
dcfg = cfg.data.test if hasattr(cfg.data, "test") else cfg.data.val
dcfg.test_mode = True
ds = build_dataset(dcfg)
root = os.getcwd()
for F in frames:
    info = ds.data_infos[F]
    outd = "/scratch/ANON/cams_%04d" % F; os.makedirs(outd, exist_ok=True)
    print("frame %d cams:" % F, list(info["cams"].keys()))
    for cam, c in info["cams"].items():
        p = c["data_path"]
        if not os.path.isabs(p): p = os.path.normpath(os.path.join(root, p))
        if not os.path.exists(p):  # try data root fallbacks
            for cand in [p.replace("./", ""), os.path.join(root, "data/nuscenes", p.split("nuscenes/")[-1] if "nuscenes/" in p else p)]:
                if os.path.exists(cand): p = cand; break
        if os.path.exists(p):
            shutil.copy(p, os.path.join(outd, cam + ".jpg")); print("  ok", cam)
        else:
            print("  MISSING", cam, p)
print("CAMS_DONE", flush=True)
