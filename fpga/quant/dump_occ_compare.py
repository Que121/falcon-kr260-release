#!/usr/bin/env python
"""Dump GT semantics + FP32 prediction for a few val frames, to build the qualitative occ comparison
(GT vs FP32 vs on-board INT8). Run from the FlashOCC dir (fbocc env).
  python dump_occ_compare.py 254 306 276
Writes /scratch/ANON/occ_cmp_<F>.npz {gt, gt_mask, fp32} per frame (200x200x16 uint8).
"""
import os, sys, importlib, numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset

frames = [int(x) for x in sys.argv[1:]] or [254]
cfg = Config.fromfile("projects/configs/flashocc/flashocc-r50.py")
if getattr(cfg, "plugin", False):
    importlib.import_module(".".join(cfg.plugin_dir.rstrip("/").split("/")))
dcfg = cfg.data.test if hasattr(cfg.data, "test") else cfg.data.val
dcfg.test_mode = True
ds = build_dataset(dcfg)

info0 = ds.data_infos[frames[0]]
print("data_infos[%d] keys:" % frames[0], list(info0.keys()), flush=True)
for k in info0:
    if "occ" in k.lower() or "path" in k.lower(): print("   ", k, "=", info0[k])

def load_gt(info):
    # FlashOcc Occ3D: info['occ_path'] -> a dir or a labels.npz with 'semantics' + 'mask_camera'
    p = info.get("occ_path") or info.get("occ_gt_path") or info.get("pts_semantic_mask_path")
    if p is None: return None, None
    f = p if p.endswith(".npz") else os.path.join(p, "labels.npz")
    g = np.load(f)
    return g["semantics"].astype(np.uint8), (g["mask_camera"].astype(bool) if "mask_camera" in g else None)

for F in frames:
    info = ds.data_infos[F]
    gt, mask = load_gt(info)
    fp = np.load("/scratch/ANON/buildB_io_full/frame_%04d.npz" % F)["occ"].astype(np.uint8)
    out = "/scratch/ANON/occ_cmp_%04d.npz" % F
    np.savez(out, gt=(gt if gt is not None else np.zeros((200,200,16),np.uint8)),
             gt_mask=(mask if mask is not None else np.ones((200,200,16),bool)), fp32=fp)
    print("frame %d: gt %s  fp32 %s -> %s" % (F, None if gt is None else gt.shape, fp.shape, out), flush=True)
print("OCC_CMP_DONE", flush=True)
