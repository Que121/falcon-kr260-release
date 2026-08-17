#!/usr/bin/env python
"""Localize WHICH part of the on-board gather recipe costs vt fidelity (pure HPC, no board).
Per frame: model FP32 vt (CUDA bev_pool, ground truth) vs numpy recipes -> cos. Pinpoints feat-INT8 vs
vt-clip vs depth-Q0.7 vs arithmetic. (Q0.8 depth already ruled out on-board.)
  python gather_cos_sweep.py --cap 128
"""
import argparse, importlib, numpy as np, torch, torch.nn as nn
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet3d.apis import single_gpu_test

ap = argparse.ArgumentParser()
ap.add_argument("--cap", type=int, default=128)
ap.add_argument("--config", default="projects/configs/flashocc/flashocc-r50.py")
ap.add_argument("--ckpt", default="ckpts/flashocc-r50-256x704.pth")
args = ap.parse_args()

cfg = Config.fromfile(args.config)
if getattr(cfg, "plugin", False):
    importlib.import_module(".".join(cfg.plugin_dir.rstrip("/").split("/")))
dcfg = cfg.data.test if hasattr(cfg.data, "test") else cfg.data.val
dcfg.test_mode = True
dataset = build_dataset(dcfg)
if args.cap: dataset.data_infos = dataset.data_infos[:args.cap]
loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=4, dist=False, shuffle=False)
model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
load_checkpoint(model, args.ckpt, map_location="cpu")
model = MMDataParallel(model.cuda(), [0]); model.eval()
mm = model.module; vtm = mm.img_view_transformer

R = {}
def ng(depth, feat, dscale=128.0, fscale=4.0, fbits=8, clip_vt=True, vbits=8):
    # depth (6,88,16,44) softmax; feat (6,64,16,44)
    dflat = depth.reshape(-1).astype(np.float64)
    fflat = feat.transpose(0, 2, 3, 1).reshape(-1, 64).astype(np.float64)
    fqmax = 2 ** (fbits - 1) - 1; fqmin = -2 ** (fbits - 1)
    draw = np.clip(np.round(dflat * dscale), 0, 255 if dscale <= 256 else 2 * dscale - 1)
    fraw = np.clip(np.round(fflat * fscale), fqmin, fqmax)
    dq = (draw / dscale)[R["rd"]]        # (N_POINTS,) gathered depth weights
    fq = (fraw / fscale)[R["rf"]]        # (N_POINTS, 64) gathered feat
    contrib = dq[:, None] * fq           # (N_POINTS, 64)
    bev = np.zeros((40000, 64), np.float64)
    np.add.at(bev, R["rb"], contrib)
    if clip_vt:
        vqmax = 2 ** (vbits - 1) - 1; vqmin = -2 ** (vbits - 1)
        bev = np.clip(np.round(bev), vqmin, vqmax)
    return bev   # (40000,64)

import glob
_rf = sorted(glob.glob("/scratch/ANON/buildB_io_full/frame_*.npz"))[0]
_rd = np.load(_rf)
R["rd"] = _rd["ranks_depth"].astype(np.int64); R["rf"] = _rd["ranks_feat"].astype(np.int64)
R["rb"] = _rd["ranks_bev"].astype(np.int64)
print("ranks (from %s) rd%s rf%s rb%s  rb-max %d" % (_rf, R["rd"].shape, R["rf"].shape, R["rb"].shape, R["rb"].max()), flush=True)

ACC = {}
orig = vtm.view_transform
def patched(input, depth, tran_feat):
    out = orig(input, depth, tran_feat)             # real pool -> FP32 ground truth
    bev_feat = out[0] if isinstance(out, (tuple, list)) else out   # (1,64,200,200) FP32 vt = ground truth
    gt = bev_feat.detach().cpu().numpy()[0].transpose(1, 2, 0).reshape(40000, 64).astype(np.float64)
    d = depth.detach().cpu().numpy(); f = tran_feat.detach().cpu().numpy()
    # contrib uses ranks into depth(B,N,D,H,W) & feat(B,N,H,W,C); replicate ground-truth via numpy too for sanity
    variants = {
        "fp_numpy(sanity)": ng(d, f, clip_vt=False),
        "board Q0.7+INT8feat+clip8": ng(d, f, dscale=128, fscale=4, fbits=8, clip_vt=True, vbits=8),
        "Q0.8 depth": ng(d, f, dscale=256, fscale=4, fbits=8, clip_vt=True, vbits=8),
        "INT16 feat": ng(d, f, dscale=128, fscale=4, fbits=16, clip_vt=True, vbits=8),
        "INT16 vt(no clip)": ng(d, f, dscale=128, fscale=4, fbits=8, clip_vt=False),
        "INT16 vt+feat+Q0.8": ng(d, f, dscale=256, fscale=4, fbits=16, clip_vt=False),
    }
    def cos(a, b):
        a = a.ravel(); b = b.ravel(); return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    for k, v in variants.items():
        ACC.setdefault(k, []).append(cos(gt, v))
    return out
vtm.view_transform = patched

_ = single_gpu_test(model, loader)
print("==== gather recipe vt-cos to FP32 ground-truth (mean over %d frames) ====" % args.cap, flush=True)
for k in ["fp_numpy(sanity)", "board Q0.7+INT8feat+clip8", "Q0.8 depth", "INT16 feat", "INT16 vt(no clip)", "INT16 vt+feat+Q0.8"]:
    print("  %-26s cos %.4f" % (k, float(np.mean(ACC[k]))), flush=True)
print("SWEEP_DONE", flush=True)
