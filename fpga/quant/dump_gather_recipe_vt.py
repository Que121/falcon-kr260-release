#!/usr/bin/env python
"""DIAGNOSTIC: replicate the EXACT on-board gather recipe in numpy (FP accumulate) on FP32 feat/depth,
dump vt per frame -> to be fed to the REAL board bev_fpc. If that mIoU ~= board full (16.23) the loss is
the RECIPE (depth Q0.7 + feat INT8 + vt clip); if ~= 21 the loss is the IP HARDWARE arithmetic.
Recipe (board_gather_from_featdepth / board_pipeline_batched L121-133, bev_gather.hpp):
  depth_raw = round(depth_softmax*128)  (ap_ufixed<8,1> Q0.7, value=raw/128)
  feat_raw  = clip(round(feat*4), -128,127)  (ap_int8, fp_feat=2)
  acc[cell,c] = sum over points  (depth_raw/128) * feat_raw           (ap_fixed<24,12> ~ FP)
  vt_int8 = clip(round(acc * 0.25), -128,127)   (out_scale=1024/4096=0.25, fp_vt=0)
  vt = vt_int8  (host reconstruct x 2^-0)
  python dump_gather_recipe_vt.py --cap 256 --out /scratch/ANON/gather_recipe_vt
"""
import argparse, importlib, os, numpy as np, torch, torch.nn as nn
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet3d.apis import single_gpu_test

ap = argparse.ArgumentParser()
ap.add_argument("--cap", type=int, default=256)
ap.add_argument("--out", default="/scratch/ANON/gather_recipe_vt")
ap.add_argument("--config", default="projects/configs/flashocc/flashocc-r50.py")
ap.add_argument("--ckpt", default="ckpts/flashocc-r50-256x704.pth")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

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
mm = model.module
vtm = mm.img_view_transformer

RANKS = {}
def board_gather(depth, feat):
    # depth (6,88,16,44) softmax ; feat (6,64,16,44) -> NHWC (6,16,44,64)
    dflat = depth.reshape(-1).astype(np.float64)                       # (6*88*16*44,)
    fflat = feat.transpose(0, 2, 3, 1).reshape(-1, 64).astype(np.float64)  # (6*16*44, 64)
    draw = np.round(dflat * 128.0)                                     # Q0.7 raw
    fraw = np.clip(np.round(fflat * 4.0), -128, 127)                   # int8 raw, fp_feat=2
    rd = RANKS["rd"]; rf = RANKS["rf"]; rb = RANKS["rb"]
    contrib = (draw[rd] / 128.0)[:, None] * fraw[rf]                   # (N_POINTS, 64)
    bev = np.zeros((40000, 64), np.float64)
    np.add.at(bev, rb, contrib)
    vt = np.clip(np.round(0.25 * bev), -128, 127).astype(np.int8)      # out_scale 0.25, clip@127
    return vt.reshape(200, 200, 64).transpose(2, 0, 1)                 # (64,200,200), value==fp0

CNT = [0]
orig = vtm.view_transform
def patched(input, depth, tran_feat):
    if not RANKS:
        RANKS["rd"] = vtm.ranks_depth.detach().cpu().numpy().astype(np.int64)
        RANKS["rf"] = vtm.ranks_feat.detach().cpu().numpy().astype(np.int64)
        RANKS["rb"] = vtm.ranks_bev.detach().cpu().numpy().astype(np.int64)
        print("ranks: rd %s rf %s rb %s  bev-max %d" % (RANKS["rd"].shape, RANKS["rf"].shape, RANKS["rb"].shape, RANKS["rb"].max()), flush=True)
    d = depth.detach().cpu().numpy(); f = tran_feat.detach().cpu().numpy()
    vt = board_gather(d, f)
    np.savez(os.path.join(args.out, "frame_%04d.npz" % CNT[0]), vt_out=vt.astype(np.float16))
    CNT[0] += 1
    if CNT[0] % 32 == 0: print("  dumped %d" % CNT[0], flush=True)
    return orig(input, depth, tran_feat)
vtm.view_transform = patched

_ = single_gpu_test(model, loader)
print("GATHER_RECIPE_DONE dumped %d frames -> %s" % (CNT[0], args.out), flush=True)
