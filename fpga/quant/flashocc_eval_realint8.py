#!/usr/bin/env python
"""The REAL DPU-INT8 reference: full per-tensor power-of-2 INT8 fake-quant of WEIGHTS *and* activations
on ALL DPU-mapped stages (backbone + FPN + depth_net + BEV encoder + neck + occ final-conv), incl. the
no-ReLU FPN/depth conv outputs. gather (LSS) + softplus predicter stay FP (off-DPU, exactly like the
board). This is what the DPU actually computes -- the honest baseline the on-board number should match
(NOT #7, which kept all weights + FPN + depth in FP32). --mode fp32 = baseline on the same frames.

  python flashocc_eval_realint8.py --mode realint8 --clamp 16 --cap 256
"""
import argparse, importlib, numpy as np, torch, torch.nn as nn
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet3d.apis import single_gpu_test

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["fp32", "realint8"], default="realint8")
ap.add_argument("--clamp", type=float, default=16.0)
ap.add_argument("--cap", type=int, default=256)
ap.add_argument("--config", default="projects/configs/flashocc/flashocc-r50.py")
ap.add_argument("--ckpt", default="ckpts/flashocc-r50-256x704.pth")
args = ap.parse_args()

def pow2_int8(x):
    mx = x.detach().abs().max() + 1e-9
    s = torch.pow(torch.tensor(2.0, device=x.device), torch.ceil(torch.log2(mx / 127.0)))
    return torch.clamp(torch.round(x / s), -128, 127) * s

def mk_hook(clamp_c):
    def h(m, i, o):
        if not torch.is_tensor(o): return o
        x = torch.clamp(o, max=clamp_c) if clamp_c is not None else o
        return pow2_int8(x)
    return h

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

if args.mode == "realint8":
    no_clamp = [mm.img_backbone, mm.img_neck, mm.img_view_transformer.depth_net]
    do_clamp = [mm.img_bev_encoder_backbone, mm.img_bev_encoder_neck]
    nw = na = 0
    # (1) STATIC per-tensor pow2 INT8 on ALL DPU conv WEIGHTS (this is the real DPU; #7 skipped it)
    for grp in [no_clamp, do_clamp, [mm.occ_head.final_conv]]:
        for root in grp:
            for md in root.modules():
                if isinstance(md, nn.Conv2d):
                    md.weight.data = pow2_int8(md.weight.data); nw += 1
    # (2) activation FQ: ReLU outputs (per #7) + the no-ReLU FPN/depth conv outputs (the bottleneck)
    for grp, c in [(no_clamp, None), (do_clamp, args.clamp)]:
        for root in grp:
            relu_seen = False
            for md in root.modules():
                if isinstance(md, nn.ReLU): md.register_forward_hook(mk_hook(c)); na += 1; relu_seen = True
            if not relu_seen:   # FPN / depth_net -> no ReLU -> FQ the conv outputs
                for md in root.modules():
                    if isinstance(md, nn.Conv2d): md.register_forward_hook(mk_hook(c)); na += 1
    try:
        mm.occ_head.final_conv.activate.register_forward_hook(mk_hook(args.clamp)); na += 1
    except Exception as e: print("final_conv act hook skip:", e)
    print("[realint8] weight-FQ convs: %d | activation-FQ hooks: %d (clamp=%.0f on BEV/occ)" % (nw, na, args.clamp), flush=True)
else:
    print("[fp32] no quant", flush=True)

outputs = single_gpu_test(model, loader)
res = dataset.evaluate(outputs, metric=['mIoU'])
print("==== %s cap=%d ====" % (args.mode, args.cap), flush=True)
if isinstance(res, dict):
    for k, v in res.items(): print("  %s: %s" % (k, v), flush=True)
print("EVAL_DONE", flush=True)
