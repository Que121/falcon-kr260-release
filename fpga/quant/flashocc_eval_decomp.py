#!/usr/bin/env python
"""Decompose the real-DPU-INT8 loss (and the on-board 23->16 residual) BY STAGE, to localize the
bottleneck the gather-seam ablation ruled OUT (gather costs ~0). Same per-tensor-pow2 INT8 fake-quant
as flashocc_eval_realint8.py, but applied SELECTIVELY:

  --mode all    : quantize EVERY DPU stage (img backbone+FPN+depth_net + BEV enc + occ) == realint8 (23.13/256)
  --mode img    : quantize ONLY the image path (backbone+FPN+depth_net); BEV+occ stay FP32  -> image-INT8 loss
  --mode bev    : quantize ONLY BEV-encoder+occ (clamp16); image stays FP32                 -> BEV-INT8 loss (continuous)
  --mode bev_rz : bev + ALSO INT8-quantize the two nn.Upsample outputs in the BEV neck (= the on-board
                  resize-IP seam the realint8 sim omits) -> tells us if the INT8 resize/subgraph deploy
                  explains the board's extra ~7 pt drop.
  --mode none   : FP32 everywhere (sanity == 25.0/256)

Combined with the gather ablation (gather~0) and the on-board BEV-only (19.97, FP32 vt), this triangulates
image-deploy vs BEV-deploy as the dominant on-board seam.

  python flashocc_eval_decomp.py --mode bev_rz --cap 256
"""
import argparse, importlib, numpy as np, torch, torch.nn as nn
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet3d.apis import single_gpu_test

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["none", "all", "img", "bev", "bev_rz"], default="all")
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

IMG = [mm.img_backbone, mm.img_neck, mm.img_view_transformer.depth_net]   # no clamp (not heavy-tailed)
BEV = [mm.img_bev_encoder_backbone, mm.img_bev_encoder_neck]               # clamp16 (heavy-tailed)
nw = na = nrz = 0

def quant_group(roots, clamp_c):
    global nw, na
    for root in roots:
        for md in root.modules():
            if isinstance(md, nn.Conv2d):
                md.weight.data = pow2_int8(md.weight.data); nw += 1
        relu_seen = False
        for md in root.modules():
            if isinstance(md, nn.ReLU): md.register_forward_hook(mk_hook(clamp_c)); na += 1; relu_seen = True
        if not relu_seen:
            for md in root.modules():
                if isinstance(md, nn.Conv2d): md.register_forward_hook(mk_hook(clamp_c)); na += 1

if args.mode in ("all", "img"):
    quant_group(IMG, None)
if args.mode in ("all", "bev", "bev_rz"):
    quant_group(BEV, args.clamp)
    for md in mm.occ_head.final_conv.modules():
        if isinstance(md, nn.Conv2d): md.weight.data = pow2_int8(md.weight.data); nw += 1
    try:
        mm.occ_head.final_conv.activate.register_forward_hook(mk_hook(args.clamp)); na += 1
    except Exception as e: print("final_conv act hook skip:", e)

# --- on-board RESIZE-IP seam: INT8-quantize the BEV-neck nn.Upsample outputs (FP in realint8 sim) ---
if args.mode == "bev_rz":
    for md in mm.img_bev_encoder_neck.modules():
        if isinstance(md, nn.Upsample): md.register_forward_hook(mk_hook(args.clamp)); nrz += 1

print("[decomp] mode=%s weight-FQ=%d act-FQ=%d upsample-FQ=%d (clamp=%.0f on BEV/occ)"
      % (args.mode, nw, na, nrz, args.clamp), flush=True)

outputs = single_gpu_test(model, loader)
res = dataset.evaluate(outputs, metric=['mIoU'])
print("==== mode=%s cap=%d ====" % (args.mode, args.cap), flush=True)
if isinstance(res, dict):
    for k, v in res.items(): print("  %s: %s" % (k, v), flush=True)
print("EVAL_DONE", flush=True)
