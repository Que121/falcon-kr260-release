#!/usr/bin/env python
"""WHY does the board (16.37) fall short of the idealized-INT8 sim (23.13)? This makes the sim FAITHFUL
to the real Vitis DPU and measures how much each realization effect costs:
  --wq/--aq ceil  : NON-clipping pow2 scale (s=2^ceil(log2(max/127)), never saturates)  [my idealized sim]
  --wq/--aq round : SATURATING calibrated pow2 fix-point (fp=round(log2(127/max)), clamps ±127) [real DPU]
  --fold 1        : fold BatchNorm into the preceding conv BEFORE weight-quant (real DPU does this; widens
                    the per-tensor weight range -> coarser per-tensor scale). Emulated by quantizing the
                    folded weight and dividing the BN scale back out (BN still applied -> function ~same,
                    only the quant granularity changes).
Same coverage as flashocc_eval_realint8.py (img backbone+FPN+depth_net + BEV enc + occ final-conv;
gather+predicter FP, BEV/occ clamp16). If round+fold drops toward ~16-18, the gap is the faithful
INT8-with-saturation/BN-fold ceiling (and 16.37 is near the real INT8 reality); if it stays ~22, the
board is leaving points on the table (deployment under-optimized -> recoverable via QAT/fix-point tuning).

  python flashocc_eval_vitissat.py --wq round --aq round --fold 1 --cap 256
"""
import argparse, importlib, numpy as np, torch, torch.nn as nn
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet3d.apis import single_gpu_test

ap = argparse.ArgumentParser()
ap.add_argument("--wq", choices=["ceil", "round"], default="round")   # weight scale rule
ap.add_argument("--aq", choices=["ceil", "round"], default="round")   # activation scale rule
ap.add_argument("--fold", type=int, default=1)                         # BN-fold before weight quant
ap.add_argument("--clamp", type=float, default=16.0)
ap.add_argument("--cap", type=int, default=256)
ap.add_argument("--config", default="projects/configs/flashocc/flashocc-r50.py")
ap.add_argument("--ckpt", default="ckpts/flashocc-r50-256x704.pth")
args = ap.parse_args()

def q_pow2(x, rule):
    mx = x.detach().abs().max() + 1e-9
    if rule == "ceil":   # never clips (idealized): step >= max/127
        fp = torch.floor(torch.log2(torch.tensor(127.0, device=x.device) / mx))
    else:                # round (Vitis-like): finer step, SATURATES the tail
        fp = torch.round(torch.log2(torch.tensor(127.0, device=x.device) / mx))
    s = torch.pow(torch.tensor(2.0, device=x.device), -fp)
    return torch.clamp(torch.round(x / s), -128, 127) * s

def mk_hook(clamp_c, rule):
    def h(m, i, o):
        if not torch.is_tensor(o): return o
        x = torch.clamp(o, max=clamp_c) if clamp_c is not None else o
        return q_pow2(x, rule)
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

no_clamp = [mm.img_backbone, mm.img_neck, mm.img_view_transformer.depth_net]
do_clamp = [mm.img_bev_encoder_backbone, mm.img_bev_encoder_neck]
ALL = no_clamp + do_clamp + [mm.occ_head.final_conv]

# ---- (optional) fold BN into the preceding conv, then quantize the FOLDED weight ----
nfold = nw = na = 0
if args.fold:
    for root in ALL:
        last_conv = None
        for md in root.modules():
            if isinstance(md, nn.Conv2d):
                last_conv = md
            elif isinstance(md, (nn.BatchNorm2d,)) and last_conv is not None \
                    and md.num_features == last_conv.out_channels and md.affine and md.running_var is not None:
                sc = (md.weight / torch.sqrt(md.running_var + md.eps)).detach()   # (Cout,)
                fw = last_conv.weight.data * sc[:, None, None, None]
                last_conv.weight.data = q_pow2(fw, args.wq) / sc[:, None, None, None]
                last_conv._folded = True
                nfold += 1; nw += 1; last_conv = None
# quantize any remaining (unfolded) conv weights per-tensor (folded ones already quantized above)
for root in ALL:
    for md in root.modules():
        if isinstance(md, nn.Conv2d) and not getattr(md, "_folded", False):
            md.weight.data = q_pow2(md.weight.data, args.wq); nw += 1

# activation FQ (same placement as realint8): ReLU outputs; for no-ReLU FPN/depth -> conv outputs
for grp, c in [(no_clamp, None), (do_clamp, args.clamp)]:
    for root in grp:
        relu_seen = False
        for md in root.modules():
            if isinstance(md, nn.ReLU): md.register_forward_hook(mk_hook(c, args.aq)); na += 1; relu_seen = True
        if not relu_seen:
            for md in root.modules():
                if isinstance(md, nn.Conv2d): md.register_forward_hook(mk_hook(c, args.aq)); na += 1
try:
    mm.occ_head.final_conv.activate.register_forward_hook(mk_hook(args.clamp, args.aq)); na += 1
except Exception as e: print("final_conv act hook skip:", e)

print("[vitissat] wq=%s aq=%s fold=%d | bn-folded=%d weight-FQ=%d act-FQ=%d clamp=%.0f"
      % (args.wq, args.aq, args.fold, nfold, nw, na, args.clamp), flush=True)

outputs = single_gpu_test(model, loader)
res = dataset.evaluate(outputs, metric=['mIoU'])
print("==== wq=%s aq=%s fold=%d cap=%d ====" % (args.wq, args.aq, args.fold, args.cap), flush=True)
if isinstance(res, dict):
    for k, v in res.items(): print("  %s: %s" % (k, v), flush=True)
print("EVAL_DONE", flush=True)
