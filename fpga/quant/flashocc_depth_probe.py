#!/usr/bin/env python
"""Probe: does the IDEALIZED per-tensor-pow2 INT8 sim (basis of the 29.08 'INT8 ceiling') preserve the
88-bin DEPTH head much better than the board's real deployed image xmodel (measured depth-argmax ~0.75)?
If sim depth-argmax agreement >> 0.75, the realint8 sim is OPTIMISTIC about the image stage and 29.08 is
not a realistic on-board target -- the depth head is the true limiter (consistent with the project memory
'image at DPU-INT8 ceiling'). If ~0.75, the sim is faithful and the residual gap is BEV-deploy.

Quantizes ONLY the image path (backbone+FPN+depth_net) exactly like flashocc_eval_realint8.py, captures
the depth softmax (B*N,88,H,W) via a view_transform hook, dumps per-frame argmax + max-prob.

  python flashocc_depth_probe.py --mode int8 --cap 64 --out depth_int8.npz
"""
import argparse, importlib, numpy as np, torch, torch.nn as nn
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet3d.apis import single_gpu_test

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["fp32", "int8"], default="int8")
ap.add_argument("--cap", type=int, default=64)
ap.add_argument("--out", default="depth_probe.npz")
ap.add_argument("--config", default="projects/configs/flashocc/flashocc-r50.py")
ap.add_argument("--ckpt", default="ckpts/flashocc-r50-256x704.pth")
args = ap.parse_args()

def pow2_int8(x):
    mx = x.detach().abs().max() + 1e-9
    s = torch.pow(torch.tensor(2.0, device=x.device), torch.ceil(torch.log2(mx / 127.0)))
    return torch.clamp(torch.round(x / s), -128, 127) * s

def mk_hook():
    def h(m, i, o):
        if not torch.is_tensor(o): return o
        return pow2_int8(o)
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

if args.mode == "int8":
    IMG = [mm.img_backbone, mm.img_neck, mm.img_view_transformer.depth_net]   # the image path only
    nw = na = 0
    for root in IMG:
        for md in root.modules():
            if isinstance(md, nn.Conv2d): md.weight.data = pow2_int8(md.weight.data); nw += 1
        relu_seen = False
        for md in root.modules():
            if isinstance(md, nn.ReLU): md.register_forward_hook(mk_hook()); na += 1; relu_seen = True
        if not relu_seen:
            for md in root.modules():
                if isinstance(md, nn.Conv2d): md.register_forward_hook(mk_hook()); na += 1
    print("[probe int8] image weight-FQ=%d act-FQ=%d" % (nw, na), flush=True)
else:
    print("[probe fp32] no quant", flush=True)

ARG = []; MAXP = []; DEP = []; FEAT = []   # depth argmax, peak conf, full softmax depth + tran_feat (for cos)
vt = mm.img_view_transformer
orig = vt.view_transform
def patched(input, depth, tran_feat):
    d = depth.detach()                      # (B*N=6, 88, H, W) softmax probs
    ARG.append(d.argmax(1).to(torch.uint8).cpu().numpy())   # (6,H,W)
    MAXP.append(float(d.amax(1).mean().item()))             # mean peak confidence
    DEP.append(d.to(torch.float16).cpu().numpy())           # (6,88,H,W) softmax
    FEAT.append(tran_feat.detach().to(torch.float16).cpu().numpy())  # (6,64,H,W) context
    return orig(input, depth, tran_feat)
vt.view_transform = patched

_ = single_gpu_test(model, loader)
arg = np.stack(ARG, 0)                       # (N,6,H,W)
np.savez(args.out, argmax=arg, maxp=np.array(MAXP),
         depth=np.stack(DEP, 0), feat=np.stack(FEAT, 0))
print("[probe %s] dumped %s argmax%s mean-peak-conf %.3f -> %s"
      % (args.mode, arg.shape[0], arg.shape, float(np.mean(MAXP)), args.out), flush=True)
print("PROBE_DONE", flush=True)
