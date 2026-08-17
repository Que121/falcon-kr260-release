#!/usr/bin/env python
"""Final faithfulness refinement: the real DPU uses a STATIC per-tensor fix-point (calibrated ONCE,
frozen for all frames). My per-frame-adaptive sim re-optimizes the scale every frame = optimistic. Here
we CALIBRATE activation fix-points on the first --calib frames (max over them), FREEZE, then evaluate
with saturating quant at the frozen fix-points. Weights: static round-pow2 + BN-fold (DPU-faithful).
If this stays ~21-22, the board's 16.37 is NOT an INT8 ceiling (recoverable deployment); if it drops
toward ~17, static-fix-point + cross-frame variation is a real chunk (recover via robust calibration/QAT).

  python flashocc_eval_staticfp.py --calib 32 --cap 256
"""
import argparse, importlib, numpy as np, torch, torch.nn as nn
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet3d.apis import single_gpu_test

ap = argparse.ArgumentParser()
ap.add_argument("--calib", type=int, default=32)
ap.add_argument("--cap", type=int, default=256)
ap.add_argument("--clamp", type=float, default=16.0)
ap.add_argument("--config", default="projects/configs/flashocc/flashocc-r50.py")
ap.add_argument("--ckpt", default="ckpts/flashocc-r50-256x704.pth")
args = ap.parse_args()

def fp_round(mx):
    return torch.round(torch.log2(torch.tensor(127.0) / (mx + 1e-9)))

def q_static(x, fp):                      # saturating quant at a FROZEN fix-point fp
    s = 2.0 ** (-fp.item() if torch.is_tensor(fp) else -fp)
    return torch.clamp(torch.round(x / s), -128, 127) * s

class ActQ:
    """forward-hook: calib mode records max (clamped); frozen mode quantizes at the calibrated fp."""
    def __init__(self, clamp_c): self.clamp_c = clamp_c; self.mx = 0.0; self.fp = None; self.calib = True
    def __call__(self, m, i, o):
        if not torch.is_tensor(o): return o
        x = torch.clamp(o, max=self.clamp_c) if self.clamp_c is not None else o
        if self.calib:
            self.mx = max(self.mx, float(x.detach().abs().max().item())); return o
        return q_static(x, self.fp)

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

# static round-pow2 weights + BN-fold (faithful)
nfold = nw = 0
for root in ALL:
    last_conv = None
    for md in root.modules():
        if isinstance(md, nn.Conv2d):
            last_conv = md
        elif isinstance(md, nn.BatchNorm2d) and last_conv is not None \
                and md.num_features == last_conv.out_channels and md.affine and md.running_var is not None:
            sc = (md.weight / torch.sqrt(md.running_var + md.eps)).detach()
            fw = last_conv.weight.data * sc[:, None, None, None]
            fp = fp_round(fw.abs().max()); s = 2.0 ** (-fp.item())
            last_conv.weight.data = (torch.clamp(torch.round(fw / s), -128, 127) * s) / sc[:, None, None, None]
            last_conv._folded = True; nfold += 1; nw += 1; last_conv = None
for root in ALL:
    for md in root.modules():
        if isinstance(md, nn.Conv2d) and not getattr(md, "_folded", False):
            fp = fp_round(md.weight.data.abs().max()); s = 2.0 ** (-fp.item())
            md.weight.data = torch.clamp(torch.round(md.weight.data / s), -128, 127) * s; nw += 1

# activation hooks (ActQ) — same placement as realint8
hooks = []
def add(root, c):
    relu_seen = False
    for md in root.modules():
        if isinstance(md, nn.ReLU): hk = ActQ(c); md.register_forward_hook(hk); hooks.append(hk); relu_seen = True
    if not relu_seen:
        for md in root.modules():
            if isinstance(md, nn.Conv2d): hk = ActQ(c); md.register_forward_hook(hk); hooks.append(hk)
for root in no_clamp: add(root, None)
for root in do_clamp: add(root, args.clamp)
try:
    hk = ActQ(args.clamp); mm.occ_head.final_conv.activate.register_forward_hook(hk); hooks.append(hk)
except Exception as e: print("final_conv act hook skip:", e)
print("[staticfp] bn-folded=%d weight-FQ=%d act-hooks=%d | calibrating on %d frames"
      % (nfold, nw, len(hooks), args.calib), flush=True)

# ---- calibration pass (first --calib frames) ----
with torch.no_grad():
    for idx, data in enumerate(loader):
        if idx >= args.calib: break
        model(return_loss=False, rescale=True, **data)
for hk in hooks:
    hk.fp = fp_round(torch.tensor(hk.mx)); hk.calib = False
print("[staticfp] frozen %d activation fix-points (median fp=%.1f)"
      % (len(hooks), float(np.median([float(h.fp) for h in hooks]))), flush=True)

outputs = single_gpu_test(model, loader)
res = dataset.evaluate(outputs, metric=['mIoU'])
print("==== staticfp calib=%d cap=%d ====" % (args.calib, args.cap), flush=True)
if isinstance(res, dict):
    for k, v in res.items(): print("  %s: %s" % (k, v), flush=True)
print("EVAL_DONE", flush=True)
