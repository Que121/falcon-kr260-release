#!/usr/bin/env python
"""Board-FAITHFUL sim: real DPU-INT8 (weights+acts, flashocc_eval_realint8.py) PLUS the on-board
view-transform GATHER seam, so we can localize the 23.13(real-INT8 sim) -> 16.37(board) gap WITHOUT
the board and ablate the fix on the A100.

The board's gather (fpga/board/board_pipeline_batched.py L121-133, fpga/hls/bev_gather*.cpp):
  depth  -> ap_ufixed<8,1>  : round(d*128)/128, clamp [0, 255/128]          (Q0.7, step 1/128)
  feat   -> ap_int<8>       : round(f*2^fp_feat)/2^fp_feat, clamp [-128,127] (fp_feat HARDCODED = 2)
  vt_out -> ap_int<8>       : round(v*2^fp_vt)/2^fp_vt,     clamp [-128,127] (fp_vt   HARDCODED = 0)
        => vt_out is INT8 with STEP 1.0 across all 64 channels (single global scale). Prime suspect.

--seam modes (DPU weight+act INT8 is ALWAYS on; the seam is added on top):
  none      : gather kept FP32  -> reproduces realint8 == 23.13
  board     : feat fp=2, depth Q0.7, vt INT8 fp=0   (EXACT board reproduction; expect ~16)
  board_optfp: feat fp=2, depth Q0.7, vt INT8 fp=per-frame-optimal pow2 (CHEAP fix: just out_scale)
  vt16      : feat fp=2, depth Q0.7, vt INT16 optimal fp (ALL-PL fix: widen gather BEV output)
  vt8pc     : feat fp=2, depth Q0.7, vt INT8 per-CHANNEL optimal fp (per-channel out_scale fix)
  featdepth : feat fp=2, depth Q0.7, vt FP32  (isolate how much the INPUT quant alone costs)
Also prints the FP32 vt_out abs-max distribution (justifies the fp choice).

  python flashocc_eval_boardsim.py --seam board --cap 256
"""
import argparse, importlib, numpy as np, torch, torch.nn as nn
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet3d.apis import single_gpu_test

ap = argparse.ArgumentParser()
ap.add_argument("--seam", choices=["none", "board", "board_optfp", "vt16", "vt8pc", "featdepth"], default="board")
ap.add_argument("--feat_fp", type=int, default=2)     # board hardcoded
ap.add_argument("--vt_fp", type=int, default=0)       # board hardcoded
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

# ---- gather-seam quantizers ----
def q_depth(d):                                  # board Q0.7 (ap_ufixed<8,1>)
    return torch.clamp(torch.round(d * 128.0), 0, 255) / 128.0

def q_fixfp(x, fp, bits=8):                       # INT @ fixed fractional bits fp
    qmax = 2 ** (bits - 1) - 1; qmin = -2 ** (bits - 1)
    s = 2.0 ** (-fp)
    return torch.clamp(torch.round(x / s), qmin, qmax) * s

def opt_fp(x, bits=8):                             # per-tensor pow2 fp so max maps near qmax (no clip)
    qmax = 2 ** (bits - 1) - 1
    mx = x.detach().abs().max() + 1e-9
    return int(torch.floor(torch.log2(torch.tensor(qmax / mx.item()))).item())

def q_optfp(x, bits=8):
    return q_fixfp(x, opt_fp(x, bits), bits)

def q_perchan(x, bits=8):                          # per-CHANNEL pow2 (dim=1 = C), INT8
    qmax = 2 ** (bits - 1) - 1; qmin = -2 ** (bits - 1)
    mx = x.detach().abs().amax(dim=(0, 2, 3), keepdim=True) + 1e-9   # (1,C,1,1)
    fp = torch.floor(torch.log2(qmax / mx))                          # per-channel fractional bits
    s = torch.pow(torch.tensor(2.0, device=x.device), -fp)
    return torch.clamp(torch.round(x / s), qmin, qmax) * s

VT_ABSMAX = []   # collect FP32 vt_out abs-max per frame

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

# ---- (always) real DPU-INT8: per-tensor pow2 INT8 on ALL DPU conv WEIGHTS + activation FQ ----
no_clamp = [mm.img_backbone, mm.img_neck, mm.img_view_transformer.depth_net]
do_clamp = [mm.img_bev_encoder_backbone, mm.img_bev_encoder_neck]
nw = na = 0
for grp in [no_clamp, do_clamp, [mm.occ_head.final_conv]]:
    for root in grp:
        for md in root.modules():
            if isinstance(md, nn.Conv2d):
                md.weight.data = pow2_int8(md.weight.data); nw += 1
for grp, c in [(no_clamp, None), (do_clamp, args.clamp)]:
    for root in grp:
        relu_seen = False
        for md in root.modules():
            if isinstance(md, nn.ReLU): md.register_forward_hook(mk_hook(c)); na += 1; relu_seen = True
        if not relu_seen:
            for md in root.modules():
                if isinstance(md, nn.Conv2d): md.register_forward_hook(mk_hook(c)); na += 1
try:
    mm.occ_head.final_conv.activate.register_forward_hook(mk_hook(args.clamp)); na += 1
except Exception as e: print("final_conv act hook skip:", e)
print("[boardsim] DPU weight-FQ convs: %d | act-FQ hooks: %d | seam=%s feat_fp=%d vt_fp=%d"
      % (nw, na, args.seam, args.feat_fp, args.vt_fp), flush=True)

# ---- inject the on-board gather seam by wrapping view_transform ----
vt_mod = mm.img_view_transformer
orig_view_transform = vt_mod.view_transform   # bound method
def patched_view_transform(input, depth, tran_feat):
    if args.seam != "none":
        depth = q_depth(depth)
        tran_feat = q_fixfp(tran_feat, args.feat_fp)
    bev_feat, d2 = orig_view_transform(input, depth, tran_feat)
    VT_ABSMAX.append(float(bev_feat.detach().abs().max().item()))
    if args.seam == "board":
        bev_feat = q_fixfp(bev_feat, args.vt_fp, bits=8)
    elif args.seam == "board_optfp":
        bev_feat = q_optfp(bev_feat, bits=8)
    elif args.seam == "vt16":
        bev_feat = q_optfp(bev_feat, bits=16)
    elif args.seam == "vt8pc":
        bev_feat = q_perchan(bev_feat, bits=8)
    # featdepth / none: vt_out stays FP
    return bev_feat, d2
vt_mod.view_transform = patched_view_transform   # instance attr (no self bound) matches forward's call

outputs = single_gpu_test(model, loader)
res = dataset.evaluate(outputs, metric=['mIoU'])
print("==== seam=%s cap=%d ====" % (args.seam, args.cap), flush=True)
if isinstance(res, dict):
    for k, v in res.items(): print("  %s: %s" % (k, v), flush=True)
vm = np.array(VT_ABSMAX)
if vm.size:
    print("[vt_out FP32 abs-max] n=%d min=%.3f p50=%.3f p90=%.3f max=%.3f  (board fp_vt=0 => step 1.0 across all 64ch)"
          % (vm.size, vm.min(), np.median(vm), np.percentile(vm, 90), vm.max()), flush=True)
print("EVAL_DONE", flush=True)
