#!/usr/bin/env python
"""Dump FP32 conv_only (occ_head.final_conv output) from the dumped vt_out -- lightweight, CPU-OK
(no image backbone; just BEV encoder + final_conv). Localizes the on-board BEV-stage value error.

For each buildB_io16 frame: vt_out(64,200,200) -> img_bev_encoder_backbone -> _neck -> final_conv
=> conv_only_fp32 (256,200,200). Self-checks that the numpy predicter on conv_only reproduces the
dumped occ. Saves convonly_%04d.npy + predicter_head.npz.

  python dump_bevref.py --io /scratch/ANON/buildB_io16 --n 16 --out /scratch/ANON/buildB_bevref
"""
import os, importlib, argparse, numpy as np, torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--io", default="/scratch/ANON/buildB_io16")
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--out", default="/scratch/ANON/buildB_bevref")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

cfg = Config.fromfile("projects/configs/flashocc/flashocc-r50.py")
if getattr(cfg, "plugin", False):
    importlib.import_module(".".join(cfg.plugin_dir.rstrip("/").split("/")))
model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
load_checkpoint(model, "ckpts/flashocc-r50-256x704.pth", map_location="cpu")
model.eval()
bb = model.img_bev_encoder_backbone; nk = model.img_bev_encoder_neck
oh = model.occ_head; fc = oh.final_conv; pr = oh.predicter

# save predicter weights for the board
sd = {k: v.detach().cpu().numpy() for k, v in pr.state_dict().items()}
np.savez(os.path.join(args.out, "predicter_head.npz"), **sd)
W0, b0 = sd["0.weight"], sd["0.bias"]; W2, b2 = sd["2.weight"], sd["2.bias"]

def np_predict(conv):                              # conv (256,200,200) -> occ (200,200,16)
    x = conv.transpose(2, 1, 0).reshape(-1, 256) @ W0.T + b0
    x = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
    return (x @ W2.T + b2).reshape(200, 200, 16, 18).argmax(-1).astype(np.uint8)

with torch.no_grad():
    for i in range(args.n):
        f = os.path.join(args.io, "frame_%04d.npz" % i)
        if not os.path.exists(f): break
        d = np.load(f)
        vt = torch.from_numpy(d["vt_out"].astype(np.float32))[None]   # (1,64,200,200)
        x = bb(vt); x = nk(x)                                         # (1,256,200,200)
        conv = fc(x)                                                  # (1,256,200,200)
        conv_np = conv[0].cpu().numpy().astype(np.float32)
        np.save(os.path.join(args.out, "convonly_%04d.npy" % i), conv_np)
        if i == 0:
            occ_np = np_predict(conv_np)
            ref = d["occ"].astype(np.uint8)
            print("conv_only", conv_np.shape, "range [%.2f,%.2f] mean|%.3f" % (
                conv_np.min(), conv_np.max(), np.abs(conv_np).mean()), flush=True)
            print("predicter self-check vs dump occ: agree %.4f (should be ~1.0)" % float((occ_np == ref).mean()), flush=True)
print("BEVREF_DONE", flush=True)
