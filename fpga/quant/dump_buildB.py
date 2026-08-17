#!/usr/bin/env python
"""Build-B comprehensive dump (HPC, fbocc env, A100): EVERYTHING the board needs per val frame
to run the FULL image->occ pipeline on the FPGA, in val order (shuffle=False => frame i = Occ3D GT i).

Per frame_%04d.npz:
  img         (6,3,256,704) f16  - normalized backbone input (exactly the DPU image input, pre-int8)
  ranks_depth (Np,) i32 | ranks_feat (Np,) i32 | ranks_bev (Np,) i32  - gather index tables
  interval_starts (Npil,) i32 | interval_lengths (Npil,) i32
  bev_feat_shape (5,) i32
  vt_out      (64,200,200) f16 - FP32 view-transform reference (gather-output check)
  occ         (200,200,16) u8  - FP32 argmax occupancy reference (mIoU/retention ref)
  [--withfeat] feat (6,16,44,64) f16 + depth (6,88,16,44) f16  - FP32 seam refs (small sets only)

  python dump_buildB.py --n 16 --withfeat --out /scratch/ANON/buildB_io16
  python dump_buildB.py --n 6019          --out /scratch/ANON/buildB_io_full
"""
import os, importlib, argparse, numpy as np, torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--out", default="/scratch/ANON/buildB_io16")
ap.add_argument("--withfeat", action="store_true")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

cfg = Config.fromfile("projects/configs/flashocc/flashocc-r50.py")
if getattr(cfg, "plugin", False):
    importlib.import_module(".".join(cfg.plugin_dir.rstrip("/").split("/")))
data_cfg = cfg.data.test if hasattr(cfg.data, "test") else cfg.data.val
data_cfg.test_mode = True
dataset = build_dataset(data_cfg)
loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=4, dist=False, shuffle=False)
model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
load_checkpoint(model, "ckpts/flashocc-r50-256x704.pth", map_location="cpu")
model = MMDataParallel(model.cuda(), [0]); model.eval()
m = model.module
vt = m.img_view_transformer

cap = {}
# backbone input = the exact normalized image fed to the DPU
def pre_bb(mod, inp): cap["img"] = inp[0].detach().float().cpu().numpy().astype(np.float16)
m.img_backbone.register_forward_pre_hook(pre_bb)
# occ_head output (argmax later)
def occ_hook(mod, inp, out): cap["occ_logit"] = out
m.occ_head.register_forward_hook(occ_hook)

# wrap voxel_pooling_v2 to grab depth, feat, ranks, vt_out
orig_vp = vt.voxel_pooling_v2
def wrap_vp(coor, depth, feat):
    rb, rd, rf, ist, iln = vt.voxel_pooling_prepare_v2(coor)
    cap["ranks_bev"] = rb.int().cpu().numpy()
    cap["ranks_depth"] = rd.int().cpu().numpy()
    cap["ranks_feat"] = rf.int().cpu().numpy()
    cap["interval_starts"] = ist.int().cpu().numpy()
    cap["interval_lengths"] = iln.int().cpu().numpy()
    if args.withfeat:
        cap["feat_fp"] = feat.detach().float().cpu().numpy().astype(np.float16)   # (B,N,C,fH,fW)
        cap["depth_fp"] = depth.detach().float().cpu().numpy().astype(np.float16)  # (B,N,D,fH,fW)
    out = orig_vp(coor, depth, feat)
    cap["vt_out"] = out.detach().float().cpu().numpy().astype(np.float16)
    return out
vt.voxel_pooling_v2 = wrap_vp

with torch.no_grad():
    for i, data in enumerate(loader):
        cap.clear()
        model(return_loss=False, rescale=True, **data)
        save = dict(
            img=cap["img"].reshape(6, 3, 256, 704),
            ranks_depth=cap["ranks_depth"].astype(np.int32),
            ranks_feat=cap["ranks_feat"].astype(np.int32),
            ranks_bev=cap["ranks_bev"].astype(np.int32),
            interval_starts=cap["interval_starts"].astype(np.int32),
            interval_lengths=cap["interval_lengths"].astype(np.int32),
            vt_out=cap["vt_out"].reshape(64, 200, 200),
            occ=cap["occ_logit"].squeeze().argmax(-1).byte().cpu().numpy().astype(np.uint8),
        )
        if args.withfeat:
            save["feat"] = cap["feat_fp"].reshape(6, 64, 16, 44).transpose(0, 2, 3, 1)  # (6,16,44,64)
            save["depth"] = cap["depth_fp"].reshape(6, 88, 16, 44)
        np.savez_compressed(os.path.join(args.out, "frame_%04d.npz" % i), **save)
        if i == 0:
            for k, v in save.items(): print("f0", k, v.shape, v.dtype, flush=True)
        if (i + 1) % 16 == 0: print("dumped %d/%d" % (i + 1, args.n), flush=True)
        if (i + 1) >= args.n: break
print("DUMP_DONE", flush=True)
