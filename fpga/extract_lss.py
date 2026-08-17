#!/usr/bin/env python
"""Capture FlashOcc's BEVPoolv2 view-transform tensors (HPC, fbocc) to design the HLS gather IP.

Wraps bev_pool_v2 to save, for one frame: the precomputed STATIC index tables (ranks_depth/ranks_feat/
ranks_bev, interval_starts/lengths) + the inputs (depth, feat) + the CUDA output. This lets us
(1) reimplement the op as a plain index-gather and verify bit-exactness vs the CUDA kernel, and
(2) size + write the synthesizable HLS kernel (fixed N_points -> bounded cycles -> WCET).
Saves /scratch/ANON/lss_dump.npz. Run from the FlashOCC repo.
"""
import importlib
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model

cfg = Config.fromfile("projects/configs/flashocc/flashocc-r50.py")
importlib.import_module(".".join(cfg.plugin_dir.rstrip("/").split("/")))
vt_mod = importlib.import_module("projects.mmdet3d_plugin.models.necks.view_transformer")
orig = vt_mod.bev_pool_v2
saved = {}
def wrap(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape, interval_starts, interval_lengths):
    out = orig(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape, interval_starts, interval_lengths)
    if not saved:
        saved.update(
            depth=depth.detach().cpu().numpy(), feat=feat.detach().cpu().numpy(),
            ranks_depth=ranks_depth.detach().cpu().numpy().astype(np.int64),
            ranks_feat=ranks_feat.detach().cpu().numpy().astype(np.int64),
            ranks_bev=ranks_bev.detach().cpu().numpy().astype(np.int64),
            interval_starts=interval_starts.detach().cpu().numpy().astype(np.int64),
            interval_lengths=interval_lengths.detach().cpu().numpy().astype(np.int64),
            bev_feat_shape=np.array(bev_feat_shape, dtype=np.int64),
            out=out.detach().cpu().numpy())
        print("captured: depth", saved["depth"].shape, "feat", saved["feat"].shape,
              "| N_points", saved["ranks_bev"].shape[0], "N_pillar", saved["interval_starts"].shape[0],
              "| bev_feat_shape", tuple(bev_feat_shape), "| out", saved["out"].shape, flush=True)
    return out
vt_mod.bev_pool_v2 = wrap

dcfg = cfg.data.test; dcfg.test_mode = True
dataset = build_dataset(dcfg); dataset.data_infos = dataset.data_infos[:1]
loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
load_checkpoint(model, "ckpts/flashocc-r50-256x704.pth", map_location="cpu")
model = MMDataParallel(model.cuda(), [0]); model.eval()
with torch.no_grad():
    for data in loader:
        model(return_loss=False, rescale=True, **data); break

np.savez_compressed("/scratch/ANON/lss_dump.npz", **saved)
print("saved /scratch/ANON/lss_dump.npz ; keys:", list(saved.keys()), flush=True)
print("LSS_DUMP_DONE", flush=True)
