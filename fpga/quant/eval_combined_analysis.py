#!/usr/bin/env python
"""Combined analysis over the on-board full-val predictions (one pass, one job).

From board_argmax_full.npy (N,200,200,16 uint8, val order) + Occ3D GT:
  (1) per-frame per-class intersection/union/gt-count table  -> bootstrap CI for mIoU
      (frame-level and scene-clustered)
  (2) condition subsets (night / rain / day) from nuScenes scene descriptions
  (3) bicycle error analysis: where bicycle GT voxels go (confusion), voxel counts
  (4) fallback degradation: ego-warp the previous frame's prediction to the current frame
      (SE2 on the BEV grid, nearest), VRU recall + mIoU vs fresh, 1- and 2-frame stale

Run from the FlashOCC repo, env fbocc, on HPC:
  python eval_combined_analysis.py <board_argmax_full.npy> <out.npz> [N]
"""
import os, sys, json, importlib, numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset
from pyquaternion import Quaternion

BD  = sys.argv[1]
OUT = sys.argv[2]
N   = int(sys.argv[3]) if len(sys.argv) > 3 else 6019

NCLS, FREE = 18, 17
VRU = (2, 6, 7)
GRID, RNG = 200, 40.0   # 0.4 m cells, [-40, 40)

cfg = Config.fromfile("projects/configs/flashocc/flashocc-r50.py")
if getattr(cfg, "plugin", False):
    importlib.import_module(".".join(cfg.plugin_dir.rstrip("/").split("/")))
dc = cfg.data.test if hasattr(cfg.data, "test") else cfg.data.val
dc.test_mode = True
ds = build_dataset(dc)
board = np.load(BD, mmap_mode="r")
M = min(N, board.shape[0], len(ds.data_infos))
print("analysis over %d frames" % M, flush=True)

def gt_path(info):
    for k in ("occ_gt_path", "occ_path"):
        p = info.get(k)
        if p:
            return p if p.endswith(".npz") else os.path.join(p, "labels.npz")
    raise KeyError("no occ GT path")

# nuScenes tables for scene mapping + descriptions (robust to layout)
def find_tables():
    for root in ("data/nuscenes/v1.0-trainval", "/scratch/ANON/occ3d_raw/v1.0-trainval",
                 "/scratch/ANON/nuscenes/v1.0-trainval"):
        if os.path.exists(os.path.join(root, "scene.json")):
            return root
    return None
TB = find_tables()
tok2scene, scene_desc = {}, {}
if TB:
    samples = json.load(open(os.path.join(TB, "sample.json")))
    scenes = json.load(open(os.path.join(TB, "scene.json")))
    for s in samples: tok2scene[s["token"]] = s["scene_token"]
    for s in scenes: scene_desc[s["token"]] = s["description"].lower()
    print("nuScenes tables:", TB, flush=True)
else:
    print("WARNING: no nuScenes tables found; subsets skipped", flush=True)

inter = np.zeros((M, NCLS), np.int64)
union = np.zeros((M, NCLS), np.int64)
gtcnt = np.zeros((M, NCLS), np.int64)
bike_conf = np.zeros(NCLS, np.int64)          # what the board predicts on bicycle GT voxels
scene_ids = np.full(M, -1, np.int32)
flags = np.zeros(M, np.uint8)                 # 1=night 2=rain 3=both
sid_map = {}

# fallback accumulators: [stale_k][class] tp/fn ; plus miou-style inter/union for warped
fb_tp = {1: np.zeros(NCLS, np.int64), 2: np.zeros(NCLS, np.int64)}
fb_fn = {1: np.zeros(NCLS, np.int64), 2: np.zeros(NCLS, np.int64)}
fb_occtp = {1: np.zeros(NCLS, np.int64), 2: np.zeros(NCLS, np.int64)}
fb_frames = {1: 0, 2: 0}

def ego_se2(info):
    t = np.asarray(info["ego2global_translation"], float)
    q = Quaternion(info["ego2global_rotation"])
    yaw = q.yaw_pitch_roll[0]
    return t[0], t[1], yaw

# precompute grid centers
ii, jj = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
xc = (ii + 0.5) * (2 * RNG / GRID) - RNG    # x along dim0 (BEVDet convention)
yc = (jj + 0.5) * (2 * RNG / GRID) - RNG

def warp_pred(pred, info_from, info_to):
    """nearest-resample pred (200,200,16) from frame A's ego grid into frame B's ego grid."""
    xa, ya, tha = ego_se2(info_from)
    xb, yb, thb = ego_se2(info_to)
    # target cell center in B ego -> global -> A ego
    cb, sb = np.cos(thb), np.sin(thb)
    gx = xb + cb * xc - sb * yc
    gy = yb + sb * xc + cb * yc
    ca, sa = np.cos(-tha), np.sin(-tha)
    ax = ca * (gx - xa) - sa * (gy - ya)
    ay = sa * (gx - xa) + ca * (gy - ya)
    si = np.floor((ax + RNG) / (2 * RNG / GRID)).astype(np.int32)
    sj = np.floor((ay + RNG) / (2 * RNG / GRID)).astype(np.int32)
    ok = (si >= 0) & (si < GRID) & (sj >= 0) & (sj < GRID)
    out = np.full((GRID, GRID, 16), FREE, np.uint8)   # out-of-view -> free (the conservative gap)
    out[ok] = pred[si[ok], sj[ok]]
    return out

prev = {}   # scene_id -> list of (idx, pred, info)
for i in range(M):
    info = ds.data_infos[i]
    z = np.load(gt_path(info))
    sem = z["semantics"]; mask = z["mask_camera"].astype(bool)
    pr = np.asarray(board[i])
    st = tok2scene.get(info.get("token", ""), None)
    if st is not None:
        sid = sid_map.setdefault(st, len(sid_map)); scene_ids[i] = sid
        d = scene_desc.get(st, "")
        flags[i] = (1 if "night" in d else 0) | (2 if "rain" in d else 0)
    pm, gm = pr[mask], sem[mask]
    for c in range(NCLS):
        pc, gc = pm == c, gm == c
        it = int((pc & gc).sum())
        inter[i, c] = it
        union[i, c] = int(pc.sum()) + int(gc.sum()) - it
        gtcnt[i, c] = int(gc.sum())
    bb = gm == 2
    if bb.any():
        bike_conf += np.bincount(pm[bb], minlength=NCLS)
    # fallback: warp predictions from k frames back (same scene only)
    hist = prev.setdefault(scene_ids[i], [])
    for k in (1, 2):
        if len(hist) >= k:
            j, pj, infoj = hist[-k]
            w = warp_pred(pj, infoj, info)
            wm = w[mask]
            for c in VRU:
                gc = gm == c
                fb_tp[k][c] += int(((wm == c) & gc).sum())
                fb_fn[k][c] += int(((wm != c) & gc).sum())
                fb_occtp[k][c] += int(((wm != FREE) & gc).sum())
            fb_frames[k] += 1
    hist.append((i, pr, info))
    if len(hist) > 2: hist.pop(0)
    if (i + 1) % 500 == 0: print("  ...%d" % (i + 1), flush=True)

def miou_from(idx):
    I = inter[idx].sum(0).astype(float); U = union[idx].sum(0).astype(float)
    valid = U[:FREE] > 0
    return float(np.mean((I[:FREE] / np.maximum(U[:FREE], 1))[valid]))

full_idx = np.arange(M)
res = {"miou_full": miou_from(full_idx)}
# bootstrap CI (frame-level and scene-clustered), 2000 resamples
rng = np.random.default_rng(0)
for name, units in (("frame", [np.array([i]) for i in range(M)]),
                    ("scene", [np.where(scene_ids == s)[0] for s in range(scene_ids.max() + 1)])):
    stats = []
    for _ in range(2000):
        pick = rng.integers(0, len(units), len(units))
        idx = np.concatenate([units[p] for p in pick])
        stats.append(miou_from(idx))
    res["ci_%s" % name] = (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))
# subsets
for name, m in (("night", flags & 1 > 0), ("rain", flags & 2 > 0), ("day", flags == 0)):
    idx = np.where(m)[0]
    if len(idx): res["miou_%s" % name] = (miou_from(idx), int(len(idx)))
print(json.dumps(res, indent=1), flush=True)
print("bicycle GT voxel confusion (board):", flush=True)
tot = bike_conf.sum()
for c in np.argsort(bike_conf)[::-1][:8]:
    if bike_conf[c]: print("  class %2d: %8d (%.1f%%)" % (c, bike_conf[c], 100 * bike_conf[c] / tot), flush=True)
for k in (1, 2):
    print("fallback stale k=%d over %d frames:" % (k, fb_frames[k]), flush=True)
    for c in VRU:
        den = max(fb_tp[k][c] + fb_fn[k][c], 1)
        print("  class %d recall(class)=%.4f recall(occ)=%.4f" %
              (c, fb_tp[k][c] / den, fb_occtp[k][c] / den), flush=True)
np.savez(OUT, inter=inter, union=union, gtcnt=gtcnt, scene_ids=scene_ids, flags=flags,
         bike_conf=bike_conf,
         fb_tp1=fb_tp[1], fb_fn1=fb_fn[1], fb_occtp1=fb_occtp[1],
         fb_tp2=fb_tp[2], fb_fn2=fb_fn[2], fb_occtp2=fb_occtp[2])
print("saved", OUT, flush=True)
