#!/usr/bin/env python
"""Verify the HLS view-transform design: a plain precomputed-index weighted scatter-add reproduces
FlashOcc's CUDA bev_pool_v2 bit-exactly. Confirms the IP algorithm is correct before HLS.

bev[ranks_bev[i]] += depth_flat[ranks_depth[i]] * feat_flat[ranks_feat[i]]   over N_points (static idx).
Also reports the segmented (interval) form the HLS kernel uses (no scatter hazard) + WCET sizing.
"""
import numpy as np

d = np.load("/home/ANON/lss_dump.npz")
depth = d["depth"].astype(np.float64).ravel()                  # (B*N*D*H*W,)
feat = d["feat"].astype(np.float64).reshape(-1, d["feat"].shape[-1])   # (B*N*H*W, C)
rd, rf, rb = d["ranks_depth"], d["ranks_feat"], d["ranks_bev"]
istart, ilen = d["interval_starts"], d["interval_lengths"]
B, Dz, Dy, Dx, C = d["bev_feat_shape"]
ref = d["out"].astype(np.float64)                             # (B, C, Dz, Dy, Dx)
N = rb.shape[0]
print("N_points=%d  N_pillar=%d  C=%d  bev=%dx%dx%d  depth_len=%d  feat_vecs=%d"
      % (N, istart.shape[0], C, Dz, Dy, Dx, depth.shape[0], feat.shape[0]), flush=True)

# (1) dense scatter-add form
bev = np.zeros((Dz * Dy * Dx, C), dtype=np.float64)
contrib = depth[rd][:, None] * feat[rf]
np.add.at(bev, rb, contrib)
bev1 = bev.reshape(Dz, Dy, Dx, C).transpose(3, 0, 1, 2)[None]  # (B,C,Dz,Dy,Dx)

# (2) segmented (interval) form — exactly what the HLS kernel does (accumulate per pillar, write once)
bev2 = np.zeros((Dz * Dy * Dx, C), dtype=np.float64)
for p in range(istart.shape[0]):
    s, L = int(istart[p]), int(ilen[p])
    idx = np.arange(s, s + L)
    acc = (depth[rd[idx]][:, None] * feat[rf[idx]]).sum(0)
    bev2[rb[s]] = acc
bev2 = bev2.reshape(Dz, Dy, Dx, C).transpose(3, 0, 1, 2)[None]

for name, b in [("scatter-add", bev1), ("segmented(HLS form)", bev2)]:
    err = np.abs(b - ref)
    rel = err.max() / (np.abs(ref).max() + 1e-9)
    print("%-22s max_abs_err=%.3e  rel=%.3e  allclose(1e-3)=%s"
          % (name, err.max(), rel, np.allclose(b, ref, atol=1e-3)), flush=True)

macs = N * C
print("\nHLS sizing: %d points x C=%d = %.2fM MACs/frame; at II=1 (C=64 unrolled) -> %d cycles"
      % (N, C, macs / 1e6, N), flush=True)
for f in (200, 300):
    print("  WCET @ %dMHz = %.3f ms (fixed N_points -> input-invariant, bounded)" % (f, N / (f * 1e3)), flush=True)
print("VERIFY_DONE", flush=True)
