#!/usr/bin/env python
"""Gold FP32 reference for the BEV stage + intermediate capture, to debug the on-board walker."""
import os, sys, numpy as np, torch
D = os.path.expanduser("~/occfpga_quant_bevclamp")
os.chdir(D); sys.path.insert(0, D)
from bev_stage import BEVStage

sd = torch.load("bev_stage_sd.pth", map_location="cpu")
m = BEVStage(conv_only=True, clamp_c=32.0)
m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("predicter.")}, strict=False)
m.eval()

x = torch.from_numpy(np.load("bev_test_input.npy").astype(np.float32))   # (1,64,200,200)

inter = {}
with torch.no_grad():
    feats = m.backbone(x)
    inter["feat0_128x100"] = feats[0].numpy()
    inter["feat2_512x25"]  = feats[2].numpy()
    x2, x1 = feats[0], feats[2]
    x1u = m.neck.up(x1)
    inter["up_512x100"] = x1u.numpy()
    cat = torch.cat([x2, x1u], dim=1)
    inter["cat_640x100"] = cat.numpy()
    nc = m.neck.conv(cat)
    inter["neckconv_512x100"] = nc.numpy()
    up2 = m.neck.up2(nc)
    inter["up2_256x200"] = up2.numpy()
    final = m.final_relu(m.final_conv(up2))
    inter["final_256x200"] = final.numpy()

ref = np.load("bev_clampref_fp32.npy").astype(np.float32)
a, b = inter["final_256x200"].ravel(), ref.ravel()
cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
print("torch conv_only vs bev_clampref_fp32  cosine = %.6f  (max %.2f / ref max %.2f)"
      % (cos, inter["final_256x200"].max(), ref.max()))
np.savez("/home/ANON/bev_ref_intermediates.npz", **inter)
for k, v in inter.items():
    print(f"  {k:18s} {v.shape}  min {v.min():.3f} max {v.max():.3f} L2 {np.linalg.norm(v):.1f}")

# also dump the predicter head weights for the on-board occupancy argmax (task 3)
pr = {k[len('predicter.'):]: v for k, v in sd.items() if k.startswith('predicter.')}
np.savez("/home/ANON/predicter_head.npz", **{k: v.numpy() for k, v in pr.items()})
print("predicter keys:", list(pr.keys()))
