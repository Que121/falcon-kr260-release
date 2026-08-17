#!/usr/bin/env python
"""Realize the clamp on the DPU WITHOUT fragmenting: plain ReLU (DPU-native, 3 subgraphs) + override
activation fix_points to >= FPCLAMP. The DPU's INT8 saturation at fp=FPCLAMP IS the clamp
(127*2^-3 = 15.9 ~ clamp16), and ReLU fuses into conv so the graph stays 3-DPU + 2-resize.
Mount occfpga_quant_bev as /work.  FPCLAMP=3 python quantize_bev_fpclamp.py [NSAMP=48] [export=1]
"""
import os, sys, json, numpy as np, torch, torch.nn as nn
from pytorch_nndct.apis import torch_quantizer
from bev_stage import BEVStage
WORK = "/work"
FPC = int(os.environ.get("FPCLAMP", "3")); NS = int(sys.argv[1]) if len(sys.argv) > 1 else 48
EXPORT = (len(sys.argv) > 2 and sys.argv[2] == "1")
full_sd = torch.load(os.path.join(WORK, "bev_stage_sd.pth"), map_location="cpu")
m = BEVStage(conv_only=True)
m.load_state_dict({k: v for k, v in full_sd.items() if not k.startswith("predicter.")}, strict=False); m.eval()
calib = np.load(os.path.join(WORK, "calib_bev.npy"))[:NS].astype(np.float32)
ref = np.load(os.path.join(WORK, "convonly_0000.npy")).astype(np.float32)
frame = np.load(os.path.join(WORK, "frame_0000.npz")); vt0 = torch.from_numpy(frame["vt_out"].astype(np.float32))[None]
occ_ref = frame["occ"].astype(np.int64)
W0 = full_sd["predicter.0.weight"].numpy(); b0 = full_sd["predicter.0.bias"].numpy()
W2 = full_sd["predicter.2.weight"].numpy(); b2 = full_sd["predicter.2.bias"].numpy()
def occ_of(c):
    x = c.transpose(2, 1, 0).reshape(-1, 256) @ W0.T + b0
    x = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
    return (x @ W2.T + b2).reshape(200, 200, 16, 18).argmax(-1).astype(np.int64)
def cos(a, b): return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
qdir = os.path.join(WORK, "quantize_result_bev_fpc")

q = torch_quantizer("calib", m, (torch.randn(1, 64, 200, 200),), output_dir=qdir)
qm = q.quant_model
with torch.no_grad():
    for i in range(0, NS, 2): qm(torch.from_numpy(calib[i:i + 2]))
q.export_quant_config()

# --- override activation output fix_points: fp = max(fp, FPC)  (clamp heavy-tailed tensors) ---
qi = os.path.join(qdir, "quant_info.json"); J = json.load(open(qi))
n = 0
for k, v in J.get("output", {}).items():
    if isinstance(v, list) and v and isinstance(v[0], list) and len(v[0]) == 2:
        if v[0][1] < FPC: v[0][1] = FPC; n += 1
json.dump(J, open(qi, "w"))
print("raised %d activation fix_points to >=%d (clamp ~%.1f)" % (n, FPC, 127 / 2**FPC), flush=True)

q = torch_quantizer("test", m, (torch.randn(1, 64, 200, 200),), output_dir=qdir)
qm = q.quant_model
with torch.no_grad(): conv = qm(vt0)[0].numpy()
occ = occ_of(conv)
print("FPCLAMP(fp>=%d) conv cos %.4f | occ-agree %.4f | free %.3f (ref %.3f) | conv range %.1f"
      % (FPC, cos(conv, ref), float((occ == occ_ref).mean()), (occ == 17).mean(), (occ_ref == 17).mean(), conv.max()), flush=True)
if EXPORT:
    q.export_xmodel(deploy_check=False, output_dir=qdir); print("FPC_XMODEL_EXPORTED", flush=True)
