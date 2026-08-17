#!/usr/bin/env python
"""Deploy the QAT-distilled BEV: rebuild BEVStage with Hardtanh(0, learned-alpha) clamps + QAT weights,
Vitis PTQ (now quant-robust -> respects the clamps), report conv cos + occ agree, export xmodel.
Mount occfpga_quant_bev as /work; needs bev_stage_qatd_sd.pth + predicter_qatd.npz + frame_0000.npz +
convonly_0000.npy in /work.   python quantize_bev_qatd.py [NSAMP=32] [export=1]
"""
import os, sys, numpy as np, torch, torch.nn as nn
from pytorch_nndct.apis import torch_quantizer
from bev_stage import BEVStage
WORK = "/work"; NS = int(sys.argv[1]) if len(sys.argv) > 1 else 32
EXPORT = (len(sys.argv) > 2 and sys.argv[2] == "1")

ck = torch.load(os.path.join(WORK, "bev_stage_qatd_sd.pth"), map_location="cpu")
clean_sd, alphas = ck["clean_sd"], ck["alphas"]
print("alphas(first8)", [round(a, 1) for a in alphas[:8]], "n=%d" % len(alphas), flush=True)

model = BEVStage(conv_only=True)
ait = iter(alphas)
def rep(m):
    for n, c in m.named_children():
        if isinstance(c, nn.ReLU): setattr(m, n, nn.Hardtanh(0.0, float(next(ait)), inplace=False))
        else: rep(c)
rep(model)
miss, unexp = model.load_state_dict(clean_sd, strict=False)
print("load missing(non-bn)", [k for k in miss if "num_batches" not in k][:4], "unexp", list(unexp)[:4], flush=True)
model.eval()

calib = np.load(os.path.join(WORK, "calib_bev.npy"))[:NS].astype(np.float32)
ref = np.load(os.path.join(WORK, "convonly_0000.npy")).astype(np.float32)
frame = np.load(os.path.join(WORK, "frame_0000.npz"))
vt0 = torch.from_numpy(frame["vt_out"].astype(np.float32))[None]; occ_ref = frame["occ"].astype(np.int64)
ph = np.load(os.path.join(WORK, "predicter_qatd.npz"))
W0, b0, W2, b2 = ph["0.weight"], ph["0.bias"], ph["2.weight"], ph["2.bias"]
def occ_of(conv):
    x = conv.transpose(2, 1, 0).reshape(-1, 256) @ W0.T + b0
    x = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
    return (x @ W2.T + b2).reshape(200, 200, 16, 18).argmax(-1).astype(np.int64)
qdir = os.path.join(WORK, "quantize_result_bev_qatd")

q = torch_quantizer("calib", model, (torch.randn(1, 64, 200, 200),), output_dir=qdir)
qm = q.quant_model
with torch.no_grad():
    for i in range(0, NS, 2): qm(torch.from_numpy(calib[i:i + 2]))
q.export_quant_config()
q = torch_quantizer("test", model, (torch.randn(1, 64, 200, 200),), output_dir=qdir)
qm = q.quant_model
with torch.no_grad(): conv = qm(vt0)[0].numpy()
occ = occ_of(conv.transpose(0, 1, 2) if conv.ndim == 3 else conv)
def cos(a, b): return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
print("QATD DPU-INT8 conv cos %.4f | occ-agree %.4f | free %.3f (ref %.3f) | conv range %.1f"
      % (cos(conv, ref), float((occ == occ_ref).mean()), (occ == 17).mean(), (occ_ref == 17).mean(), conv.max()), flush=True)
if EXPORT:
    q.export_xmodel(deploy_check=False, output_dir=qdir); print("QATD_XMODEL_EXPORTED", flush=True)
