#!/usr/bin/env python
"""Per-LAYER clamp PTQ: set each ReLU's Hardtanh ceiling to that layer's own p99.x (from calib), so
early heavy-tail layers (p99.9~5) clamp hard while later layers (p99.9~21) keep their range. Then
Vitis PTQ. Reports conv cos + occ. Mount occfpga_quant_bev as /work.
  python quantize_bev_clamp_pl.py [PCT=99.5] [NSAMP=32] [export=1]
"""
import os, sys, numpy as np, torch, torch.nn as nn
from pytorch_nndct.apis import torch_quantizer
from bev_stage import BEVStage
WORK = "/work"
PCT = float(os.environ.get("PCT", "99.5"))
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 32
EXPORT = (len(sys.argv) > 2 and sys.argv[2] == "1")
full_sd = torch.load(os.path.join(WORK, "bev_stage_sd.pth"), map_location="cpu")
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

# 1) record per-ReLU p99.x on calib
m0 = BEVStage(conv_only=True); m0.load_state_dict({k: v for k, v in full_sd.items() if not k.startswith("predicter.")}, strict=False); m0.eval()
relus = [mod for mod in m0.modules() if isinstance(mod, nn.ReLU)]
acc = [[] for _ in relus]
idx = {id(r): i for i, r in enumerate(relus)}
def hook(mod, inp, out):
    a = out.detach().abs().flatten()
    acc[idx[id(mod)]].append(float(torch.quantile(a[:100000], PCT / 100.0)))
hs = [r.register_forward_hook(hook) for r in relus]
with torch.no_grad():
    for i in range(0, min(8, NS)): m0(torch.from_numpy(calib[i:i + 1]))
for h in hs: h.remove()
clamps = [max(2.0, float(np.mean(a))) for a in acc]
print("per-layer clamps:", [round(c, 1) for c in clamps], flush=True)

# 2) build model with per-layer Hardtanh
m = BEVStage(conv_only=True); m.load_state_dict({k: v for k, v in full_sd.items() if not k.startswith("predicter.")}, strict=False)
cit = iter(clamps)
def rep(mm):
    for n, c in mm.named_children():
        if isinstance(c, nn.ReLU): setattr(mm, n, nn.Hardtanh(0.0, float(next(cit)), inplace=False))
        else: rep(c)
rep(m); m.eval()
qdir = os.path.join(WORK, "quantize_result_bev_clpl")
q = torch_quantizer("calib", m, (torch.randn(1, 64, 200, 200),), output_dir=qdir)
qm = q.quant_model
with torch.no_grad():
    for i in range(0, NS, 2): qm(torch.from_numpy(calib[i:i + 2]))
q.export_quant_config()
q = torch_quantizer("test", m, (torch.randn(1, 64, 200, 200),), output_dir=qdir)
qm = q.quant_model
with torch.no_grad(): conv = qm(vt0)[0].numpy()
occ = occ_of(conv)
print("CLPL(p%.1f) conv cos %.4f | occ-agree %.4f | free %.3f (ref %.3f) | conv range %.1f"
      % (PCT, cos(conv, ref), float((occ == occ_ref).mean()), (occ == 17).mean(), (occ_ref == 17).mean(), conv.max()), flush=True)
if EXPORT:
    q.export_xmodel(deploy_check=False, output_dir=qdir); print("CLPL_XMODEL_EXPORTED", flush=True)
