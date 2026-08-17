#!/usr/bin/env python
"""BEV INT8 with activation CLAMP (ReLU->ReLU6, DPU-native) -- tames the heavy-tailed activations so
per-tensor-pow2 8-bit resolves them. Sim predicts conv cos ~0.92 (vs 0.59 unclamped). Re-quantizes,
reports conv cosine + occ agreement vs FP32, exports xmodel. Mount occfpga_quant_bev as /work.
  python quantize_bev_clamp.py [NSAMP=32] [export=1]
"""
import os, sys, numpy as np, torch, torch.nn as nn
from pytorch_nndct.apis import torch_quantizer
from bev_stage import BEVStage
WORK = "/work"
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 32
EXPORT = (len(sys.argv) > 2 and sys.argv[2] == "1")

CLAMPC = float(os.environ.get("CLAMPC", "8"))
KIND = os.environ.get("KIND", "ht")    # ht=Hardtanh(0,C), r6=ReLU6
def replace_relu(m):
    for name, ch in m.named_children():
        if isinstance(ch, nn.ReLU):
            setattr(m, name, nn.ReLU6(inplace=False) if KIND == "r6" else nn.Hardtanh(0.0, CLAMPC, inplace=False))
        else: replace_relu(ch)

model = BEVStage(conv_only=True)
full_sd = torch.load(os.path.join(WORK, "bev_stage_sd.pth"), map_location="cpu")
model.load_state_dict({k: v for k, v in full_sd.items() if not k.startswith("predicter.")}, strict=False)
replace_relu(model); model.eval()
print("KIND=%s CLAMPC=%g" % (KIND, CLAMPC), flush=True)

calib = np.load(os.path.join(WORK, "calib_bev.npy"))[:NS].astype(np.float32)
ref = np.load(os.path.join(WORK, "convonly_0000.npy")).astype(np.float32)
frame = np.load(os.path.join(WORK, "frame_0000.npz"))
vt0 = torch.from_numpy(frame["vt_out"].astype(np.float32))[None]
occ_ref = frame["occ"].astype(np.int64)
pr = BEVStage(conv_only=False); pr.load_state_dict(full_sd); pr = pr.predicter.eval()
qdir = os.path.join(WORK, "quantize_result_bev_clamp")

q = torch_quantizer("calib", model, (torch.randn(1, 64, 200, 200),), output_dir=qdir)
qm = q.quant_model
with torch.no_grad():
    for i in range(0, NS, 2): qm(torch.from_numpy(calib[i:i + 2]))
q.export_quant_config()
q = torch_quantizer("test", model, (torch.randn(1, 64, 200, 200),), output_dir=qdir)
qm = q.quant_model
with torch.no_grad():
    conv = qm(vt0)[0].numpy()
    o = pr(torch.from_numpy(conv)[None].permute(0, 3, 2, 1))
    occ = o.view(1, 200, 200, 16, 18).argmax(-1).numpy()[0]
def cos(a, b): return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
print("CLAMP(ReLU6) DPU-INT8 conv cos %.4f | occ-agree %.4f | free %.3f (ref %.3f) | conv range %.1f"
      % (cos(conv, ref), float((occ == occ_ref).mean()), (occ == 17).mean(), (occ_ref == 17).mean(), conv.max()), flush=True)
if EXPORT:
    q.export_xmodel(deploy_check=False, output_dir=qdir); print("CLAMP_XMODEL_EXPORTED", flush=True)
