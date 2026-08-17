#!/usr/bin/env python
"""BEV-stage INT8 with fast_finetune (AdaQuant), REDUCED calib for tractable CPU time.
Goal: recover the 0.59 DPU-PTQ conv cosine. Exports quantize_result_bev_ft + prints conv cosine
on frame_0000 vs FP32. Mount occfpga_quant_bev as /work.
  python quantize_bev_ft_fast.py [NSAMP=24]
"""
import os, sys, numpy as np, torch
from pytorch_nndct.apis import torch_quantizer
from bev_stage import BEVStage

WORK = "/work"
NSAMP = int(sys.argv[1]) if len(sys.argv) > 1 else 24
model = BEVStage(conv_only=True)
full_sd = torch.load(os.path.join(WORK, "bev_stage_sd.pth"), map_location="cpu")
model.load_state_dict({k: v for k, v in full_sd.items() if not k.startswith("predicter.")}, strict=False)
model.eval()

calib = np.load(os.path.join(WORK, "calib_bev.npy"))[:NSAMP]
ref = np.load(os.path.join(WORK, "convonly_0000.npy")).astype(np.float32)
frame = np.load(os.path.join(WORK, "frame_0000.npz"))
vt0 = torch.from_numpy(frame["vt_out"].astype(np.float32))[None]
dummy = torch.randn(1, 64, 200, 200)
qdir = os.path.join(WORK, "quantize_result_bev_ft")

def run_fn(m, n):
    with torch.no_grad():
        for i in range(0, min(n, NSAMP), 2):
            m(torch.from_numpy(calib[i:i + 2]))

q = torch_quantizer("calib", model, (dummy,), output_dir=qdir)
qm = q.quant_model
print("BEV fast_finetune (NSAMP=%d) ..." % NSAMP, flush=True)
q.fast_finetune(run_fn, (qm, NSAMP))
run_fn(qm, NSAMP)
q.export_quant_config()

q = torch_quantizer("test", model, (dummy,), output_dir=qdir)
q.load_ft_param()
qm = q.quant_model
with torch.no_grad():
    conv = qm(vt0)[0].numpy()
def cos(a, b): return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
print("BEV-FT conv vs FP32: cos %.4f  (PTQ was 0.59)" % cos(conv, ref), flush=True)
q.export_xmodel(deploy_check=False, output_dir=qdir)
print("BEV_FT_XMODEL_EXPORTED", flush=True)
