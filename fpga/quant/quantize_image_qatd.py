#!/usr/bin/env python
"""Deploy the QAT-distilled IMAGE path: ImageFull with Hardtanh(0,alpha) + QAT weights -> Vitis PTQ ->
export xmodel. Mount 03_OccFPGA_Work as /work; needs occfpga_image/image_full_qatd_sd.pth + occfpga_quant/calib.npy.
  python quantize_image_qatd.py [NSAMP=64] [export=1]
"""
import os, sys, numpy as np, torch, torch.nn as nn
from pytorch_nndct.apis import torch_quantizer
sys.path.insert(0, "/work/occfpga_image")
from image_full_model import ImageFull
W = "/work/occfpga_image"; Q = "/work/occfpga_quant"
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 64
EXPORT = (len(sys.argv) > 2 and sys.argv[2] == "1")

ck = torch.load(os.path.join(W, "image_full_qatd_sd.pth"), map_location="cpu")
clean_sd, alphas = ck["clean_sd"], ck["alphas"]
print("alphas(first8)", [round(a, 1) for a in alphas[:8]], "n=%d" % len(alphas), flush=True)
m = ImageFull()
ait = iter(alphas)
def rep(mm):
    for n, c in mm.named_children():
        if isinstance(c, nn.ReLU): setattr(mm, n, nn.Hardtanh(0.0, float(next(ait)), inplace=False))
        else: rep(c)
rep(m)
miss, unexp = m.load_state_dict(clean_sd, strict=False)
print("missing(non-bn)", [k for k in miss if "num_batches" not in k][:4], "unexp", list(unexp)[:4], flush=True)
m.eval()
calib = np.load(os.path.join(Q, "calib.npy")).astype(np.float32)[:NS]
test_in = torch.from_numpy(np.load(os.path.join(Q, "test_input.npy")))
ref = np.load(os.path.join(W, "ref_fp32_depthnet.npy"))
qdir = os.path.join(W, "quantize_result_qatd")

q = torch_quantizer("calib", m, (torch.randn(1, 3, 256, 704),), output_dir=qdir)
qm = q.quant_model
with torch.no_grad():
    for i in range(0, len(calib), 8): qm(torch.from_numpy(calib[i:i + 8]))
q.export_quant_config()
q = torch_quantizer("test", m, (torch.randn(1, 3, 256, 704),), output_dir=qdir)
qm = q.quant_model
with torch.no_grad(): out = qm(test_in).numpy()
def sm(x): e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def cos(a, b): return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
df, dq = sm(ref[:, :88]), sm(out[:, :88])
am = float((df.argmax(1) == dq.argmax(1)).mean())
print("QATD-IMG INT8 vs FP32: depth-softmax cos %.4f | feat cos %.4f | depth-argmax %.3f"
      % (cos(df, dq), cos(ref[:, 88:], out[:, 88:]), am), flush=True)
if EXPORT:
    q.export_xmodel(deploy_check=False, output_dir=qdir); print("QATD_IMG_XMODEL_EXPORTED", flush=True)
