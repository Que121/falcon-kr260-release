#!/usr/bin/env python
"""Image-path INT8 with activation CLAMP (ReLU->Hardtanh(0,C)) to tame ResNet50's heavy-tailed
activations for the DPU per-tensor-pow2-8bit. Reports depth-argmax match + feat/depth-softmax cos.
Mount 03_OccFPGA_Work as /work.  CLAMPC=16 python quantize_image_clamp.py [NSAMP=64] [export=1]
"""
import os, sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/work/occfpga_image")
from image_full_model import ImageFull
from pytorch_nndct.apis import torch_quantizer
W = "/work/occfpga_image"; Q = "/work/occfpga_quant"
C = float(os.environ.get("CLAMPC", "16")); NS = int(sys.argv[1]) if len(sys.argv) > 1 else 64
EXPORT = (len(sys.argv) > 2 and sys.argv[2] == "1")
def rep(m):
    for n, ch in m.named_children():
        if isinstance(ch, nn.ReLU): setattr(m, n, nn.Hardtanh(0.0, C, inplace=False))
        else: rep(ch)
m = ImageFull(); m.load_state_dict(torch.load(os.path.join(W, "image_full_sd.pth"), map_location="cpu"))
rep(m); m.eval(); print("CLAMPC=%g" % C, flush=True)
calib = np.load(os.path.join(Q, "calib.npy")).astype(np.float32)[:NS]
test_in = torch.from_numpy(np.load(os.path.join(Q, "test_input.npy")))
ref = np.load(os.path.join(W, "ref_fp32_depthnet.npy"))
qdir = os.path.join(W, "quantize_result_clamp")
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
df, dq = sm(ref[:, :88]), sm(out[:, :88]); am = float((df.argmax(1) == dq.argmax(1)).mean())
print("CLAMP-IMG INT8: depth-softmax cos %.4f | feat cos %.4f | depth-argmax %.3f"
      % (cos(df, dq), cos(ref[:, 88:], out[:, 88:]), am), flush=True)
if EXPORT:
    q.export_xmodel(deploy_check=False, output_dir=qdir); print("CLAMP_IMG_XMODEL_EXPORTED", flush=True)
