#!/usr/bin/env python
"""Deploy QAT-distilled IMAGE with fix-point clamp (plain ReLU DPU-native + override activation
fix_points >= FPC; the QAT weights were trained for clamp~8 so FPC=4). Avoids Hardtanh fragmentation.
Mount 03_OccFPGA_Work as /work.  FPCLAMP=4 python quantize_image_qatd_fpc.py [NSAMP=64] [export=1]
"""
import os, sys, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/work/occfpga_image")
from image_full_model import ImageFull
from pytorch_nndct.apis import torch_quantizer
W = "/work/occfpga_image"; Q = "/work/occfpga_quant"
FPC = int(os.environ.get("FPCLAMP", "4")); NS = int(sys.argv[1]) if len(sys.argv) > 1 else 64
EXPORT = (len(sys.argv) > 2 and sys.argv[2] == "1")
ck = torch.load(os.path.join(W, "image_full_qatd_sd.pth"), map_location="cpu")
clean_sd = ck["clean_sd"]
m = ImageFull(); miss, unexp = m.load_state_dict(clean_sd, strict=False)
print("missing(non-bn)", [k for k in miss if "num_batches" not in k][:4], flush=True); m.eval()
calib = np.load(os.path.join(Q, "calib.npy")).astype(np.float32)[:NS]
test_in = torch.from_numpy(np.load(os.path.join(Q, "test_input.npy")))
ref = np.load(os.path.join(W, "ref_fp32_depthnet.npy"))
qdir = os.path.join(W, "quantize_result_qatd_fpc")
q = torch_quantizer("calib", m, (torch.randn(1, 3, 256, 704),), output_dir=qdir)
qm = q.quant_model
with torch.no_grad():
    for i in range(0, len(calib), 8): qm(torch.from_numpy(calib[i:i + 8]))
q.export_quant_config()
qi = os.path.join(qdir, "quant_info.json"); J = json.load(open(qi)); n = 0
for k, v in J.get("output", {}).items():
    if isinstance(v, list) and v and isinstance(v[0], list) and len(v[0]) == 2 and v[0][1] < FPC:
        v[0][1] = FPC; n += 1
json.dump(J, open(qi, "w")); print("raised %d image activation fps to >=%d" % (n, FPC), flush=True)
q = torch_quantizer("test", m, (torch.randn(1, 3, 256, 704),), output_dir=qdir)
qm = q.quant_model
with torch.no_grad(): out = qm(test_in).numpy()
def sm(x): e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def cos(a, b): return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
df, dq = sm(ref[:, :88]), sm(out[:, :88]); am = float((df.argmax(1) == dq.argmax(1)).mean())
print("QATD-IMG-FPC(fp>=%d): depth-softmax cos %.4f | feat cos %.4f | depth-argmax %.3f"
      % (FPC, cos(df, dq), cos(ref[:, 88:], out[:, 88:]), am), flush=True)
if EXPORT:
    q.export_xmodel(deploy_check=False, output_dir=qdir); print("QATD_IMG_FPC_XMODEL_EXPORTED", flush=True)
