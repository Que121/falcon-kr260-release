#!/usr/bin/env python
"""Deploy the QAT split image head: ImageSplit + QAT weights -> Vitis PTQ (no override) -> 2-output
xmodel (depth_conv 88 + feat_conv 64, SEPARATE fix_points). Reports depth-argmax + feat cos.
Mount 03_OccFPGA_Work as /work.  SD=image_split_split_sd.pth python quantize_image_split_deploy.py [NSAMP=64] [export=1]
"""
import os, sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/work/occfpga_image")
from image_full_split import ImageSplit
from pytorch_nndct.apis import torch_quantizer
W = "/work/occfpga_image"; Q = "/work/occfpga_quant"
SD = os.environ.get("SD", "image_split_split_sd.pth")
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 64
EXPORT = (len(sys.argv) > 2 and sys.argv[2] == "1")
ck = torch.load(os.path.join(W, SD), map_location="cpu")
clean_sd, use_bn = ck["clean_sd"], ck.get("use_bn", False)
m = ImageSplit(use_bn=use_bn)
miss, unexp = m.load_state_dict(clean_sd, strict=False)
print("use_bn", use_bn, "missing(non-bn)", [k for k in miss if "num_batches" not in k][:4], flush=True); m.eval()
calib = np.load(os.path.join(Q, "calib.npy")).astype(np.float32)[:NS]
test_in = torch.from_numpy(np.load(os.path.join(Q, "test_input.npy")))
ref = np.load(os.path.join(W, "ref_fp32_depthnet.npy"))   # (1,152,16,44) FP32: [:88] logits, [88:] feat
qdir = os.path.join(W, "quantize_result_split")
q = torch_quantizer("calib", m, (torch.randn(1, 3, 256, 704),), output_dir=qdir)
qm = q.quant_model
with torch.no_grad():
    for i in range(0, len(calib), 8): qm(torch.from_numpy(calib[i:i + 8]))
q.export_quant_config()
q = torch_quantizer("test", m, (torch.randn(1, 3, 256, 704),), output_dir=qdir)
qm = q.quant_model
with torch.no_grad():
    dlog, feat = qm(test_in); dlog = dlog.numpy(); feat = feat.numpy()
def sm(x): e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def cos(a, b): return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
df, dq = sm(ref[:, :88]), sm(dlog); am = float((df.argmax(1) == dq.argmax(1)).mean())
print("SPLIT-DEPLOY INT8: depth-softmax cos %.4f | feat cos %.4f | depth-argmax %.3f"
      % (cos(df, dq), cos(ref[:, 88:], feat), am), flush=True)
if EXPORT:
    q.export_xmodel(deploy_check=False, output_dir=qdir); print("SPLIT_XMODEL_EXPORTED", flush=True)
