#!/usr/bin/env python
"""Deploy image_split at the THEORETICAL per-tensor-INT8 CEILING by forcing NON-CLIPPING calibration.
The deployed image_split lost ~5.4 mIoU because Vitis default activation calib = method 'diffs' +
calib_statistic_method 'modal' (error-minimizing -> CLIPS the feat/depth tail; board feat capped ~16
vs FP 35.7). The idealized per-tensor-pow2 sim (feat/depth cos 0.983 -> img 24.31) NEVER clips. Forcing
method='maxmin' + calib_statistic_method='max' (cover the true max) replicates that idealized scheme on
the DPU by construction -> should hit ~0.983. Reimpl is FP-perfect (verify_reimpl_fp32 cos 1.0) so plain
FP weights + non-clip calib is all that's needed (no QAT).
  CMETHOD=maxmin CSTAT=max python quantize_image_maxcalib.py [NSAMP=64] [export=1]
"""
import os, sys, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/work/occfpga_image")
from image_full_model import ImageFull
from pytorch_nndct.apis import torch_quantizer
W = "/work/occfpga_image"; Q = "/work/occfpga_quant"
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 64
EXPORT = (len(sys.argv) > 2 and sys.argv[2] == "1")
METHOD = os.environ.get("CMETHOD", "maxmin"); CSTAT = os.environ.get("CSTAT", "max")

cfg = {"target_device": "DPU",
       "quantizable_data_type": ["input", "weights", "bias", "activation"],
       "tensor_quantize_config": {
           "activation": {"method": METHOD, "calib_statistic_method": CSTAT},
           "input": {"method": METHOD, "calib_statistic_method": CSTAT},
           "weights": {"method": METHOD}}}
cfgp = os.path.join(W, "qcfg_maxcalib.json"); json.dump(cfg, open(cfgp, "w"))
print("quant_config: method=%s calib_statistic_method=%s (non-clipping target)" % (METHOD, CSTAT), flush=True)

m = ImageFull(); m.load_state_dict(torch.load(os.path.join(W, "image_full_sd.pth"), map_location="cpu")); m.eval()
calib = np.load(os.path.join(Q, "calib.npy")).astype(np.float32)[:NS]
test_in = torch.from_numpy(np.load(os.path.join(Q, "test_input.npy")))
ref = np.load(os.path.join(W, "ref_fp32_depthnet.npy"))
qdir = os.path.join(W, "quantize_result_maxcalib")

q = torch_quantizer("calib", m, (torch.randn(1, 3, 256, 704),), output_dir=qdir, quant_config_file=cfgp)
qm = q.quant_model
with torch.no_grad():
    for i in range(0, len(calib), 8): qm(torch.from_numpy(calib[i:i + 8]))
q.export_quant_config()

q = torch_quantizer("test", m, (torch.randn(1, 3, 256, 704),), output_dir=qdir, quant_config_file=cfgp)
qm = q.quant_model
with torch.no_grad(): out = qm(test_in).numpy()
def sm(x): e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def cos(a, b): return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
df, dq = sm(ref[:, :88]), sm(out[:, :88])
print("MAXCALIB(%s/%s): depth-softmax cos %.4f | feat cos %.4f | depth-argmax %.3f  (target ~0.983; board was ~0.94)"
      % (METHOD, CSTAT, cos(df, dq), cos(ref[:, 88:], out[:, 88:]), float((df.argmax(1) == dq.argmax(1)).mean())), flush=True)
if EXPORT:
    q.export_xmodel(deploy_check=False, output_dir=qdir); print("MAXCALIB_XMODEL_EXPORTED", flush=True)
    for f in sorted(os.listdir(qdir)):
        if f.endswith(".xmodel"): print("  xmodel:", f, flush=True)
