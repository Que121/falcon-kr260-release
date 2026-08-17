#!/usr/bin/env python
"""Deploy the QAT-v3 weights (image_full_qatv3_sd.pth, trained per-tensor-INT8-faithful, depth-softmax
cos 0.988 in QAT) through Vitis -> xmodel. The QAT made the weights robust to per-tensor pow2 INT8, so a
plain Vitis PTQ (CLE+BC on, diffs/modal, NO clamp - image not heavy-tailed) should now transfer the 0.988.
Compares the deployed INT8 output to the ORIGINAL FP teacher (ref_fp32_depthnet.npy) since QAT trained the
student's INT8 forward to match the teacher's FP.
  python quantize_image_qatv3_deploy.py [NSAMP=128] [export=1]
"""
import os, sys, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/work/occfpga_image")
from image_full_model import ImageFull
from pytorch_nndct.apis import torch_quantizer
W = "/work/occfpga_image"; Q = "/work/occfpga_quant"
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 128
EXPORT = (len(sys.argv) > 2 and sys.argv[2] == "1")

cfg = {"convert_relu6_to_relu": True, "include_cle": True, "include_bias_corr": True,
       "keep_first_last_layer_accuracy": True, "target_device": "DPU",
       "quantizable_data_type": ["input", "weights", "bias", "activation"]}
cfgp = os.path.join(W, "qcfg_image.json"); json.dump(cfg, open(cfgp, "w"))

ck = torch.load(os.path.join(W, "image_full_qatv3_sd.pth"), map_location="cpu")
clean = ck["clean_sd"] if "clean_sd" in ck else ck
m = ImageFull(); miss, unexp = m.load_state_dict(clean, strict=False)
print("loaded qatv3; missing(non-bn) %s unexp %s" % ([k for k in miss if "num_batches" not in k][:4], unexp[:4]), flush=True)
m.eval()
calib = np.load(os.path.join(Q, "calib.npy")).astype(np.float32)[:NS]
test_in = torch.from_numpy(np.load(os.path.join(Q, "test_input.npy")))
ref = np.load(os.path.join(W, "ref_fp32_depthnet.npy"))   # ORIGINAL FP teacher
qdir = os.path.join(W, "quantize_result_qatv3")

q = torch_quantizer("calib", m, (torch.randn(1, 3, 256, 704),), output_dir=qdir, quant_config_file=cfgp)
qm = q.quant_model
with torch.no_grad():
    for i in range(0, len(calib), 8): qm(torch.from_numpy(calib[i:i + 8]))
q.export_quant_config()
q = torch_quantizer("test", m, (torch.randn(1, 3, 256, 704),), output_dir=qdir, quant_config_file=cfgp)
with torch.no_grad(): out = q.quant_model(test_in).numpy()
def sm(x): e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def cos(a, b): return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
df, dq = sm(ref[:, :88]), sm(out[:, :88])
print("QATV3-DEPLOY vs FP teacher: depth-softmax cos %.4f | feat cos %.4f | depth-argmax %.3f  (QAT train cos 0.988; deployed-PTQ-FP was 0.949/0.940)"
      % (cos(df, dq), cos(ref[:, 88:], out[:, 88:]), float((df.argmax(1) == dq.argmax(1)).mean())), flush=True)
if EXPORT:
    q.export_xmodel(deploy_check=False, output_dir=qdir); print("QATV3_XMODEL_EXPORTED", flush=True)
    for f in sorted(os.listdir(qdir)):
        if f.endswith(".xmodel"): print("  xmodel:", f, flush=True)
