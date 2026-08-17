#!/usr/bin/env python
"""A1+A2 (SOTA-research #1 experiment): deploy image_split with the Vitis accuracy knobs the deployed
xmodel NEVER used + the proven BEV fix-point-clamp ported to image, picking max POST-SOFTMAX cos.
A1 = quant_config_file {convert_relu6_to_relu, include_cle (cross-layer equalization = THE per-tensor
   rescue, Nagel 2019), include_bias_corr (fixes the depth-argmax mean shift), keep_first_last_layer_accuracy
   (protects depth_net)}. method/calib left at DPU-legal diffs/modal.
A2 = after calib, sweep raising output activation fix_points to >=FPC (FPC in {none,3,4,5}); pick the one
   maximizing depth-softmax cos vs the FP ref. Exports the best xmodel.
  python quantize_image_cle_clamp.py [NSAMP=128]
"""
import os, sys, json, copy, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/work/occfpga_image")
from image_full_model import ImageFull
from pytorch_nndct.apis import torch_quantizer
W = "/work/occfpga_image"; Q = "/work/occfpga_quant"
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 128

cfg = {"convert_relu6_to_relu": True, "include_cle": True, "include_bias_corr": True,
       "keep_first_last_layer_accuracy": True, "keep_add_layer_accuracy": True,
       "target_device": "DPU",
       "quantizable_data_type": ["input", "weights", "bias", "activation"]}
cfgp = os.path.join(W, "qcfg_image.json"); json.dump(cfg, open(cfgp, "w"))
print("A1 config: CLE=%s bias_corr=%s keep_first_last=%s relu6->relu=%s" %
      (cfg["include_cle"], cfg["include_bias_corr"], cfg["keep_first_last_layer_accuracy"], cfg["convert_relu6_to_relu"]), flush=True)

m = ImageFull(); m.load_state_dict(torch.load(os.path.join(W, "image_full_sd.pth"), map_location="cpu")); m.eval()
calib = np.load(os.path.join(Q, "calib.npy")).astype(np.float32)[:NS]
test_in = torch.from_numpy(np.load(os.path.join(Q, "test_input.npy")))
ref = np.load(os.path.join(W, "ref_fp32_depthnet.npy"))
qdir = os.path.join(W, "quantize_result_cleclamp")

q = torch_quantizer("calib", m, (torch.randn(1, 3, 256, 704),), output_dir=qdir, quant_config_file=cfgp)
qm = q.quant_model
with torch.no_grad():
    for i in range(0, len(calib), 8): qm(torch.from_numpy(calib[i:i + 8]))
q.export_quant_config()
qi = os.path.join(qdir, "quant_info.json")
base = json.load(open(qi))   # baseline (CLE+BC) fix-points

def sm(x): e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def cos(a, b): return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
df = sm(ref[:, :88]); reff = ref[:, 88:]

def test_with(J):
    json.dump(J, open(qi, "w"))
    qt = torch_quantizer("test", m, (torch.randn(1, 3, 256, 704),), output_dir=qdir, quant_config_file=cfgp)
    with torch.no_grad(): out = qt.quant_model(test_in).numpy()
    dq = sm(out[:, :88])
    return cos(df, dq), cos(reff, out[:, 88:]), float((df.argmax(1) == dq.argmax(1)).mean()), qt

results = {}
# FPC=None => baseline (CLE+BC only)
for FPC in [None, 3, 4, 5]:
    J = copy.deepcopy(base); n = 0
    if FPC is not None:
        for k, v in J.get("output", {}).items():
            if isinstance(v, list) and v and isinstance(v[0], list) and len(v[0]) == 2 and v[0][1] < FPC:
                v[0][1] = FPC; n += 1
    dcos, fcos, am, qt = test_with(J)
    results[str(FPC)] = (dcos, fcos, am)
    print("A1%s : depth-softmax cos %.4f | feat cos %.4f | argmax %.3f  (raised %d fps)  (board base 0.949/0.940/0.71)"
          % ("" if FPC is None else "+clamp>=%d" % FPC, dcos, fcos, am, n), flush=True)

best = max(results, key=lambda k: results[k][0])
print("BEST = A1%s  depth-softmax cos %.4f" % ("" if best == "None" else "+clamp>=%s" % best, results[best][0]), flush=True)
# rebuild best + export
Jb = copy.deepcopy(base)
if best != "None":
    for k, v in Jb.get("output", {}).items():
        if isinstance(v, list) and v and isinstance(v[0], list) and len(v[0]) == 2 and v[0][1] < int(best):
            v[0][1] = int(best)
json.dump(Jb, open(qi, "w"))
qt = torch_quantizer("test", m, (torch.randn(1, 3, 256, 704),), output_dir=qdir, quant_config_file=cfgp)
with torch.no_grad(): qt.quant_model(test_in)
qt.export_xmodel(deploy_check=False, output_dir=qdir)
print("CLECLAMP_XMODEL_EXPORTED (best=A1%s)" % ("" if best == "None" else "+clamp>=%s" % best), flush=True)
for f in sorted(os.listdir(qdir)):
    if f.endswith(".xmodel"): print("  xmodel:", f, flush=True)
