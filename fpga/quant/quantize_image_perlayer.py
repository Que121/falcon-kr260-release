#!/usr/bin/env python
"""DIG DEEPER: per-layer INT8-vs-FP cosine on the REAL Vitis quant model, to localize WHERE the deep
image net loses fidelity (the way the BEV win came from finding its heavy-tail layers). Runs the FP
reimpl and the Vitis test-mode quant model on the same input, hooks every Conv2d output in forward order
(both have identical ImageFull structure -> align by index), prints per-layer cos sorted worst-first +
the activation range per layer. Worst layers -> targeted fix-point treatment / where QAT must focus.
  python quantize_image_perlayer.py [NSAMP=64]
"""
import os, sys, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/work/occfpga_image")
from image_full_model import ImageFull
from pytorch_nndct.apis import torch_quantizer
W = "/work/occfpga_image"; Q = "/work/occfpga_quant"
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 64
test_in = torch.from_numpy(np.load(os.path.join(Q, "test_input.npy")))

def cos(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def collect(model, x):
    outs = []; names = []
    hs = []
    for nm, md in model.named_modules():
        if isinstance(md, nn.Conv2d):
            def mk(n):
                def h(m, i, o): outs.append(o.detach().float().cpu().numpy()); names.append(n)
                return h
            hs.append(md.register_forward_hook(mk(nm)))
    with torch.no_grad(): model(x)
    for h in hs: h.remove()
    return names, outs

# FP reimpl
mfp = ImageFull(); mfp.load_state_dict(torch.load(os.path.join(W, "image_full_sd.pth"), map_location="cpu")); mfp.eval()
fp_names, fp_outs = collect(mfp, test_in)

# Vitis quant (default diffs+modal = the deployed-equivalent), test mode
m = ImageFull(); m.load_state_dict(torch.load(os.path.join(W, "image_full_sd.pth"), map_location="cpu")); m.eval()
calib = np.load(os.path.join(Q, "calib.npy")).astype(np.float32)[:NS]
qdir = os.path.join(W, "quantize_result_perlayer")
q = torch_quantizer("calib", m, (torch.randn(1, 3, 256, 704),), output_dir=qdir)
qm = q.quant_model
with torch.no_grad():
    for i in range(0, len(calib), 8): qm(torch.from_numpy(calib[i:i + 8]))
q.export_quant_config()
q = torch_quantizer("test", m, (torch.randn(1, 3, 256, 704),), output_dir=qdir)
qm = q.quant_model
q_names, q_outs = collect(qm, test_in)

print("FP convs=%d  Vitis convs=%d" % (len(fp_outs), len(q_outs)), flush=True)
n = min(len(fp_outs), len(q_outs))
rows = []
for i in range(n):
    c = cos(fp_outs[i], q_outs[i]); rng = float(np.abs(fp_outs[i]).max())
    rows.append((c, fp_names[i], rng))
print("==== per-layer INT8-vs-FP cosine (WORST first) ====", flush=True)
for c, nm, rng in sorted(rows)[:20]:
    print("  cos %.4f  |fp|max %7.2f  %s" % (c, rng, nm), flush=True)
print("---- final output cos ----", flush=True)
print("  last conv (depth_net) cos %.4f" % rows[-1][0], flush=True)
print("PERLAYER_DONE", flush=True)
