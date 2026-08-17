#!/usr/bin/env python
"""Probe ImageFull intermediate activation ranges (esp. FPN convs, which have NO ReLU so were never
clamped) + sim: does clamping them help the per-tensor-pow2-8bit depth-softmax? Pro6000 ANONPROJ_310."""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision
sys.path.insert(0, "/home/ANON/03_OccFPGA_Work/occfpga_image")
from image_full_model import ImageFull
sd = torch.load("/home/ANON/03_OccFPGA_Work/occfpga_image/image_full_sd.pth", map_location="cpu")
ti = torch.from_numpy(np.load("/home/ANON/03_OccFPGA_Work/occfpga_quant/test_input.npy"))
m = ImageFull(); m.load_state_dict(sd); m.eval()

# probe: hook every conv/relu/add output, report mean|/max|/p99.9
stats = []
def mk(nm):
    def h(mod, i, o):
        a = o.detach().abs().flatten()
        stats.append((nm, float(a.mean()), float(a.max()), float(torch.quantile(a[:200000], 0.999))))
    return h
hs = []
for n, mod in m.named_modules():
    if isinstance(mod, (nn.Conv2d, nn.ReLU)) and n: hs.append(mod.register_forward_hook(mk(n)))
with torch.no_grad(): m(ti)
for h in hs: h.remove()
print("=== top heavy-tail activations (max/mean) ===")
for nm, me, mx, p in sorted(stats, key=lambda s: -s[2]/(s[1]+1e-6))[:12]:
    print("  %-30s mean|%6.2f max|%7.1f p99.9 %6.1f  max/mean %6.1fx" % (nm, me, mx, p, mx/(me+1e-6)))

# sim: per-tensor-pow2-8bit on all conv/relu/add outputs, with optional clamp; FP32 depth_net for clean readout
ref_d = None
def q8(x):
    mx = x.abs().max().clamp(min=1e-8); s = torch.pow(2.0, torch.floor(torch.log2(127.0/mx)))
    return torch.clamp(torch.round(x*s), -128, 127)/s
dn_w = sd["depth_net.weight"][:88]; dn_b = sd["depth_net.bias"][:88]
def run(clamp):
    mm = ImageFull(); mm.load_state_dict(sd); mm.eval()
    for mod in mm.modules():
        if isinstance(mod, nn.Conv2d):
            with torch.no_grad(): mod.weight.copy_(q8(mod.weight))
    hooks = []
    def hh(mod, i, o):
        o2 = o if clamp is None else torch.clamp(o, -clamp, clamp)
        return q8(o2)
    for n, mod in mm.named_modules():
        # clamp+quant conv/relu/upsample outputs EXCEPT the final depth_net (keep logits clean)
        if isinstance(mod, (nn.Conv2d, nn.ReLU, nn.Upsample)) and n and n != "depth_net":
            hooks.append(mod.register_forward_hook(hh))
    with torch.no_grad():
        out = mm(q8(ti)); dlog = out[:, :88]
    for h in hooks: h.remove()
    return F.softmax(dlog, 1).numpy()
with torch.no_grad():
    fp = F.softmax(m(ti)[:, :88], 1).numpy()
def cos(a, b): return float(a.ravel()@b.ravel()/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
print("=== clamp sweep: depth-softmax cos vs FP32 (clamp backbone+FPN, FP32 depth_net) ===")
for c in [None, 64, 32, 16, 8]:
    print("  clamp=%-5s depth-softmax cos %.4f" % (str(c), cos(run(c), fp)), flush=True)
