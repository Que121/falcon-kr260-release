#!/usr/bin/env python
"""Does clamping activations let per-tensor-pow2 8-bit (the DPU scheme) recover? If yes, an
activation-clamp finetune is the fix. Also reports per-layer max/mean (heavy-tail locator).
Runs on Pro6000 ANONPROJ_310 (no docker)."""
import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/home/ANON/03_OccFPGA_Work/occfpga_quant_bev")
from bev_stage import BEVStage
sd = torch.load("/home/ANON/03_OccFPGA_Work/occfpga_quant_bev/bev_stage_sd.pth", map_location="cpu")
ref = np.load("/home/ANON/convonly_0000.npy").astype(np.float32)
vt0 = torch.from_numpy(np.load("/home/ANON/frame_0000.npz")["vt_out"].astype(np.float32))[None]
def cos(a, b): return float((a.ravel() @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def q_pt_pow2(x, bits=8):
    qmax = 2 ** (bits - 1) - 1; mx = x.abs().max().clamp(min=1e-8)
    s = torch.pow(2.0, torch.floor(torch.log2(qmax / mx)))
    return torch.clamp(torch.round(x * s), -qmax - 1, qmax) / s

# per-layer activation stats (no quant)
m = BEVStage(conv_only=True); m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("predicter.")}, strict=False); m.eval()
stats = []
def stat_hook(name):
    def h(mod, inp, out):
        a = out.detach(); stats.append((name, float(a.abs().mean()), float(a.abs().max()), float(torch.quantile(a.abs().flatten()[:200000], 0.999))))
    return h
hs = [mod.register_forward_hook(stat_hook("relu%d" % i)) for i, mod in enumerate(m.modules()) if isinstance(mod, nn.ReLU)]
with torch.no_grad(): m(vt0)
for h in hs: h.remove()
print("per-ReLU |mean  |max   p99.9   max/mean")
for nm, me, mx, p in stats: print("  %-7s %5.2f %6.1f %6.1f   %5.1fx" % (nm, me, mx, p, mx / (me + 1e-6)))

# clamp sweep: clamp each activation to CLAMP then pt_pow2_8
for clamp in [None, 64, 32, 16, 8, 6, 4]:
    m = BEVStage(conv_only=True); m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("predicter.")}, strict=False); m.eval()
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            with torch.no_grad(): mod.weight.copy_(q_pt_pow2(mod.weight, 8))
    hooks = []
    def mkhook(c):
        def h(mod, inp, out):
            o = out if c is None else torch.clamp(out, -c, c)
            return q_pt_pow2(o, 8)
        return h
    for mod in m.modules():
        if isinstance(mod, (nn.ReLU, nn.Upsample)): hooks.append(mod.register_forward_hook(mkhook(clamp)))
    with torch.no_grad():
        x = q_pt_pow2(torch.clamp(vt0, -(clamp or 1e9), (clamp or 1e9)), 8)
        conv = m(x)[0].numpy()
    for h in hooks: h.remove()
    print("clamp=%-4s  pt_pow2_8 conv cos %.4f" % (str(clamp), cos(conv, ref)), flush=True)
print("SIM_DONE", flush=True)
