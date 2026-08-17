#!/usr/bin/env python
"""Pinpoint the DPU-INT8 accuracy bottleneck: fake-quantize the BEV encoder's activations+weights
under different schemes and compare conv_only cosine vs FP32. Runs on Pro6000 ANONPROJ_310 (no docker).

  fp32        - no quant (sanity 1.0)
  pt_pow2_8   - per-tensor power-of-2 8-bit  (== the DPU's activation scheme; expect ~0.6)
  pt_arb_8    - per-tensor arbitrary-scale 8-bit
  pc_arb_8    - per-CHANNEL arbitrary-scale 8-bit  (what the framework sim can do)
  pt_pow2_16  - per-tensor power-of-2 16-bit
If pc_arb_8 / pt_pow2_16 jump to ~0.95 while pt_pow2_8 stays ~0.6, the per-tensor-pow2 8-bit
constraint is proven to be the culprit (not a bug, not the model).
"""
import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/home/ANON/03_OccFPGA_Work/occfpga_quant_bev")
from bev_stage import BEVStage

sd = torch.load("/home/ANON/03_OccFPGA_Work/occfpga_quant_bev/bev_stage_sd.pth", map_location="cpu")
ref = np.load("/home/ANON/convonly_0000.npy").astype(np.float32)
vt0 = torch.from_numpy(np.load("/home/ANON/frame_0000.npz")["vt_out"].astype(np.float32))[None]

def q_pt_pow2(x, bits=8):
    qmax = 2 ** (bits - 1) - 1
    mx = x.abs().max().clamp(min=1e-8)
    fp = torch.floor(torch.log2(qmax / mx))
    s = torch.pow(2.0, fp)
    return torch.clamp(torch.round(x * s), -qmax - 1, qmax) / s
def q_pt_arb(x, bits=8):
    qmax = 2 ** (bits - 1) - 1
    s = x.abs().max().clamp(min=1e-8) / qmax
    return torch.clamp(torch.round(x / s), -qmax - 1, qmax) * s
def q_pc_arb(x, bits=8):                       # per output-channel (dim1) for activations
    qmax = 2 ** (bits - 1) - 1
    mx = x.abs().amax(dim=(0, 2, 3), keepdim=True).clamp(min=1e-8) if x.dim() == 4 else x.abs().max()
    s = mx / qmax
    return torch.clamp(torch.round(x / s), -qmax - 1, qmax) * s

QFN = {"fp32": None, "pt_pow2_8": lambda x: q_pt_pow2(x, 8), "pt_arb_8": lambda x: q_pt_arb(x, 8),
       "pc_arb_8": lambda x: q_pc_arb(x, 8), "pt_pow2_16": lambda x: q_pt_pow2(x, 16)}

def cos(a, b): return float((a.ravel() @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

for mode, qf in QFN.items():
    m = BEVStage(conv_only=True)
    m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("predicter.")}, strict=False)
    m.eval()
    hooks = []
    if qf is not None:
        # quantize WEIGHTS of every conv (per-channel for weights is allowed even on DPU; here match mode)
        for mod in m.modules():
            if isinstance(mod, nn.Conv2d):
                with torch.no_grad():
                    mod.weight.copy_(q_pc_arb(mod.weight, 8) if mode.startswith("pc") else
                                     (q_pt_pow2(mod.weight, 16) if "16" in mode else q_pt_pow2(mod.weight, 8)))
        # quantize ACTIVATIONS at every ReLU / Upsample output (the scheme under test)
        def mkhook(f):
            def h(mod, inp, out): return f(out)
            return h
        for mod in m.modules():
            if isinstance(mod, (nn.ReLU, nn.Upsample)):
                hooks.append(mod.register_forward_hook(mkhook(qf)))
    with torch.no_grad():
        x = qf(vt0) if qf is not None else vt0
        conv = m(x)[0].numpy()
    for h in hooks: h.remove()
    print("%-12s conv cos %.4f  (range %.1f..%.1f)" % (mode, cos(conv, ref), conv.min(), conv.max()), flush=True)
print("SIM_DONE", flush=True)
