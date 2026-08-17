"""Accuracy of a re-quantized BEVStage variant (sim-INT8) vs FP32 occupancy. ACT env selects variant."""
import os, sys, numpy as np, torch
import torch.nn as nn
sys.path.insert(0, "/work")
from bev_stage import BEVStage, _set_submodule
from pytorch_nndct.apis import torch_quantizer

ACT = os.environ["ACT"]
sd = torch.load("/work/bev_stage_sd.pth", map_location="cpu")
def make():
    if ACT == "hardtanh32":
        m = BEVStage(conv_only=True, clamp_c=32.0)
    else:
        m = BEVStage(conv_only=True, clamp_c=None)
        if ACT == "relu6":
            for parent in (m.backbone, m.neck):
                for n, mod in list(parent.named_modules()):
                    if isinstance(mod, nn.ReLU):
                        _set_submodule(parent, n, nn.ReLU6(inplace=True))
            m.final_relu = nn.ReLU6(inplace=True)
    m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("predicter.")}, strict=False)
    return m.eval()

dummy = torch.randn(1, 64, 200, 200)
q = torch_quantizer("test", make(), (dummy,), output_dir=f"/work/frag_{ACT}")
qm = q.quant_model
x = torch.from_numpy(np.load("/work/bev_test_input.npy").astype("float32"))
with torch.no_grad():
    conv = qm(x).numpy()

pred = BEVStage(conv_only=False); pred.load_state_dict(sd); pred = pred.predicter.eval()
def occ(c):
    t = torch.from_numpy(c.astype("float32")).permute(0, 3, 2, 1)
    with torch.no_grad():
        o = pred(t)
    return o.view(1, 200, 200, 16, 18).argmax(-1).numpy().ravel()

fp = np.load("/work/bev_clampref_fp32.npy").astype("float32")
ow, of = occ(conv), occ(fp)
VRU = {2, 6, 7}
vm = np.isin(of, list(VRU)); gr = of != 17
print(f"{ACT}: conv max {conv.max():.1f} | occ overall {(ow==of).mean():.4f} "
      f"VRU {(ow[vm]==of[vm]).mean():.4f} geom {(ow[gr]!=17).mean():.4f}")
