"""Quantize+export a BEVStage variant with a chosen activation, to test graph fragmentation.
ACT env: hardtanh32 (baseline) | relu | relu6 . Runs in Vitis-AI container, /work = quant dir."""
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
        m = BEVStage(conv_only=True, clamp_c=None)          # keep original nn.ReLU
        if ACT == "relu6":
            for parent in (m.backbone, m.neck):
                for name, mod in list(parent.named_modules()):
                    if isinstance(mod, nn.ReLU):
                        _set_submodule(parent, name, nn.ReLU6(inplace=True))
            m.final_relu = nn.ReLU6(inplace=True)
    m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("predicter.")}, strict=False)
    return m.eval()

calib = np.load("/work/calib_bev.npy")[:16].astype("float32")
dummy = torch.randn(1, 64, 200, 200)
qdir = f"/work/frag_{ACT}"
q = torch_quantizer("calib", make(), (dummy,), output_dir=qdir)
qm = q.quant_model
with torch.no_grad():
    for i in range(0, len(calib), 4):
        qm(torch.from_numpy(calib[i:i + 4]))
q.export_quant_config()
q = torch_quantizer("test", make(), (dummy,), output_dir=qdir)
with torch.no_grad():
    q.quant_model(torch.from_numpy(calib[:1]))
q.export_xmodel(deploy_check=False, output_dir=qdir)
print("EXPORTED", ACT, "->", qdir)
