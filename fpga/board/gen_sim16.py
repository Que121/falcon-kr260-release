"""INT8-simulator conv_only for all 16 eval frames (clamp-32 model), in the Vitis-AI container."""
import sys, numpy as np, torch
sys.path.insert(0, "/work")
from bev_stage import BEVStage
from pytorch_nndct.apis import torch_quantizer

sd = torch.load("/work/bev_stage_sd.pth", map_location="cpu")
def make():
    m = BEVStage(conv_only=True, clamp_c=32.0)
    m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("predicter.")}, strict=False)
    return m.eval()
q = torch_quantizer("test", make(), (torch.randn(1, 64, 200, 200),), output_dir="/work/quantize_result_clamp")
qm = q.quant_model
vt = np.load("/work/eval_vt.npy").astype("float32")          # (16,64,200,200)
out = np.empty((vt.shape[0], 256, 200, 200), np.float32)
with torch.no_grad():
    for i in range(vt.shape[0]):
        out[i] = qm(torch.from_numpy(vt[i:i+1]))[0].numpy()
np.save("/work/eval_sim16_convonly.npy", out)
print("sim16 convonly", out.shape, float(out.min()), float(out.max()))
