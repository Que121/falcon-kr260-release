import sys, numpy as np, torch
sys.path.insert(0, "/work")
from bev_stage import BEVStage
from pytorch_nndct.apis import torch_quantizer
sd = torch.load("/work/bev_stage_sd.pth", map_location="cpu")
def make():
    m = BEVStage(conv_only=True, clamp_c=None)
    m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("predicter.")}, strict=False)
    return m.eval()
q = torch_quantizer("test", make(), (torch.randn(1, 64, 200, 200),), output_dir="/work/frag_reluc")
qm = q.quant_model
x = torch.from_numpy(np.load("/work/bev_test_input.npy").astype("float32"))
with torch.no_grad():
    y = qm(x).numpy()
np.save("/work/bev_reluc_sim_convonly.npy", y)
print("reluc sim convonly", y.shape, float(y.min()), float(y.max()))
