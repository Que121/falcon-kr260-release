import os, numpy as np, torch, sys
sys.path.insert(0, "/work")
from bev_stage import BEVStage
from pytorch_nndct.apis import torch_quantizer
sd = torch.load("/work/bev_stage_sd.pth", map_location="cpu")
m = BEVStage(conv_only=True)
m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("predicter.")}, strict=False)
m.eval()
qdir = sys.argv[1]
q = torch_quantizer("test", m, (torch.randn(1, 64, 200, 200),), output_dir=qdir)
qm = q.quant_model
d = np.load("/work/frame_0000.npz"); vt = torch.from_numpy(d["vt_out"].astype(np.float32))[None]
with torch.no_grad():
    conv = qm(vt)[0].numpy()
np.save("/work/host_int8_conv_0000.npy", conv.astype(np.float16))
ref = np.load("/work/convonly_0000.npy").astype(np.float32)
def cos(a, b): return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
print("HOST INT8-sim conv vs FP32: cos {:.4f} range[{:.1f},{:.1f}]".format(cos(conv, ref), float(conv.min()), float(conv.max())))
