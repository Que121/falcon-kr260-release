import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/home/ANON/03_OccFPGA_Work/occfpga_quant_bev")
from bev_stage import BEVStage
sd = torch.load("/home/ANON/03_OccFPGA_Work/occfpga_quant_bev/bev_stage_sd.pth", map_location="cpu")
vt0 = torch.from_numpy(np.load("/home/ANON/frame_0000.npz")["vt_out"].astype(np.float32))[None]
ref = np.load("/home/ANON/convonly_0000.npy").astype(np.float32)
def cos(a, b): return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def replace(m, kind, C):
    for n, c in m.named_children():
        if isinstance(c, nn.ReLU):
            setattr(m, n, nn.ReLU6(inplace=False) if kind == "relu6" else nn.Hardtanh(0.0, C, inplace=False))
        else:
            replace(c, kind, C)

for tag, kind, C in [("relu", None, 0), ("relu6", "relu6", 6), ("hardtanh8", "ht", 8.0), ("hardtanh6", "ht", 6.0)]:
    m = BEVStage(conv_only=True)
    m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("predicter.")}, strict=False)
    if kind: replace(m, kind, C)
    m.eval()
    with torch.no_grad():
        conv = m(vt0)[0].numpy()
    print("{:10s} float conv max {:.2f}  cos-vs-unclamped {:.4f}".format(tag, float(conv.max()), cos(conv, ref)))
