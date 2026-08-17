import os, sys, numpy as np, torch
sys.path.insert(0, "/work")
from bev_stage import BEVStage
from pytorch_nndct.apis import torch_quantizer
SD = sys.argv[1]                                  # e.g. bev_stage_qatv4_sd.pth
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
m = BEVStage(conv_only=True)
sd = torch.load(os.path.join("/work", SD), map_location="cpu")
m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("predicter.")}, strict=False)
m.eval()
calib = np.load("/work/calib_bev.npy")[:NS].astype(np.float32)
ref = np.load("/work/convonly_0000.npy").astype(np.float32)
vt0 = torch.from_numpy(np.load("/work/frame_0000.npz")["vt_out"].astype(np.float32))[None]
qd = "/work/qat_eval_" + SD.replace(".pth", "")
q = torch_quantizer("calib", m, (torch.randn(1, 64, 200, 200),), output_dir=qd)
qm = q.quant_model
with torch.no_grad():
    for i in range(0, NS, 2): qm(torch.from_numpy(calib[i:i + 2]))
q.export_quant_config()
q = torch_quantizer("test", m, (torch.randn(1, 64, 200, 200),), output_dir=qd)
qm = q.quant_model
with torch.no_grad(): conv = qm(vt0)[0].numpy()
def cos(a, b): return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
print("QAT[%s] DPU-INT8 conv vs FP32: cos %.4f (PTQ/cfg was 0.59)" % (SD, cos(conv, ref)), flush=True)
