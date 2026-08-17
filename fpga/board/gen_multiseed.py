"""Clamp-32 PTQ retention over M calibration seeds (robustness). Vitis-AI container, /work mount."""
import sys, numpy as np, torch
sys.path.insert(0, "/work")
from bev_stage import BEVStage
from pytorch_nndct.apis import torch_quantizer

sd = torch.load("/work/bev_stage_sd.pth", map_location="cpu")
def make():
    m = BEVStage(conv_only=True, clamp_c=32.0)
    m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("predicter.")}, strict=False)
    return m.eval()

calib = np.load("/work/calib_bev.npy")
eval_vt = np.load("/work/eval_vt.npy").astype("float32")
fp = np.load("/work/eval_fp_argmax.npy").astype("int64")
predm = BEVStage(conv_only=False); predm.load_state_dict(sd); pred = predm.predicter.eval()
def occ(c):
    x = torch.from_numpy(c[None].astype("float32")).permute(0, 3, 2, 1)
    with torch.no_grad():
        o = pred(x)
    return o.view(1, 200, 200, 16, 18).argmax(-1).numpy()[0]
VRU = {2, 6, 7}

res = []
for seed in range(3):
    rng = np.random.RandomState(seed)
    cal = calib[rng.permutation(len(calib))[:32]].astype("float32")
    qdir = f"/work/qseed{seed}"
    q = torch_quantizer("calib", make(), (torch.randn(1, 64, 200, 200),), output_dir=qdir)
    qm = q.quant_model
    with torch.no_grad():
        for i in range(0, len(cal), 4):
            qm(torch.from_numpy(cal[i:i + 4]))
    q.export_quant_config()
    q = torch_quantizer("test", make(), (torch.randn(1, 64, 200, 200),), output_dir=qdir)
    qm = q.quant_model
    ov, vr, ge = [], [], []
    with torch.no_grad():
        for k in range(len(eval_vt)):
            c = qm(torch.from_numpy(eval_vt[k:k + 1]))[0].numpy()
            so, fo = occ(c).ravel(), fp[k].ravel()
            vm = np.isin(fo, list(VRU)); gr = fo != 17
            ov.append((so == fo).mean()); vr.append((so[vm] == fo[vm]).mean()); ge.append((so[gr] != 17).mean())
    res.append((np.mean(ov), np.mean(vr), np.mean(ge)))
    print(f"seed {seed}: overall {np.mean(ov):.3f} VRU {np.mean(vr):.3f} geom {np.mean(ge):.3f}", flush=True)

res = np.array(res)
print(f"SEEDS(M=3) overall {res[:,0].mean():.3f}+/-{res[:,0].std():.3f}  "
      f"VRU {res[:,1].mean():.3f}+/-{res[:,1].std():.3f}  geom {res[:,2].mean():.3f}+/-{res[:,2].std():.3f}")
