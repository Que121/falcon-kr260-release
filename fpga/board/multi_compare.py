"""Per-frame on-board fidelity over 16 eval frames -> mean +/- std. Run on Pro6000 (ANONPROJ_310)."""
import os, numpy as np, torch
D = os.path.expanduser("~/occfpga_quant_bevclamp")
sd = torch.load(os.path.join(D, "bev_stage_sd.pth"), map_location="cpu")
import torch.nn as nn
pred = nn.Sequential(nn.Linear(256, 512), nn.Softplus(), nn.Linear(512, 16 * 18))
pred.load_state_dict({k[len("predicter."):]: v for k, v in sd.items() if k.startswith("predicter.")})
pred.eval()

def occ(c):                                   # c: (256,200,200) -> (200,200,16) argmax
    x = torch.from_numpy(c[None].astype("float32")).permute(0, 3, 2, 1)
    with torch.no_grad():
        o = pred(x)
    return o.view(1, 200, 200, 16, 18).argmax(-1).numpy()[0]

def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

board = np.load(os.path.join(D, "board16_convonly.npy")).astype(np.float32)   # (16,256,200,200)
sim = np.load(os.path.join(D, "eval_sim16_convonly.npy")).astype(np.float32)
fp = np.load(os.path.join(D, "eval_fp_argmax.npy")).astype(np.int64)           # (16,200,200,16)
VRU = {2, 6, 7}
N = board.shape[0]

def agree(a, b):
    vm = np.isin(b, list(VRU)); gr = b != 17
    return ((a == b).mean(),
            (a[vm] == b[vm]).mean() if vm.sum() else np.nan,
            (a[gr] != 17).mean() if gr.sum() else np.nan)

cosines, bs, bf, sf = [], [], [], []
for k in range(N):
    bo, so, fo = occ(board[k]).ravel(), occ(sim[k]).ravel(), fp[k].ravel()
    cosines.append(cos(board[k], sim[k]))
    bs.append(agree(bo, so)); bf.append(agree(bo, fo)); sf.append(agree(so, fo))

cosines = np.array(cosines); bs, bf, sf = map(np.array, (bs, bf, sf))
def ms(x): return f"{np.nanmean(x):.3f} +/- {np.nanstd(x):.3f}"
print(f"frames N={N}")
print(f"board-vs-sim conv cosine : {ms(cosines)}   (deployment fidelity)")
print(f"board-vs-sim   occ overall/VRU/geom : {ms(bs[:,0])} / {ms(bs[:,1])} / {ms(bs[:,2])}")
print(f"board-vs-FP32  occ overall/VRU/geom : {ms(bf[:,0])} / {ms(bf[:,1])} / {ms(bf[:,2])}")
print(f"sim-vs-FP32    occ overall/VRU/geom : {ms(sf[:,0])} / {ms(sf[:,1])} / {ms(sf[:,2])}")
