"""Deployment fidelity: INT8 reference (off-board simulator) vs on-board INT8 (KR260), with a
light-blue error panel (ANONPROJ style). The error here is the FPGA deployment/hardware error only,
NOT quantisation (that is FP32->INT8, an accepted cost). Run on Pro6000 (ANONPROJ_310)."""
import os, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch.nn as nn
D = os.path.expanduser("~/occfpga_quant_bevclamp")

OCC_COLORS = np.array([
    [0,0,0],[255,120,50],[255,192,203],[255,255,0],[0,150,245],[0,255,255],[255,127,0],
    [255,0,0],[255,240,150],[135,60,0],[160,32,240],[255,0,255],[139,137,137],[75,0,75],
    [150,240,80],[230,230,250],[0,175,0],[255,255,255]], np.float32) / 255.0
NAMES = ['others','barrier','bicycle','bus','car','constr.veh','motorcycle','pedestrian','traffic_cone',
         'trailer','truck','drive_surf','other_flat','sidewalk','terrain','manmade','vegetation','free']
VRU = {2, 6, 7}
LIGHTBLUE = np.array([0.40, 0.68, 0.92])      # light blue for the deployment-error voxels
AGREE_GREY = np.array([0.91, 0.91, 0.93])     # faint grey for agreeing-occupied cells

sd = torch.load(D + "/bev_stage_sd.pth", map_location="cpu")
pred = nn.Sequential(nn.Linear(256, 512), nn.Softplus(), nn.Linear(512, 16 * 18))
pred.load_state_dict({k[len("predicter."):]: v for k, v in sd.items() if k.startswith("predicter.")}); pred.eval()
def occ(c):
    x = torch.from_numpy(c[None].astype("float32")).permute(0, 3, 2, 1)
    with torch.no_grad():
        o = pred(x)
    return o.view(1, 200, 200, 16, 18).argmax(-1).numpy()[0]

board = np.load(D + "/board16_convonly.npy").astype("float32")     # on-board INT8
sim = np.load(D + "/eval_sim16_convonly.npy").astype("float32")     # off-board INT8 reference
fpa = np.load(D + "/eval_fp_argmax.npy").astype("int64")            # only to pick VRU-rich frames

def bevtop(g, zfree=17):
    H, W, Z = g.shape; top = np.full((H, W), -1, np.int64)
    for z in range(Z):
        m = (g[:, :, z] != zfree) & (g[:, :, z] != 0); top[m] = g[:, :, z][m]
    img = np.ones((H, W, 3), np.float32); mm = top >= 0; img[mm] = OCC_COLORS[top[mm]]
    return img, top
def occ_mask(g): return (g != 17).any(-1)

cand = sorted(((int(np.isin(fpa[k], list(VRU)).sum()), k) for k in range(16)), reverse=True)
pick = [k for _, k in cand[:2]]
HEADERS = ["INT8 reference", "on-board (KR260)", "error"]
fig, axes = plt.subplots(2, 3, figsize=(3.35, 2.55))     # single column
plt.subplots_adjust(wspace=0.06, hspace=0.10, left=0.06, right=0.99, top=0.90, bottom=0.20)
for row, k in enumerate(pick):
    so, bo = occ(sim[k]), occ(board[k])
    img_s, top_s = bevtop(so); img_b, _ = bevtop(bo)
    ms, mb = occ_mask(so), occ_mask(bo)
    vox_err = float((so != bo).mean()) * 100.0
    emap = np.ones((200, 200, 3), np.float32)
    emap[ms & mb] = AGREE_GREY
    emap[ms != mb] = LIGHTBLUE
    for col, (im, top, hdr) in enumerate([(img_s, top_s, HEADERS[0]),
                                          (img_b, None, HEADERS[1]),
                                          (emap, None, f"{HEADERS[2]} {vox_err:.1f}%")]):
        ax = axes[row, col]
        ax.imshow(np.transpose(im, (1, 0, 2)), origin="lower")
        if top is not None:
            ys, xs = np.nonzero(np.isin(top, list(VRU)).T)
            ax.scatter(xs, ys, s=8, facecolors="none", edgecolors="red", linewidths=0.7)
        if row == 0:
            ax.set_title(hdr, fontsize=7)
        if col == 0:
            ax.set_ylabel(f"frame {k}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("0.8")
present = sorted(set(np.unique(np.concatenate([occ(sim[k]).ravel() for k in pick]))) - {0, 17})
handles = [Patch(facecolor=OCC_COLORS[i], edgecolor='gray', label=NAMES[i]) for i in present]
handles += [Patch(facecolor=LIGHTBLUE, edgecolor='gray', label='mismatch')]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=5.6,
           frameon=False, bbox_to_anchor=(0.5, -0.02), handlelength=1.1, columnspacing=0.9)
plt.savefig(D + "/occ_error.png", dpi=150, bbox_inches="tight")
plt.savefig(D + "/occ_error.pdf", bbox_inches="tight")
print("rendered occ_error (board vs sim) frames", pick)
