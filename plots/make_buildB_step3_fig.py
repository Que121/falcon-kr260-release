#!/usr/bin/env python3
"""Build-B step 3 viz: the FULL on-board front half image(DPU)->depth_net->gather(IP)->vt_out.

(a) BEV energy map of the on-board vt_out vs FP32 reference (per-cell L2 over 64 ch).
(b) the fidelity chain: where the cosine goes -- gather-only (step1, FP32 in) 0.985,
    -> full front-half on silicon (step3, INT8 DPU in) 0.915, attributed to the seam cosines.
"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "experiments/results/buildB"
vt_board = np.load(os.path.join(R, "step3_image_to_vt.npy")).astype(np.float32)      # (64,200,200)
fr = np.load(os.path.join(R, "frame_0000.npz"))
vt_ref = fr["vt_out"].astype(np.float32)                                              # (64,200,200)
fd = np.load(os.path.join(R, "step3_board_featdepth_0000.npz"))
feat_b, depth_b = fd["feat"].astype(np.float32), fd["depth"].astype(np.float32)
feat_f, depth_f = fr["feat"].astype(np.float32), fr["depth"].astype(np.float32)

def cos(a, b): return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
eb = np.linalg.norm(vt_board, axis=0); ef = np.linalg.norm(vt_ref, axis=0)
metrics = dict(
    image_to_vt_cos=cos(vt_board, vt_ref),
    seam_feat_cos=cos(feat_b, feat_f),
    seam_depth_softmax_cos=cos(depth_b, depth_f),
    seam_depth_argmax_match=float((depth_b.argmax(1) == depth_f.argmax(1)).mean()),
    gather_only_cos_step1=0.985, fp_feat=2, fp_vt=0)
json.dump(metrics, open(os.path.join(R, "step3_metrics.json"), "w"), indent=2)
print(metrics)

fig = plt.figure(figsize=(13, 4.2))
g = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25])
a0 = fig.add_subplot(g[0]); a1 = fig.add_subplot(g[1]); a2 = fig.add_subplot(g[2])
vmax = float(max(eb.max(), ef.max()))
a0.imshow(ef, cmap="magma", vmax=vmax, origin="lower"); a0.set_title("FP32 vt_out\n|feature| per BEV cell"); a0.axis("off")
a1.imshow(eb, cmap="magma", vmax=vmax, origin="lower")
a1.set_title("on-board image->DPU->gather\ncos=%.3f vs FP32" % metrics["image_to_vt_cos"]); a1.axis("off")
# chain bars
labels = ["gather only\n(FP32 in)\nstep1", "image DPU\nfeat seam", "image DPU\ndepth seam", "FULL front-half\non silicon\nstep3"]
vals = [0.985, metrics["seam_feat_cos"], metrics["seam_depth_softmax_cos"], metrics["image_to_vt_cos"]]
cols = ["#3b6ea5", "#9a3b3b", "#c46a1f", "#2e7d32"]
b = a2.bar(labels, vals, color=cols); a2.set_ylim(0.8, 1.0); a2.set_ylabel("cosine vs FP32")
a2.set_title("(c) on-board fidelity chain")
for r, v in zip(b, vals): a2.text(r.get_x()+r.get_width()/2, v+0.003, "%.3f" % v, ha="center", fontsize=9)
a2.tick_params(axis="x", labelsize=7.5)
fig.suptitle("Build-B Step 3 — full image->vt_out on KR260 (DPU image xmodel + HLS gather IP), PTQ", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
for ext in ("png", "pdf"):
    fig.savefig(os.path.join("docs/figs/buildB", "step3_image_to_vt." + ext), dpi=140, bbox_inches="tight")
print("saved docs/figs/buildB/step3_image_to_vt.{png,pdf}")
