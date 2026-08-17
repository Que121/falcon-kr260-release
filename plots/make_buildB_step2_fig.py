#!/usr/bin/env python3
"""Build-B step 2 viz: INT8 vs FP32 of the FULL image path (backbone+FPN+depth_net) on KR260 DPU.

The depth_net output (1,152,16,44) = 88 depth-logits + 64 context/feat. What actually matters
downstream is (a) softmax(depth) -- the per-pixel depth distribution that splats feat into BEV,
and (b) the feat itself. Raw-logit cosine (0.88) understates fidelity because softmax is invariant
to logit shift/scale; this figure shows the post-softmax depth + feat agreement.
"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "experiments/results/buildB"
fp = np.load(os.path.join(R, "step2_ref_fp32_depthnet.npy"))[0]   # (152,16,44)
q8 = np.load(os.path.join(R, "step2_ref_int8_depthnet.npy"))[0]
dlog_fp, feat_fp = fp[:88], fp[88:]
dlog_q,  feat_q  = q8[:88], q8[88:]

def softmax(x, ax=0):
    e = np.exp(x - x.max(ax, keepdims=True)); return e / e.sum(ax, keepdims=True)
d_fp = softmax(dlog_fp, 0)   # (88,16,44) depth prob
d_q  = softmax(dlog_q, 0)
def cos(a, b): return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

cos_logit = cos(dlog_fp, dlog_q)
cos_soft  = cos(d_fp, d_q)
cos_feat  = cos(feat_fp, feat_q)
argmax_match = float((d_fp.argmax(0) == d_q.argmax(0)).mean())
exp_fp = (d_fp * np.arange(88)[:, None, None]).sum(0)   # expected depth-bin
exp_q  = (d_q  * np.arange(88)[:, None, None]).sum(0)
exp_mae = float(np.abs(exp_fp - exp_q).mean())

metrics = dict(cos_depth_logits=cos_logit, cos_depth_softmax=cos_soft, cos_feat=cos_feat,
               depth_argmax_match=argmax_match, exp_depthbin_mae=exp_mae,
               input_fixpoint=5, output_fixpoint=3)
json.dump(metrics, open(os.path.join(R, "step2_metrics.json"), "w"), indent=2)
print(metrics)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
# (a) expected-depth-bin map FP32 vs INT8 (one camera-feature grid), side by side as scatter
ax[0].scatter(exp_fp.ravel(), exp_q.ravel(), s=6, alpha=0.4, color="#3b6ea5")
lim = [0, 88]; ax[0].plot(lim, lim, "k--", lw=0.8)
ax[0].set_xlabel("FP32 expected depth-bin"); ax[0].set_ylabel("INT8 expected depth-bin")
ax[0].set_title("(a) per-pixel expected depth\nMAE=%.2f bin, argmax match=%.1f%%" % (exp_mae, 100*argmax_match))
# (b) feat scatter
i = np.random.default_rng(0).choice(feat_fp.size, 4000, replace=False)
ax[1].scatter(feat_fp.ravel()[i], feat_q.ravel()[i], s=5, alpha=0.3, color="#9a3b3b")
l2 = [feat_fp.min(), feat_fp.max()]; ax[1].plot(l2, l2, "k--", lw=0.8)
ax[1].set_xlabel("FP32 context/feat"); ax[1].set_ylabel("INT8 context/feat")
ax[1].set_title("(b) context feature\ncos=%.3f" % cos_feat)
# (c) cosine bars
labels = ["depth\nlogits", "depth\nsoftmax", "context\nfeat"]
vals = [cos_logit, cos_soft, cos_feat]
b = ax[2].bar(labels, vals, color=["#bbb", "#3b6ea5", "#9a3b3b"])
ax[2].set_ylim(0.8, 1.0); ax[2].set_ylabel("cosine INT8 vs FP32")
ax[2].set_title("(c) fidelity by signal")
for r, v in zip(b, vals): ax[2].text(r.get_x()+r.get_width()/2, v+0.003, "%.3f" % v, ha="center", fontsize=9)
fig.suptitle("Build-B Step 2 — INT8 image path (ResNet50+FPN+depth_net) on KR260 B4096 DPU", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("png", "pdf"):
    fig.savefig(os.path.join("docs/figs/buildB", "step2_image_int8." + ext), dpi=140, bbox_inches="tight")
print("saved docs/figs/buildB/step2_image_int8.{png,pdf}")
