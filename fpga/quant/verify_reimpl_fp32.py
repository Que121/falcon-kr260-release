#!/usr/bin/env python
"""DECISIVE A-vs-B test (board-independent, local): does the reimpl ImageFull FP32 match the REAL FlashOcc
FP32 feat/depth? frame_0000.npz holds the real-FlashOcc feat (6,16,44,64) + depth (6,88,16,44, post-softmax)
+ the exact normalized img (6,3,256,704). Run reimpl on that img, compare. >=0.999 => reimpl faithful
(gap is 100% Vitis deploy = Hyp B); <0.98 => reimpl FP bug (Hyp A) that ALSO poisoned calib + QAT teacher."""
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "fpga/quant")
from image_full_model import ImageFull

d = np.load("traces/buildB/frame_0000.npz")
img = torch.from_numpy(d["img"].astype(np.float32))          # (6,3,256,704)
ref_feat = d["feat"].astype(np.float32)                      # (6,16,44,64) NHWC
ref_depth = d["depth"].astype(np.float32)                    # (6,88,16,44) NDHW
print("ref_depth sum over bins (should be ~1 if post-softmax): mean=%.4f min=%.4f max=%.4f"
      % (ref_depth.sum(1).mean(), ref_depth.sum(1).min(), ref_depth.sum(1).max()))

m = ImageFull()
m.load_state_dict(torch.load("traces/buildB/image_full_sd.pth", map_location="cpu"))
m.eval()
with torch.no_grad():
    y = m(img)                                               # (6,152,16,44)
    re_depth = F.softmax(y[:, :88], dim=1).numpy()           # (6,88,16,44)
    re_feat = y[:, 88:].permute(0, 2, 3, 1).numpy()          # (6,16,44,64) NHWC
    re_logits = y[:, :88].numpy()

def cos(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

print("==== reimpl-FP32 vs REAL-FlashOcc-FP32 (frame 0, 6 cams) ====")
print("  depth-softmax cos : %.5f" % cos(re_depth, ref_depth))
print("  feat cos          : %.5f" % cos(re_feat, ref_feat))
print("  depth-argmax match: %.4f" % float((re_depth.argmax(1) == ref_depth.argmax(1)).mean()))
print("  feat absmax reimpl %.3f / ref %.3f | depth peak-conf reimpl %.3f / ref %.3f"
      % (np.abs(re_feat).max(), np.abs(ref_feat).max(), re_depth.max(1).mean(), ref_depth.max(1).mean()))
print("VERIFY_DONE")
