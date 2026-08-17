#!/usr/bin/env python3
"""Quick: does the board image_split feat exceed the gather's fp_feat=2 range (+-31.75)? And depth range?
No DPU. Reads the dumped board featdepth npz(s)."""
import sys, glob, numpy as np
paths = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob("/home/ubuntu/buildB/featdepth_*.npz"))
if not paths: paths = sorted(glob.glob("/home/ubuntu/buildB/step3_board_featdepth_*.npz"))
for p in paths[:4]:
    d = np.load(p); ks = list(d.keys())
    feat = d["feat"].astype(np.float32) if "feat" in d else None
    dep = d["depth"].astype(np.float32) if "depth" in d else None
    print(p, "keys=", ks)
    if feat is not None:
        a = np.abs(feat)
        print("  feat: shape", feat.shape, "absmax %.3f p99 %.3f p50 %.3f | clip@31.75 frac %.4f"
              % (a.max(), np.percentile(a, 99), np.percentile(a, 50), float((a > 31.75).mean())))
    if dep is not None:
        print("  depth: shape", dep.shape, "min %.4f max %.4f (Q0.7 step=1/128=0.0078, range [0,2))" % (dep.min(), dep.max()))
print("CHECK_DONE")
