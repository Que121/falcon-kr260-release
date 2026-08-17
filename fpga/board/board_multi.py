#!/usr/bin/env python3
"""Run the hybrid BEV walker over all 16 eval frames -> board16_convonly.npy (for error bars)."""
import sys, time
import numpy as np, xir
sys.path.insert(0, "/home/ubuntu")
import board_bev_walk as W
from pynq_dpu import DpuOverlay

ov = DpuOverlay("dpu.bit")
print("DPU overlay loaded")
g = xir.Graph.deserialize(W.XM)
subs = g.get_root_subgraph().toposort_child_subgraph()

def walk_one(vt_nhwc):
    vmap = {}
    in_t = list(subs[0].get_output_tensors())[0]
    fp = W.t_fp(in_t)
    vmap[in_t.name] = W.q_dq(vt_nhwc, fp) if fp is not None else vt_nhwc
    for sg in subs[1:]:
        d = W.dev(sg)
        if d == "DPU":
            W.run_dpu_subgraph(sg, vmap)
        elif d == "CPU":
            W.run_cpu_subgraph(sg, vmap)
    out_t = list(subs[-1].get_output_tensors())[0]
    return vmap[out_t.name].transpose(0, 3, 1, 2)        # NHWC -> NCHW (1,256,200,200)

vt = np.load("/home/ubuntu/bev/eval_vt.npy").astype(np.float32)   # (16,64,200,200)
N = vt.shape[0]
board = np.empty((N, 256, 200, 200), np.float32)
t0 = time.time()
for k in range(N):
    vt_nhwc = np.ascontiguousarray(vt[k:k + 1].transpose(0, 2, 3, 1))
    board[k] = walk_one(vt_nhwc)[0]
    print(f"frame {k+1}/{N} done ({time.time()-t0:.0f}s)", flush=True)
np.save("/home/ubuntu/bev/board16_convonly.npy", board)
print("saved board16_convonly", board.shape)
