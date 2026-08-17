#!/usr/bin/env python3
"""F1: the predicter head on the DPU. head.xmodel = DPU conv1 | CPU softplus | DPU conv2.
Times each part and checks numerics vs the FP32 numpy head on the real conv features.
  sudo ... python3 e7_head_dpu.py [runs=20]
"""
import os, sys, time
os.environ["XILINX_XRT"] = "/usr"
import numpy as np, xir, vart
from pynq_dpu import DpuOverlay

RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
ov = DpuOverlay("dpu.bit")
g = xir.Graph.deserialize("/home/ubuntu/bev/head.xmodel")
subs = g.get_root_subgraph().toposort_child_subgraph()
dev = lambda s: s.get_attr("device") if s.has_attr("device") else "?"
print("subgraphs:", [(i, dev(s), s.get_name()[:40]) for i, s in enumerate(subs)], flush=True)
runners = {i: vart.Runner.create_runner(s, "run") for i, s in enumerate(subs) if dev(s) == "DPU"}

conv = np.load("/home/ubuntu/bev_allpl/allpl_convonly.npy").astype(np.float32)
if conv.ndim == 3: conv = conv[None]
z = np.load("/home/ubuntu/bev/predicter_head.npz")
W1, b1 = z["0.weight"].astype(np.float32), z["0.bias"].astype(np.float32)
W2, b2 = z["2.weight"].astype(np.float32), z["2.bias"].astype(np.float32)
x2 = conv[0].transpose(1, 2, 0).reshape(-1, 256)
h = x2 @ W1.T + b1
h = np.log1p(np.exp(-np.abs(h))) + np.maximum(h, 0.0)
ref_occ = (h @ W2.T + b2).reshape(200, 200, 16, 18).argmax(-1).astype(np.uint8)

def t_fp(t): return t.get_attr("fix_point") if t.has_attr("fix_point") else None
LUT = [None]
def build_lut(fp_in_sp, fp_out_sp):
    v = np.arange(-128, 128, dtype=np.float32) * (2.0 ** (-fp_in_sp))
    sp = np.log1p(np.exp(-np.abs(v))) + np.maximum(v, 0.0)
    return np.clip(np.round(sp * (2.0 ** fp_out_sp)), -128, 127).astype(np.int8)

def run_dpu(i, xin_int8):
    r = runners[i]; it, ot = r.get_input_tensors()[0], r.get_output_tensors()[0]
    ins = [np.ascontiguousarray(xin_int8.reshape([int(d) for d in it.dims]), np.int8)]
    outs = [np.empty([int(d) for d in ot.dims], np.int8)]
    jid = r.execute_async(ins, outs); r.wait(jid)
    return outs[0], t_fp(ot)

dpu_ids = sorted(runners)
in_t = list(subs[0].get_output_tensors())[0]
fp_in = t_fp(in_t) if t_fp(in_t) is not None else 4
xin = conv[0].transpose(1, 2, 0)[None]        # NHWC (1,200,200,256)
q_in = np.clip(np.round(xin * (2.0 ** fp_in)), -128, 127).astype(np.int8)

ts = {"c1": [], "sp": [], "c2": []}
occ_dpu = None
for it_ in range(RUNS):
    t0 = time.perf_counter()
    o1, fp1 = run_dpu(dpu_ids[0], q_in)
    t1 = time.perf_counter()
    it2 = runners[dpu_ids[1]].get_input_tensors()[0]
    fp2i = t_fp(it2) if t_fp(it2) is not None else 4
    if LUT[0] is None:
        LUT[0] = build_lut(fp1, fp2i)
    q2 = LUT[0][(o1.astype(np.int16) + 128)]
    t2 = time.perf_counter()
    o2, fp2 = run_dpu(dpu_ids[1], q2)
    t3 = time.perf_counter()
    ts["c1"].append(1e3*(t1-t0)); ts["sp"].append(1e3*(t2-t1)); ts["c2"].append(1e3*(t3-t2))
    if it_ == 0:
        logits = o2.astype(np.float32) * (2.0 ** (-fp2))
        occ_dpu = logits[0].reshape(200, 200, 16, 18).argmax(-1).astype(np.uint8)

for k in ("c1", "sp", "c2"):
    a = np.array(ts[k]); print("%s: p50 %.1f max %.1f ms" % (k, np.percentile(a, 50), a.max()), flush=True)
tot = np.array(ts["c1"]) + np.array(ts["sp"]) + np.array(ts["c2"])
print("HEAD TOTAL: p50 %.1f max %.1f ms  (numpy fp32 reference: ~6700 ms)" % (np.percentile(tot, 50), tot.max()), flush=True)
agree = float((occ_dpu == ref_occ).mean())
print("occ argmax agreement vs FP32 head: %.4f" % agree, flush=True)
