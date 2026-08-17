#!/usr/bin/env python3
"""Build-B step3 phase A: run the FULL image xmodel (backbone+FPN+depth_net) on the KR260 DPU.

Input: a buildB dump frame_%04d.npz (img (6,3,256,704) f16 normalized).
DPU image xmodel: per-cam (256,704,3) int8 @fix5 -> (16,44,152) int8 @fix3.
Split 152 = 88 depth-logits + 64 context/feat; softmax(depth); emit feat(6,16,44,64)+depth(6,88,16,44)
exactly in the gather's expected layout. Also reports the seam cosine vs the dump's FP32 feat/depth.

  python3 board_image_dpu.py <frame.npz> <out_featdepth.npz>
"""
import os, sys, time
os.environ["XILINX_XRT"] = "/usr"
import numpy as np, xir, vart
from pynq_dpu import DpuOverlay

FIN  = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/buildB/frame_0000.npz"
FOUT = sys.argv[2] if len(sys.argv) > 2 else "/home/ubuntu/buildB/featdepth_0000.npz"
XM = sys.argv[3] if len(sys.argv) > 3 else "/home/ubuntu/flashocc/flashocc_r50_image.xmodel"

def softmax(x, ax):
    e = np.exp(x - x.max(ax, keepdims=True)); return e / e.sum(ax, keepdims=True)

def main():
    ov = DpuOverlay("dpu.bit")
    g = xir.Graph.deserialize(XM)
    subs = g.get_root_subgraph().toposort_child_subgraph()
    dpu_sub = [s for s in subs if s.has_attr("device") and s.get_attr("device") == "DPU"][0]
    r = vart.Runner.create_runner(dpu_sub, "run")
    it = r.get_input_tensors()[0]; ot = r.get_output_tensors()[0]
    idims = [int(x) for x in it.dims]; odims = [int(x) for x in ot.dims]
    fpi = it.get_attr("fix_point"); fpo = ot.get_attr("fix_point")
    print("DPU in", idims, "fix", fpi, "| out", odims, "fix", fpo, flush=True)

    d = np.load(FIN)
    img = d["img"].astype(np.float32)                 # (6,3,256,704) normalized
    feat_o = np.empty((6, 16, 44, 64), np.float32)
    depth_o = np.empty((6, 88, 16, 44), np.float32)
    t0 = time.time()
    for c in range(6):
        x = img[c].transpose(1, 2, 0)                 # (256,704,3)
        xi = np.clip(np.round(x * (2.0 ** fpi)), -128, 127).astype(np.int8)[None]  # (1,256,704,3)
        ins = [np.ascontiguousarray(xi.reshape(idims), np.int8)]
        outs = [np.empty(odims, np.int8)]
        jid = r.execute_async(ins, outs); r.wait(jid)
        of = outs[0].reshape(16, 44, 152).astype(np.float32) * (2.0 ** -fpo)
        dlog = of[..., :88]; feat = of[..., 88:]                      # (16,44,88),(16,44,64)
        depth = softmax(dlog, -1).transpose(2, 0, 1)                  # (88,16,44)
        feat_o[c] = feat; depth_o[c] = depth
    dt = time.time() - t0
    np.savez(FOUT, feat=feat_o.astype(np.float16), depth=depth_o.astype(np.float16))
    print("saved %s  (6 cams in %.2fs)" % (FOUT, dt), flush=True)

    if "feat" in d.files and "depth" in d.files:
        ff = d["feat"].astype(np.float32); df = d["depth"].astype(np.float32)
        def cos(a, b): return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        am = float((df.argmax(1) == depth_o.argmax(1)).mean())
        print("SEAM vs FP32: feat-cos %.4f | depth-softmax-cos %.4f | depth-argmax-match %.3f"
              % (cos(ff, feat_o), cos(df, depth_o), am), flush=True)

if __name__ == "__main__":
    main()
