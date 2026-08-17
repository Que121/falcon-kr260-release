#!/usr/bin/env python3
"""Fast on-board occupancy over N frames -> a REAL on-board mIoU (no simulation, no xclbin needed).

Stock dpu.bit (works, fingerprint-matched to bev_reluc) loaded ONCE + bev_reluc's 3 DPU conv subgraphs
on the B4096 DPU + the 2 bilinear upsamples in numpy (align_corners=True; the resize IP is for the
latency bound, not accuracy -- float numpy bilinear is the exact reference) + the Linear-Softplus-Linear
predicter + argmax. No per-frame PL reconfig -> a few hundred ms/frame, so the full 6019-frame Occ3D
val set is feasible on the board. Saves per-frame occupancy argmax for mIoU-vs-GT.

  python3 board_mIoU_walk.py <Nframes> <eval_vt.npy> <out_argmax.npy>
"""
import os, sys, time
os.environ["XILINX_XRT"] = "/usr"
import numpy as np, xir
import pynq.pl_server.embedded_device as _ed
_DX = os.path.join(os.path.dirname(_ed.__file__), "default.xclbin")
if os.path.exists(_DX):
    _ed._create_xclbin = lambda m: open(_DX, "rb").read()

XM   = "/home/ubuntu/bev_allpl/bev_reluc.xmodel"
HEAD = "/home/ubuntu/bev/predicter_head.npz"

def t_fp(t):
    return t.get_attr("fix_point") if t.has_attr("fix_point") else None

def bilinear_up(xf, Hout, Wout):                     # xf (Hin,Win,C) -> (Hout,Wout,C), align_corners
    Hin, Win, C = xf.shape
    def ax(out, inn):
        if out == 1: return np.zeros(out, int), np.zeros(out, int), np.zeros(out)
        s = np.arange(out) * (inn - 1) / (out - 1)
        f = np.floor(s).astype(int); f1 = np.minimum(f + 1, inn - 1)
        return f, f1, (s - f)
    y0, y1, wy = ax(Hout, Hin); x0, x1, wx = ax(Wout, Win)
    a = xf[y0][:, x0]; b = xf[y0][:, x1]; c = xf[y1][:, x0]; d = xf[y1][:, x1]
    wy = wy[:, None, None]; wx = wx[None, :, None]
    return (a*(1-wx)+b*wx)*(1-wy) + (c*(1-wx)+d*wx)*wy

def main():
    import vart
    from pynq_dpu import DpuOverlay
    N   = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    VT  = sys.argv[2] if len(sys.argv) > 2 else "/home/ubuntu/bev/eval_vt.npy"
    OUT = sys.argv[3] if len(sys.argv) > 3 else "/home/ubuntu/board_argmax.npy"
    ov = DpuOverlay("dpu.bit")                        # stock DPU, loads, no xclbin issue
    g = xir.Graph.deserialize(XM)
    subs = g.get_root_subgraph().toposort_child_subgraph()
    dev = lambda s: s.get_attr("device") if s.has_attr("device") else "?"
    runners = {i: vart.Runner.create_runner(s, "run") for i, s in enumerate(subs) if dev(s) == "DPU"}
    h = np.load(HEAD); W0, b0, W2, b2 = h["0.weight"], h["0.bias"], h["2.weight"], h["2.bias"]
    in_t = list(subs[0].get_output_tensors())[0]; fp_in = t_fp(in_t)
    vt = np.load(VT, mmap_mode="r")
    M = min(N, vt.shape[0]); print(f"frames {M} | DPU runners {sorted(runners)}", flush=True)

    def predicter(conv):                              # conv (256,200,200) -> argmax (200,200,16)
        x = conv.transpose(2, 1, 0).reshape(-1, 256) @ W0.T + b0
        x = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
        return (x @ W2.T + b2).reshape(200, 200, 16, 18).argmax(-1).astype(np.uint8)

    def one(vtf):                                     # vtf (64,200,200) -> argmax (200,200,16)
        nh = vtf.transpose(1, 2, 0)[None]
        store = {in_t.name: (np.clip(np.round(nh * 2.0**fp_in), -128, 127).astype(np.int8), fp_in)}
        for i, s in enumerate(subs[1:], 1):
            if dev(s) == "DPU":
                r = runners[i]; its, ots = r.get_input_tensors(), r.get_output_tensors()
                ins = [np.ascontiguousarray(store[it.name][0].reshape([int(x) for x in it.dims]), np.int8) for it in its]
                outs = [np.empty([int(x) for x in ot.dims], np.int8) for ot in ots]
                jid = r.execute_async(ins, outs); r.wait(jid)
                for k, ot in enumerate(ots): store[ot.name] = (outs[k].copy(), ot.get_attr("fix_point"))
            else:
                it = list(s.get_input_tensors())[0]; ot = list(s.get_output_tensors())[0]
                xin, fpi = store[it.name]
                if int(ot.dims[1]) > int(it.dims[1]):
                    Ho, Wo = int(ot.dims[1]), int(ot.dims[2]); fpo = t_fp(ot) if t_fp(ot) is not None else fpi
                    up = bilinear_up(xin[0].astype(np.float32) * 2.0**(-fpi), Ho, Wo)
                    store[ot.name] = (np.clip(np.round(up * 2.0**fpo), -128, 127).astype(np.int8)[None], fpo)
                else:
                    store[ot.name] = (xin, fpi)
        ot = list(subs[-1].get_output_tensors())[0]; arr, fp = store[ot.name]
        return predicter((arr[0].astype(np.float32) * 2.0**(-fp)).transpose(2, 0, 1))

    out = np.empty((M, 200, 200, 16), np.uint8); t0 = time.time()
    for i in range(M):
        out[i] = one(np.asarray(vt[i], np.float32))
        if (i + 1) % 20 == 0: print(f"{i+1}/{M}  {(time.time()-t0)/(i+1):.2f}s/frame", flush=True)
    np.save(OUT, out)
    print(f"DONE {M} frames in {time.time()-t0:.0f}s -> {OUT} {out.shape}", flush=True)

if __name__ == "__main__":
    main()
