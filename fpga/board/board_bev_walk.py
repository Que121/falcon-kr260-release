#!/usr/bin/env python3
"""Full end-to-end on-board BEV-stage forward pass on the KR260.

Walks the 21-subgraph fragmented bev_clamp32.xmodel:
  * DPU subgraphs  -> run on the real B4096 DPU via vart.Runner (int8 in/out at each tensor's fix_point)
  * CPU subgraphs  -> evaluated in numpy on the ARM PS (conv2d / add / Hardtanh(0,32) / bilinear
                      upsample(align_corners) / concat), with int8 quant->dequant applied at every
                      tensor that carries a fix_point, exactly where the compiler inserted fix nodes.

Input : bev_test_input.npy  (1,64,200,200 NCHW float32) = a real LSS view-transform output (vt_out).
Output: conv_only BEV (1,200,200,256 NHWC) -> transposed -> compared to bev_clampref_fp32.npy (FP32 ref).

This is the first end-to-end on-board run of the deployed INT8 BEV stage (DPU + PS on one KR260 chip).
"""
import sys, time, os as _os
import numpy as np
import xir, vart

XM   = "/home/ubuntu/bev/bev_clamp32.xmodel"
INP  = "/home/ubuntu/bev/bev_test_input.npy"
REF  = "/home/ubuntu/bev/bev_clampref_fp32.npy"
CLAMP = 32.0  # Hardtanh(0, C) baked in by the quantiser (bev_clamp32); FP ref max == 32 confirms

# ---------------- numpy op kernels ----------------

def q_dq(x, fp):
    """Vitis-AI int8 quantise->dequantise at fix_point fp:  q=clip(round(r*2^fp)), r=q*2^-fp."""
    s = 2.0 ** fp
    q = np.round(x * s)
    q = np.clip(q, -128.0, 127.0)
    return (q / s).astype(np.float32)


def conv2d_nhwc(x, w, b, stride, pad):
    """x:(1,H,W,Cin) f32 ; w:(Cout,Kh,Kw,Cin) OHWI ; b:(Cout,) ; shift-accumulate, FLOOR output."""
    Cout, Kh, Kw, Cin = w.shape
    pt, pl, pb, pr = pad[0], pad[1], pad[2], pad[3]
    xp = np.pad(x, ((0, 0), (pt, pb), (pl, pr), (0, 0)))
    H, W = xp.shape[1], xp.shape[2]
    sh, sw = stride
    Hout = (H - Kh) // sh + 1
    Wout = (W - Kw) // sw + 1
    out = np.empty((1, Hout, Wout, Cout), np.float32)
    out[:] = b.reshape(1, 1, 1, Cout)
    for di in range(Kh):
        for dj in range(Kw):
            xs = xp[:, di:di + sh * Hout:sh, dj:dj + sw * Wout:sw, :]   # (1,Hout,Wout,Cin)
            wk = w[:, di, dj, :].T                                       # (Cin,Cout)
            out += (xs.reshape(-1, Cin) @ wk).reshape(1, Hout, Wout, Cout)
    return out


def bilinear_align_corners(x, scale):
    """x:(1,H,W,C) -> (1,H*scale,W*scale,C), align_corners=True."""
    sh, sw = scale
    H, W, C = x.shape[1], x.shape[2], x.shape[3]
    Ho, Wo = int(round(H * sh)), int(round(W * sw))

    def axis_idx(n_in, n_out):
        if n_out == 1:
            return np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros(1, np.float32)
        pos = np.arange(n_out, dtype=np.float64) * (n_in - 1) / (n_out - 1)
        i0 = np.floor(pos).astype(np.int64)
        i1 = np.minimum(i0 + 1, n_in - 1)
        frac = (pos - i0).astype(np.float32)
        return i0, i1, frac

    y0, y1, fy = axis_idx(H, Ho)
    x0, x1, fx = axis_idx(W, Wo)
    a = x[0][y0][:, x0]          # (Ho,Wo,C)  top-left
    b = x[0][y0][:, x1]
    c = x[0][y1][:, x0]
    d = x[0][y1][:, x1]
    fy_ = fy[:, None, None]; fx_ = fx[None, :, None]
    top = a * (1 - fx_) + b * fx_
    bot = c * (1 - fx_) + d * fx_
    out = top * (1 - fy_) + bot * fy_
    return out[None].astype(np.float32)


# ---------------- graph helpers ----------------

def dev(sg):
    try: return sg.get_attr("device")
    except Exception: return "?"

def t_fp(t):
    try:
        if t.has_attr("fix_point"): return t.get_attr("fix_point")
    except Exception: pass
    return None

def op_fp(o):
    try:
        if o.has_attr("fix_point"): return o.get_attr("fix_point")
    except Exception: pass
    return t_fp(o.get_output_tensor())

def const_data(o):
    t = o.get_output_tensor()
    raw = o.get_attr("data")
    return np.frombuffer(raw, np.float32).reshape(list(t.dims)).copy()


# ---------------- CPU subgraph interpreter ----------------

def run_cpu_subgraph(sg, vmap):
    """Evaluate a CPU subgraph in numpy. Reads/writes real f32 tensors in vmap by tensor name."""
    ops = _toposort_ops(sg)
    for o in ops:
        ot = o.get_type()
        out_t = o.get_output_tensor()
        name = out_t.name
        ins = o.get_input_ops()  # arg -> [op]

        def arr_list():               # all input arrays in declared (key, then list) order
            xs = []
            for a in ins:
                for io in ins[a]:
                    xs.append(vmap[io.get_output_tensor().name])
            return xs

        def arg(a):                   # named single input (conv2d: input/weights/bias)
            return vmap[ins[a][0].get_output_tensor().name]

        if ot == "const":
            val = const_data(o)
        elif ot == "data-fix":
            val = vmap[name]                      # provided by caller (USER input)
        elif ot in ("fix", "float2fix", "fix2float"):
            val = arr_list()[0]
        elif ot == "conv2d":
            x = arg("input"); w = arg("weights"); b = arg("bias")
            val = conv2d_nhwc(x, w, b, o.get_attr("stride"), o.get_attr("pad"))
        elif ot == "add":
            xs = arr_list()
            val = xs[0].astype(np.float32).copy()
            for extra in xs[1:]:
                val = val + extra
        elif ot == "nndct_hardtanh":
            val = np.clip(arr_list()[0], 0.0, CLAMP).astype(np.float32)
        elif ot in ("upsample-fix", "upsample"):
            val = bilinear_align_corners(arr_list()[0], o.get_attr("scale"))
        elif ot in ("concat-fix", "concat"):
            parts = arr_list()
            if _os.environ.get("FLIP_CONCAT") == "1":
                parts = parts[::-1]
            val = np.concatenate(parts, axis=o.get_attr("axis")).astype(np.float32)
        else:
            raise RuntimeError("unhandled CPU op type: " + ot)
        fp = op_fp(o)
        # fix2float is dequant-only (already real in our convention) -> never requant/clip it.
        # const carries raw weights. Everything else with a fix_point gets int8 quant->dequant.
        if fp is not None and ot not in ("const", "fix2float"):
            if not (_os.environ.get("NO_CPU_QUANT") == "1" and ot in ("conv2d", "add", "nndct_hardtanh",
                                                "upsample-fix", "concat-fix", "float2fix", "fix")):
                val = q_dq(val, fp)
        vmap[name] = val


def _toposort_ops(sg):
    ops = list(sg.get_ops())
    name2op = {o.get_output_tensor().name: o for o in ops}
    visited, order = set(), []
    def visit(o):
        n = o.get_output_tensor().name
        if n in visited: return
        visited.add(n)
        for a, lst in o.get_input_ops().items():
            for io in lst:
                if io.get_output_tensor().name in name2op:
                    visit(io)
        order.append(o)
    for o in ops:
        visit(o)
    return order


# ---------------- DPU subgraph via vart ----------------

def run_dpu_subgraph(sg, vmap):
    runner = vart.Runner.create_runner(sg, "run")
    its = runner.get_input_tensors()
    ots = runner.get_output_tensors()
    in_arrs = []
    for it in its:
        fp = it.get_attr("fix_point")
        real = vmap[it.name]                                  # f32 NHWC real
        q = np.clip(np.round(real * (2.0 ** fp)), -128, 127).astype(np.int8)
        in_arrs.append(np.ascontiguousarray(q.reshape([int(d) for d in it.dims])))
    out_arrs = [np.empty([int(d) for d in ot.dims], np.int8) for ot in ots]
    jid = runner.execute_async(in_arrs, out_arrs)
    runner.wait(jid)
    for ot, oa in zip(ots, out_arrs):
        fp = ot.get_attr("fix_point")
        vmap[ot.name] = (oa.astype(np.float32) * (2.0 ** (-fp)))
    del runner


# ---------------- main ----------------

def main():
    from pynq_dpu import DpuOverlay
    ov = DpuOverlay("dpu.bit")          # bring up the B4096 DPU on PL before any vart runner
    print("DPU overlay loaded")

    g = xir.Graph.deserialize(XM)
    subs = g.get_root_subgraph().toposort_child_subgraph()
    vmap = {}

    vt = np.load(INP).astype(np.float32)            # (1,64,200,200) NCHW
    vt_nhwc = np.ascontiguousarray(vt.transpose(0, 2, 3, 1))   # (1,200,200,64)

    t0 = time.time()
    for i, sg in enumerate(subs):
        d = dev(sg)
        if d == "USER":
            out_t = list(sg.get_output_tensors())[0]
            fp = t_fp(out_t)
            v = q_dq(vt_nhwc, fp) if fp is not None else vt_nhwc
            vmap[out_t.name] = v
            print(f"[{i}] USER input -> {out_t.name[:50]} fp={fp} shape={v.shape}")
        elif d == "DPU":
            run_dpu_subgraph(sg, vmap)
            ot = list(sg.get_output_tensors())[0]
            v = vmap[ot.name]
            print(f"[{i}] DPU  -> {ot.name[-42:]} {v.shape} min{v.min():.2f} max{v.max():.2f} L2{np.linalg.norm(v):.1f}")
        elif d == "CPU":
            run_cpu_subgraph(sg, vmap)
            ot = list(sg.get_output_tensors())[0]
            v = vmap[ot.name]
            print(f"[{i}] CPU  -> {ot.name[-42:]} {v.shape} min{v.min():.2f} max{v.max():.2f} L2{np.linalg.norm(v):.1f}")
        else:
            print(f"[{i}] dev={d} skipped")
    print(f"walk done in {time.time()-t0:.1f}s")

    if _os.environ.get("SAVE_VMAP") == "1":
        save = {}
        for k, v in vmap.items():
            if v.ndim == 4 and v.shape[0] == 1 and v.shape[1] >= 25 and v.shape[2] >= 25:
                save[k] = v.transpose(0, 3, 1, 2)        # NHWC -> NCHW to match torch gold
        np.savez("/home/ubuntu/bev/bev_walker_vmap.npz", **save)
        print("saved", len(save), "activation tensors to bev_walker_vmap.npz")

    # final conv_only output (NHWC) -> NCHW
    out_t = list(subs[-1].get_output_tensors())[0]
    y = vmap[out_t.name]                              # (1,200,200,256) NHWC real
    y_nchw = y.transpose(0, 3, 1, 2)                  # (1,256,200,200)
    np.save("/home/ubuntu/bev/bev_onboard_convonly.npy", y_nchw)

    ref = np.load(REF).astype(np.float32)            # (1,256,200,200)
    print("\n=== on-board conv_only vs FP32 reference ===")
    print("shapes", y_nchw.shape, ref.shape, " out range", float(y_nchw.min()), float(y_nchw.max()))
    a, b = y_nchw.ravel(), ref.ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    rel = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
    mae = float(np.abs(a - b).mean())
    print(f"cosine        {cos:.4f}")
    print(f"rel L2 error  {rel:.4f}")
    print(f"MAE           {mae:.4f}")
    # voxel-occupancy-relevant: argmax over the 256-d feature channel agreement (proxy for head input)
    am_on = y_nchw[0].argmax(0); am_ref = ref[0].argmax(0)
    print(f"channel-argmax agreement (feature proxy)  {float((am_on==am_ref).mean()):.4f}")


if __name__ == "__main__":
    main()
