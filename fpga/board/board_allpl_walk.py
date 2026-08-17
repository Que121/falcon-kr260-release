#!/usr/bin/env python3
"""All-PL end-to-end BEV stage on the KR260: 3 DPU subgraphs + 2 resize-IP upsamples.

The de-fragmented `bev_reluc.xmodel` (DPU-native ReLU, clamp via INT8 saturation) compiles to 7
subgraphs: 3 DPU conv blocks, 2 bilinear upsamples, a USER input and a trivial cast. This walker runs
the 3 DPU subgraphs on the B4096 DPU (via vart) and the 2 upsamples on the custom resize IP (PL), so
NO convolution or activation runs on the ARM PS. DPU and resize live in separate overlays, so we swap
the PL bitstream between stages (fine for a one-shot correctness run); steady-state would use the
unified DPU+IP bitstream.

  [1] DPU  vt_out -> feat0(100,100,128), feat2(25,25,512)
  [2] IP   resize x4  feat2 25->100  (overlay rovl_2510)
  [3] DPU  concat(feat0,up1)+neck -> (100,100,512)
  [4] IP   resize x2  100->200       (overlay resize_ovl)
  [5] DPU  up2+final  -> (200,200,256)
  [6] cast -> conv_only

Run (sudo, pynq venv, XRT sourced):
  python3 board_allpl_walk.py
"""
import sys, time
import numpy as np
import xir

XM   = "/home/ubuntu/bev_allpl/bev_reluc.xmodel"
DPU_BIT   = "dpu.bit"
RZ_BIT_25 = "/home/ubuntu/rovl_2510/rovl_2510.bit"      # 25->100 x4 (built this session)
RZ_BIT_100= "/home/ubuntu/resize_ovl/resize_ovl.bit"    # 100->200 x2 (already on board)
INP  = "/home/ubuntu/bev/bev_test_input.npy"

# resize_deploy.cpp s_axilite register map
AP_CTRL = 0x00
RZREG = {"in":0x10, "y0":0x1c, "y1":0x28, "wy":0x34, "x0":0x40, "x1":0x4c, "wx":0x58, "out":0x64}
WB = 64  # 512-bit word = 64 INT8 channels

# ---- xclbinutil-segfault workaround (return PYNQ's shipped default.xclbin topology) ----
import pynq.pl_server.embedded_device as _ed, os as _os
_DX = _os.path.join(_os.path.dirname(_ed.__file__), "default.xclbin")
if _os.path.exists(_DX):
    _B = open(_DX, "rb").read(); _ed._create_xclbin = lambda _m: _B


def t_fp(t):
    return t.get_attr("fix_point") if t.has_attr("fix_point") else None


def bilinear_taps(out_n, in_n):
    i0 = np.zeros(out_n, np.uint16); i1 = np.zeros(out_n, np.uint16); w = np.zeros(out_n, np.uint16)
    for o in range(out_n):
        s = o * (in_n - 1) / (out_n - 1) if out_n > 1 else 0.0
        f = int(np.floor(s)); f1 = min(f + 1, in_n - 1)
        i0[o] = f; i1[o] = f1; w[o] = int(round((s - f) * 256)) & 0x1ff
    return i0, i1, w


# ================= DPU stage =================

def run_dpu_subgraph(sg, feeds):
    """feeds: {tensor_name: int8 NHWC array}. Returns {out_name: (int8 array, fp)}.
    Loads the DPU overlay fresh (PL was reconfigured by a resize overlay before this)."""
    import vart
    from pynq_dpu import DpuOverlay
    ov = DpuOverlay(DPU_BIT)                      # reconfigure PL to the DPU
    runner = vart.Runner.create_runner(sg, "run")
    its, ots = runner.get_input_tensors(), runner.get_output_tensors()
    in_arrs = []
    for it in its:
        a = feeds[it.name]
        in_arrs.append(np.ascontiguousarray(a.reshape([int(d) for d in it.dims]), dtype=np.int8))
    out_arrs = [np.empty([int(d) for d in ot.dims], np.int8) for ot in ots]
    jid = runner.execute_async(in_arrs, out_arrs); runner.wait(jid)
    res = {ot.name: (out_arrs[i].copy(), ot.get_attr("fix_point")) for i, ot in enumerate(ots)}
    del runner, ov
    return res


# ================= resize IP stage =================

def run_resize_ip(bit, x_int8_nhwc, fp_in, fp_out, out_hw):
    """x_int8_nhwc: (1,Hin,Win,C) int8. Upsample to (1,Hout,Wout,C) via the resize IP on `bit`.
    Repacks to the IP's [NTILE][H][W][64] layout, loads align_corners taps, rescales fp if needed."""
    from pynq import Overlay, allocate
    _, Hin, Win, C = x_int8_nhwc.shape
    Hout, Wout = out_hw
    NTILE = C // WB
    ol = Overlay(bit)
    ip = None
    for nm in ("resize_0", "resize_bilinear_0"):
        ip = getattr(ol, nm, None)
        if ip is not None: break
    if ip is None:
        key = [k for k in ol.ip_dict if "resize" in k.lower()][0].split('/')[-1]
        ip = getattr(ol, key)

    # NHWC (1,H,W,C) -> [NTILE][H][W][64] packed bytes
    packed = x_int8_nhwc[0].reshape(Hin, Win, NTILE, WB).transpose(2, 0, 1, 3)   # (NTILE,Hin,Win,64)
    in_buf  = allocate(shape=(NTILE * Hin * Win * WB,),  dtype=np.uint8)
    out_buf = allocate(shape=(NTILE * Hout * Wout * WB,), dtype=np.uint8)
    in_buf[:] = packed.astype(np.int8).view(np.uint8).ravel(); in_buf.flush(); out_buf.flush()

    y0, y1, wy = bilinear_taps(Hout, Hin); x0, x1, wx = bilinear_taps(Wout, Win)
    bufs = {}
    for nm, arr in (("y0", y0), ("y1", y1), ("wy", wy), ("x0", x0), ("x1", x1), ("wx", wx)):
        b = allocate(shape=arr.shape, dtype=np.uint16); b[:] = arr; b.flush(); bufs[nm] = b

    def setptr(reg, buf):
        a = buf.device_address; ip.write(reg, a & 0xffffffff); ip.write(reg + 4, (a >> 32) & 0xffffffff)
    setptr(RZREG["in"], in_buf); setptr(RZREG["out"], out_buf)
    for nm in ("y0", "y1", "wy", "x0", "x1", "wx"):
        setptr(RZREG[nm], bufs[nm])
    ip.write(AP_CTRL, 1)
    while (ip.read(AP_CTRL) & 0x2) == 0: pass

    out = np.frombuffer(bytes(out_buf), np.int8).reshape(NTILE, Hout, Wout, WB)
    out = out.transpose(1, 2, 0, 3).reshape(1, Hout, Wout, C)                    # -> NHWC
    if fp_out != fp_in:                                                          # rescale fp if needed
        out = np.clip(np.round(out.astype(np.float32) * (2.0 ** (fp_out - fp_in))), -128, 127).astype(np.int8)
    del ol
    return np.ascontiguousarray(out)


# ================= orchestration =================

def main():
    g = xir.Graph.deserialize(XM)
    subs = g.get_root_subgraph().toposort_child_subgraph()
    dev = lambda sg: sg.get_attr("device") if sg.has_attr("device") else "?"

    # name the boundary tensors from the known structure
    vt = np.load(INP).astype(np.float32)                       # (1,64,200,200) NCHW
    vt_nhwc = vt.transpose(0, 2, 3, 1)                         # (1,200,200,64)
    store = {}                                                 # tensor_name -> (int8 NHWC, fp)

    # [0] USER input
    in_t = list(subs[0].get_output_tensors())[0]
    fp_in = t_fp(in_t)
    q = np.clip(np.round(vt_nhwc * (2.0 ** fp_in)), -128, 127).astype(np.int8)
    store[in_t.name] = (q, fp_in)
    print(f"[0] USER {in_t.name[-30:]} fp={fp_in} {q.shape}")

    t0 = time.time()
    for i, sg in enumerate(subs[1:], 1):
        d = dev(sg)
        if d == "DPU":
            feeds = {it.name: store[it.name][0] for it in sg.get_input_tensors()}
            res = run_dpu_subgraph(sg, feeds)
            for nm, (arr, fp) in res.items():
                store[nm] = (arr, fp)
                print(f"[{i}] DPU  -> {nm[-30:]} fp={fp} {arr.shape}")
        elif d == "CPU":
            ins = list(sg.get_input_tensors()); outs = list(sg.get_output_tensors())
            ot = outs[0]; it = ins[0]
            xin, fpi = store[it.name]
            if "upsample" in ot.name.lower() or list(ot.dims)[1] > list(it.dims)[1]:
                Hout, Wout = int(ot.dims[1]), int(ot.dims[2])
                bit = RZ_BIT_25 if Hout == 100 else RZ_BIT_100
                fpo = t_fp(ot) if t_fp(ot) is not None else fpi
                up = run_resize_ip(bit, xin, fpi, fpo, (Hout, Wout))
                store[ot.name] = (up, fpo)
                print(f"[{i}] IP   -> {ot.name[-30:]} fp={fpo} {up.shape}  (resize {it.dims[1]}->{Hout})")
            else:                                              # trivial cast (fix2float)
                store[ot.name] = (xin, fpi)
                print(f"[{i}] cast -> {ot.name[-30:]} {xin.shape}")
    print(f"all-PL walk done in {time.time()-t0:.1f}s")

    out_t = list(subs[-1].get_output_tensors())[0]
    arr, fp = store[out_t.name]
    conv = (arr.astype(np.float32) * (2.0 ** (-fp))).transpose(0, 3, 1, 2)        # NHWC->NCHW
    np.save("/home/ubuntu/bev_allpl/allpl_convonly.npy", conv)
    print("saved conv_only", conv.shape, "range", float(conv.min()), float(conv.max()))


if __name__ == "__main__":
    main()
