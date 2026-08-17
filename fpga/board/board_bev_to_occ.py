#!/usr/bin/env python3
"""Build-B step4: vt_out -> occupancy, FULLY on PL. All-PL BEV walk (3 DPU conv + 2 resize-IP
upsamples, overlay-swapped) + the Linear-Softplus-Linear predicter (off-DPU by design) + argmax.

Takes the ON-BOARD vt_out from step3 (image->DPU->gather), produces occ argmax (200,200,16), and
compares to the FP32 occ reference in the dump frame -> per-frame voxel agreement (camera-mask voxels).
NO numpy upsample: every conv on the B4096 DPU, every upsample on the HLS bilinear resize IP.

  python3 board_bev_to_occ.py <vt.npy(64,200,200)> <frame.npz(occ ref)> <out_occ.npy>
"""
import sys, time, numpy as np, xir
import pynq.pl_server.embedded_device as _ed, os as _os
_os.environ["XILINX_XRT"] = "/usr"
_DX = _os.path.join(_os.path.dirname(_ed.__file__), "default.xclbin")
if _os.path.exists(_DX):
    _B = open(_DX, "rb").read(); _ed._create_xclbin = lambda _m: _B

XM = sys.argv[4] if len(sys.argv) > 4 else _os.environ.get("BEV_XM", "/home/ubuntu/bev_allpl/bev_reluc.xmodel")
DPU_BIT = "dpu.bit"
RZ_BIT_25 = "/home/ubuntu/rovl_2510/rovl_2510.bit"
RZ_BIT_100 = "/home/ubuntu/resize_ovl/resize_ovl.bit"
HEAD = "/home/ubuntu/bev/predicter_head.npz"
AP_CTRL = 0x00
RZREG = {"in":0x10, "y0":0x1c, "y1":0x28, "wy":0x34, "x0":0x40, "x1":0x4c, "wx":0x58, "out":0x64}
WB = 64

def t_fp(t): return t.get_attr("fix_point") if t.has_attr("fix_point") else None

def bilinear_taps(out_n, in_n):
    i0 = np.zeros(out_n, np.uint16); i1 = np.zeros(out_n, np.uint16); w = np.zeros(out_n, np.uint16)
    for o in range(out_n):
        s = o * (in_n - 1) / (out_n - 1) if out_n > 1 else 0.0
        f = int(np.floor(s)); f1 = min(f + 1, in_n - 1)
        i0[o] = f; i1[o] = f1; w[o] = int(round((s - f) * 256)) & 0x1ff
    return i0, i1, w

def run_dpu_subgraph(sg, feeds):
    import vart
    from pynq_dpu import DpuOverlay
    ov = DpuOverlay(DPU_BIT)
    runner = vart.Runner.create_runner(sg, "run")
    its, ots = runner.get_input_tensors(), runner.get_output_tensors()
    in_arrs = [np.ascontiguousarray(feeds[it.name].reshape([int(d) for d in it.dims]), np.int8) for it in its]
    out_arrs = [np.empty([int(d) for d in ot.dims], np.int8) for ot in ots]
    jid = runner.execute_async(in_arrs, out_arrs); runner.wait(jid)
    res = {ot.name: (out_arrs[i].copy(), ot.get_attr("fix_point")) for i, ot in enumerate(ots)}
    del runner, ov
    return res

def run_resize_ip(bit, x, fp_in, fp_out, out_hw):
    from pynq import Overlay, allocate
    _, Hin, Win, C = x.shape; Hout, Wout = out_hw; NTILE = C // WB
    ol = Overlay(bit)
    ip = None
    for nm in ("resize_0", "resize_bilinear_0"):
        ip = getattr(ol, nm, None)
        if ip is not None: break
    if ip is None:
        ip = getattr(ol, [k for k in ol.ip_dict if "resize" in k.lower()][0].split('/')[-1])
    packed = x[0].reshape(Hin, Win, NTILE, WB).transpose(2, 0, 1, 3)
    in_buf = allocate(shape=(NTILE*Hin*Win*WB,), dtype=np.uint8)
    out_buf = allocate(shape=(NTILE*Hout*Wout*WB,), dtype=np.uint8)
    in_buf[:] = packed.astype(np.int8).view(np.uint8).ravel(); in_buf.flush(); out_buf.flush()
    y0, y1, wy = bilinear_taps(Hout, Hin); x0, x1, wx = bilinear_taps(Wout, Win)
    bufs = {}
    for nm, arr in (("y0",y0),("y1",y1),("wy",wy),("x0",x0),("x1",x1),("wx",wx)):
        b = allocate(shape=arr.shape, dtype=np.uint16); b[:] = arr; b.flush(); bufs[nm] = b
    def setptr(reg, buf):
        a = buf.device_address; ip.write(reg, a & 0xffffffff); ip.write(reg+4, (a >> 32) & 0xffffffff)
    setptr(RZREG["in"], in_buf); setptr(RZREG["out"], out_buf)
    for nm in ("y0","y1","wy","x0","x1","wx"): setptr(RZREG[nm], bufs[nm])
    ip.write(AP_CTRL, 1)
    while (ip.read(AP_CTRL) & 0x2) == 0: pass
    out = np.frombuffer(bytes(out_buf), np.int8).reshape(NTILE, Hout, Wout, WB).transpose(1,2,0,3).reshape(1, Hout, Wout, C)
    if fp_out != fp_in:
        out = np.clip(np.round(out.astype(np.float32) * (2.0 ** (fp_out - fp_in))), -128, 127).astype(np.int8)
    del ol
    return np.ascontiguousarray(out)

def main():
    VT = sys.argv[1]; FR = sys.argv[2]
    OUT = sys.argv[3] if len(sys.argv) > 3 else "/home/ubuntu/buildB/step4_occ.npy"
    g = xir.Graph.deserialize(XM)
    subs = g.get_root_subgraph().toposort_child_subgraph()
    dev = lambda sg: sg.get_attr("device") if sg.has_attr("device") else "?"
    vt = np.load(VT).astype(np.float32)
    if vt.ndim == 3: vt = vt[None]                       # (1,64,200,200)
    vt_nhwc = vt.transpose(0, 2, 3, 1)
    store = {}
    in_t = list(subs[0].get_output_tensors())[0]; fp_in = t_fp(in_t)
    store[in_t.name] = (np.clip(np.round(vt_nhwc * (2.0**fp_in)), -128, 127).astype(np.int8), fp_in)
    t0 = time.time()
    for i, sg in enumerate(subs[1:], 1):
        d = dev(sg)
        if d == "DPU":
            feeds = {it.name: store[it.name][0] for it in sg.get_input_tensors()}
            for nm, (arr, fp) in run_dpu_subgraph(sg, feeds).items(): store[nm] = (arr, fp)
        elif d == "CPU":
            it = list(sg.get_input_tensors())[0]; ot = list(sg.get_output_tensors())[0]
            xin, fpi = store[it.name]
            if "upsample" in ot.name.lower() or int(ot.dims[1]) > int(it.dims[1]):
                Hout, Wout = int(ot.dims[1]), int(ot.dims[2])
                bit = RZ_BIT_25 if Hout == 100 else RZ_BIT_100
                fpo = t_fp(ot) if t_fp(ot) is not None else fpi
                store[ot.name] = (run_resize_ip(bit, xin, fpi, fpo, (Hout, Wout)), fpo)
            else:
                store[ot.name] = (xin, fpi)
    ot = list(subs[-1].get_output_tensors())[0]; arr, fp = store[ot.name]
    conv = (arr[0].astype(np.float32) * (2.0**(-fp))).transpose(2, 0, 1)        # (256,200,200)
    print("BEV all-PL walk done in %.1fs -> conv_only %s fp=%d range[%.2f,%.2f]" % (
        time.time()-t0, conv.shape, fp, float(conv.min()), float(conv.max())), flush=True)
    np.save(OUT.replace(".npy", "_conv.npy"), conv.astype(np.float16))

    h = np.load(HEAD); W0, b0, W2, b2 = h["0.weight"], h["0.bias"], h["2.weight"], h["2.bias"]
    x = conv.transpose(2, 1, 0).reshape(-1, 256) @ W0.T + b0
    x = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
    occ = (x @ W2.T + b2).reshape(200, 200, 16, 18).argmax(-1).astype(np.uint8)
    np.save(OUT, occ); print("saved occ", occ.shape, "-> %s" % OUT, flush=True)

    fr = np.load(FR)
    if "occ" in fr.files:
        ref = fr["occ"].astype(np.uint8)                  # (200,200,16) FP32 argmax
        agree = float((occ == ref).mean())
        nz = (ref != 17)                                  # 17 = free/empty class in Occ3D (18-class)
        occ_acc = float((occ[nz] == ref[nz]).mean()) if nz.any() else 0.0
        print("vs FP32 occ: all-voxel agree %.4f | non-free-voxel agree %.4f" % (agree, occ_acc), flush=True)

if __name__ == "__main__":
    main()
