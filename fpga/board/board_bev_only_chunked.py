#!/usr/bin/env python3
"""BEV-only, CHUNKED (memory-safe): FP32 vt_out -> BEV xmodel (3 DPU + 2 resize-IP) -> predicter -> occ
argmax, over N frames in chunks of CHUNK to bound RAM (the 25->100 resize output is ~5MB/frame; 256
frames in one chunk = ~1.3GB and OOM-wedged the 4GB board before). Per chunk: load DPU/resize overlays,
walk subgraphs, predicter+argmax, free. Isolates BEV-stage deploy quality on the SAME frames as the sim.
  python3 board_bev_only_chunked.py <frames_dir> <N> <out_occ.npy> <bev_xmodel> [predicter.npz] [chunk=48]
"""
import os, sys, time, glob, numpy as np, xir
os.environ["XILINX_XRT"] = "/usr"
import pynq.pl_server.embedded_device as _ed
_DX = os.path.join(os.path.dirname(_ed.__file__), "default.xclbin")
if os.path.exists(_DX): _ed._create_xclbin = lambda _m: open(_DX, "rb").read()
import vart
from pynq import Overlay, allocate
from pynq_dpu import DpuOverlay

FR = sys.argv[1]; N = int(sys.argv[2]); OUT = sys.argv[3]
BEV_XM = sys.argv[4]
PRED = sys.argv[5] if len(sys.argv) > 5 else "/home/ubuntu/bev/predicter_head.npz"
CHUNK = int(sys.argv[6]) if len(sys.argv) > 6 else 48
VTCLIP = int(sys.argv[7]) if len(sys.argv) > 7 else 0   # 1 = clip FP32 vt at 127 (mimic gather fp_vt=0)
RZ25 = "/home/ubuntu/rovl_2510/rovl_2510.bit"; RZ100 = "/home/ubuntu/resize_ovl/resize_ovl.bit"
WB = 64; AP = 0x00
RZREG = {"in":0x10, "y0":0x1c, "y1":0x28, "wy":0x34, "x0":0x40, "x1":0x4c, "wx":0x58, "out":0x64}
def t_fp(t): return t.get_attr("fix_point") if t.has_attr("fix_point") else None
def taps(o, i):
    a = np.zeros(o, np.uint16); b = np.zeros(o, np.uint16); w = np.zeros(o, np.uint16)
    for k in range(o):
        s = k * (i - 1) / (o - 1) if o > 1 else 0.0; f = int(np.floor(s))
        a[k] = f; b[k] = min(f + 1, i - 1); w[k] = int(round((s - f) * 256)) & 0x1ff
    return a, b, w
def resize(bit, x, fpi, fpo, hw, _cache={}):
    Hin, Win, C = x.shape[1:]; Ho, Wo = hw; NT = C // WB
    if _cache.get("bit") != bit:
        _cache["ol"] = Overlay(bit); _cache["bit"] = bit
        ip = None
        for nm in ("resize_0", "resize_bilinear_0"):
            ip = getattr(_cache["ol"], nm, None)
            if ip: break
        if ip is None: ip = getattr(_cache["ol"], [k for k in _cache["ol"].ip_dict if "resize" in k.lower()][0].split('/')[-1])
        _cache["ip"] = ip
    ip = _cache["ip"]
    packed = x[0].reshape(Hin, Win, NT, WB).transpose(2, 0, 1, 3)
    ib = allocate(shape=(NT*Hin*Win*WB,), dtype=np.uint8); ob = allocate(shape=(NT*Ho*Wo*WB,), dtype=np.uint8)
    ib[:] = packed.astype(np.int8).view(np.uint8).ravel(); ib.flush(); ob.flush()
    y0, y1, wy = taps(Ho, Hin); x0, x1, wx = taps(Wo, Win); tb = {}
    for nm, arr in (("y0",y0),("y1",y1),("wy",wy),("x0",x0),("x1",x1),("wx",wx)):
        bb = allocate(shape=arr.shape, dtype=np.uint16); bb[:] = arr; bb.flush(); tb[nm] = bb
    def sp(r, bu): a = bu.device_address; ip.write(r, a & 0xffffffff); ip.write(r+4, (a >> 32) & 0xffffffff)
    sp(RZREG["in"], ib); sp(RZREG["out"], ob)
    for nm in ("y0","y1","wy","x0","x1","wx"): sp(RZREG[nm], tb[nm])
    ip.write(AP, 1)
    while (ip.read(AP) & 0x2) == 0: pass
    o = np.frombuffer(bytes(ob), np.int8).reshape(NT, Ho, Wo, WB).transpose(1,2,0,3).reshape(1, Ho, Wo, C)
    if fpo != fpi: o = np.clip(np.round(o.astype(np.float32) * (2.0**(fpo-fpi))), -128, 127).astype(np.int8)
    ib.freebuffer(); ob.freebuffer()
    for bb in tb.values(): bb.freebuffer()
    return np.ascontiguousarray(o)

def run_chunk(frames_c, g_order, in_t, fp_in, live, W0, b0, W2, b2, vtclip=0):
    M = len(frames_c); st = [dict() for _ in range(M)]
    for k in range(M):
        vt = np.load(frames_c[k])["vt_out"].astype(np.float32)[None].transpose(0, 2, 3, 1)
        if vtclip:   # mimic the gather IP fp_vt=0 output: INT8 step 1.0, hard-clip at 127 (true vt reaches ~250)
            vt = np.clip(np.round(vt), -128, 127)
        st[k][in_t.name] = (np.clip(np.round(vt * 2.0**fp_in), -128, 127).astype(np.int8), fp_in)
    cur = None; dov = None
    for si, sg in enumerate(g_order):
        d = sg.get_attr("device") if sg.has_attr("device") else "?"
        if d == "DPU":
            if cur != "dpu":
                if dov is not None: del dov
                dov = DpuOverlay("dpu.bit"); cur = "dpu"
            r = vart.Runner.create_runner(sg, "run"); its = r.get_input_tensors(); ots = r.get_output_tensors()
            for k in range(M):
                ins = [np.ascontiguousarray(st[k][t.name][0].reshape([int(x) for x in t.dims]), np.int8) for t in its]
                outs = [np.empty([int(x) for x in t.dims], np.int8) for t in ots]
                jid = r.execute_async(ins, outs); r.wait(jid)
                for j, t in enumerate(ots): st[k][t.name] = (outs[j].copy(), t.get_attr("fix_point"))
            del r
        else:
            it = list(sg.get_input_tensors())[0]; ot = list(sg.get_output_tensors())[0]
            Hin = int(it.dims[1]); Ho = int(ot.dims[1])
            if Ho > Hin:
                bit = RZ25 if Ho == 100 else RZ100; cur = bit
                Wo = int(ot.dims[2]); fpo = t_fp(ot)
                for k in range(M):
                    x, fpx = st[k][it.name]
                    st[k][ot.name] = (resize(bit, x, fpx, fpo if fpo is not None else fpx, (Ho, Wo)), fpo if fpo is not None else fpx)
            else:
                fpo = t_fp(ot)
                for k in range(M):
                    xin, fpi = st[k][it.name]
                    if fpo is not None and fpo != fpi:
                        v = xin.astype(np.float32) * (2.0 ** (-fpi))
                        st[k][ot.name] = (np.clip(np.round(v * (2.0 ** fpo)), -128, 127).astype(np.int8), fpo)
                    else:
                        st[k][ot.name] = (xin, fpi)
        for nm, lu in live.items():
            if lu == si:
                for k in range(M): st[k].pop(nm, None)
    if dov is not None: del dov
    ot = list(g_order[-1].get_output_tensors())[0]
    res = np.empty((M, 200, 200, 16), np.uint8)
    for k in range(M):
        arr, fp = st[k][ot.name]
        conv = (arr[0].astype(np.float32) * 2.0**(-fp)).transpose(2, 0, 1)
        x = conv.transpose(2, 1, 0).reshape(-1, 256) @ W0.T + b0
        x = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
        res[k] = (x @ W2.T + b2).reshape(200, 200, 16, 18).argmax(-1).astype(np.uint8)
    return res

def main():
    frames = sorted(glob.glob(os.path.join(FR, "frame_*.npz")))[:N]
    M = len(frames); print("BEV-only CHUNKED %d frames | chunk %d | %s" % (M, CHUNK, os.path.basename(BEV_XM)), flush=True)
    ph = np.load(PRED); W0, b0, W2, b2 = ph["0.weight"], ph["0.bias"], ph["2.weight"], ph["2.bias"]
    g = xir.Graph.deserialize(BEV_XM); subs = g.get_root_subgraph().toposort_child_subgraph()
    in_t = list(subs[0].get_output_tensors())[0]; fp_in = t_fp(in_t); order = subs[1:]
    live = {in_t.name: -1}
    for si, sg in enumerate(order):
        for t in sg.get_input_tensors(): live[t.name] = si
    print("VTCLIP=%d (1=mimic gather fp_vt=0 clip@127)" % VTCLIP, flush=True)
    out = np.empty((M, 200, 200, 16), np.uint8); t0 = time.time()
    for cs in range(0, M, CHUNK):
        ce = min(cs + CHUNK, M)
        out[cs:ce] = run_chunk(frames[cs:ce], order, in_t, fp_in, live, W0, b0, W2, b2, VTCLIP)
        print("  chunk %d-%d done (%.0fs, %d/%d)" % (cs, ce-1, time.time()-t0, ce, M), flush=True)
    np.save(OUT, out); print("DONE %d -> %s in %.0fs" % (M, OUT, time.time()-t0), flush=True)

if __name__ == "__main__":
    main()
