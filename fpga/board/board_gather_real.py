#!/usr/bin/env python3
"""Build-B step 1: gather IP on REAL data (lss_dump) -> reconcile vs the float vt_out reference.

Proves the on-board gather IP produces the right view-transform output from the REAL trunk/depth-net
feat+depth (not synthetic). Quantises lss_dump's float feat/depth to INT8 at power-of-2 fix-points,
drives the gather IP with the REAL rank/interval tables, dequantises the INT8 BEV output, and compares
to lss_dump['out'] (the float vt_out reference) by cosine + relative error.

  out_scale_reg = round(2**(fp_vt - fp_depth - fp_feat) * 4096)   # acc_t Q12.12 -> INT8 BEV @ fp_vt
  sudo XILINX_XRT=/usr venv/python3 board_gather_real.py [fp_feat=1 fp_depth=6 fp_vt=-1]
"""
import sys, numpy as np
from pynq import Overlay, allocate
import pynq.pl_server.embedded_device as _ed, os as _os
_DX = _os.path.join(_os.path.dirname(_ed.__file__), "default.xclbin")
if _os.path.exists(_DX):
    _ed._create_xclbin = lambda m: open(_DX, "rb").read()

AP_CTRL = 0x00
REG = {"feat":0x10, "depth":0x1c, "rank_depth":0x28, "rank_feat":0x34,
       "rank_bev":0x40, "interval_start":0x4c, "interval_len":0x58, "bev":0x6c}
OUT_SCALE = 0x64
N_POINTS, N_PILLAR, C = 302558, 21853, 64
DEPTH_LEN, FEAT_VECS, NUM_BEV, WB = 371712, 4224, 40000, 64
BIT = "/home/ubuntu/gather_ovl/gather_ovl.bit"

def q8(x, fp):
    return np.clip(np.round(x * (2.0 ** fp)), -128, 127).astype(np.int8)

def main():
    fp_feat = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    fp_dep  = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    fp_vt   = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    out_scale = int(round(2.0 ** (fp_vt - fp_dep - fp_feat) * 4096))
    print(f"fp_feat={fp_feat} fp_dep={fp_dep} fp_vt={fp_vt} -> out_scale_reg={out_scale}", flush=True)

    d = np.load("/home/ubuntu/lss_dump.npz")
    feat = d["feat"].reshape(FEAT_VECS, C)            # (4224,64) float
    depth = d["depth"].reshape(DEPTH_LEN)             # (371712,) float
    rdep = d["ranks_depth"].astype(np.uint32)
    rfea = d["ranks_feat"].astype(np.uint32)
    istart = d["interval_starts"].astype(np.uint32)
    ilen = d["interval_lengths"].astype(np.uint32)
    rbev_pt = d["ranks_bev"].astype(np.int64)         # per-point -> per-pillar at interval starts
    rbev = rbev_pt[d["interval_starts"].astype(np.int64)].astype(np.uint32)
    vt_ref = d["out"].reshape(C, 200, 200)            # float reference

    ol = Overlay(BIT)
    ip = getattr(ol, "gather_0", None) or getattr(ol, "bev_gather_0", None)
    if ip is None:
        ip = getattr(ol, [k for k in ol.ip_dict if "gather" in k.lower()][0].split('/')[-1])

    bf = {}
    def buf(name, shape, dt, src=None):
        b = allocate(shape=shape, dtype=dt)
        if src is not None: b[:] = src
        b.flush(); bf[name] = b; return b
    fbuf = buf("feat", (FEAT_VECS*WB,), np.uint8)
    dbuf = buf("depth", (DEPTH_LEN,), np.uint8)
    buf("rank_depth", (N_POINTS,), np.uint32, rdep)
    buf("rank_feat", (N_POINTS,), np.uint32, rfea)
    buf("rank_bev", (N_PILLAR,), np.uint32, rbev)
    buf("interval_start", (N_PILLAR,), np.uint32, istart)
    buf("interval_len", (N_PILLAR,), np.uint32, ilen)
    bev = buf("bev", (NUM_BEV*WB,), np.uint8)
    def setptr(reg, b):
        a = b.device_address; ip.write(reg, a & 0xffffffff); ip.write(reg+4, (a >> 32) & 0xffffffff)
    for nm, reg in REG.items(): setptr(reg, bf[nm])

    refn = np.linalg.norm(vt_ref.ravel().astype(np.float64))
    # depth_t = ap_ufixed<8,1> (Q0.7): raw = round(depth*128), UNSIGNED, IP reads it as the [0,1) float.
    dbuf[:] = np.clip(np.round(depth * 128.0), 0, 255).astype(np.uint8); dbuf.flush()
    # acc = 2^fp_feat * vt_float  ->  out_scale = 2^(fp_vt - fp_feat).  acc<=2048 => fp_feat<=3 for vt~195.
    best = None
    for ff in range(0, 4):
        fbuf[:] = q8(feat, ff).ravel().view(np.uint8); fbuf.flush()
        for fvt in (-2, -1, 0, 1):
            osc = int(round(2.0 ** (fvt - ff) * 4096))
            if osc < 1: continue
            ip.write(OUT_SCALE, osc)
            ip.write(AP_CTRL, 1)
            while (ip.read(AP_CTRL) & 0x2) == 0: pass
            bev.invalidate()
            bev_i = np.array(bev).view(np.int8).reshape(NUM_BEV, WB)
            vt = (bev_i.astype(np.float32) * (2.0 ** (-fvt))).reshape(200, 200, C).transpose(2, 0, 1)
            a = vt.ravel().astype(np.float64); b = vt_ref.ravel().astype(np.float64)
            cos = float(a @ b / (np.linalg.norm(a) * refn + 1e-12))
            print(f"  fp_feat={ff} fp_vt={fvt:2d} osc={osc:5d}: cos {cos:.4f}  "
                  f"recon|max {np.abs(vt).max():7.1f}  nz {(np.abs(vt)>1e-3).mean()*100:4.1f}%", flush=True)
            if best is None or cos > best[0]: best = (cos, ff, fvt, vt.copy())
    print(f"BEST: cos {best[0]:.4f} at fp_feat={best[1]} fp_vt={best[2]} (ref|max {np.abs(vt_ref).max():.1f})", flush=True)
    np.save("/home/ubuntu/gather_real_vt.npy", best[3])

if __name__ == "__main__":
    main()
