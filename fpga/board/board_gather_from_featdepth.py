#!/usr/bin/env python3
"""Build-B step3 phase B: gather IP on the ON-BOARD DPU feat/depth -> vt_out, reconcile vs FP32 ref.

Reads featdepth_%04d.npz (from board_image_dpu.py) + frame_%04d.npz (gather ranks + vt_out ref).
Same validated gather recipe as Build-B step1: depth ap_ufixed<8,1> Q0.7 (round*128 unsigned),
feat ap_int<8> @fp_feat, out_scale=2^(fp_vt-fp_feat), bev.invalidate() before read. This closes the
FULL on-board front half: image(DPU) -> depth_net -> gather(IP) -> vt_out, vs the FP32 vt_out.

  sudo XILINX_XRT=/usr venv/python3 board_gather_from_featdepth.py <featdepth.npz> <frame.npz> <out_vt.npy>
"""
import sys, numpy as np
from pynq import Overlay, allocate
import pynq.pl_server.embedded_device as _ed, os as _os
_DX = _os.path.join(_os.path.dirname(_ed.__file__), "default.xclbin")
if _os.path.exists(_DX):
    _ed._create_xclbin = lambda m: open(_DX, "rb").read()

REG = {"feat":0x10, "depth":0x1c, "rank_depth":0x28, "rank_feat":0x34,
       "rank_bev":0x40, "interval_start":0x4c, "interval_len":0x58, "bev":0x6c}
OUT_SCALE = 0x64; AP_CTRL = 0x00
C, WB, NUM_BEV = 64, 64, 40000
BIT = "/home/ubuntu/gather_ovl/gather_ovl.bit"

def q8(x, fp): return np.clip(np.round(x * (2.0 ** fp)), -128, 127).astype(np.int8)

def main():
    FD = sys.argv[1]; FR = sys.argv[2]
    OUT = sys.argv[3] if len(sys.argv) > 3 else "/home/ubuntu/buildB/step3_vt.npy"
    fd = np.load(FD); fr = np.load(FR)
    feat = fd["feat"].astype(np.float32).reshape(-1, C)            # (4224,64)
    depth = fd["depth"].astype(np.float32).reshape(-1)            # (371712,)
    FEAT_VECS = feat.shape[0]; DEPTH_LEN = depth.shape[0]
    rdep = fr["ranks_depth"].astype(np.uint32); rfea = fr["ranks_feat"].astype(np.uint32)
    istart = fr["interval_starts"].astype(np.uint32); ilen = fr["interval_lengths"].astype(np.uint32)
    rbev = fr["ranks_bev"].astype(np.int64)[fr["interval_starts"].astype(np.int64)].astype(np.uint32)
    N_POINTS = rdep.shape[0]; N_PILLAR = istart.shape[0]
    vt_ref = fr["vt_out"].astype(np.float32).reshape(C, 200, 200)
    print("Np", N_POINTS, "Npil", N_PILLAR, "FEAT_VECS", FEAT_VECS, "DEPTH_LEN", DEPTH_LEN, flush=True)

    ol = Overlay(BIT)
    ip = getattr(ol, [k for k in ol.ip_dict if "gather" in k.lower()][0].split('/')[-1])
    bf = {}
    def buf(name, shape, dt, src=None):
        b = allocate(shape=shape, dtype=dt)
        if src is not None: b[:] = src
        b.flush(); bf[name] = b; return b
    fbuf = buf("feat", (FEAT_VECS*WB,), np.uint8)
    dbuf = buf("depth", (DEPTH_LEN,), np.uint8)
    buf("rank_depth", (N_POINTS,), np.uint32, rdep); buf("rank_feat", (N_POINTS,), np.uint32, rfea)
    buf("rank_bev", (N_PILLAR,), np.uint32, rbev)
    buf("interval_start", (N_PILLAR,), np.uint32, istart); buf("interval_len", (N_PILLAR,), np.uint32, ilen)
    bev = buf("bev", (NUM_BEV*WB,), np.uint8)
    def setptr(reg, b):
        a = b.device_address; ip.write(reg, a & 0xffffffff); ip.write(reg+4, (a >> 32) & 0xffffffff)
    for nm, reg in REG.items(): setptr(reg, bf[nm])

    dbuf[:] = np.clip(np.round(depth * 128.0), 0, 255).astype(np.uint8); dbuf.flush()
    refn = np.linalg.norm(vt_ref.ravel().astype(np.float64))
    best = None
    for ff in (1, 2, 3):
        fbuf[:] = q8(feat, ff).ravel().view(np.uint8); fbuf.flush()
        for fvt in (-1, 0, 1):
            osc = int(round(2.0 ** (fvt - ff) * 4096))
            if osc < 1: continue
            ip.write(OUT_SCALE, osc); ip.write(AP_CTRL, 1)
            while (ip.read(AP_CTRL) & 0x2) == 0: pass
            bev.invalidate()
            bev_i = np.array(bev).view(np.int8).reshape(NUM_BEV, WB)
            vt = (bev_i.astype(np.float32) * (2.0 ** (-fvt))).reshape(200, 200, C).transpose(2, 0, 1)
            a = vt.ravel().astype(np.float64); b = vt_ref.ravel().astype(np.float64)
            cos = float(a @ b / (np.linalg.norm(a) * refn + 1e-12))
            print("  fp_feat=%d fp_vt=%2d osc=%5d: cos %.4f" % (ff, fvt, osc, cos), flush=True)
            if best is None or cos > best[0]: best = (cos, ff, fvt, vt.copy())
    print("BEST image->vt_out cos %.4f at fp_feat=%d fp_vt=%d" % (best[0], best[1], best[2]), flush=True)
    np.save(OUT, best[3])

if __name__ == "__main__":
    main()
