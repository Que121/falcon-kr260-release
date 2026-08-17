#!/usr/bin/env python3
"""Sanity: load the INT16-vt gather overlay, run on AdaQuant featdepth (frame 0), reconstruct vt at
fp_vt=7 (int16 * 2^-7), cos vs FP32 vt_out. Verifies the new bitstream loads + the INT16 scale is right.
  sudo .../python3 board_gather_i16_test.py <featdepth.npz> <frame.npz>
"""
import sys, numpy as np
from pynq import Overlay, allocate
import pynq.pl_server.embedded_device as _ed, os as _os
_DX = _os.path.join(_os.path.dirname(_ed.__file__), "default.xclbin")
if _os.path.exists(_DX): _ed._create_xclbin = lambda m: open(_DX, "rb").read()

REG = {"feat":0x10, "depth":0x1c, "rank_depth":0x28, "rank_feat":0x34,
       "rank_bev":0x40, "interval_start":0x4c, "interval_len":0x58, "bev":0x6c}
OUT_SCALE = 0x64; AP = 0x00
C, WB, NUM_BEV = 64, 64, 40000
BIT = "/home/ubuntu/gather_ovl_i16/gather_ovl_i16.bit"

FD = sys.argv[1]; FR = sys.argv[2]
fd = np.load(FD); fr = np.load(FR)
feat = fd["feat"].astype(np.float32).reshape(-1, C); depth = fd["depth"].astype(np.float32).reshape(-1)
rdep = fr["ranks_depth"].astype(np.uint32); rfea = fr["ranks_feat"].astype(np.uint32)
ist = fr["interval_starts"].astype(np.uint32); iln = fr["interval_lengths"].astype(np.uint32)
rbev = fr["ranks_bev"].astype(np.int64)[fr["interval_starts"].astype(np.int64)].astype(np.uint32)
vt_ref = fr["vt_out"].astype(np.float32).reshape(C, 200, 200)
FV = feat.shape[0]; DL = depth.shape[0]; NP = rdep.shape[0]; NPI = ist.shape[0]

ol = Overlay(BIT); print("overlay loaded:", [k for k in ol.ip_dict], flush=True)
ip = getattr(ol, [k for k in ol.ip_dict if "gather" in k.lower()][0].split('/')[-1])
bf = {}
def buf(name, shape, dt, src=None):
    b = allocate(shape=shape, dtype=dt)
    if src is not None: b[:] = src
    b.flush(); bf[name] = b; return b
buf("feat", (FV*WB,), np.uint8, np.clip(np.round(feat*4.0), -128, 127).astype(np.int8).ravel().view(np.uint8))
buf("depth", (DL,), np.uint8, np.clip(np.round(depth*128.0), 0, 255).astype(np.uint8))
buf("rank_depth", (NP,), np.uint32, rdep); buf("rank_feat", (NP,), np.uint32, rfea)
buf("rank_bev", (NPI,), np.uint32, rbev)
buf("interval_start", (NPI,), np.uint32, ist); buf("interval_len", (NPI,), np.uint32, iln)
buf("bev", (NUM_BEV*WB*2,), np.uint8)            # INT16: 128 bytes/cell
for nm, reg in REG.items():
    a = bf[nm].device_address; ip.write(reg, a & 0xffffffff); ip.write(reg+4, (a >> 32) & 0xffffffff)
ip.write(OUT_SCALE, 2097152); ip.write(AP, 1)
while (ip.read(AP) & 0x2) == 0: pass
bf["bev"].invalidate()
vt = np.array(bf["bev"]).view(np.int16).reshape(NUM_BEV, WB).astype(np.float32) * (2.0**-7)
vt = vt.reshape(200, 200, C).transpose(2, 0, 1)
a = vt.ravel().astype(np.float64); b = vt_ref.ravel().astype(np.float64)
cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
print("INT16 gather vt: cos %.4f vs FP32 | vt absmax %.2f (ref %.2f) | clip@127? max>127: %s"
      % (cos, np.abs(vt).max(), np.abs(vt_ref).max(), bool(np.abs(vt).max() > 127)), flush=True)
print("I16_TEST_DONE", flush=True)
