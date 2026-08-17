#!/usr/bin/env python3
"""On-board latency-determinism probe for the custom view-transform gather IP on the KR260.

Loads the PS+gather overlay, fills DDR buffers (per-frame tensors + the five static index tables),
drives the deployable gather IP over its s_axilite register map, and times host-visible per-frame
latency over many runs -> the same determinism metrics as the resize IP and the DPU P1 probe.
The IP's WCET is input-invariant by construction (a fixed N_POINTS MACs + NUM_BEV writeback regardless
of the index VALUES), so we use valid synthetic index tables here: a real-shaped partition of the
points into pillars with in-range depth/feat/bev indices. This measures the real on-silicon determinism
of the gather datapath (host submission path + the fixed-cycle PL gather).

Run on the KR260:
  sudo /usr/local/share/pynq-venv/bin/python3 gather_onboard.py /home/ubuntu/gather_ovl/gather_ovl.bit
"""
import sys, time
import numpy as np
from pynq import Overlay, allocate

# This board's XRT 2.13 xclbinutil segfaults, so PYNQ's _create_xclbin (which packages a
# memory-topology xclbin for buffer allocation) fails for every overlay. The topology we need is
# just the PS DDR, which PYNQ already ships as default.xclbin -> patch _create_xclbin to return it.
import pynq.pl_server.embedded_device as _ed
import os as _os
_DX = _os.path.join(_os.path.dirname(_ed.__file__), "default.xclbin")
if _os.path.exists(_DX):
    _BYTES = open(_DX, "rb").read()
    _ed._create_xclbin = lambda _mem_dict: _BYTES
    print("patched _create_xclbin -> default.xclbin (xclbinutil segfault workaround)")

# xbev_gather_hw.h s_axilite register map (64-bit pointers: lo at REG, hi at REG+4)
AP_CTRL = 0x00
REG = {"feat":0x10, "depth":0x1c, "rank_depth":0x28, "rank_feat":0x34,
       "rank_bev":0x40, "interval_start":0x4c, "interval_len":0x58, "bev":0x6c}
OUT_SCALE = 0x64

# sizing (must match bev_gather.hpp, full rig)
N_POINTS, N_PILLAR, C = 302558, 21853, 64
DEPTH_LEN, FEAT_VECS, NUM_BEV = 371712, 4224, 40000
WB = C  # 512-bit wide word = 64 bytes

def main():
    bit  = sys.argv[1] if len(sys.argv) > 1 else "gather_ovl.bit"
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    ol = Overlay(bit)
    ip = getattr(ol, "gather_0", None) or getattr(ol, "bev_gather_0", None)
    if ip is None:
        key = [k for k in ol.ip_dict if "gather" in k.lower()][0].split('/')[-1]
        ip = getattr(ol, key)
    print("IP:", ip)

    # ---- DDR buffers ----
    feat = allocate(shape=(FEAT_VECS*WB,), dtype=np.uint8)
    depth= allocate(shape=(DEPTH_LEN,),    dtype=np.uint8)
    rdep = allocate(shape=(N_POINTS,),     dtype=np.uint32)
    rfea = allocate(shape=(N_POINTS,),     dtype=np.uint32)
    rbev = allocate(shape=(N_PILLAR,),     dtype=np.uint32)
    istart=allocate(shape=(N_PILLAR,),     dtype=np.uint32)
    ilen = allocate(shape=(N_PILLAR,),     dtype=np.uint32)
    bev  = allocate(shape=(NUM_BEV*WB,),   dtype=np.uint8)

    # valid synthetic tables: even partition of the points into pillars
    base, rem = N_POINTS // N_PILLAR, N_POINTS % N_PILLAR
    lens = np.full(N_PILLAR, base, np.uint32); lens[:rem] += 1
    starts = np.zeros(N_PILLAR, np.uint32); starts[1:] = np.cumsum(lens)[:-1]
    ilen[:] = lens; istart[:] = starts
    rdep[:] = (np.arange(N_POINTS) % DEPTH_LEN).astype(np.uint32)
    rfea[:] = (np.arange(N_POINTS) % FEAT_VECS).astype(np.uint32)
    rbev[:] = (np.arange(N_PILLAR) % NUM_BEV).astype(np.uint32)
    feat[:] = np.random.randint(0,256,size=feat.shape,dtype=np.uint8)
    depth[:]= np.random.randint(0,256,size=depth.shape,dtype=np.uint8)
    for b in (feat,depth,rdep,rfea,rbev,istart,ilen,bev): b.flush()

    def setptr(reg, buf):
        a = buf.device_address
        ip.write(reg, a & 0xffffffff); ip.write(reg+4, (a>>32) & 0xffffffff)
    setptr(REG["feat"],feat); setptr(REG["depth"],depth)
    setptr(REG["rank_depth"],rdep); setptr(REG["rank_feat"],rfea)
    setptr(REG["rank_bev"],rbev); setptr(REG["interval_start"],istart); setptr(REG["interval_len"],ilen)
    setptr(REG["bev"],bev)
    ip.write(OUT_SCALE, 4096)   # acc_t ap_fixed<24,12>: 1.0 == 1<<12

    for _ in range(5):  # warmup
        ip.write(AP_CTRL, 1)
        while (ip.read(AP_CTRL) & 0x2) == 0: pass
    t = np.empty(runs)
    for i in range(runs):
        t0 = time.perf_counter()
        ip.write(AP_CTRL, 1)
        while (ip.read(AP_CTRL) & 0x2) == 0: pass
        t[i] = time.perf_counter() - t0
    ms = t*1e3
    p50,p99,p999,mx = np.percentile(ms,50),np.percentile(ms,99),np.percentile(ms,99.9),ms.max()
    print(f"gather on-board ({runs} runs): mean {ms.mean():.3f} ms  p50 {p50:.3f}  p99 {p99:.3f}  "
          f"p99.9 {p999:.3f}  max {mx:.3f}  CV {ms.std()/ms.mean()*100:.2f}%  max/p50 {mx/p50:.3f}")
    np.save("/home/ubuntu/gather_onboard_lat.npy", ms)

if __name__ == "__main__":
    main()
