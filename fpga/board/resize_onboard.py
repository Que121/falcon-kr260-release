#!/usr/bin/env python3
"""On-board latency-determinism probe for the custom resize IP on the KR260.

Loads the PS+resize overlay, drives the IP over its s_axilite register map (MMIO), and times
host-visible per-frame latency over many runs -> the same determinism metrics as the DPU P1 probe
(p50/p99/p99.9/max, CV, max/p50). The IP's WCET is input-invariant by construction, so this measures
the real on-silicon determinism (host submission path + the fixed-cycle PL datapath).

Run on the KR260 (needs root for the FPGA manager / /dev/mem):
  sudo /usr/local/share/pynq-venv/bin/python3 resize_onboard.py /home/ubuntu/resize_ovl/resize_ovl.bit
"""
import sys, time
import numpy as np
from pynq import Overlay, allocate

# resize_deploy.cpp s_axilite register map (from xresize_bilinear_hw.h)
AP_CTRL = 0x00
REG = {"in":0x10, "y0":0x1c, "y1":0x28, "wy":0x34, "x0":0x40, "x1":0x4c, "wx":0x58, "out":0x64}

# UP2 instance sizing (must match resize.hpp)
NTILE, HIN, WIN, HOUT, WOUT = 8, 100, 100, 200, 200
WB = 64  # 512-bit wide word = 64 bytes

def taps(out_n, in_n):
    # align_corners=True upsampling: src = o*(in-1)/(out-1)
    i0 = np.zeros(out_n, np.uint16); i1 = np.zeros(out_n, np.uint16); w = np.zeros(out_n, np.uint16)
    for o in range(out_n):
        s = o*(in_n-1)/(out_n-1) if out_n > 1 else 0.0
        f = int(np.floor(s)); f1 = min(f+1, in_n-1); frac = s-f
        i0[o] = f; i1[o] = f1; w[o] = int(round(frac*256)) & 0x1ff   # ap_ufixed<9,1>, 8 frac bits
    return i0, i1, w

def main():
    bit = sys.argv[1] if len(sys.argv) > 1 else "resize_ovl.bit"
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    ol = Overlay(bit)
    # locate the IP (DefaultIP -> MMIO via .read/.write)
    ip = None
    for name in ("resize_0", "resize_bilinear_0"):
        ip = getattr(ol, name, None)
        if ip is not None: break
    if ip is None:
        key = [k for k in ol.ip_dict if "resize" in k.lower()][0]
        ip = ol.ip_dict[key]  # fall back; use ol.<key> if available
        ip = getattr(ol, key.split('/')[-1], ip)
    print("IP:", ip)

    in_buf  = allocate(shape=(NTILE*HIN*WIN*WB,),  dtype=np.uint8)
    out_buf = allocate(shape=(NTILE*HOUT*WOUT*WB,), dtype=np.uint8)
    y0,y1,wy = taps(HOUT, HIN); x0,x1,wx = taps(WOUT, WIN)
    bufs = {}
    for nm, arr in (("y0",y0),("y1",y1),("wy",wy),("x0",x0),("x1",x1),("wx",wx)):
        b = allocate(shape=arr.shape, dtype=np.uint16); b[:] = arr; b.flush(); bufs[nm]=b
    in_buf[:] = np.random.randint(0,256,size=in_buf.shape,dtype=np.uint8); in_buf.flush()
    out_buf.flush()

    # 64-bit pointers: each arg is lo at REG[nm], hi at REG[nm]+4 (DDR < 4 GB so hi = 0)
    ip.write(REG["in"],  in_buf.device_address & 0xffffffff); ip.write(REG["in"]+4,  in_buf.device_address>>32)
    ip.write(REG["out"], out_buf.device_address & 0xffffffff); ip.write(REG["out"]+4, out_buf.device_address>>32)
    for nm in ("y0","y1","wy","x0","x1","wx"):
        a = bufs[nm].device_address
        ip.write(REG[nm], a & 0xffffffff)
        ip.write(REG[nm]+4, (a>>32) & 0xffffffff)

    # warmup
    for _ in range(5):
        ip.write(AP_CTRL, 1)
        while (ip.read(AP_CTRL) & 0x2) == 0: pass
    # timed runs
    t = np.empty(runs)
    for i in range(runs):
        t0 = time.perf_counter()
        ip.write(AP_CTRL, 1)                       # ap_start
        while (ip.read(AP_CTRL) & 0x2) == 0: pass  # poll ap_done
        t[i] = time.perf_counter() - t0
    ms = t*1e3
    p50,p99,p999,mx = np.percentile(ms,50),np.percentile(ms,99),np.percentile(ms,99.9),ms.max()
    print(f"resize on-board ({runs} runs): mean {ms.mean():.3f} ms  p50 {p50:.3f}  p99 {p99:.3f}  "
          f"p99.9 {p999:.3f}  max {mx:.3f}  CV {ms.std()/ms.mean()*100:.2f}%  max/p50 {mx/p50:.3f}")
    np.save("/home/ubuntu/resize_onboard_lat.npy", ms)

if __name__ == "__main__":
    main()
