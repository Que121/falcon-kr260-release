#!/usr/bin/env python3
"""Semi-analytic bound kit: how DDR, DMA, and the CPU tail are bounded.

(a) Deployed gather path split: per-frame input flush / ap_start..ap_done kernel window (PL compute
    + DDR table streaming) / output readback, idle and under load+iso.
(b) Static byte ledger: every DDR-resident table and stream is a compile-time constant; printed so
    WCET_gather <= cycles/f + bytes/BW_min composes analytically.
(c) Stressed DDR bandwidth floor: sustained big-block copy bandwidth on the isolated core while
    3 memory-hog burners saturate the other cores. The MINIMUM observed sustained bandwidth is the
    conservative BW_min for (b).
(d) ARM predicter tail under load: the fixed-size Linear-Softplus-Linear head on the A53, worst
    case over repeated runs with burners active.

  sudo ... python3 e6_semianalytic.py [runs=2000]
"""
import os, sys, time, subprocess, signal
import numpy as np
from pynq import Overlay, allocate
import pynq.pl_server.embedded_device as _ed
_DX = os.path.join(os.path.dirname(_ed.__file__), "default.xclbin")
if os.path.exists(_DX):
    _B = open(_DX, "rb").read(); _ed._create_xclbin = lambda _m: _B

AP = 0x00
REG = {"feat":0x10, "depth":0x1c, "rank_depth":0x28, "rank_feat":0x34,
       "rank_bev":0x40, "interval_start":0x4c, "interval_len":0x58, "bev":0x6c}
OUT_SCALE = 0x64
N_POINTS, N_PILLAR, C = 302558, 21853, 64
DEPTH_LEN, FEAT_VECS, NUM_BEV = 371712, 4224, 64 * 625
WB = C
NUM_BEV = 40000
BIT = "/home/ubuntu/gather_ovl_i16/gather_ovl_i16.bit"
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000

def burners_mem(n, cores):
    procs = []
    for k in range(n):
        code = "import numpy as np\na=np.zeros(1<<24,np.uint8)\nb=np.zeros(1<<24,np.uint8)\nwhile True: b[:]=a"
        procs.append(subprocess.Popen(["taskset", "-c", str(cores[k % len(cores)]), "python3", "-c", code],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    return procs

def kill(procs):
    for p in procs:
        try: p.send_signal(signal.SIGKILL)
        except Exception: pass

def stats(tag, ms):
    ms = np.asarray(ms)
    print("[%s] n=%d p50 %.3f p99 %.3f p99.9 %.3f max %.3f ms" %
          (tag, len(ms), np.percentile(ms,50), np.percentile(ms,99),
           np.percentile(ms,99.9), ms.max()), flush=True)
    return ms

def main():
    ncpu = os.cpu_count(); iso = ncpu - 1
    # ---------- (b) static byte ledger ----------
    ledger = {
        "rank_depth": N_POINTS*4, "rank_feat": N_POINTS*4, "rank_bev": N_PILLAR*4,
        "interval_start": N_PILLAR*4, "interval_len": N_PILLAR*4,
        "depth_stream": DEPTH_LEN, "feat_stream": FEAT_VECS*WB, "bev_out": NUM_BEV*WB*2,
    }
    tot = sum(ledger.values())
    print("STATIC BYTE LEDGER (per frame, compile-time constants):", flush=True)
    for k, v in ledger.items(): print("  %-15s %10d B" % (k, v), flush=True)
    print("  TOTAL           %10d B = %.2f MB" % (tot, tot/1e6), flush=True)

    # ---------- (a) gather split ----------
    ol = Overlay(BIT)
    key = [k for k in ol.ip_dict if "gather" in k.lower()][0].split('/')[-1]
    ip = getattr(ol, key); print("IP:", key, flush=True)
    bufs = {}
    def mk(name, shape, dt, src=None):
        b = allocate(shape=shape, dtype=dt)
        if src is not None: b[:] = src
        b.flush(); bufs[name] = b; return b
    base, rem = N_POINTS // N_PILLAR, N_POINTS % N_PILLAR
    lens = np.full(N_PILLAR, base, np.uint32); lens[:rem] += 1
    starts = np.zeros(N_PILLAR, np.uint32); starts[1:] = np.cumsum(lens)[:-1]
    mk("feat",(FEAT_VECS*WB,),np.uint8,np.random.randint(0,256,FEAT_VECS*WB,np.uint8))
    mk("depth",(DEPTH_LEN,),np.uint8,np.random.randint(0,256,DEPTH_LEN,np.uint8))
    mk("rank_depth",(N_POINTS,),np.uint32,(np.arange(N_POINTS)%DEPTH_LEN).astype(np.uint32))
    mk("rank_feat",(N_POINTS,),np.uint32,(np.arange(N_POINTS)%FEAT_VECS).astype(np.uint32))
    mk("rank_bev",(N_PILLAR,),np.uint32,(np.arange(N_PILLAR)%NUM_BEV).astype(np.uint32))
    mk("interval_start",(N_PILLAR,),np.uint32,starts)
    mk("interval_len",(N_PILLAR,),np.uint32,lens)
    mk("bev",(NUM_BEV*WB*2,),np.uint8)
    for nm, reg in REG.items():
        a = bufs[nm].device_address; ip.write(reg, a & 0xffffffff); ip.write(reg+4,(a>>32)&0xffffffff)
    ip.write(OUT_SCALE, 2097152)

    def one_split():
        t0 = time.perf_counter()
        bufs["feat"].flush(); bufs["depth"].flush()          # per-frame inputs
        t1 = time.perf_counter()
        ip.write(AP, 1)
        while (ip.read(AP) & 0x2) == 0: pass
        t2 = time.perf_counter()
        bufs["bev"].invalidate(); _ = np.frombuffer(bytes(bufs["bev"][:4096]), np.int16).sum()
        t3 = time.perf_counter()
        return (1e3*(t1-t0), 1e3*(t2-t1), 1e3*(t3-t2))

    for _ in range(5): one_split()
    for cond in ("idle", "load_iso"):
        b = []
        if cond == "load_iso":
            b = burners_mem(3, list(range(ncpu-1))); time.sleep(2)
            try: os.sched_setaffinity(0, {iso})
            except Exception as e: print("affinity:", e, flush=True)
        rec = np.array([one_split() for _ in range(RUNS)])
        stats("gather_prep_%s" % cond, rec[:,0])
        stats("gather_kernel_%s" % cond, rec[:,1])
        stats("gather_readback_%s" % cond, rec[:,2])
        np.save("/home/ubuntu/e1/gather_split_%s.npy" % cond, rec)
        kill(b)

    # ---------- (c) stressed DDR bandwidth floor ----------
    b = burners_mem(3, list(range(ncpu-1))); time.sleep(2)
    try: os.sched_setaffinity(0, {iso})
    except Exception: pass
    src = np.zeros(1 << 26, np.uint8); dst = np.zeros(1 << 26, np.uint8)   # 64 MB
    bw = []
    for _ in range(60):
        t0 = time.perf_counter(); dst[:] = src; dt = time.perf_counter() - t0
        bw.append((len(src) * 2 / dt) / 1e9)   # read+write bytes
    kill(b)
    bw = np.array(bw)
    print("DDR BW under stress (64MB copy, r+w): p50 %.2f GB/s min %.2f GB/s" %
          (np.percentile(bw, 50), bw.min()), flush=True)
    print("SEMI-ANALYTIC: table+stream %.2f MB / BW_min %.2f GB/s = %.2f ms transfer ceiling; "
          "+ compute 4.7 ms (cycle count) => gather bound %.2f ms" %
          (tot/1e6, bw.min(), 1e3*tot/ (bw.min()*1e9), 1e3*tot/(bw.min()*1e9) + 4.7), flush=True)

    # ---------- (d) ARM predicter under load ----------
    z = np.load("/home/ubuntu/bev/predicter_head.npz")
    keys = sorted(z.files)
    print("predicter npz keys:", keys, flush=True)
    Ws = [z[k].astype(np.float32) for k in keys if z[k].ndim == 2]
    bs = [z[k].astype(np.float32) for k in keys if z[k].ndim == 1]
    if len(Ws) >= 2:
        W1, W2 = Ws[0], Ws[1]
        b1 = bs[0] if bs else np.zeros(W1.shape[-1] if W1.shape[-1] != NUM_BEV else W1.shape[0], np.float32)
        cin = W1.shape[0] if W1.shape[0] != W1.shape[1] else W1.shape[1]
        try:
            x = np.random.randn(NUM_BEV, W1.shape[0]).astype(np.float32)
            bb = burners_mem(3, list(range(ncpu-1))); time.sleep(2)
            ts = []
            for _ in range(50):
                t0 = time.perf_counter()
                h = x @ W1
                h = np.log1p(np.exp(np.clip(h, -30, 30)))
                _ = h @ W2
                ts.append(1e3*(time.perf_counter()-t0))
            kill(bb)
            stats("predicter_load", ts)
        except Exception as e:
            print("predicter timing failed:", e, flush=True)
    print("SEMIANALYTIC_DONE", flush=True)

if __name__ == "__main__":
    main()
