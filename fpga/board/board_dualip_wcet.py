#!/usr/bin/env python3
"""#2 -- on-board WCET of the deployed INT16 gather IP under interference (idle / +3 CPU burners /
+3 burners with whole-process core-isolation), the leg-3 RT-isolation story extended to the gather IP
(the paper already has it for the DPU and the resize IP). The gather IP's PL datapath is fixed-cycle /
input-invariant; what a co-tenant load perturbs is the host submission/poll thread, and core-isolation
(taskset) collapses the tail back to idle -- the same mechanism and fix as the DPU host path.

Combined dual-IP serial WCET = gather + resize, both characterised under the same conditions (resize
numbers from resize_onboard.py / the paper). Saves per-condition latency arrays for the figure/table.

  sudo /usr/local/share/pynq-venv/bin/python3 board_dualip_wcet.py <gather_i16.bit> [runs=3000]
"""
import sys, time, os, subprocess, signal
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
DEPTH_LEN, FEAT_VECS, NUM_BEV = 371712, 4224, 40000
WB = C

BIT  = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/gather_ovl_i16/gather_ovl_i16.bit"
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000

def spawn_burners(n=3, pin=None):
    # pin = list of cores to taskset burners onto (explicit, so they do NOT inherit the parent's
    # affinity -- the bug that put burners on the measurement core). None = unpinned (float).
    procs = []
    for k in range(n):
        cmd = ["python3", "-c", "x=0\nwhile True: x+=1"]
        if pin is not None:
            cmd = ["taskset", "-c", str(pin[k % len(pin)])] + cmd
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(p)
    return procs

def kill_burners(procs):
    for p in procs:
        try: p.send_signal(signal.SIGKILL)
        except Exception: pass
    for p in procs:
        try: p.wait(timeout=2)
        except Exception: pass

def measure(ip, runs):
    for _ in range(5):
        ip.write(AP, 1)
        while (ip.read(AP) & 0x2) == 0: pass
    t = np.empty(runs)
    for i in range(runs):
        t0 = time.perf_counter()
        ip.write(AP, 1)
        while (ip.read(AP) & 0x2) == 0: pass
        t[i] = time.perf_counter() - t0
    return t * 1e3

def report(tag, ms):
    p50 = np.percentile(ms, 50); mx = ms.max()
    print("[%s] n=%d mean %.3f ms | p50 %.3f | p99 %.3f | p99.9 %.3f | max %.3f | CV %.3f%% | max/p50 %.3f"
          % (tag, len(ms), ms.mean(), p50, np.percentile(ms,99), np.percentile(ms,99.9), mx,
             ms.std()/ms.mean()*100, mx/p50), flush=True)
    np.save("/home/ubuntu/gather_wcet_%s.npy" % tag, ms)

def main():
    ol = Overlay(BIT)
    key = [k for k in ol.ip_dict if "gather" in k.lower()][0].split('/')[-1]
    ip = getattr(ol, key); print("IP:", key, flush=True)
    # buffers (deployed INT16 gather)
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
    mk("bev",(NUM_BEV*WB*2,),np.uint8)             # INT16 output
    for nm,reg in REG.items():
        a = bufs[nm].device_address; ip.write(reg, a & 0xffffffff); ip.write(reg+4,(a>>32)&0xffffffff)
    ip.write(OUT_SCALE, 2097152)

    ncpu = os.cpu_count()
    others = list(range(ncpu - 1))   # cores 0..ncpu-2 for burners
    isocore = ncpu - 1               # last core reserved for the measurement under iso
    # 1) idle
    report("idle", measure(ip, RUNS))
    # 2) load (3 burners, no isolation -- float on all cores incl. the measurement's)
    b = spawn_burners(3); time.sleep(2)
    report("load", measure(ip, RUNS)); kill_burners(b)
    # 3) load + core isolation: burners EXPLICITLY taskset to cores 0..n-2, measurement pinned to core n-1
    b = spawn_burners(3, pin=others); time.sleep(2)
    try: os.sched_setaffinity(0, {isocore})
    except Exception as e: print("affinity set failed:", e, flush=True)
    report("load_iso", measure(ip, RUNS)); kill_burners(b)
    print("DUALIP_WCET_DONE", flush=True)

if __name__ == "__main__":
    main()
