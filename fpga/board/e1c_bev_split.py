#!/usr/bin/env python3
"""E1: per-stage timing decomposition of the steady-state all-PL BEV stage (dpu2rz).

Same datapath as board_allpl_steady.py (one bitstream, zero per-frame reconfig), but every
subgraph is timed separately, and each resize call is split into host pack / IP kernel wait /
host unpack. Board power is sampled from the INA260 (hwmon) in a background thread.

  sudo ... python3 e1_bev_steady_stages.py <cond_name> [frames=500] [burners=0]

Writes /home/ubuntu/e1/<cond_name>_stages.npz with per-stage ms arrays + power trace.
"""
import os, sys, time, threading, subprocess
os.environ["XILINX_XRT"] = "/usr"
import numpy as np
import xir
import pynq.pl_server.embedded_device as _ed
_DX = os.path.join(os.path.dirname(_ed.__file__), "default.xclbin")
if os.path.exists(_DX):
    _B = open(_DX, "rb").read(); _ed._create_xclbin = lambda m: _B

XM  = "/home/ubuntu/bev_allpl/bev_reluc.xmodel"
BIT = "dpu.bit"
INP = "/home/ubuntu/bev/bev_test_input.npy"
AP_CTRL = 0x00
RZ_IN, RZ_OUT = 0x10, 0x1c
WB = 64
PWR = "/sys/class/hwmon/hwmon2/power1_input"   # INA260 u14, microwatts

def t_fp(t):
    return t.get_attr("fix_point") if t.has_attr("fix_point") else None

class PowerSampler(threading.Thread):
    def __init__(self, period=0.05):
        super().__init__(daemon=True); self.period = period; self.samples = []; self.on = True
    def run(self):
        while self.on:
            try: self.samples.append(int(open(PWR).read()))
            except Exception: pass
            time.sleep(self.period)
    def stop(self):
        self.on = False; self.join()
        return np.array(self.samples, dtype=np.float64) / 1e6   # W

def main():
    import vart
    from pynq_dpu import DpuOverlay
    from pynq import allocate
    cond = sys.argv[1] if len(sys.argv) > 1 else "idle"
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    nburn = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    os.makedirs("/home/ubuntu/e1", exist_ok=True)
    idle_pw = []
    for _ in range(40):
        idle_pw.append(int(open(PWR).read())); time.sleep(0.05)
    idle_w = float(np.median(idle_pw)) / 1e6

    burn = []
    if nburn:
        burn = [subprocess.Popen(["taskset", "-c", "0-2", "sh", "-c", "while true; do :; done"])
                for _ in range(nburn)]

    ov = DpuOverlay(BIT)
    rz25 = rz100 = None   # standard overlay: no resize IPs; CPU bilinear stands in (not part of the IP budget)
    g = xir.Graph.deserialize(XM)
    subs = g.get_root_subgraph().toposort_child_subgraph()
    dev = lambda s: s.get_attr("device") if s.has_attr("device") else "?"
    runners = {i: __import__("vart").Runner.create_runner(s, "run")
               for i, s in enumerate(subs) if dev(s) == "DPU"}

    in_t = list(subs[0].get_output_tensors())[0]; fp_in = t_fp(in_t)
    vt = np.load(INP).astype(np.float32).transpose(0, 2, 3, 1)
    q0 = np.clip(np.round(vt * (2.0 ** fp_in)), -128, 127).astype(np.int8)

    # persistent resize buffers per (shape) so allocation cost is not re-paid per frame;
    # a separate first-call record captures the allocation price once
    bufs = {}
    def run_resize_timed(ip, x_nhwc, out_hw, fp_i, fp_o, rec):
        _, Hin, Win, C = x_nhwc.shape; Hout, Wout = out_hw; NT = C // WB
        key = (Hin, Win, Hout, Wout)
        t0 = time.perf_counter()
        if key not in bufs:
            bufs[key] = (allocate(shape=(NT*Hin*Win*WB,), dtype=np.uint8),
                         allocate(shape=(NT*Hout*Wout*WB,), dtype=np.uint8))
        ib, ob = bufs[key]
        packed = x_nhwc[0].reshape(Hin, Win, NT, WB).transpose(2, 0, 1, 3)
        ib[:] = packed.astype(np.int8).view(np.uint8).ravel(); ib.flush(); ob.flush()
        for reg, b in ((RZ_IN, ib), (RZ_OUT, ob)):
            a = b.device_address; ip.write(reg, a & 0xffffffff); ip.write(reg+4, (a >> 32) & 0xffffffff)
        t1 = time.perf_counter()
        ip.write(AP_CTRL, 1)
        while (ip.read(AP_CTRL) & 0x2) == 0: pass
        t2 = time.perf_counter()
        out = np.frombuffer(bytes(ob), np.int8).reshape(NT, Hout, Wout, WB).transpose(1, 2, 0, 3).reshape(1, Hout, Wout, C)
        if fp_o != fp_i:
            out = np.clip(np.round(out.astype(np.float32) * (2.0 ** (fp_o - fp_i))), -128, 127).astype(np.int8)
        out = np.ascontiguousarray(out)
        t3 = time.perf_counter()
        rec.append((1000*(t1-t0), 1000*(t2-t1), 1000*(t3-t2)))   # pack+flush, kernel, unpack
        return out

    stage_names, per_stage = [], {}
    rz_detail = {"rz25": [], "rz100": []}

    def one_frame(record):
        store = {in_t.name: (q0, fp_in)}
        for i, s in enumerate(subs[1:], 1):
            t0 = time.perf_counter()
            if dev(s) == "DPU":
                r = runners[i]; its, ots = r.get_input_tensors(), r.get_output_tensors()
                ins = [np.ascontiguousarray(store[it.name][0].reshape([int(x) for x in it.dims]), np.int8) for it in its]
                outs = [np.empty([int(x) for x in ot.dims], np.int8) for ot in ots]
                te0 = time.perf_counter()
                jid = r.execute_async(ins, outs); r.wait(jid)
                te1 = time.perf_counter()
                for k, ot in enumerate(ots): store[ot.name] = (outs[k].copy(), ot.get_attr("fix_point"))
                label = "DPU[%d]" % i
                if record: per_stage.setdefault(label + ".exec", []).append(1000*(te1-te0))
            else:
                it = list(s.get_input_tensors())[0]; ot = list(s.get_output_tensors())[0]
                xin, fpi = store[it.name]
                if int(ot.dims[1]) > int(it.dims[1]):
                    Hout, Wout = int(ot.dims[1]), int(ot.dims[2])
                    tag = "rz25" if Hout == 100 else "rz100"
                    fpo = t_fp(ot) if t_fp(ot) is not None else fpi
                    x = xin.astype(np.float32)
                    sc = Hout // int(it.dims[1])
                    up = x.repeat(sc, axis=1).repeat(sc, axis=2)   # nearest stand-in for datapath continuity
                    if fpo != fpi:
                        up = np.clip(np.round(up * (2.0 ** (fpo - fpi))), -128, 127)
                    store[ot.name] = (np.ascontiguousarray(up.astype(np.int8)), fpo)
                    label = "cpu_upsample[%s]" % tag
                else:
                    store[ot.name] = (xin, fpi); label = "cast[%d]" % i
            if record:
                per_stage.setdefault(label, []).append(1000*(time.perf_counter()-t0))
                if label not in stage_names: stage_names.append(label)

    one_frame(False)   # warmup / first-call allocations
    ps = PowerSampler(); ps.start()
    total = np.empty(frames)
    for i in range(frames):
        t0 = time.perf_counter(); one_frame(True); total[i] = 1000*(time.perf_counter()-t0)
    pw = ps.stop()
    for p in burn: p.kill()

    print("== cond=%s frames=%d affinity=%s" % (cond, frames, sorted(os.sched_getaffinity(0))))
    print("%-14s %8s %8s %8s %8s" % ("stage", "p50", "p99", "max", "share%"))
    tot50 = np.percentile(total, 50)
    for n in stage_names:
        a = np.array(per_stage[n])
        print("%-14s %8.2f %8.2f %8.2f %7.1f%%" % (n, np.percentile(a,50), np.percentile(a,99), a.max(),
                                                   100*np.percentile(a,50)/tot50))
    for tag, rec in rz_detail.items():
        if rec:
            a = np.array(rec)
            print("resize[%s] split p50 ms: pack+flush %.2f | kernel %.2f | unpack %.2f" %
                  (tag, *np.percentile(a, 50, axis=0)))
    print("TOTAL p50 %.2f p99 %.2f max %.2f CV %.2f%% max/p50 %.3f" %
          (tot50, np.percentile(total,99), total.max(), 100*total.std()/total.mean(), total.max()/tot50))
    print("POWER idle %.2f W | run p50 %.2f W | run max %.2f W (n=%d)" %
          (idle_w, np.median(pw), pw.max(), len(pw)))
    np.savez("/home/ubuntu/e1/%s_stages.npz" % cond,
             total=total, power=pw, idle_w=idle_w,
             **{("s_"+n.replace("[","_").replace("]","")): np.array(v) for n, v in per_stage.items()},
             **{("rzd_"+k): np.array(v) for k, v in rz_detail.items() if v})

if __name__ == "__main__":
    main()
