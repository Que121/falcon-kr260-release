#!/usr/bin/env python3
"""Steady-state all-PL BEV stage on ONE bitstream (dpu2rz) -- NO per-frame PL reconfig.

The swap-based walk (board_allpl_walk.py) proved the all-PL datapath is correct but reprograms the
PL between the DPU and resize stages. This driver loads the unified DPU+2-resize bitstream (dpu2rz)
ONCE and runs the whole BEV stage on it -- 3 DPU subgraphs (persistent VART runners) + the two
bilinear upsamples on resize_1 (25->100) and resize_0 (100->200), both fixed-tap IPs in the same
overlay -- so per-frame latency has zero reconfiguration. This is the certifiable steady-state WCET.

Pipeline: USER vt_out(200,200,64) -> DPU[1] -> {feat2(25),feat0(100)}
          -> resize_1 25->100 -> DPU[3] concat+neck -> resize_0 100->200 -> DPU[5] final -> cast

Run (sudo, pynq venv, XRT):  python3 board_allpl_steady.py [frames=200]
"""
import os, sys, time
os.environ["XILINX_XRT"] = "/usr"
import numpy as np
import xir
import pynq.pl_server.embedded_device as _ed
_DX = os.path.join(os.path.dirname(_ed.__file__), "default.xclbin")
if os.path.exists(_DX):
    _ed._create_xclbin = lambda m: open(_DX, "rb").read()

XM  = "/home/ubuntu/bev_allpl/bev_reluc.xmodel"
BIT = "/home/ubuntu/dpu2rz/dpu2rz.bit"
INP = "/home/ubuntu/bev/bev_test_input.npy"
REF = "/home/ubuntu/bev_allpl/allpl_convonly.npy"   # swap-based result, for correctness
AP_CTRL = 0x00
RZ_IN, RZ_OUT = 0x10, 0x1c                           # 64-bit ptrs (lo at REG, hi at REG+4)
WB = 64

def t_fp(t):
    return t.get_attr("fix_point") if t.has_attr("fix_point") else None

def run_resize(ip, x_nhwc, out_hw, fp_in, fp_out):
    from pynq import allocate
    _, Hin, Win, C = x_nhwc.shape; Hout, Wout = out_hw; NT = C // WB
    packed = x_nhwc[0].reshape(Hin, Win, NT, WB).transpose(2, 0, 1, 3)            # (NT,Hin,Win,64)
    ib = allocate(shape=(NT*Hin*Win*WB,),  dtype=np.uint8)
    ob = allocate(shape=(NT*Hout*Wout*WB,), dtype=np.uint8)
    ib[:] = packed.astype(np.int8).view(np.uint8).ravel(); ib.flush(); ob.flush()
    for reg, b in ((RZ_IN, ib), (RZ_OUT, ob)):
        a = b.device_address; ip.write(reg, a & 0xffffffff); ip.write(reg+4, (a >> 32) & 0xffffffff)
    ip.write(AP_CTRL, 1)
    while (ip.read(AP_CTRL) & 0x2) == 0: pass
    out = np.frombuffer(bytes(ob), np.int8).reshape(NT, Hout, Wout, WB).transpose(1, 2, 0, 3).reshape(1, Hout, Wout, C)
    if fp_out != fp_in:
        out = np.clip(np.round(out.astype(np.float32) * (2.0 ** (fp_out - fp_in))), -128, 127).astype(np.int8)
    del ib, ob
    return np.ascontiguousarray(out)

def main():
    import vart
    from pynq_dpu import DpuOverlay
    frames = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    ov = DpuOverlay(BIT)                                  # program PL ONCE, no further reconfig
    rz25, rz100 = ov.resize_1, ov.resize_0               # 25->100, 100->200
    g = xir.Graph.deserialize(XM)
    subs = g.get_root_subgraph().toposort_child_subgraph()
    dev = lambda s: s.get_attr("device") if s.has_attr("device") else "?"
    runners = {i: vart.Runner.create_runner(s, "run") for i, s in enumerate(subs) if dev(s) == "DPU"}
    print("DPU runners:", sorted(runners), "| resize: rz25, rz100 ready", flush=True)

    in_t = list(subs[0].get_output_tensors())[0]; fp_in = t_fp(in_t)
    vt = np.load(INP).astype(np.float32).transpose(0, 2, 3, 1)                    # (1,200,200,64)
    q0 = np.clip(np.round(vt * (2.0 ** fp_in)), -128, 127).astype(np.int8)

    def one_frame():
        store = {in_t.name: (q0, fp_in)}
        for i, s in enumerate(subs[1:], 1):
            if dev(s) == "DPU":
                r = runners[i]; its, ots = r.get_input_tensors(), r.get_output_tensors()
                ins = [np.ascontiguousarray(store[it.name][0].reshape([int(x) for x in it.dims]), np.int8) for it in its]
                outs = [np.empty([int(x) for x in ot.dims], np.int8) for ot in ots]
                jid = r.execute_async(ins, outs); r.wait(jid)
                for k, ot in enumerate(ots): store[ot.name] = (outs[k].copy(), ot.get_attr("fix_point"))
            else:
                it = list(s.get_input_tensors())[0]; ot = list(s.get_output_tensors())[0]
                xin, fpi = store[it.name]
                if int(ot.dims[1]) > int(it.dims[1]):                            # upsample
                    Hout, Wout = int(ot.dims[1]), int(ot.dims[2])
                    ip = rz25 if Hout == 100 else rz100
                    fpo = t_fp(ot) if t_fp(ot) is not None else fpi
                    store[ot.name] = (run_resize(ip, xin, (Hout, Wout), fpi, fpo), fpo)
                else:                                                            # trivial cast
                    store[ot.name] = (xin, fpi)
        ot = list(subs[-1].get_output_tensors())[0]
        return store[ot.name]

    arr, fp = one_frame()
    conv = (arr.astype(np.float32) * (2.0 ** (-fp))).transpose(0, 3, 1, 2)
    np.save("/home/ubuntu/bev_allpl/steady_convonly.npy", conv)
    if os.path.exists(REF):
        ref = np.load(REF).astype(np.float32)
        cos = float(conv.ravel() @ ref.ravel() / (np.linalg.norm(conv) * np.linalg.norm(ref) + 1e-9))
        print(f"correctness: steady vs swap-based conv cosine = {cos:.4f}", flush=True)

    lat = np.empty(frames)
    for i in range(frames):
        t0 = time.perf_counter(); one_frame(); lat[i] = (time.perf_counter() - t0) * 1000.0
    p50 = np.percentile(lat, 50)
    print(f"STEADY all-PL {frames} frames: mean {lat.mean():.2f}ms p50 {p50:.2f} "
          f"p99 {np.percentile(lat,99):.2f} p99.9 {np.percentile(lat,99.9):.2f} max {lat.max():.2f} "
          f"CV {100*lat.std()/lat.mean():.2f}% max/p50 {lat.max()/p50:.3f}", flush=True)
    np.save("/home/ubuntu/allpl_steady_lat.npy", lat)

if __name__ == "__main__":
    main()
