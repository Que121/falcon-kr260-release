#!/usr/bin/env python3
"""E4: minimal on-board watchdog enforcement prototype 

Loop: the BEV-stage DPU pipeline (3 subgraphs on the standard dpu.bit overlay) runs each frame
under a hard deadline D. A monitor thread arms a timer per frame; if the frame is not complete at
the deadline, it delivers the FALLBACK immediately: the previous frame's output ego-warped by a
constant-velocity SE2 (nearest, precomputed during the previous frame's slack), tagged stale.
Measured quantities:
  - enforcement latency: deadline expiry -> fallback delivered (should be a buffer publish, ~us-ms)
  - fallback preparation cost: the ego-warp of a (200,200,C) int8 map on the A53
  - trigger statistics under injected host interference (a CPU spike thread), and the delivered
    output timeline: every frame gets AN output at or before D + enforcement latency.

  sudo ... python3 e4_watchdog_proto.py <deadline_ms> [frames=2000] [spike_every=50]
"""
import os, sys, time, threading
os.environ["XILINX_XRT"] = "/usr"
import numpy as np
import xir
import pynq.pl_server.embedded_device as _ed
_DX = os.path.join(os.path.dirname(_ed.__file__), "default.xclbin")
if os.path.exists(_DX):
    _B = open(_DX, "rb").read(); _ed._create_xclbin = lambda m: _B

XM  = "/home/ubuntu/bev_allpl/bev_reluc.xmodel"
INP = "/home/ubuntu/bev/bev_test_input.npy"
GRID = 200

def main():
    import vart
    from pynq_dpu import DpuOverlay
    D = float(sys.argv[1])
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    spike_every = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    ov = DpuOverlay("dpu.bit")
    g = xir.Graph.deserialize(XM)
    subs = g.get_root_subgraph().toposort_child_subgraph()
    dev = lambda s: s.get_attr("device") if s.has_attr("device") else "?"
    runners = {i: vart.Runner.create_runner(s, "run") for i, s in enumerate(subs) if dev(s) == "DPU"}
    in_t = list(subs[0].get_output_tensors())[0]
    fp_in = in_t.get_attr("fix_point") if in_t.has_attr("fix_point") else 4
    vt = np.load(INP).astype(np.float32).transpose(0, 2, 3, 1)
    q0 = np.clip(np.round(vt * (2.0 ** fp_in)), -128, 127).astype(np.int8)

    # constant-velocity SE2 warp index table (precomputed once here; per-frame in a real stack)
    v, dt = 5.0, 0.5           # 5 m/s forward, one 2 Hz keyframe of staleness
    cell = 80.0 / GRID
    shift = int(round(v * dt / cell))
    def ego_warp(x):
        out = np.zeros_like(x)
        if shift > 0:
            out[:-shift or None] = x[shift:]
        else:
            out[:] = x
        return out

    # frame kernel: run the 3 DPU subgraphs sequentially (host marshaling minimal)
    def run_frame():
        store = {in_t.name: q0}
        for i, s in enumerate(subs[1:], 1):
            if dev(s) != "DPU":
                it = list(s.get_input_tensors())[0]; ot = list(s.get_output_tensors())[0]
                x = store[it.name]
                if int(ot.dims[1]) > int(it.dims[1]):
                    sc = int(ot.dims[1]) // int(it.dims[1])
                    x = x.repeat(sc, axis=1).repeat(sc, axis=2)
                store[ot.name] = np.ascontiguousarray(x)
                continue
            r = runners[i]; its, ots = r.get_input_tensors(), r.get_output_tensors()
            ins = [np.ascontiguousarray(store[it.name].reshape([int(d) for d in it.dims]), np.int8) for it in its]
            outs = [np.empty([int(d) for d in ot.dims], np.int8) for ot in ots]
            jid = r.execute_async(ins, outs); r.wait(jid)
            for k, ot in enumerate(ots): store[ot.name] = outs[k]
        return store[list(subs[-1].get_output_tensors())[0].name]

    # measure fallback preparation cost standalone
    dummy = np.zeros((GRID, GRID, 16), np.int8)
    wp = []
    for _ in range(200):
        t0 = time.perf_counter(); ego_warp(dummy); wp.append(1000 * (time.perf_counter() - t0))
    wp = np.array(wp)

    delivered = np.empty(frames)        # time from frame start to SOME output being published
    enforce_lat = []                    # deadline expiry -> fallback publish
    triggers = 0
    last_out = {"buf": ego_warp(dummy), "stale": True}
    publish_lock = threading.Lock()

    def spike():
        end = time.time() + 0.08
        while time.time() < end: pass

    for i in range(frames):
        if spike_every and i % spike_every == spike_every - 1:
            threading.Thread(target=spike, daemon=True).start()
        fallback = ego_warp(last_out["buf"])          # prepared up front (previous slack in a real stack)
        published = {"t": None, "stale": None}
        t0 = time.perf_counter()
        def enforcer():
            slack = D / 1000.0 - (time.perf_counter() - t0)
            if slack > 0: time.sleep(slack)
            with publish_lock:
                if published["t"] is None:
                    te = time.perf_counter()
                    published["t"], published["stale"] = te, True
                    last_out["buf"] = fallback
                    enforce_lat.append(1000 * (te - t0) - D)
        th = threading.Timer(0, enforcer); th.start()
        out = run_frame()
        with publish_lock:
            if published["t"] is None:
                published["t"], published["stale"] = time.perf_counter(), False
                last_out["buf"] = out
        th.join()
        if published["stale"]: triggers += 1
        delivered[i] = 1000 * (published["t"] - t0)

    print("frames %d deadline %.1f ms | watchdog triggers %d (%.2f%%)" %
          (frames, D, triggers, 100 * triggers / frames))
    print("fallback ego-warp prep: p50 %.3f ms max %.3f ms" % (np.percentile(wp, 50), wp.max()))
    if enforce_lat:
        el = np.array(enforce_lat)
        print("enforcement latency past deadline: p50 %.3f ms p99 %.3f max %.3f" %
              (np.percentile(el, 50), np.percentile(el, 99), el.max()))
    print("delivered-output time: p50 %.2f p99 %.2f max %.2f ms (deadline+enforcement bound: %.2f)" %
          (np.percentile(delivered, 50), np.percentile(delivered, 99), delivered.max(),
           D + (np.array(enforce_lat).max() if enforce_lat else 0)))
    np.savez("/home/ubuntu/e1/watchdog_D%d.npz" % int(D),
             delivered=delivered, warp=wp, enforce=np.array(enforce_lat), triggers=triggers)

if __name__ == "__main__":
    main()
