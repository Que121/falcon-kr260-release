#!/usr/bin/env python3
"""E5: the delivery-timeline run. Six-inference DPU frame-budget probe under a HARD absolute
100 ms deadline with the watchdog enforcing delivery: raw inference latency AND delivered-output
latency recorded per frame over 100k frames. Expected: rare raw spikes past 100 ms (3.2e-4 from
E2+), zero delivered outputs past deadline + enforcement latency.

  sudo ... taskset -c 3 python3 e5_delivery_100k.py [deadline_ms=100] [frames=100000] [burners=3]
"""
import os, sys, time, threading, subprocess
os.environ["XILINX_XRT"] = "/usr"
import numpy as np

def main():
    import vart, xir
    from pynq_dpu import DpuOverlay
    D = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
    nburn = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    burn = [subprocess.Popen(["taskset", "-c", "0-2", "sh", "-c", "while true; do :; done"])
            for _ in range(nburn)]
    MODEL = "/usr/local/share/pynq-venv/lib/python3.10/site-packages/pynq_dpu/notebooks/dpu_resnet50.xmodel.d/dpu_resnet50.KR260.xmodel"
    ov = DpuOverlay("dpu.bit"); ov.load_model(MODEL)
    d = ov.runner
    it, ot = d.get_input_tensors(), d.get_output_tensors()
    inp = [np.zeros(tuple(it[0].dims), np.float32, order="C")]
    out = [np.zeros(tuple(ot[0].dims), np.float32, order="C")]
    raw = np.empty(frames); delivered = np.empty(frames)
    stale = np.zeros(frames, np.uint8); enforce = []
    for _ in range(100):
        jid = d.execute_async(inp, out); d.wait(jid)
    for i in range(frames):
        published = {"t": None}
        lock = threading.Lock()
        t0 = time.perf_counter()
        def enforcer():
            slack = D / 1000.0 - (time.perf_counter() - t0)
            if slack > 0: time.sleep(slack)
            with lock:
                if published["t"] is None:
                    te = time.perf_counter(); published["t"] = te
                    stale[i] = 1; enforce.append(1000 * (te - t0) - D)
        th = threading.Timer(0, enforcer); th.start()
        for _ in range(6):
            jid = d.execute_async(inp, out); d.wait(jid)
        t1 = time.perf_counter()
        raw[i] = 1000 * (t1 - t0)
        with lock:
            if published["t"] is None: published["t"] = t1
        th.join()
        delivered[i] = 1000 * (published["t"] - t0)
        if (i + 1) % 10000 == 0:
            print("  %d/%d raw_max %.2f delivered_max %.2f stale %d" %
                  (i + 1, frames, raw[:i+1].max(), delivered[:i+1].max(), int(stale[:i+1].sum())), flush=True)
    for p in burn: p.kill()
    el = np.array(enforce) if enforce else np.zeros(1)
    print("E5 DELIVERY D=%.0fms frames=%d" % (D, frames))
    print("raw:       p50 %.2f p99.9 %.2f max %.2f | >D: %d (%.4f%%)" %
          (np.median(raw), np.percentile(raw, 99.9), raw.max(), (raw > D).sum(), 100 * (raw > D).mean()))
    print("delivered: p50 %.2f max %.2f | fallbacks %d | enforcement p50 %.3f max %.3f ms" %
          (np.median(delivered), delivered.max(), int(stale.sum()), np.median(el), el.max()))
    print("DELIVERY BOUND: every output within D + %.3f ms" % max(0.0, delivered.max() - D))
    np.savez("/home/ubuntu/e1/e5_delivery_D%d.npz" % int(D), raw=raw, delivered=delivered, stale=stale, enforce=el)

if __name__ == "__main__":
    main()
