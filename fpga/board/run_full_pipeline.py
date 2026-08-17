#!/usr/bin/env python3
"""Windows orchestrator: stream frame chunks HPC -> board, run the all-PL pipeline per chunk,
collect occ argmax. ROBUST + RESUMABLE for unattended overnight runs.

The board holds only ONE stream-chunk of frames at a time (13GB disk), so the 6019-frame / 41GB dump
never lands whole on-board. Per stream-chunk [cs,ce):
  tar frames on HPC -> scp Pro6000 -> get local -> put board -> extract ->
  LAUNCH board_pipeline_batched DETACHED (nohup setsid) -> POLL log for DONE ->
  get chunk occ -> accumulate -> delete board chunk.

Robustness: each transfer step retried; the whole chunk wrapped in try/except (a failed chunk is
skipped, NOT fatal -- re-run resumes it via progress.txt). Board pipeline launched detached + polled
(no 30-min synchronous SSH channel to drop). STREAM_CHUNK = frames transferred together; BOARD_CHUNK =
board_pipeline internal RAM batch (keep 32 -- larger OOMs the 4GB board).

  python run_full_pipeline.py <N> [stream_chunk=512] [io_dir=buildB_io_full] [bev_xm] [out] [img_xm] [board_chunk=32]
"""
import sys, os, subprocess, numpy as np, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "D:/miniconda/python.exe"
JUMP = os.path.join(HERE, "_jump.py"); BOARD = os.path.join(HERE, "_board.py")
LOCAL = os.path.join(HERE, "..", "..", "experiments", "results", "buildB", "fullrun")
os.makedirs(LOCAL, exist_ok=True)

N      = int(sys.argv[1]) if len(sys.argv) > 1 else 6019
STREAM = int(sys.argv[2]) if len(sys.argv) > 2 else 512
IODIR  = sys.argv[3] if len(sys.argv) > 3 else "buildB_io_full"
BEVXM  = sys.argv[4] if len(sys.argv) > 4 else "/home/ubuntu/bev_allpl/bev_fpc.xmodel"
OUT    = sys.argv[5] if len(sys.argv) > 5 else os.path.join(LOCAL, "board_argmax_full.npy")
IMGXM  = sys.argv[6] if len(sys.argv) > 6 else "/home/ubuntu/flashocc/flashocc_r50_image_adaquant.xmodel"
BCHUNK = int(sys.argv[7]) if len(sys.argv) > 7 else 32
SAP_IO = "/scratch/ANON/" + IODIR
LOG = os.path.join(LOCAL, "orchestrator.log")

def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def sh(args, tmo):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=tmo)
    except subprocess.TimeoutExpired as e:
        class R: returncode = 124; stdout = (e.stdout or b"").decode("utf-8","replace") if isinstance(e.stdout,bytes) else (e.stdout or ""); stderr = "TIMEOUT"
        return R()

def jump(where, cmd, tmo=900): return sh([PY, JUMP, where, cmd, str(tmo)], tmo + 60)
def jump_io(verb, a, b):       return sh([PY, JUMP, verb, a, b], 5400)   # slow WAN (~0.6MB/s): allow 90min
def board(verb, *a, tmo=1800): return sh([PY, BOARD, verb, *a], tmo)

def retry(fn, ok, tries=3, wait=15, what=""):
    for t in range(tries):
        r = fn()
        if ok(r): return r
        log("  retry %d/%d %s rc=%s err=%s" % (t+1, tries, what, getattr(r,"returncode","?"), (getattr(r,"stderr","") or "")[-120:]))
        time.sleep(wait)
    return r

def do_chunk(cs, ce, occ):
    names = " ".join("frame_%04d.npz" % i for i in range(cs, ce))
    # 1) tar on HPC
    retry(lambda: jump("HPC", "cd %s && tar cf /scratch/ANON/chunk.tar %s && echo TAROK" % (SAP_IO, names), 900),
          lambda r: "TAROK" in (r.stdout or ""), what="HPC-tar")
    # 2) HPC -> pro
    retry(lambda: jump("pro", "scp -q HPC:/scratch/ANON/chunk.tar ~/chunk.tar < /dev/null && echo SCPOK", 900),
          lambda r: "SCPOK" in (r.stdout or ""), what="HPC->pro")
    # 3) pro -> local
    lt = os.path.join(LOCAL, "chunk.tar")
    if os.path.exists(lt): os.remove(lt)
    retry(lambda: jump_io("get", "chunk.tar", lt), lambda r: os.path.exists(lt) and os.path.getsize(lt) > 1e6, what="pro->local")
    # 4) local -> board, extract
    board("run", "rm -rf ~/buildB/chunk ~/buildB/work ~/buildB/chunk_occ.npy; mkdir -p ~/buildB/chunk")
    retry(lambda: board("put", lt, "chunk.tar"), lambda r: r.returncode == 0, what="local->board")
    r = retry(lambda: board("run", "cd ~/buildB/chunk && tar xf ~/chunk.tar && ls | wc -l"),
              lambda r: (r.stdout or "").strip().splitlines() and (r.stdout or "").strip().splitlines()[-1].strip().isdigit()
                        and int((r.stdout).strip().splitlines()[-1].strip()) == (ce - cs), what="board-extract")
    nf = int((r.stdout or "0").strip().splitlines()[-1].strip()) if (r.stdout or "").strip().splitlines() else 0
    if nf != (ce - cs): raise RuntimeError("extract count %d != %d" % (nf, ce - cs))
    # 5) LAUNCH detached
    lf = "/home/ubuntu/buildB/chunk_%d.log" % cs
    launch = ("cd /home/ubuntu && nohup setsid python3 /home/ubuntu/board_pipeline_batched.py "
              "/home/ubuntu/buildB/chunk %d /home/ubuntu/buildB/chunk_occ.npy %d %s %s > %s 2>&1 < /dev/null & disown; "
              "sleep 4; echo LAUNCH_RC=$?" % (ce - cs, BCHUNK, IMGXM, BEVXM, lf))
    r = board("srun", launch, tmo=120)
    if "LAUNCH_RC=0" not in (r.stdout or ""): raise RuntimeError("launch failed: " + (r.stdout or "")[-200:] + (r.stderr or "")[-200:])
    # 6) poll for DONE (up to 120 min)
    t0 = time.time()
    while time.time() - t0 < 7200:
        time.sleep(75)
        r = board("srun", "tail -n 2 %s; ls -la /home/ubuntu/buildB/chunk_occ.npy 2>/dev/null && echo HAVE_OCC" % lf, tmo=120)
        s = (r.stdout or "")
        if "DONE %d frames" % (ce - cs) in s or "HAVE_OCC" in s:
            break
        if "Traceback" in s or "Error" in s:
            raise RuntimeError("board pipeline error: " + s[-300:])
    else:
        raise RuntimeError("chunk timed out after 120min")
    # 7) collect
    lo = os.path.join(LOCAL, "chunk_occ.npy")
    if os.path.exists(lo): os.remove(lo)
    retry(lambda: board("get", "buildB/chunk_occ.npy", lo), lambda r: os.path.exists(lo) and os.path.getsize(lo) > 1e6, what="board-get-occ")
    co = np.load(lo)
    occ[cs:ce] = co[:ce - cs]

def main():
    occ = np.zeros((N, 200, 200, 16), np.uint8)
    done = set()
    prog = OUT + ".progress"
    if os.path.exists(prog):
        done = set(int(x) for x in open(prog).read().split())
        if os.path.exists(OUT):
            prev = np.load(OUT); occ[:min(prev.shape[0], N)] = prev[:N]
    log("FULLRUN start N=%d stream=%d bchunk=%d IMG=%s BEV=%s done=%d" %
        (N, STREAM, BCHUNK, os.path.basename(IMGXM), os.path.basename(BEVXM), len(done)))
    t0 = time.time()
    for cs in range(0, N, STREAM):
        ce = min(cs + STREAM, N)
        if cs in done:
            log("chunk %d-%d already done, skip" % (cs, ce - 1)); continue
        tc = time.time()
        try:
            do_chunk(cs, ce, occ)
            np.save(OUT, occ[:ce])
            done.add(cs); open(prog, "w").write(" ".join(str(x) for x in sorted(done)))
            log("chunk %d-%d DONE in %.0fs (elapsed %.0fmin, %d/%d frames)" %
                (cs, ce - 1, time.time() - tc, (time.time() - t0)/60, ce, N))
        except Exception as e:
            log("chunk %d-%d FAILED (%s) -- skipping, resumable" % (cs, ce - 1, str(e)[:200]))
            continue
    np.save(OUT, occ)
    log("FULLRUN COMPLETE %d frames -> %s in %.0fmin (chunks done %d)" % (N, OUT, (time.time()-t0)/60, len(done)))

if __name__ == "__main__":
    main()
