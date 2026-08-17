#!/usr/bin/env python3
"""Wait for the full-6019 board run to finish, then gzip the .npy so it's ready for the HPC eval.

Polls the appended board_argmax_full.npy until it reaches 6019 frames (3.85 GB), then streams it to
.npy.gz (~136 MB, 28x). Self-exits; the harness re-invokes the agent when this background task ends.
12 h safety timeout. No board/WAN load (only a local stat() every 5 min).
"""
import os, time, subprocess, sys

NPY = "traces/buildB/fullrun/board_argmax_full.npy"
TARGET = 6019 * 200 * 200 * 16          # 3,852,160,000 bytes = all 6019 frames written
PY = "D:/miniconda/python.exe"
DEADLINE = time.time() + 12 * 3600

print("watch_and_compress: waiting for %s to reach %d bytes (6019 frames)" % (NPY, TARGET), flush=True)
while True:
    sz = os.path.getsize(NPY) if os.path.exists(NPY) else 0
    frames = sz // (200 * 200 * 16)
    if sz >= TARGET:
        print("RUN COMPLETE: %d frames (%d bytes). Compressing..." % (frames, sz), flush=True)
        break
    if time.time() > DEADLINE:
        print("TIMEOUT after 12 h at %d frames; not compressing." % frames, flush=True)
        sys.exit(2)
    print("  ... %d/6019 frames (%.2f GB); sleeping 300 s" % (frames, sz / 1e9), flush=True)
    time.sleep(300)

rc = subprocess.call([PY, "fpga/board/compress_board.py", NPY])
gz = NPY + ".gz"
if rc == 0 and os.path.exists(gz):
    print("COMPRESS DONE -> %s (%.0f MB). Ready for HPC transfer." % (gz, os.path.getsize(gz) / 1e6), flush=True)
    sys.exit(0)
print("COMPRESS FAILED rc=%d" % rc, flush=True)
sys.exit(1)
