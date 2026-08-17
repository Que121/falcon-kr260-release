#!/usr/bin/env python3
"""Stream-gzip a big board_argmax .npy -> .npy.gz (28x smaller; occupancy is mostly free space).

Keeps eval_board_miou.py unchanged: gunzip on HPC restores the exact .npy. Streams in chunks so
the 3.85 GB full-6019 file never loads into RAM.

  python compress_board.py <in.npy> [out.npy.gz]
"""
import sys, gzip, os, shutil

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else src + ".gz"
raw = os.path.getsize(src)
with open(src, "rb") as fi, gzip.open(dst, "wb", compresslevel=6) as fo:
    shutil.copyfileobj(fi, fo, length=64 * 1024 * 1024)
comp = os.path.getsize(dst)
print("compressed %s (%.2f GB) -> %s (%.0f MB, %.1fx)" % (src, raw / 1e9, dst, comp / 1e6, raw / comp))
