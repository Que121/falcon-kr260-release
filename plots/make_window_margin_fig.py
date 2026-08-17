#!/usr/bin/env python3
"""FIG 1 (single-column) -- worst-case latency margin vs. observation window, log y.

Each platform's WORST run (max margin at each window) is a BOLD line; the faint shading is its full
min--max spread over every run measured. This is the WCET-relevant comparison (worst-case vs worst-case):
the KR260 FPGA DPU is the FLOOR (bounded ~1.8x at any window, cycle count + watchdog), while the Jetson
Orin NX at edge power budgets reaches 2.6x (sustained 3.7x), the workstation GPU 4.4x, and the datacenter
GPUs ~20x. The Orin's best mode (MAXN, unlimited power) is tighter at 1.37x but is not a deployable edge
budget and is still uncertifiable; it appears as the lower edge of the orange band. All traces are 100k
frames. Arial, one WACV column.
"""
import os, sys, glob, numpy as np, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, FuncFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style as S; S.apply()

RES = "experiments/results"
WINDOWS = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 50000, 100000]
W = np.array(WINDOWS, float)

# each platform = every run measured; bold line = its worst run, faint fill = full spread.
# order: drawn back-to-front so KR260 (floor) ends on top.
FAMS = [
    ("5 datacenter GPUs", S.PLUM,
     sorted(glob.glob(f"{RES}/hpc/ms/ms_*_rep*_idle.npy")) + sorted(glob.glob(f"{RES}/hpc/ms/ms_*_rep*_loaded.npy"))),
    ("Workstation GPU", S.CRIMSON,
     [f"{RES}/occfpga_gpu_long_idle.npy", f"{RES}/occfpga_gpu_long_loaded.npy"]),
    ("RTX 4090 (consumer)", "#C25FA3",
     [f"{RES}/rtx4090/occfpga_4090_100k_idle.npy", f"{RES}/rtx4090/occfpga_4090_100k_loaded.npy"]),
    ("Jetson Orin NX", S.ORANGE,
     [f"{RES}/orin/occfpga_orin_100k_%s.npy" % m for m in ["MAXN", "10W", "15W", "25W"]]
     + sorted(glob.glob(f"{RES}/orin/occfpga_orin_100k_*rep*.npy"))   # MAXN x5 seeds at 100k (swings to 4.4)
     + sorted(glob.glob(f"{RES}/orin/ms_orin_*_rep*.npy"))            # all 4 modes x 5 seeds (30k)
     + [f"{RES}/orin/occfpga_orin_sustained_15W_locked.npy"]),
    ("KR260 FPGA DPU", S.TEAL,
     [f"{RES}/kr260/kr260_long_idle_100k.npy"]),
]

def margin_curve(x, rng):
    N = x.size; out = []
    for w in WINDOWS:
        if w > N: out.append(np.nan); continue
        st = np.array([0]) if w == N else rng.integers(0, N - w + 1, size=min(600, N - w + 1))
        out.append(np.mean([x[s:s+w].max() / np.median(x[s:s+w]) for s in st]))
    return np.array(out, float)

def curves(paths, rng):
    return [margin_curve(np.load(p).astype(float).ravel(), rng) for p in paths if os.path.exists(p)]

rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(3.3, 2.95))

handles = []
for i, (label, col, paths) in enumerate(FAMS):
    cs = curves(paths, rng)
    # thin dashed line per individual run (each power mode x seed / idle-load), so the spread is visible
    for c in cs:
        m = ~np.isnan(c)
        ax.plot(W[m], c[m], color=col, lw=0.5, ls=(0, (2, 2.2)), alpha=0.22, zorder=2 + i, solid_capstyle="round")
    # bold worst-case envelope; non-decreasing in window (a 100k look includes every sub-window), so cumulative max
    hi = np.nanmax(np.array(cs), 0)
    himono = np.maximum.accumulate(np.where(np.isnan(hi), -np.inf, hi))
    m = ~np.isnan(hi); himono = np.where(m, himono, np.nan)
    ax.plot(W[m], himono[m], color=col, lw=2.4 if "KR260" in label else 2.1, zorder=7 + i, solid_capstyle="round")
    handles.append(Line2D([0], [0], color=col, lw=2.4 if "KR260" in label else 2.1, label=label))

ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(90, 1.3e5); ax.set_ylim(0.99, 26)
ax.set_yticks([1, 1.5, 2, 3, 5, 10, 20]); ax.yaxis.set_minor_locator(NullLocator())
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: ("%g" % v) + r"$\times$"))
ax.axhline(1.0, color="#BBB", lw=0.8, ls=(0, (1, 2)), zorder=1)
ax.set_xlabel("observation window (consecutive frames)", fontsize=8.5)
ax.set_ylabel("worst-case margin  (max / median)", fontsize=8.5)
ax.tick_params(labelsize=8)
ax.grid(True, which="major", color="#EEF0F2"); ax.set_axisbelow(True)

# legend order: KR260 (floor) first; last entry explains the thin dashed individual-run lines
dash_h = Line2D([0], [0], color=S.SUBTLE, lw=0.7, ls=(0, (2, 2.2)), label="individual modes $\\times$ seeds")
ax.legend(handles=[handles[3], handles[2], handles[1], handles[0], dash_h], loc="upper left", fontsize=6.3,
          handlelength=1.8, borderpad=0.4, labelspacing=0.3, framealpha=0.93, facecolor="white", edgecolor="#DDD")

# notes in clear zones (no solid line passes through them)
ax.text(2.7e3, 23, "datacenter GPU\nworst case ${\\sim}20\\times$", color=S.SUBTLE,
        fontsize=7, ha="left", va="top")
from matplotlib.transforms import offset_copy
_gfs = 5.95  # 0.85x of the 7pt base
_gt = offset_copy(ax.transData, fig=fig, x=-5 * _gfs, y=-1 * _gfs, units="points")  # 5 font-sizes left, 1 down
ax.text(1.9e4, 1.83, "KR260 DPU bounded\n$\\leq$1.5$\\times$ at any window", color=S.TEAL, fontsize=_gfs,
        ha="left", va="center", fontweight="semibold", linespacing=1.15, transform=_gt)

fig.tight_layout(pad=0.25)
out = "docs/figs/fig_window_margin"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out + ".png", dpi=220, bbox_inches="tight", pad_inches=0.01)
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.01)
print("wrote", out)
