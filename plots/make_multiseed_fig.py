#!/usr/bin/env python3
"""FIG -- multi-seed worst-case margin: the FPGA is bounded, every GPU is a lottery.

The single-window numbers (KR260 1.06, Pro6000 1.07) looked similar. Running each platform as
N=5 INDEPENDENT long windows (seeds) settles it: the KR260 DPU's worst-case margin (max/median)
is the SAME tight 1.03-1.17 on every seed (the bound is set by the cycle count), while every
GPU and the Orin scatter wildly from seed to seed -- the rare host-stack spike is real but its
size is unpredictable, up to 23x on the L4. A bound you cannot reproduce is a bound you cannot
certify.

Left  : worst-case margin per seed (strip), log-x. Each dot is one independent long window.
Right : the same as expected-running-max vs observation window (mean +/- seed min-max band),
        so you see the margin both scatter across seeds AND grow with the window.

Data: KR260 DPU 5x30k (idle); Orin 15W/10W 5x30k each; HPC A100/H100/L4/L40S/P100 5x100k (idle).
"""
import os, sys, glob, numpy as np, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullLocator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style as S; S.apply()

RES = "experiments/results"
GREEN, VERM, ORANGE, BLUE = S.TEAL, S.CRIMSON, S.GOLD, S.PLUM   # KR260 / Orin15W / Orin10W / GPUs

# (label, glob, color)  -- ordered worst (top) to best (bottom); DPU last so it sits at the bottom
PLATS = [
    ("L4 GPU",       f"{RES}/hpc/ms/ms_L4_rep*_idle.npy",        BLUE),
    ("L40S GPU",     f"{RES}/hpc/ms/ms_L40S_rep*_idle.npy",      BLUE),
    ("P100 GPU",     f"{RES}/hpc/ms/ms_P100_rep*_idle.npy",      BLUE),
    ("H100 GPU",     f"{RES}/hpc/ms/ms_H100_rep*_idle.npy",      BLUE),
    ("A100 GPU",     f"{RES}/hpc/ms/ms_A100_rep*_idle.npy",      BLUE),
    ("Orin NX 10W",  f"{RES}/orin/ms_orin_10W_rep*.npy",         ORANGE),
    ("Orin NX 15W",  f"{RES}/orin/ms_orin_15W_rep*.npy",         VERM),
    ("KR260 DPU",    f"{RES}/kr260/kr260_ms_idle_rep*.npy",      GREEN),
]
WINDOWS = [1000, 2000, 5000, 10000, 20000, 30000, 50000, 100000]

def load_seeds(g):
    return [np.load(f).astype(np.float64).ravel() for f in sorted(glob.glob(g))]

def win_margin(x, w, rng, S=300):
    N = x.size
    if w > N: return np.nan
    if w == N: starts = np.array([0])
    else: starts = rng.integers(0, N - w + 1, size=min(S, N - w + 1))
    return np.mean([x[s:s+w].max() / np.median(x[s:s+w]) for s in starts])

rng = np.random.default_rng(0)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.2, 3.6), gridspec_kw={"width_ratios": [1, 1.15]})
fig.subplots_adjust(wspace=0.32)

# ---- LEFT: per-seed worst-case margin strip ----
yt, ylab = [], []
for yi, (lab, g, col) in enumerate(PLATS):
    seeds = load_seeds(g)
    if not seeds: print("[skip]", lab); continue
    m = np.array([s.max() / np.median(s) for s in seeds])
    jit = (np.arange(len(m)) - (len(m)-1)/2) * 0.045
    axL.scatter(m, np.full(len(m), yi) + jit, s=26, color=col, alpha=0.85, zorder=3, edgecolor="white", linewidth=0.4)
    axL.plot([m.min(), m.max()], [yi, yi], color=col, lw=1.0, alpha=0.5, zorder=2)
    axL.scatter([m.mean()], [yi], marker="|", s=180, color=col, zorder=4, linewidth=1.6)
    yt.append(yi); ylab.append(lab)
axL.axvspan(1.0, 1.2, color=GREEN, alpha=0.07, zorder=0)
axL.set_xscale("log"); axL.set_xlim(1.0, 28)
axL.set_xticks([1, 1.5, 2, 3, 5, 10, 20]); axL.xaxis.set_major_formatter(ScalarFormatter()); axL.xaxis.set_minor_locator(NullLocator())
axL.set_yticks(yt); axL.set_yticklabels(ylab); axL.set_ylim(-0.6, len(PLATS)-0.4)
axL.axvline(1.0, color="0.7", lw=0.8, ls=(0,(1,2)))
axL.set_xlabel(r"worst-case margin  (max $/$ median),  per seed")
axL.set_title("each dot = one independent long window", pad=5)
axL.grid(axis="x", color="0.92", lw=0.6); axL.set_axisbelow(True)

# ---- RIGHT: margin vs observation window, mean +/- seed band ----
W = np.array(WINDOWS, float)
for lab, g, col in PLATS:
    seeds = load_seeds(g)
    if not seeds: continue
    curves = np.array([[win_margin(s, w, rng) for w in WINDOWS] for s in seeds])  # (nseed, nwin)
    mean = np.nanmean(curves, 0); lo = np.nanmin(curves, 0); hi = np.nanmax(curves, 0)
    msk = ~np.isnan(mean)
    axR.fill_between(W[msk], lo[msk], hi[msk], color=col, alpha=0.13, zorder=1)
    axR.plot(W[msk], mean[msk], color=col, lw=2.0 if "KR260" in lab else 1.5,
             label=lab, marker="o", ms=2.5, zorder=3)
axR.set_xscale("log"); axR.set_yscale("log")
axR.set_xlim(900, 1.2e5); axR.set_ylim(1.0, 28)
axR.set_yticks([1, 1.5, 2, 3, 5, 10, 20]); axR.yaxis.set_major_formatter(ScalarFormatter()); axR.yaxis.set_minor_locator(NullLocator())
axR.axhline(1.0, color="0.7", lw=0.8, ls=(0,(1,2)))
axR.set_xlabel("observation window (frames)"); axR.set_ylabel(r"worst-case margin (max $/$ median)")
axR.set_title("band = seed min-max", pad=5)
axR.grid(color="0.92", lw=0.6); axR.set_axisbelow(True)
axR.legend(loc="upper left", ncol=2, handlelength=1.6, columnspacing=1.0, borderpad=0.2)

out = "docs/figs/fig_multiseed_margin"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out + ".png", dpi=300); fig.savefig(out + ".pdf")
print("wrote", out + ".pdf/.png")
# numeric summary
print(f"\n{'platform':14s} {'seeds':>5} {'max/p50 min-max':>18} {'mean':>6}")
for lab, g, col in PLATS:
    seeds = load_seeds(g)
    if not seeds: continue
    m = np.array([s.max()/np.median(s) for s in seeds])
    print(f"{lab:14s} {len(m):5d}   {m.min():6.2f} - {m.max():6.2f}   {m.mean():6.2f}")
