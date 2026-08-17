#!/usr/bin/env python
"""Determinism *discriminator* figure (WACV) -- the model-architecture axis only.

Per-frame latency normalised by each series' own median. We deliberately drop the
platform sweep (KR260 vs Orin vs GPU lives in the window-margin teaser, Fig 1) and
keep the occupancy-model zoo plus the KR260 DPU as the deterministic floor, so this
figure carries exactly one message Fig 1 does not: the discriminator is
*input-dependent computation, not sparsity*. OPUS is sparse yet fixed-query and stays
a tight vertical line like the KR260 floor; SparseOcc is sparse AND input-dependent
and grows a tail past the deadline; dense FlashOcc sits between.

Left: bulk (CDF). Right: tail (CCDF, log y) where the deadline-relevant mass lives.
All real measured data (KR260 DPU on-device; FlashOcc/SparseOcc/OPUS per-frame CSVs
on a shared A100, FlashOcc as a 3-seed band).
"""
import argparse, csv, os, sys, glob
import numpy as np, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, ScalarFormatter
from matplotlib.patheffects import withStroke
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style as S; S.apply()

WARMUP = 15

# (label, filename, kind, color, linewidth)  -- 4 series, distinct hues, all solid.
# KR260 = teal hero (deterministic floor); OPUS plum (sparse but fixed); FlashOcc amber
# (dense); SparseOcc crimson (sparse + input-dependent -> the cautionary tail).
CURVES = [
    ("KR260 FPGA DPU",      "kr260_rt_idle.npy",             "npy",   S.TEAL,    3.0),
    ("OPUS (fixed-query)",  "opus_latency_density.csv",      "csv",   S.PLUM,    2.2),
    ("FlashOcc (dense)",    "focc_ms_A100_rep*.csv",         "mscsv", S.AMBER,   2.2),
    ("SparseOcc (sparse)",  "sparseocc_latency_density.csv", "csv",   S.CRIMSON, 2.6),
]

def load(path, kind):
    if kind == "npy":
        return np.load(path).astype(np.float64).ravel()
    vals = [float(r["latency_ms"]) for r in csv.DictReader(open(path, newline=""))]
    return np.asarray(vals[WARMUP:], np.float64)

ap = argparse.ArgumentParser()
ap.add_argument("--datadir", default="traces/_figstage")
ap.add_argument("--out", default="figs/determinism_cdf")
a = ap.parse_args()

fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 3.05))
fig.subplots_adjust(wspace=0.24, bottom=0.155, top=0.88, left=0.085, right=0.985)

stats, handles = {}, []
for label, fn, kind, color, lw in CURVES:
    if kind == "mscsv":
        reps = sorted(glob.glob(os.path.join("traces/hpc/ms", fn)))
        if not reps:
            print("  [skip]", fn); continue
        seeds = [load(r, "csv") for r in reps]
        allv = np.concatenate(seeds)
        grid = np.linspace(1.0, max(s.max() / np.median(s) for s in seeds), 600)
        C = np.array([np.interp(grid, np.sort(s / np.median(s)), np.arange(1, s.size + 1) / s.size,
                                left=0.0, right=1.0) for s in seeds])
        lo, hi, mid = C.min(0), C.max(0), C.mean(0)
        axL.fill_between(grid, lo, hi, color=color, alpha=0.18, lw=0, zorder=2)
        axL.plot(grid, mid, color=color, lw=lw, label=label, solid_capstyle="round", zorder=4)
        axR.fill_between(grid, np.clip(1 - hi, 6e-5, 1), np.clip(1 - lo, 6e-5, 1), color=color, alpha=0.18, lw=0, zorder=2)
        axR.semilogy(grid, np.clip(1 - mid, 6e-5, 1), color=color, lw=lw, solid_capstyle="round", zorder=4)
        x = allv
    else:
        p = os.path.join(a.datadir if kind == "npy" else "traces", fn)
        if not os.path.exists(p):
            print("  [skip]", p); continue
        x = load(p, kind); xs = np.sort(x / np.median(x))
        y = np.arange(1, xs.size + 1) / xs.size
        z = 6 if "KR260" in label else (5 if "Sparse" in label else 4)
        axL.plot(xs, y, color=color, lw=lw, solid_capstyle="round", label=label, zorder=z)
        axR.semilogy(xs, 1.0 - np.arange(xs.size) / xs.size, color=color, lw=lw, solid_capstyle="round", zorder=z)
    cv = 100 * x.std() / x.mean(); mp = x.max() / np.median(x)
    stats[label] = (color, cv, mp)
    handles.append((label, color, cv))

# ---- left panel: bulk ----
axL.set_xlim(1.0, 2.0); axL.set_ylim(0, 1.005)
axL.set_xticks([1.0, 1.25, 1.5, 1.75, 2.0])
axL.set_ylabel("cumulative fraction"); axL.set_xlabel(r"per-frame latency $/$ median")
axL.set_title("bulk  (CDF)", pad=5)

# ---- right panel: tail (the money panel) ----
axR.set_xscale("log", base=2); axR.set_xlim(1.0, 3.0); axR.set_xticks([1, 1.25, 1.5, 2, 3])
fmt = ScalarFormatter(); fmt.set_scientific(False); axR.xaxis.set_major_formatter(fmt)
axR.xaxis.set_minor_locator(NullLocator())
axR.set_ylim(6e-5, 1); axR.set_ylabel(r"fraction slower than $x$")
axR.set_xlabel(r"per-frame latency $/$ median"); axR.set_title("tail  (CCDF, log scale)", pad=5)
# shade the deadline-miss region + 2x marker
axR.axvspan(2.0, 3.0, color=S.CRIMSON, alpha=0.05, zorder=0)
axR.axvline(2.0, color=S.CRIMSON, ls=(0, (2, 2)), lw=1.1, alpha=0.7, zorder=1)
axR.text(2.02, 0.42, "2$\\times$ deadline", color=S.CRIMSON, fontsize=8.5, rotation=90, va="center", alpha=0.85)

# ---- in-axes legend box, styled like Fig 1 (framed, white, upper-right of the tail panel) ----
from matplotlib.lines import Line2D
LBL = {"KR260 FPGA DPU": "KR260 FPGA DPU", "OPUS (fixed-query)": "OPUS (sparse, fixed-query)",
       "FlashOcc (dense)": "FlashOcc (dense)", "SparseOcc (sparse)": "SparseOcc (input-dependent)"}
_seq = ["KR260 FPGA DPU", "OPUS (fixed-query)", "FlashOcc (dense)", "SparseOcc (sparse)"]
_lh = [Line2D([0], [0], color=stats[k][0], lw=2.6, label=LBL[k]) for k in _seq if k in stats]
axR.legend(handles=_lh, loc="upper right", fontsize=8.2, handlelength=1.8, borderpad=0.5,
           labelspacing=0.4, framealpha=0.95, facecolor="white", edgecolor="#DDD")

for ax in (axL, axR):
    ax.grid(axis="y", color="#EAEDF0", lw=1.0); ax.set_axisbelow(True)

# curves are labelled in-axes above; no separate legend box.
order = ["KR260 FPGA DPU", "OPUS (fixed-query)", "FlashOcc (dense)", "SparseOcc (sparse)"]

os.makedirs(os.path.dirname(a.out), exist_ok=True)
fig.savefig(a.out + ".png", dpi=300)
fig.savefig(a.out + ".pdf")
print("wrote", a.out + ".pdf")
print(f"{'series':24s} {'p50':>8} {'CV%':>7} {'max/p50':>8}")
for k in order:
    if k in stats:
        c, cv, mp = stats[k]; print(f"{k:24s} {'':8} {cv:7.2f} {mp:8.3f}")
