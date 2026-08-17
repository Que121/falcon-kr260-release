#!/usr/bin/env python3
"""Combined determinism figure (Figs 4/5/6 -> one ABC panel), modern publication style.
Rich harmonious palette, used semantically and consistently across panels:
  teal = FPGA / KR260 / bounded-good (hero) ; blue = OPUS (sparse, fixed-query) ;
  amber = Jetson Orin / FlashOcc / moderate ; red = unbounded / RT-backfire / cautionary ;
  plum = datacenter GPU.
  A: per-frame latency CDF normalized by each series' median  (architecture leg)
  B: per-run WCET margin (max/median) across the full hardware sweep  (platform leg)
  C: KR260 host-path leg, WCET margin under five submission policies  (host-path leg)
-> figs/determinism_panel.{png,pdf,svg}
"""
import os, sys, csv, glob
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = "traces"
WARMUP = 15

# ---- rich, harmonious palette (semantic) ----
TEAL = "#0FA08C"   # FPGA / KR260 / bounded-good (hero)
BLUE = "#2F66C4"   # OPUS (sparse, fixed-query)
AMBER= "#EBA13A"   # Jetson Orin / FlashOcc / moderate
RED  = "#E1483B"   # unbounded / RT backfire / cautionary
PLUM = "#8A52A8"   # datacenter GPU
INK  = "#15202B"
GRAYTX = "#56616C"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.weight": "bold",
    "svg.fonttype": "none",
    "font.size": 19,
    "axes.labelsize": 23, "axes.labelweight": "bold", "axes.labelcolor": INK,
    "axes.edgecolor": "#33414D", "axes.linewidth": 1.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True,
    "text.color": INK,
    "xtick.color": "#33414D", "ytick.color": "#33414D",
    "xtick.labelsize": 21, "ytick.labelsize": 21,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 6, "ytick.major.size": 6,
    "xtick.major.width": 1.5, "ytick.major.width": 1.5,
    "legend.frameon": False, "legend.fontsize": 17,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

def letter(ax, s, dx=-0.15):
    ax.text(dx, 1.015, s, transform=ax.transAxes, fontsize=30, fontweight="bold",
            va="bottom", ha="left", color=INK)

def load_lat(path, kind):
    if kind == "npy":
        return np.load(path).astype(np.float64).ravel()
    vals = [float(r["latency_ms"]) for r in csv.DictReader(open(path, newline=""))]
    return np.asarray(vals[WARMUP:], np.float64)

def margins(*globpats):
    vals = []
    for g in globpats:
        for f in sorted(glob.glob(os.path.join(R, g))):
            x = np.load(f).astype(float).ravel()
            vals.append(x.max() / np.median(x))
    return np.array(vals)

# ===================== layout =====================
fig = plt.figure(figsize=(18.8, 5.6))
gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.10, 1.06], wspace=0.36,
                      left=0.05, right=0.995, bottom=0.155, top=0.88)
axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

# --------------- Panel A: CDF (architecture leg) ---------------
CURVES = [
    ("KR260 FPGA DPU",         os.path.join(R, "kr260/kr260_rt_idle.npy"),       "npy",   TEAL,  3.9),
    ("OPUS (fixed-query)",     os.path.join(R, "opus_latency_density.csv"),      "csv",   BLUE,  3.0),
    ("FlashOcc (dense)",       "focc_ms_A100_rep*.csv",                          "mscsv", AMBER, 3.0),
    ("SparseOcc (input-dep.)", os.path.join(R, "sparseocc_latency_density.csv"), "csv",   RED,   3.3),
]
hA = []
for label, fn, kind, color, lw in CURVES:
    if kind == "mscsv":
        reps = sorted(glob.glob(os.path.join(R, "hpc/ms", fn)))
        seeds = [load_lat(r, "csv") for r in reps]
        grid = np.linspace(1.0, max(s.max()/np.median(s) for s in seeds), 600)
        C = np.array([np.interp(grid, np.sort(s/np.median(s)), np.arange(1, s.size+1)/s.size,
                                left=0.0, right=1.0) for s in seeds])
        axA.fill_between(grid, C.min(0), C.max(0), color=color, alpha=0.13, lw=0, zorder=2)
        axA.plot(grid, C.mean(0), color=color, lw=lw, solid_capstyle="round", zorder=4)
    else:
        x = load_lat(fn, kind); xs = np.sort(x/np.median(x)); y = np.arange(1, xs.size+1)/xs.size
        z = 6 if "KR260" in label else (5 if "Sparse" in label else 4)
        axA.plot(xs, y, color=color, lw=lw, solid_capstyle="round", zorder=z)
    hA.append(Line2D([0], [0], color=color, lw=4.6, label=label))

axA.axvline(2.0, color=RED, ls=(0, (3, 3)), lw=1.6, alpha=0.6, zorder=1)
axA.annotate(r"2$\times$ deadline", xy=(2.0, 1.0), xytext=(2.0, 1.03), ha="center",
             color=RED, fontsize=18, fontweight="bold", annotation_clip=False)
axA.set_xlim(1.0, 2.75); axA.set_ylim(0, 1.005)
axA.set_xticks([1.0, 1.5, 2.0, 2.5])
axA.set_xlabel("latency / median"); axA.set_ylabel("cumulative fraction")
# legend sits in the empty lower half (every CDF is at y >= 0.5, so no curve crosses it)
axA.legend(handles=hA, loc="lower left", bbox_to_anchor=(0.03, 0.02), fontsize=17,
           handlelength=1.5, labelspacing=0.45, borderpad=0.45)
letter(axA, "A")

# --------------- Panel B: platform zoo (platform leg) ---------------
FAM = {"FPGA": TEAL, "Orin": AMBER, "GPU": PLUM}
PLATS = [
    ("KR260",    "FPGA", "kr260/kr260_ms_idle_rep*.npy", "kr260/kr260_long_dedicated.npy", "kr260/kr260_long_idle.npy"),
    ("Orin 10W", "Orin", "orin/ms_orin_10W_rep*.npy"),
    ("Orin 15W", "Orin", "orin/ms_orin_15W_rep*.npy", "orin/occfpga_orin_sustained_15W_locked.npy"),
    ("L4",   "GPU", "hpc/ms/ms_L4_rep*_idle.npy",   "hpc/ms/ms_L4_rep*_loaded.npy"),
    ("P100", "GPU", "hpc/ms/ms_P100_rep*_idle.npy", "hpc/ms/ms_P100_rep*_loaded.npy"),
    ("A100", "GPU", "hpc/ms/ms_A100_rep*_idle.npy", "hpc/ms/ms_A100_rep*_loaded.npy"),
    ("H100", "GPU", "hpc/ms/ms_H100_rep*_idle.npy", "hpc/ms/ms_H100_rep*_loaded.npy"),
    ("L40S", "GPU", "hpc/ms/ms_L40S_rep*_idle.npy", "hpc/ms/ms_L40S_rep*_loaded.npy"),
]
rowsB = []
for label, fam, *globs in PLATS:
    m = margins(*globs)
    if m.size:
        rowsB.append((label, fam, m))
rowsB.sort(key=lambda r: np.median(r[2]))
rng = np.random.default_rng(0)
yt, ytl = [], []
for i, (label, fam, m) in enumerate(rowsB):
    c = FAM[fam]; yt.append(i); ytl.append(label)
    axB.plot([m.min(), m.max()], [i, i], color=c, lw=2.2, alpha=0.32, solid_capstyle="round", zorder=2)
    jit = (rng.random(m.size)-0.5)*0.22
    axB.scatter(m, i+jit, s=50, color=c, alpha=0.82, edgecolor="white", linewidth=0.9, zorder=4)
    axB.scatter([np.median(m)], [i], s=200, marker="|", color=c, linewidth=3.4, zorder=5)
    axB.text(m.max()*1.06, i, "%.1f" % m.max(), va="center", fontsize=16.5, color=GRAYTX, fontweight="bold")
axB.axvline(1.0, color="#9AA3AD", ls=":", lw=1.3, zorder=1)
axB.set_yticks(yt); axB.set_yticklabels(ytl, fontsize=21)
axB.set_xscale("log"); axB.set_xlim(0.99, 36)
axB.set_xticks([1, 1.5, 2, 3, 5, 10, 20]); axB.set_xticklabels(["1", "1.5", "2", "3", "5", "10", "20"])
axB.set_xlabel("WCET margin  (max / median)")
legB = [Line2D([0],[0], color=TEAL,  lw=4.5, label="FPGA DPU"),
        Line2D([0],[0], color=AMBER, lw=4.5, label="Jetson Orin"),
        Line2D([0],[0], color=PLUM,  lw=4.5, label="datacenter GPU"),
        Line2D([0],[0], marker="|", color=GRAYTX, lw=0, markeredgewidth=3.2, ms=15, label="median"),
        Line2D([0],[0], marker="o", color=GRAYTX, lw=0, ms=8, alpha=.6, label="one run")]
axB.legend(handles=legB, loc="lower right", fontsize=16.5, labelspacing=0.38, borderpad=0.45)
letter(axB, "B", dx=-0.10)

# --------------- Panel C: host-path leg ---------------
POLICIES = [
    ("idle",   "kr260/kr260_rt_idle.npy",      TEAL),
    ("+CPU",   "kr260/kr260_rt_loaded.npy",    AMBER),
    ("RT",     "kr260/kr260_rt_loaded_rt.npy", RED),
    ("FIFO80", "kr260/kr260_rt_fifo.npy",      RED),
    ("core",   "kr260/kr260_rt_taskset.npy",   TEAL),
]
xs, hs, cs, lbl = [], [], [], []
for i, (name, fn, c) in enumerate(POLICIES):
    x = np.load(os.path.join(R, fn)).astype(float).ravel()
    xs.append(i); hs.append(x.max()/np.median(x)); cs.append(c); lbl.append(name)
bars = axC.bar(xs, hs, width=0.70, color=cs, edgecolor="white", linewidth=1.2, zorder=3)
for b, h in zip(bars, hs):
    axC.text(b.get_x()+b.get_width()/2, h+0.12, "%.2f" % h, ha="center", va="bottom",
             fontsize=19, color=INK, fontweight="bold")
axC.axhline(1.0, color="#9AA3AD", ls=":", lw=1.4, zorder=1)
axC.set_xticks(xs); axC.set_xticklabels(lbl, fontsize=21)
axC.set_ylim(0, 4.8); axC.set_yticks([0, 1, 2, 3, 4])
axC.set_ylabel("WCET margin  (max / median)")
letter(axC, "C", dx=-0.21)

out = "figs/determinism_panel"
os.makedirs("docs/figs", exist_ok=True)
fig.savefig(out + ".png", dpi=420, bbox_inches="tight", pad_inches=0.08)
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.08)
fig.savefig(out + ".svg", bbox_inches="tight", pad_inches=0.08)
print("wrote", out + ".{png,pdf,svg}")
print("C:", list(zip(lbl, [round(h, 3) for h in hs])))
