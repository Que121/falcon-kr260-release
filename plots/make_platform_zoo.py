#!/usr/bin/env python3
"""Platform-zoo determinism summary across the FULL hardware sweep we measured -- KR260 FPGA DPU,
Jetson Orin at 10W and 15W, and five datacenter/cloud GPUs (L4, P100, A100, H100, L40S) -- each over
5 seeds x {idle, load} x 30k-100k frames. Each dot is one run's WCET margin (max/p50); the KR260 DPU
clusters tight (1.03-1.38) while GPU/Orin worst cases are large AND wildly seed-dependent (up to ~23x),
the unpredictability ISO 26262 penalises. (HPC GPU runs are on shared nodes, so even "idle" catches
co-tenant interference -- which is exactly the point: you cannot bound it.) -> figs/platform_zoo.{pdf,png}
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style as S; S.apply()
import numpy as np, matplotlib.pyplot as plt

R = "traces"

def margins(*globpats):
    vals = []
    for g in globpats:
        for f in sorted(glob.glob(os.path.join(R, g))):
            x = np.load(f).astype(float).ravel()
            vals.append(x.max() / np.median(x))
    return np.array(vals)

FAM = {"FPGA": S.TEAL, "Orin": S.ORANGE, "GPU": S.PLUM}
# (label, family, *globs)
PLATS = [
    ("KR260 FPGA DPU", "FPGA", "kr260/kr260_ms_idle_rep*.npy", "kr260/kr260_long_dedicated.npy", "kr260/kr260_long_idle.npy"),
    ("Jetson Orin 10W", "Orin", "orin/ms_orin_10W_rep*.npy"),
    ("Jetson Orin 15W", "Orin", "orin/ms_orin_15W_rep*.npy", "orin/occfpga_orin_sustained_15W_locked.npy"),
    ("L4 GPU",   "GPU", "hpc/ms/ms_L4_rep*_idle.npy",   "hpc/ms/ms_L4_rep*_loaded.npy"),
    ("P100 GPU", "GPU", "hpc/ms/ms_P100_rep*_idle.npy", "hpc/ms/ms_P100_rep*_loaded.npy"),
    ("A100 GPU", "GPU", "hpc/ms/ms_A100_rep*_idle.npy", "hpc/ms/ms_A100_rep*_loaded.npy"),
    ("H100 GPU", "GPU", "hpc/ms/ms_H100_rep*_idle.npy", "hpc/ms/ms_H100_rep*_loaded.npy"),
    ("L40S GPU", "GPU", "hpc/ms/ms_L40S_rep*_idle.npy", "hpc/ms/ms_L40S_rep*_loaded.npy"),
]

rows = []
for label, fam, *globs in PLATS:
    m = margins(*globs)
    if m.size == 0: print("  [skip]", label); continue
    rows.append((label, fam, m))
rows.sort(key=lambda r: np.median(r[2]))   # tightest (KR260) at bottom

rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(8.6, 4.6))
yt, ytl = [], []
for i, (label, fam, m) in enumerate(rows):
    c = FAM[fam]; yt.append(i); ytl.append(label)
    ax.plot([m.min(), m.max()], [i, i], color=c, lw=1.2, alpha=0.30, solid_capstyle="round", zorder=2)
    jit = (rng.random(m.size) - 0.5) * 0.22
    ax.scatter(m, i + jit, s=34, color=c, alpha=0.75, edgecolor="white", linewidth=0.6, zorder=4)
    ax.scatter([np.median(m)], [i], s=120, marker="|", color=c, linewidth=2.4, zorder=5)  # median tick
    ax.text(m.max() * 1.04, i, "max %.1f" % m.max(), va="center", fontsize=8, color=S.SUBTLE)

ax.axvline(1.0, color=S.SUBTLE, ls=":", lw=0.9, zorder=1)
ax.set_yticks(yt); ax.set_yticklabels(ytl)
ax.set_xscale("log"); ax.set_xlim(0.99, 30)
ax.set_xticks([1, 1.5, 2, 3, 5, 10, 20]); ax.set_xticklabels(["1", "1.5", "2", "3", "5", "10", "20"])
ax.set_xlabel("per-run WCET margin   (max / median),   log scale       ← tighter / more deterministic")
ax.set_title("Determinism across the full hardware sweep: KR260 FPGA vs. Jetson Orin (10W/15W) "
             "vs. 5 datacenter GPUs", loc="left", fontsize=11, fontweight="semibold")

from matplotlib.lines import Line2D
leg = [Line2D([0],[0], color=FAM["FPGA"], lw=3, label="FPGA DPU"),
       Line2D([0],[0], color=FAM["Orin"], lw=3, label="Jetson Orin"),
       Line2D([0],[0], color=FAM["GPU"], lw=3, label="datacenter GPU"),
       Line2D([0],[0], marker="|", color=S.SUBTLE, lw=0, markeredgewidth=2.4, ms=12, label="median"),
       Line2D([0],[0], marker="o", color=S.SUBTLE, lw=0, ms=6, alpha=.6, label="one run (seed x cond.)")]
ax.legend(handles=leg, loc="lower right", fontsize=8.5)
ax.grid(axis="x", which="both", color="#EAEDF0")
plt.tight_layout()
os.makedirs("docs/figs", exist_ok=True)
plt.savefig("figs/platform_zoo.png", bbox_inches="tight")
plt.savefig("figs/platform_zoo.pdf", bbox_inches="tight")
print("wrote figs/platform_zoo.{png,pdf}")
for label, fam, m in rows:
    print("  %-18s n=%2d  max/p50 median %.2f  range [%.2f, %.2f]" % (label, m.size, np.median(m), m.min(), m.max()))
