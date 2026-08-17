"""Robustness of the deadline-miss payoff across seeds/devices (not a single cherry-picked run).
Reports miss% at d=1.10 (10% WCET margin) and max/p50, per platform, across all available repeats."""
import numpy as np, glob, os
R = "experiments/results"

def stats(files, d=1.10):
    out = []
    for f in files:
        p = os.path.join(R, f)
        if not os.path.exists(p): continue
        x = np.load(p).astype(float).ravel()
        p50 = np.median(x)
        out.append((os.path.basename(f), len(x), x.max()/p50, 100*np.mean(x > d*p50)))
    return out

groups = {
 "KR260 DPU idle (5 seeds, 30k each)": [f"kr260/kr260_ms_idle_rep{i}.npy" for i in range(1,6)],
 "KR260 DPU dedicated/idle (long)":    ["kr260/kr260_long_dedicated.npy","kr260/kr260_long_idle.npy","kr260/kr260_rt_taskset.npy"],
 "Orin 10W (5 seeds, 30k each)":       [f"orin/ms_orin_10W_rep{i}.npy" for i in range(1,6)],
 "Orin 15W (5 seeds, 30k each)":       [f"orin/ms_orin_15W_rep{i}.npy" for i in range(1,6)],
 "Orin deployed/sustained/cotenant":   ["orin/occfpga_orin_idle_15W_locked.npy","orin/occfpga_orin_sustained_15W_locked.npy","orin/occfpga_orin_sustained2_15W_locked.npy","orin/occfpga_orin_cotenant_15W_locked.npy"],
 "Workstation GPU idle/loaded (long)": ["occfpga_gpu_long_idle.npy","occfpga_gpu_long_loaded.npy"],
 "HPC GPUs H100/L4/L40S idle":         ["hpc/occfpga_gpu_H100_idle.npy","hpc/occfpga_gpu_L4_idle.npy","hpc/occfpga_gpu_L40S_idle.npy"],
 "HPC GPUs H100/L4/L40S loaded":       ["hpc/occfpga_gpu_H100_loaded.npy","hpc/occfpga_gpu_L4_loaded.npy","hpc/occfpga_gpu_L40S_loaded.npy"],
}

for g, files in groups.items():
    s = stats(files)
    if not s:
        print("%-42s  (no data)" % g); continue
    mp = np.array([r[2] for r in s]); ms = np.array([r[3] for r in s])
    print("\n%s  (n=%d runs)" % (g, len(s)))
    print("   max/p50  range [%.2f, %.2f]   miss%%@1.10  range [%.3f, %.3f]  mean %.3f" %
          (mp.min(), mp.max(), ms.min(), ms.max(), ms.mean()))
    for nm, n, mpx, msx in s:
        print("     %-34s n=%-7d max/p50 %.3f   miss%%@1.10 %.3f" % (nm, n, mpx, msx))
