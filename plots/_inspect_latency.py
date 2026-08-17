import numpy as np, glob, os
R = "experiments/results"
files = [
 "kr260/kr260_long_idle.npy", "kr260/kr260_long_dedicated.npy",
 "kr260/kr260_rt_idle.npy", "kr260/kr260_rt_loaded.npy", "kr260/kr260_rt_taskset.npy",
 "kr260/kr260_6cam_idle.npy", "kr260/kr260_6cam_taskset.npy",
 "kr260/kr260_fo_idle.npy", "kr260/kr260_fo_taskset.npy",
 "orin/ms_orin_10W_rep1.npy",
 "occfpga_gpu_long_idle.npy", "occfpga_gpu_long_loaded.npy",
 "hpc/occfpga_gpu_H100_idle.npy", "hpc/occfpga_gpu_H100_loaded.npy",
]
print("%-34s %8s %8s %8s %8s %8s %8s" % ("file","n","p50ms","mean","max/p50","p99/p50","CV%"))
for f in files:
    p = os.path.join(R, f)
    if not os.path.exists(p):
        # try orin 15W variants
        print("%-34s  MISSING" % f); continue
    x = np.load(p).astype(float).ravel()
    p50 = np.median(x)
    print("%-34s %8d %8.3f %8.3f %8.3f %8.3f %8.3f" % (
        f, len(x), p50, x.mean(), x.max()/p50, np.percentile(x,99)/p50, 100*x.std()/x.mean()))
print("\n=== orin files present ===")
for f in sorted(glob.glob(os.path.join(R,"orin","*.npy"))): print(" ", os.path.basename(f))
print("=== kr260 files present ===")
for f in sorted(glob.glob(os.path.join(R,"kr260","*.npy"))): print(" ", os.path.basename(f))
