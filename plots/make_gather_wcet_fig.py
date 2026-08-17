#!/usr/bin/env python3
"""#2 gather-IP WCET-under-load figure: the deployed INT16 gather IP's compute is deterministic by
construction (p50 invariant) and only the host poll tail grows under co-tenant load -- the same host-path
mechanism as the DPU/resize (leg-3). Latency CDF (normalised by idle p50), idle vs +3 CPU burners.
Data: experiments/results/kr260/dualip/gather_wcet_{idle,load}.npy. -> docs/figs/gather_wcet_load.{pdf,png}
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style as S; S.apply()
import numpy as np, matplotlib.pyplot as plt

D = "experiments/results/kr260/dualip"
idle = np.load(os.path.join(D, "gather_wcet_idle.npy")).astype(float)
load = np.load(os.path.join(D, "gather_wcet_load.npy")).astype(float)
p50 = float(np.median(idle))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.2))

def cdf_xy(x):
    xs = np.sort(x / p50); return xs, np.arange(1, len(xs)+1)/len(xs)

# (a) CDF, filled
for x, lab, c in [(idle, "idle", S.TEAL), (load, "+3 CPU burners", S.CORAL)]:
    xs, y = cdf_xy(x)
    axL.plot(xs, y, color=c, lw=2.4, label="%s   (max/p50 %.3f)" % (lab, x.max()/p50), solid_capstyle="round")
    axL.fill_between(xs, 0, y, color=c, alpha=0.07)
axL.axvline(1.0, color=S.SUBTLE, ls=":", lw=1.0)
axL.text(1.001, 0.06, "p50 = %.1f ms\n(invariant)" % p50, fontsize=8.5, color=S.SUBTLE)
axL.set_xlabel("latency / idle median"); axL.set_ylabel("cumulative fraction")
axL.set_xlim(0.985, max(idle.max(), load.max())/p50 * 1.02); axL.set_ylim(0, 1.02)
axL.set_title("(a)  gather IP latency CDF", loc="left"); axL.legend(loc="center right")

# (b) tail CCDF, log-y
for x, lab, c in [(idle, "idle", S.TEAL), (load, "+3 CPU burners", S.CORAL)]:
    xs = np.sort(x / p50); ccdf = 1.0 - np.arange(len(xs))/len(xs)
    axR.semilogy(xs, np.clip(ccdf, 3e-4, 1), color=c, lw=2.2, label=lab, solid_capstyle="round")
axR.set_xlabel("latency / idle median"); axR.set_ylabel("P(latency > x)")
axR.set_xlim(0.985, None); axR.set_ylim(3e-4, 1)
axR.set_title("(b)  tail  (1 - CDF, log scale)", loc="left"); axR.legend(loc="upper right")
axR.grid(True, which="both", color="#EAEDF0"); axR.grid(True, which="minor", color="#F3F5F7", lw=0.7)

fig.suptitle("Gather IP: PL compute is deterministic (p50 invariant); only the host tail grows under load",
             fontsize=12.5, fontweight="bold", y=1.03, color=S.INK)
plt.tight_layout()
os.makedirs("docs/figs", exist_ok=True)
plt.savefig("docs/figs/gather_wcet_load.png", bbox_inches="tight")
plt.savefig("docs/figs/gather_wcet_load.pdf", bbox_inches="tight")
print("wrote docs/figs/gather_wcet_load.{png,pdf} | idle p50=%.2f max/p50=%.3f | load max/p50=%.3f"
      % (p50, idle.max()/p50, load.max()/p50))
