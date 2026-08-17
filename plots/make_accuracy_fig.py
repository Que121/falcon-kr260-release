#!/usr/bin/env python3
"""Accuracy figure: the on-board INT8 deployment meets/exceeds the INT8-quantized algorithm,
scale-consistently. (a) mIoU board vs INT8-algorithm sim vs FP32 across 256/1024/2048 val frames;
(b) VRU IoU on 2048 frames. Measured on the real KR260 + HPC evals on the same frames.
-> docs/figs/accuracy_int8_match.{pdf,png}
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style as S; S.apply()
import numpy as np, matplotlib.pyplot as plt

subsets  = ["256", "1024", "2048", "6019\n(full)"]
fp32     = [25.0, 30.0, 34.78, 32.19]
int8sim  = [23.13, 27.5, 31.5, 29.08]
board    = [23.77, 28.14, 32.06, 29.88]

vru_lbl  = ["bicycle", "motorcycle", "pedestrian"]   # full 6019-frame validation set
vru_fp32 = [11.47, 14.44, 16.78]
vru_sim  = [7.44, 9.12, 12.2]
vru_brd  = [5.47, 11.13, 12.9]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios":[1.25,1]})
EK = dict(edgecolor="white", linewidth=0.9)

# --- (a) mIoU across subsets ---
x = np.arange(len(subsets)); w = 0.255
b1 = axL.bar(x - w, fp32,    w, label="FP32  (cannot run on DPU)", color=S.GRAY,  **EK)
b2 = axL.bar(x,     int8sim, w, label="INT8 algorithm  (sim)",     color=S.AMBER, **EK)
b3 = axL.bar(x + w, board,   w, label="on-board INT8  (this work)",color=S.TEAL,  **EK)
S.bar_labels(axL, b2, dy=0.5, color=S.SUBTLE); S.bar_labels(axL, b3, dy=0.5, color=S.TEAL, weight="bold")
for xi, (s, b) in enumerate(zip(int8sim, board)):
    axL.text(xi + w, b + 2.7, "+%.1f" % (b - s), ha="center", fontsize=8.5,
             color=S.TEAL, weight="bold")
axL.set_xticks(x); axL.set_xticklabels([s if "\n" in s else s + "\nframes" for s in subsets])
axL.set_ylabel("mIoU"); axL.set_ylim(0, 42)
axL.set_title("(a)  On-board INT8 $\\geq$ INT8 algorithm, at every scale", loc="left")
axL.legend(loc="upper left", ncol=1)

# --- (b) VRU IoU @ 2048 ---
xv = np.arange(len(vru_lbl))
axR.bar(xv - w, vru_fp32, w, label="FP32", color=S.GRAY, **EK)
v2 = axR.bar(xv, vru_sim, w, label="INT8 algorithm", color=S.AMBER, **EK)
v3 = axR.bar(xv + w, vru_brd, w, label="on-board INT8", color=S.TEAL, **EK)
S.bar_labels(axR, v2, dy=0.2, fs=8, color=S.SUBTLE); S.bar_labels(axR, v3, dy=0.2, fs=8, color=S.TEAL, weight="bold")
axR.set_xticks(xv); axR.set_xticklabels(vru_lbl); axR.set_ylabel("IoU"); axR.set_ylim(0, 21)
axR.set_title("(b)  VRU IoU  (6019 frames, full val)", loc="left")
axR.legend(loc="upper right")

fig.suptitle("On-board INT8 occupancy on the KR260 reaches the INT8-quantized algorithm",
             fontsize=13, fontweight="bold", y=1.03, color=S.INK)
plt.tight_layout()
os.makedirs("docs/figs", exist_ok=True)
plt.savefig("docs/figs/accuracy_int8_match.png", bbox_inches="tight")
plt.savefig("docs/figs/accuracy_int8_match.pdf", bbox_inches="tight")
print("wrote docs/figs/accuracy_int8_match.{png,pdf}")
