# M4 result: full end-to-end occupancy on the KR260 (DONE, 2026-06-12)

The FlashOcc BEV stage now runs end to end on real KR260 silicon, from a real LSS view-transform
output (`vt_out`) to an occupancy volume, and the on-board result matches the deployed INT8 model.

## What runs on the board

`fpga/board/board_bev_walk.py` walks the 21-subgraph `bev_clamp32.xmodel` in one pass:

* the 10 `conv2d-fix` subgraphs run on the **B4096 DPU** via `vart.Runner` (int8 in/out at each
  tensor's `fix_point`),
* every CPU-fragment op runs in numpy on the **ARM PS** (the six `conv1` convs and the neck/up2/final
  convs the compiler could not place on the DPU, the `Hardtanh(0,32)` clamps, the residual adds, the
  two bilinear `align_corners` upsamples, and the 640-channel concat),
* the INT8 quant/dequant is applied at exactly the tensors the compiler tagged with a `fix_point`.

The graph fragments into 21 subgraphs (1 USER, 10 DPU, 10 CPU) because `nndct_hardtanh` and bilinear
upsample are not DPU ops; Kria-PYNQ ships no graph-runner and no `libvart-cpu-runner.so`, so the walk
is hand-rolled. A frame takes about 21 s (the un-accelerated numpy convs dominate).

The XRT-2.13 `xclbinutil` segfault that blocks every overlay load is patched the same way as the IP
probes (`_create_xclbin -> default.xclbin`); the DPU is brought up with `DpuOverlay("dpu.bit")` after
sourcing `/etc/profile.d/pynq_venv.sh` (sets `XILINX_XRT=/usr`).

## On-board fidelity (single real frame, `bev_test_input` = a real vt_out)

Three artifacts are compared: **board** (the on-board walk), **sim-INT8** (the vai_q_pytorch quantized
model run in the Vitis-AI container, `fpga/board/sim_int8_convonly.py`), and **FP32** (the original
`BEVStage(conv_only)`, `fpga/board/bev_ref.py`). Occupancy is the `Linear-Softplus-Linear` predicter
head + argmax over 18 classes on the 200x200x16 grid (`fpga/board/board_three_way.py`).

| comparison | conv_only cosine | occ overall | occ VRU | occ geom |
|---|---|---|---|---|
| **board vs sim-INT8** (deployment fidelity) | **0.9984** | **0.9766** | **0.9793** | **0.9920** |
| board vs FP32 (on-board retention) | 0.6573 | 0.4592 | 0.5263 | 0.9376 |
| sim-INT8 vs FP32 (quantiser retention) | 0.6586 | 0.4583 | 0.5182 | 0.9361 |

Reading:

1. **The board reproduces the deployed quantized model.** conv_only cosine 0.9984 and 97.7% voxel
   agreement (97.9% on VRU, 99.2% geometry) between the on-board walk and the INT8 simulator. The
   ~0.2% gap is DPU hardware rounding vs the simulator's rounding.
2. **The on-board run adds no error beyond quantisation.** board-vs-FP32 (0.459 / 0.526 / 0.938) is
   identical, within noise, to sim-INT8-vs-FP32 (0.458 / 0.518 / 0.936), and matches the documented
   clamp-C=32 retention (overall 0.48 / VRU 0.54 / geom 0.94 in `RESULTS-SO-FAR.md`). The moderate
   conv cosine of 0.66 is the inherent cost of min-max INT8 PTQ with no QAT; it is a property of the
   quantiser, not of the board. (mIoU *retention* stays 98.3% because mean-class IoU is far more
   forgiving than exact per-voxel argmax agreement, which is what the 0.46 measures.)

## Scope / honesty

This is a **hybrid DPU + ARM-PS** end-to-end run: the DPU does the 10 conv subgraphs, the PS does the
fragment ops. It is a genuine end-to-end on-board occupancy result (one KR260 chip, vt_out to occ),
but it is **not yet all-on-PL**. The all-PL variant, needed for the certifiable latency bound, is the
remaining step: replace the two bilinear upsamples with the resize IP (built and run standalone) and
recompile the activations into a DPU-recognised form so the graph stops fragmenting at every clamp.
The functional pipeline and the on-board fidelity are now measured; the all-PL latency path is future
work.

## Reproduce

```
# board (sudo, pynq venv, XRT sourced):
source /etc/profile.d/pynq_venv.sh
python3 board_bev_walk.py            # -> bev_onboard_convonly.npy + cosine vs FP32
python3 board_three_way.py           # board vs sim vs FP32 (needs bev_sim_int8_convonly.npy, predicter_head.npz)

# Pro6000 (ANONPROJ_310): bev_ref.py        -> FP32 reference + intermediates + predicter_head.npz
# Vitis-AI container (vitis-ai-pytorch): sim_int8_convonly.py -> bev_sim_int8_convonly.npy
```
