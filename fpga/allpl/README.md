# All-PL feasibility: de-fragmenting the BEV graph (2026-06-12)

The hybrid DPU+PS end-to-end run (`fpga/M4-RESULTS.md`) works but runs the fragment ops on the ARM
PS. The all-PL datapath, which carries the certifiable latency bound, needs the graph to stop
fragmenting at every activation so the backbone stays on the DPU and only the two upsamples move to
the resize IP. These scripts prove that is achievable.

## Root cause of the 21-subgraph fragmentation

The compiled `bev_clamp32.xmodel` fragments into 21 subgraphs (1 USER, 10 DPU, 10 CPU). The cause is
**not** the upsamples alone: it is `nndct_hardtanh`, the `Hardtanh(0,32)` clamp that fixes the LSS
outlier. The DPUCZDX8G only recognises ReLU / ReLU6 / LeakyReLU / hard-sigmoid/swish, so every
`Hardtanh(0,32)` becomes a CPU op and breaks every BasicBlock.

## Result: a DPU-native activation collapses 21 -> 7 subgraphs

Re-quantising with a DPU-native activation and counting compiled subgraphs (`vai_c_xir` for the
KR260 arch `0x101000016010407`):

| variant | subgraphs (USER/DPU/CPU) | occ overall | occ VRU | occ **geom** |
|---|---|---|---|---|
| `Hardtanh(0,32)` (baseline) | 21 (1/10/10) | 0.458 | 0.518 | 0.936 |
| plain `ReLU` | **7 (1/3/3)** | 0.374 | 0.433 | 0.575 |
| `ReLU` + forced clamp fix-points | **7 (1/3/3)** | 0.408 | **0.510** | **0.988** |

The 3 CPU subgraphs in the ReLU graph are exactly `upsample-fix`, `upsample-fix`, `fix2float`: the
two bilinear upsamples (the resize IP's job) plus a trivial cast. The whole backbone and the concat
are on the DPU.

## How the clamp is kept without the Hardtanh op

The `Hardtanh(0,32)` and the INT8 fix-point saturation are the same operation when the activation's
`fix_point` is 2 (int8 saturates at `127 * 2^-2 = 31.75 ~= 32`). Plain ReLU alone lets min-max
calibration chase the 34x LSS outlier and pick a large fix-point, so the clamp is lost and geometry
collapses (0.575). Transferring the clamp model's small per-tensor fix-points (2,2,3,3,4,4,5 ...)
onto the ReLU graph (`export_with_fps.py`) realises the clamp through saturation: the graph stays
DPU-native and de-fragmented (7 subgraphs) and geometry/VRU recover (0.99 / 0.51), matching or
beating the Hardtanh clamp model. Overall is slightly below the clamp baseline on this single frame,
attributable to the order-based fix-point transfer and the 16-frame recalibration; a full-calibration
pass is the polish step.

## DONE: the all-PL run executes on the board

The 25->100 (x4) C=512 resize overlay was built (`build_resize_2510.sh`, HLS+Vivado 2025.2, timing
closed) and, with the existing 100->200 overlay, the de-fragmented `bev_reluc.xmodel` now runs end to
end on the board with **every convolution and activation on the DPU and both bilinear upsamples on the
resize IPs** (`board_allpl_walk.py`): DPU backbone -> resize IP x4 (25->100) -> DPU concat+neck ->
resize IP x2 (100->200) -> DPU final. The walk swaps the PL bitstream between the DPU and resize
stages (a one-shot correctness run; a single DPU+2-resize bitstream is the steady-state form).

On-board fidelity on a real frame (`board_allpl_cmp.py`):

| comparison | conv cosine | occ overall | occ VRU | occ geom |
|---|---|---|---|---|
| **all-PL board vs reluc-sim** (deployment fidelity) | **0.9978** | **0.970** | 0.914 | **0.985** |
| all-PL board vs FP32 | 0.676 | 0.418 | 0.502 | 0.985 |
| reluc-sim vs FP32 | 0.659 | 0.408 | 0.510 | 0.988 |

The all-PL hardware path reproduces the simulator at 0.998 conv cosine / 97% voxel agreement, and the
resize IPs' Q0.9 fixed-point bilinear adds essentially no error beyond quantisation (geom 0.985 vs the
sim's 0.988). The VRU board-vs-sim (0.914) is a touch below the numpy-upsample hybrid (0.979) because
the IP bilinear is 8-bit-fraction vs float, but VRU retention vs FP32 (0.502) still matches the sim
(0.510). Compare the hybrid walk (numpy upsamples): board vs sim 0.998 / 97.7%; the all-PL run matches
it while moving the upsamples off the ARM and onto PL.

## Steady-state form (remaining)

The functional all-PL datapath is now demonstrated. The only remaining integration for a single-frame
per-frame determinism guarantee is one bitstream holding the DPU plus both resize instances (the
unified DPU+gather+resize bitstream is already built and timing-closed; it needs the second resize
instance added), so no PL reconfiguration happens per frame.

## Scripts (run in the Vitis-AI container `xilinx/vitis-ai-pytorch-cpu`, env `vitis-ai-pytorch`)

- `frag_quant.py` (`ACT=relu|relu6|hardtanh32`): quantise+export a BEVStage activation variant.
- `count_subgraphs.py <xmodel>`: count USER/DPU/CPU subgraphs.
- `acc_check.py` (`ACT=...`): sim-INT8 occupancy vs FP32 for a variant.
- `export_with_fps.py`: the all-PL recipe (ReLU graph + clamp fix-points) + accuracy + export.
