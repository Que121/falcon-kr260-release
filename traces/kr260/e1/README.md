# E1: per-stage timing + measured power (board session)

Standard dpu.bit overlay (dpu2rz xclbin load currently faults; DPU numbers are overlay-independent).
BEV stage = bev_reluc.xmodel, 3 DPU subgraphs, exec = execute_async..wait only (host marshaling excluded).

DPU exec, 300 frames, idle | load+core-dedicated (p50 / max ms):
  sg1 backbone   54.49 / 54.67   |  54.26 / 55.17
  sg2 + neck    143.87 / 144.43  | 143.92 / 145.47
  final conv    200.95 / 202.18  | 200.35 / 207.59
  BEV DPU total ~399 p50, ~408 max (dedicated)

Composed serial full-res budget (measured worst cases, deployed config):
  6 x trunk 49.83 + gather 53.2 + BEV DPU 408.2 + 2 x resize IP ~39.3 = ~800 ms/frame (~1.2 Hz)
  ARM predicter head not yet measured (next session). CPU upsample stand-ins measured at
  80/385 ms p50 -- 20x the resize IPs, quantifying why the IPs exist.

Power (INA260 u14, board-level): idle 4.33 W, BEV inference p50 5.80-6.58 W, max 8.70 W.

## Measured platform power

| platform | idle | under inference | peak |
|---|---|---|---|
| KR260 (INA260 u14, board) | 4.33 W | 5.80-6.58 W | 8.70 W |
| Orin NX 15W mode (VDD_IN, r50_int8 trtexec-style loop) | ~4.5 W | 7.41 W | 7.45 W |
| RTX PRO 6000 workstation (nvidia-smi, resnet50 fp16 loop, machine concurrently busy = co-tenant condition) | 272.9 W floor observed | 329 W p50 | 381.9 W |

The paper's "~5 W" for the KR260 becomes "4.3 W idle / 6.6 W inference / 8.7 W peak, measured".
