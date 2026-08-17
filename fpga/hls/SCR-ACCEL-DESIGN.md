# Custom PL SCR accelerator — fast + FP32-parity on-board localization (the "have-both" path)

Motivation: on the KR260, localization to FP32 parity is otherwise a speed↔accuracy wall — DPU INT8 is fast
(150 ms) but floored (2.42°/76cm/0% hit@10cm), ARM FP32 is accurate (0.62°/15cm/34%) but slow (14.9 s). A custom
PL SCR datapath at higher precision breaks the wall.

## P0 — precision gate PASSED (`experiments/acesplat/scr_int16_gate.py`)
Fake-quantize all 20 SCR convs (weights+activations) per-tensor, measure coord cosine + cv2-PnP on the 103 reloc
test frames:
| precision | cos vs FP32 | med rot | med trans | hit@(10cm,5°) |
|---|---|---|---|---|
| FP32 | 1.0000 | 0.62° | 0.143u | 34.0% |
| INT8 (sim) | 0.9049 | 1.48° | 0.342u | 4.9% |
| **INT16 (sim)** | **1.0000** | **0.63°** | **0.149u** | **35.0%** |
| INT12 (sim) | 0.9985 | 0.59° | 0.148u | 35.0% |
The fake-INT8 sim (cos 0.905) MATCHES the real DPU INT8 (0.90) → the sim is faithful. **INT16 (even INT12) =
FP32 parity.** So a custom INT12–INT16 PL SCR engine gives fast + FP32 localization. Gate passed.

## P1 — SCR workload (`experiments/acesplat/scr_layers.py`)
20 conv layers, **7.66 M params, 52.51 GMAC/frame**. Encoder downsamples 480×854 → 60×107 (conv2/3/4 stride 2);
heaviest layers res2_conv3 (15.1 GMAC), res2_conv1 (7.6), res1_conv1/3 (3.8 ea); heads are 1×1 512→512 (~1.7 ea).

## P2 — INT16 MAC engine synthesizes (`fpga/hls/conv16.cpp`)
Representative 1×1 INT16 conv tile, full IC unroll = 512 INT16 MACs: **512 DSP (41%), 17154 LUT (14%), 361 MHz**
on xck26 → **1 DSP per INT16 MAC** (efficient). Fits with the render IP (242 DSP) and NO DPU (the INT16 SCR
replaces the floored INT8 DPU SCR): 512+242 = 754 < 1248 DSP.
- Latency: the naive single-adder-tree gives **II=8** (~64 MAC/cyc → ~2.3 s) — wasteful.
- A proper **output-stationary systolic array (II=1)** → 512 MAC/cyc × 361 MHz = **185 GMAC/s → ~284 ms** ideal,
  ~400–570 ms with weight-stream/layer-transition overhead.
- Either way **~30–50× faster than the ARM FP (15 s) AND FP32-accurate** (vs the DPU's floored 4.9% hit).

## Status + remaining
- P0 ✅ INT16=FP32 parity. P1 ✅ workload. P2 ✅ INT16 MAC engine fits (512 DSP, 361 MHz, 1 DSP/MAC).
- **P3 (the major build)**: the full 20-layer SCR engine — output-stationary systolic dataflow (II=1), weight
  streaming from DDR (15.3 MB INT16), feature tiling in BRAM/URAM, the per-layer sequencer, requant between
  layers. This is the multi-week implementation.
- **P4**: block design + bitstream (the render-IP flow is validated) + on-board deploy → measure speed + the
  on-board INT16 localization accuracy (expect ~FP32 parity per P0).

So: **fast + FP32-parity on-board localization is FEASIBLE via a custom INT16 PL SCR accelerator** (P0–P2 prove
precision + MAC fit + throughput); P3–P4 is the build.

## P3 core — II=1 systolic conv SYNTHESIZES (`fpga/hls/conv_sys.cpp`)
Output-stationary: OCP=32 output channels in parallel × ICU=16 input-channel unroll = 512 INT16 MACs with 32
INDEPENDENT accumulators (no 512-deep adder-tree recurrence). xck26 csynth: **512 DSP (41%), 9884 LUT (8%),
379 MHz** — II driven from 8 (naive) down to **1** on the inner IC loop. Tile latency 7297 cyc = the per-pixel
acc-init/output-write overhead is not yet overlapped → ~144 MAC/cyc effective → **~960 ms** full-SCR naive;
overlapping the pixel loop (pipeline the whole tile) reaches the full 512 MAC/cyc → **~284 ms**. Both are FP32-
parity (INT16) and fit (512 DSP + render-IP 242 < 1248, no DPU).

## Feasibility VERDICT (P0–P3 core all proven)
**The custom PL SCR accelerator for fast + FP32-parity on-board localization is FEASIBLE and de-risked:**
- precision: INT16 = FP32 parity (P0, cos 1.0 / 35% hit, sim faithful vs real DPU)
- resource: 512 INT16 MACs = 512 DSP (41%), 1 DSP/MAC, 379 MHz, fits with the render IP, no DPU (P2/P3)
- throughput: ~284 ms (tuned) to ~960 ms (naive) → **~15–50× faster than the ARM FP (15 s) AND FP32-accurate**
  (vs the DPU INT8's floored 4.9% hit)

**Remaining = the multi-week BUILD** (the user authorized "不管时间"): P3 full = the 20-layer SCR engine
(pixel-loop pipelining for 512 MAC/cyc, weight streaming from DDR for the 15.3 MB INT16 weights, feature tiling
in BRAM/URAM, the per-layer sequencer + 3×3-vs-1×1 handling + inter-layer requant); P4 = block design + bitstream
(render-IP flow validated) + on-board deploy → measure the on-board INT16 localization (expect FP32 parity) and
the latency. The hard unknowns (does INT16 reach FP32? do the MACs fit? can II=1?) are all answered YES.

## P3 streaming layer — DDR weight-stream + II=1 core SYNTHESIZES (`fpga/hls/conv_stream.cpp`)
Output-stationary layer: per OC-block (OCP=32) stream OCP×IC weights DDR→BRAM ONCE, reuse across all pixels, then
the II=1 systolic core. xck26 csynth: **512 DSP (41%), 342 MHz**. Confirms weight streaming amortizes: at NP=64
(small test tile) the weight load dominates (1.52 ms / 16.7 M MAC), but at the real NP=6420 (60×107) the weights
reuse over 6420 pixels → **compute-bound → ~300 ms full SCR** (52.5 GMAC / 175 GMAC/s). DDR weight bandwidth is
fine (loaded once per OC-block, not per cycle). NB the 30% LUT is the weight array in distributed RAM — a proper
BRAM-buffered (double-buffered) weight store fixes that. So the streaming-layer architecture is validated:
compute-bound, weight-stream amortized, 512 DSP fits. Full engine = the 20-layer sequencer + BRAM weight/feature
buffering + 3×3 kernel loop (multi-week build, on track).

## S1 multi-layer dataflow — the key architecture constraint (`fpga/hls/scr_2layer.cpp`)
Chaining 2 conv layers as SEPARATE hardware instances = **1024 DSP (82%)** (2×512) — so 20 separate layers would
need ~10240 DSP >> 1248. **The full engine MUST time-multiplex ONE 512-MAC conv engine, reused across all 20
layers** (a layer-sequencer loop with `#pragma HLS ALLOCATION limit=1`), keeping DSP at ~512. Features ping-pong
between two on-chip buffers; but a 60×107×512 map = 6.6 MB and two don't fit the ~10 MB BRAM+URAM, so the engine
processes **feature tiles** (spatial/channel blocks) streamed through the layers (the standard tiled-CNN
dataflow; 3×3 layers need a halo). The intermediate features stay on-chip within a tile (no DRAM round-trip).
So the full-engine architecture is fixed: **1 time-multiplexed 512-MAC engine + per-layer weight stream + tiled
ping-pong features + a 20-layer sequencer + inter-layer requant**. This is S1's multi-week implementation; the
dataflow constraints (time-multiplex, tile) are now nailed down.

## ✅ S1 architecture VALIDATED — time-multiplexed SCR engine (`fpga/hls/scr_engine_tm.cpp`)
ONE conv engine reused across NL=6 layers (`#pragma HLS ALLOCATION limit=1`) + ping-pong BRAM features:
**514 DSP (41%), 64 BRAM (22%), 342 MHz** — DSP stays ~512 regardless of layer count (vs 6×512 per-layer). So
the full 20-layer engine = ONE time-multiplexed 512-MAC INT16 engine = **~514 DSP, fits the xck26** (+ render IP
242 = 756 < 1248), 342 MHz. **The full-engine DSP architecture is confirmed buildable and scales.**

Remaining for the complete S1 build (multi-week): the real 20-layer configs + per-layer weight DDR offsets +
3×3 kernel loop + the EARLY large layers (480×854 features → spatial tiling with halo, they exceed on-chip) +
inter-layer requant/scale; then S2 (bitstream block design + P&R, render-IP flow validated) + S3 (on-board
deploy → measure INT16 localization = FP32 parity per P0, latency ~300ms). Every architectural unknown
(precision, MAC fit, II=1, weight-stream, layer time-multiplex) is now answered — it is a pure build.

## ✅✅ S2 BITSTREAM BUILT — deployable SCR accelerator (2026-06-25)
`fpga/hls/scr_engine_deploy.cpp` (3 m_axi: weights/input/output, time-multiplexed conv core, ping-pong features)
exported (Vitis HLS) → Vivado block design (PS kr260 + scr_0 + 3×HP + GP0, adapted from the render-IP
`build_overlay3.tcl` flow) → P&R: **write_bitstream Complete, WNS +0.59 ns (timing MET), 0 Errors,
`scr_engine.bit` 7.8 MB** on Pro6000 `~/emerge/`. The deployable SCR-accelerator bitstream flow WORKS (PS+AXI+P&R,
timing closed). This is a representative engine (NL=6, NP=64 tile) proving the flow end-to-end; the full 20-layer
engine (real configs + spatial tiling) is the refinement that reuses this exact flow.

### Remaining: S3 on-board (BLOCKED on board recovery)
The KR260 is currently thrashed/unreachable (the earlier ARM-only training demo OOM-swapped it) — needs a physical
power-cycle. Once recovered: deploy `scr_engine.bit` via PYNQ, allocate weight/in/out DRAM buffers, run, measure
latency + the on-board INT16 SCR coords (expect FP32 parity per the P0 gate). The bitstream is built and waiting.

## ✅✅✅ S3 ON-BOARD — BOTH accelerators RUN on the real KR260 (2026-06-25)
After recovering the thrashed board (pkill the stuck ARM demo → memory freed, no swap), deployed BOTH custom
accelerator bitstreams via PYNQ on the real KR260:
- **SCR accelerator `scr_engine.bit`: 23.06 ms/call, power idle 3.24 W / load 3.52 W** (`deploy/deploy_scr_engine.py`,
  `deploy/power_scr.py`) — representative NL=6 engine. The bitstream loads, the IP register map matches
  (wts/in_ddr/out_ddr/nl), it runs.
- **Training-step accelerator `train_engine.bit`: 86.92 ms/step (N=2048)** (`deploy/deploy_train_step.py`) — the
  on-board fwd+bwd training core runs (register map gp_offset/gtile/gg_offset/ng).

**The full synth→bitstream→ON-BOARD-RUN flow works end-to-end on real hardware for both custom accelerators**,
at ~3.5 W (the ~5 W SWaP class = the goal's power advantage). These are REPRESENTATIVE engines (SCR NL=6, train
N=2048) that prove the deployable flow + the on-board latency/power class; the full proper engines (real 20-layer
SCR for the actual INT16=FP32 localization, full training-step for the actual per-scene training to 31) are the
multi-week refinement that reuses this exact, now-proven, on-board-validated flow. S0→S3 all done; only the
full-engine refinement remains for the real on-board metrics.

## ⚠️ HONEST CORRECTION (2026-06-25): P0's "INT16/INT12=FP32" was fake-quant-optimistic
P0 used FAKE-quant (quantize→dequantize→FP32 conv): INT12 gave coord cosine 0.9985. But the REAL INTEGER SCR
(int intermediates + per-layer requant — what the accelerator actually computes) is far worse: a single head
layer (per-channel INT12 + exact requant) = cosine ~0.91 (`prep_head4.py`); the full integer heads (per-channel
INT12 + rounding requant + residuals + fc3) = **cosine 0.71** vs FP32 (`int_heads.py`). Fake-quant does NOT
capture the integer-requant precision loss. **So producing FP32-parity coords via the custom INT accelerator is
substantially HARDER than P0 implied** — it needs DPU/Vitis-AI-class QAT quantization (per-channel + multiplier
requant + activation calibration + bias correction; even the DPU's INT8 only reaches cosine 0.90). The naive
per-channel INT12 gives 0.71. This is the honest depth: the fast+FP32 localization accelerator needs a full
QAT-level fixed-point quantization pipeline = the multi-week-to-month research, not a quick deploy. The on-board
accelerator RUNS (23ms/3.5W) and the FP32 localization metric is validated (via the ARM ref); fusing them (the
accelerator producing FP32-parity coords) is the genuine remaining quantization research.

## ✅✅ CORRECTION OF THE CORRECTION (2026-06-25): the "0.71" was a BIAS BUG — integer SCR IS FP32-parity
The previous entry's "real integer SCR = cosine 0.71, needs multi-week QAT" was WRONG: my integer sim's `iconv`
extracted `m.weight[:,:,0,0]` but **dropped the conv biases** (res3_conv1/fc1/fc3 all have bias=True). Tell-tale:
cosine was IDENTICAL 0.7133 across INT12/16/19 — bit-width-independent error = a logic bug, not quantization (the
requant `q*sOut` cancels MX, so it was already ~FP-equivalent; a 0.71 gap between two FP-equivalent paths = a
structural omission). With biases added (`int_heads2.py`), and then the FULL pipeline via a QConv monkeypatch on
every Conv2d — per-tensor act + per-channel weight INT quant + integer MAC + per-channel dequant + bias
(`int_full_scr.py`, `deploy/int_loc.py`):

| precision | FULL-SCR coords cosine vs FP32 (encoder 3x3 + heads) |
|-----------|------|
| INT16 | **0.99999** (localization: coords cos 0.99998, pose tracks FP32 per-frame) |
| INT12 | 0.99988 |
| INT10 | 0.99849 |
| INT8  | 0.95813 |

**So the custom INT16 accelerator DOES produce FP32-parity SCR coords -> FP32-parity localization. P0's
INT16=FP32 was CORRECT all along.** The accelerated localization is NOT blocked by quantization. Engine
implications: per-channel weight scales + per-layer requant + **bias add** (was missing) + INT16 needs **int64
accumulators** (INT16xINT16 x 4608-term 3x3 encoder conv = 43 bits; x 512-term 1x1 head = 39 bits — both exceed
int32). int64-accum INT16 MAC costs more DSP than the INT12/int32 path but is feasible. Remaining = BUILD the full
20-layer INT16 engine (encoder 3x3 + heads 1x1 + bias + requant + int64 accum) + ARM orchestration of
residuals/relu — a multi-day-to-week build, but de-risked: the precision is proven, not a QAT research problem.

## ✅ ENGINE ARITHMETIC LOCKED + VALIDATED (2026-06-25): exact fixed-point = FP32
The full deployable engine spec is now validated bit-exactly against FP32 (`experiments/acesplat/eng_sim.py`):
per-tensor act INT16 + per-channel weight INT16 + int64 MAC + **per-channel fixed-point multiplier requant**
M_int[c]=round((sW[c]*sA/sOut)*2^SHIFT) + **int bias** B[c]=round(bias[c]/sOut), with **per-layer SHIFT** chosen so
M_int.max ~ 2^20 (precision) AND acc*M_int stays < int64 (the earlier SHIFT=20 underflowed: scales are ~1e-5 so
M~1e-7 needs SHIFT~34-36). On res3_conv1->2->3 (real 512->512 head seq): cosine **1.00000**, rel-err 0.0006.
acc*M_int max ~ 4.6e15 << int64 9.2e18 (safe). So the engine's exact integer datapath IS FP32. Remaining is the
HLS implementation of this locked arithmetic: tiling (512x6420x2B=6.6MB feature map exceeds on-chip -> stream
pixel-chunks from DDR like the existing scr_engine) + ic-unroll pipelining + per-channel requant tables. The
ARITHMETIC is no longer a risk; the build reuses the proven scr_engine tiling/DDR-stream structure.

## ✅ HLS ENGINE csim VALIDATED (2026-06-25): scr_head16.cpp = FP32 (cosine 1.00000)
`scr_head16.cpp` implements the locked arithmetic (per-channel fixed-point multiplier requant + int bias + int48
accum + per-layer SHIFT, OCP=32xICU=16 time-mux MAC core, ping-pong). Vitis HLS 2022.2 csim on the REAL res3
head sequence (`tb_head16.cpp` reads calibrated wts/mult/bias/shifts/input + FP32 ref binaries): **cosine
1.00000, maxd 0.0011, 0 errors** (shifts 35/36/34). The actual hardware engine C++ — not just the numpy model —
is FP32-parity. Build gotcha (recorded): Vitis HLS 2022.2 on this box needs `libtinfo.so.5` which is absent
(box has 6.3); fix = `ln -sf /lib/x86_64-linux-gnu/libtinfo.so.6.3 ~/lib5/libtinfo.so.5` + prepend ~/lib5 to
LD_LIBRARY_PATH after sourcing settings64.sh, else vitis_hls hangs at startup ("couldn't load
libxv_commontasks.so"). Next: csynth (resource/timing on xck26) -> bitstream -> on-board.

## 🎯 ON-BOARD FP32-PARITY (2026-06-25): scr_head16.bit runs on REAL KR260 = FP32 (cosine 1.00000)
The FP32-parity SCR head engine is BUILT and RUN on real hardware. Bitstream `scr_head16.bit` (WNS +0.67ns,
517 DSP/41%, 128 BRAM/44%, Fmax 415MHz) deployed via PYNQ on the KR260; `deploy_head16.py` tiles the 6420
pixels through the 128-pixel engine (1x1 conv is pointwise -> 51 independent tiles), nl=3 res3 head sequence.
Board output (dequantized) vs FP32 reference a3: **cosine 1.00000, maxd 0.0029** (`deploy/cmp_head16.py`). So
the custom INT16 accelerator, on real silicon, produces FP32-parity SCR output -- closing the
"用加速器产出真实坐标" gap for the res3 sequence. Register map: wts_1/in_ddr_1/out_ddr_1/mult_1/bias_1/shifts_1/nl.
**Latency caveat (honest): 955ms total / 18.73ms per tile -- Python/PYNQ per-tile overhead dominated (sync +
AP_DONE poll + weight re-read), NOT the engine compute (~1ms/tile). Needs driver optimization (larger NPT,
interrupts, batched sync, weight caching) before the speed-advantage claim holds.** Remaining: (1) latency opt,
(2) full head orchestration (3 engine calls + ARM residuals + fc3), (3) full 5-frame localization via the
accelerator. The PRECISION path is proven end-to-end on hardware; speed is the next work.

## 🎯 FULL HEAD on-board = FP32 + STREAMING engine for speed (2026-06-25)
**FULL SCR head (all 8 conv layers + residuals + fc3) on-board via the accelerator = FP32, coords cosine
1.00000** (`deploy/deploy_fullhead.py`: 3 v2-engine calls res3|res-block|fc + ARM residual adds + fc3). The
localization-precision path is fully proven on real hardware end-to-end.

**Speed: the bottleneck chain + the streaming fix (user-directed "流式").** v1 868ms = per-call 76MB weight
re-read. v2 720ms = weights cached but input/output feature-map DDR load/store loops un-pipelined (13MB
un-bursted ~650ms). Pipelining helps but the wt-cache-reload-per-tile is the next floor. The real architecture =
**STREAMING/DATAFLOW** (`scr_conv_stream.cpp`): weights-stationary (loaded once into 256 BRAM banks), pixels
STREAM through a 256-MAC array via HLS DATAFLOW (read_px -> mac_px -> write_px run concurrently, overlapping DDR
I/O with compute), pixel-major layout for contiguous bursts, one layer per call, ARM chains layers (ping-pong
DDR) + residuals. csim FP32 cosine 1.00000, Fmax 415MHz, fits exactly (BRAM 288/288 after moving the 2 stream
FIFOs to SRL/LUTRAM). Expected ~24ms/layer -> ~71ms/res3 (vs 720ms). Drivers: `deploy/deploy_stream_res3.py`
(speed test), `deploy/deploy_stream_fullhead.py` (full head). Bitstream building.

## Speed optimization — full root-cause chain + honest floor (2026-06-25)
Chased the SCR-head latency across the deployable engines, root-causing each bottleneck on real hardware:
- v1 (per-tile separate calls): 868ms = re-reads 1.5MB weights/call x51 = 76MB DDR.
- v2 (weight-cached, internal tiling): 720ms = input/output feature-map DDR load/store loops un-pipelined.
- streaming (weights-stationary 256-MAC + DATAFLOW, narrow 16-bit m_axi): 211ms.
- streaming + partial-accumulator II=1 attempt: 698ms (WORSE — 256 parallel MACs+accs miss timing at 415MHz II=1).
- streaming + WIDE 512-bit m_axi (max_widen only reached 32-bit; explicit ap_uint<512> datapath reached 512-bit,
  csim FP32, m_axi RDATA=512 confirmed): **176ms**. Only ~16% over the narrow stream -> **the DDR was NOT the
  dominant bottleneck; the MAC inner-loop recurrence is** (acc+=ps over 16-term reduction -> II~11, ~11400
  cycles/pixel measured vs ~1056 ideal).
**Honest floor: 176ms/res3 (3 layers, ~5 GMAC) at ~3.5W on this 256-MAC engine.** Not yet a perf/W win vs the
GPU heads (~0.5ms@300W); needs the MAC at II=1, which requires a proper systolic-array datapath (separate
register-pipelined PEs) rather than the HLS auto-scheduled reduction -- a deeper engine rewrite. The PRECISION
path is fully done on-board (FP32 cosine 1.00000, full head); the perf/W-winning speed is the remaining
engine-architecture work. All engine variants + the wide datapath committed.

## MAC II — exhaustive HLS attempt + the wall (2026-06-25)
Tried 4 MAC structures for the streaming 256-MAC engine (wide 512-bit datapath, all csim FP32 cosine 1.00000):
| structure | on-board res3 | why |
|-----------|---------------|-----|
| simple acc[OCP] + ps-reduction | **176ms (best)** | HLS sweet spot |
| partial acc[OCP][ICU] (256 parallel) | 698ms | 256 parallel adds miss timing -> HLS raises II |
| interleaved acc[OCP][8] (distance-8) | 415ms | final 8-term reduce left sequential (264ms) |
| interleaved + unrolled reduce | 659ms | more logic -> WNS +1.17->+0.90 -> HLS adds pipeline cycles |
**Pattern: every II=1 idiom makes it SLOWER** — WNS drops as logic grows, so the HLS scheduler trades II for
timing; the minimal-logic simple MAC has the lowest achieved II. **176ms/res3 @3.5W is the HLS high-level floor
for this engine.** The encoder (3x3, 77% of SCR compute) is still unaccelerated, so even a faster head does not
yield a full-SCR perf/W win. Getting past 176ms needs a HAND-CODED SYSTOLIC ARRAY (explicit register-pipelined
PEs that meet timing at II=1) — a fundamentally deeper engine the HLS auto-scheduler won't produce from C idioms.
Reverted to the simple MAC as the deployed best.
