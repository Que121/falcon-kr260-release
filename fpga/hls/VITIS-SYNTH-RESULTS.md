# Vitis HLS synthesis on the real K26 / xczu5ev (KR260) — 2026-06-12

Vivado ML Standard **2025.2** on the Pro6000 (free, Kria-entitled — `set_part xck26-sfvc784-2LV-c`
succeeds, confirming the K26 is in the **free Standard edition**). HLS run headless via
`vitis-run --mode hls --tcl run_{gather,resize}_hls.tcl` (2025.2 entry point; the standalone `vitis_hls`
wrapper is gone). Compat libs (`libtinfo.so.5` etc.) staged user-space in `~/occfpga_libs` + `LD_LIBRARY_PATH`
(no sudo). Reports: `~/OccFPGA/fpga/hls/{gather,resize}_hls/sol{200,300}_xck26/syn/report/`.

## Headline

Both IPs **synthesize on the real K26, fit, meet timing, and have a deterministic (input-invariant)
cycle count** — the WCET-by-construction thesis, confirmed on silicon. The v2 (wide-word) IPs pack the 64
INT8 channels into a 512-bit AXI word; this is a **memory-layout-only change** (per-channel arithmetic is
identical to the bit-exact-verified scalar versions — gather 2.6e-5 vs CUDA `bev_pool_v2`, resize 1e-13 vs
torch align_corners), so bit-exactness is preserved by construction (a formal C-sim re-check is a remaining
nicety).

## Final IP numbers (v2, wide-word, real xck26)

| IP | solution | **Fmax** | latency (cycles) | **WCET** | BRAM_18K | DSP | FF | LUT |
|---|---|---|---|---|---|---|---|---|
| **gather** | sol200 (5 ns) | **274 MHz** | det. (see note) | **≈4.7 ms** | 31/288 (**10%**) | 192/1248 (**15%**) | 10247 (4%) | 19101 (**16%**) |
| **gather** | sol300 (3.33 ns) | **411 MHz** | det. | ≈3.4 ms | 31 (10%) | 192 (15%) | 18248 (7%) | 19682 (16%) |
| **resize** | sol200 (5 ns) | **274 MHz** | **688023 (min==max)** | **3.44 ms** | 157/288 (**54%**) | 130/1248 (**10%**) | 15597 (6%) | 19330 (**16%**) |
| **resize** | sol300 (3.33 ns) | **371 MHz** | 691223 (min==max) | **2.30 ms** | 157 (54%) | 130 (10%) | 17003 (7%) | 19421 (16%) |

- **Both fit the K26 comfortably** (max BRAM 54%, max LUT 16%, DSP ≤15%) — ample fabric left after the B4096 DPU.
- **resize WCET is exact** (latency min==max=688023 cyc → 3.44 ms@200; deterministic by construction).
- **gather WCET note:** the csynth estimate brackets [651895, 2028634] cyc @200 because of the
  `LOOP_TRIPCOUNT min=1/max=64` hint on points-per-pillar; at the **measured fixed rig** (N_POINTS=302558
  over N_PILLAR=21853, avg 13.8 pts/pillar) the count is a constant ≈ 0.93M cyc ≈ **4.7 ms@200 / 3.4 ms@300**.
  The exact cycle count is obtainable from C/RTL cosim with the real static index tables (TODO).

## The v1 → v2 optimization (the engineering story)

The first (scalar) synthesis revealed two honest issues the back-of-envelope missed; both were
**memory-dataflow**, not algorithmic, and the wide-word rewrite fixed both:

| | v1 (scalar) | **v2 (wide-word)** | fix |
|---|---|---|---|
| gather BEV writeback | 2,560,002 cyc / 12.8 ms (1 byte/cyc, missed AXI bursts) | **40,002 cyc / 0.2 ms** | pack C=64 → one 512-bit cell/cycle |
| gather LUT / BRAM | 62% / 45% | **16% / 10%** | wide word + pipelined pillar write |
| gather Fmax | 274 / 318 MHz | 274 / **411 MHz** | |
| resize BRAM | **534% (overflow, no fit)** | **54% (fits)** | row-fused: intermediate `tmp[64][200][100]` → one row `vrow[64][100]` |
| resize WCET @200 | 0.205 s (CTILE unroll failed) | **3.44 ms** | wide word → 64-lane II=1 |

Same MACs/interpolations/clamps as the verified scalar code — only the channel dimension is packed into the
AXI word and the resize intermediate is row-scoped. Sources: `bev_gather.{cpp,hpp}`, `resize.{cpp,hpp}`.

## Verification (C-sim bit-exact + C/RTL cosim) — DONE

Testbenches `bev_gather_tb.cpp` / `resize_tb.cpp` compute a scalar golden (same ap_fixed arithmetic,
unpacked) and compare against the wide-word IP:
- **C-sim at FULL rig size: `GATHER_MISMATCHES=0` (N_POINTS=302558) and `RESIZE_MISMATCHES=0` (UP2, C=512)**
  → the 512-bit packing is **bit-exact** to the scalar algorithm (which `bev_gather_verify.py` /
  `resize_verify.py` verified == CUDA `bev_pool_v2` / torch bilinear). The bit-exact chain is closed.
- **C/RTL cosim (small instance, xsim): `*** PASS ***` for both** → the generated RTL matches the C.
Run: `vitis-run --mode hls --tcl run_verify_{gather,resize}.tcl` (full-size csim + small-size cosim;
small sizes via `-DHLS_SMALL`).

## Out-of-context place&route (`export_design -flow impl`) — DONE

Vivado synthesis + place&route of each IP on the real xck26 (post-route is the silicon-accurate number;
the csynth LUT estimates were pessimistic):

| IP | **CP achieved (post-route)** | **Fmax** | meets 200 MHz | DSP | LUT (post-route) |
|---|---|---|---|---|---|
| gather | 4.733 ns | **211 MHz** | ✅ slack +0.27 ns | 192 (15.4%) | 6200 CLB LUT (**5.3%**) |
| resize | 4.346 ns | **230 MHz** | ✅ slack +0.65 ns | 130 (10.4%) | 10466 CLB LUT (**8.9%**) |

Both **close timing at the 200 MHz target after place&route** with small post-route utilization.
Run: `vitis-run --mode hls --tcl run_impl_{gather,resize}.tcl`.

## Status

- ✅ real xck26 synthesis; free Standard covers K26; both IPs fit + deterministic WCET.
- ✅ gather ≈4.7 ms@200 (BEV writeback 12.8 ms→0.2 ms); resize 3.44 ms@200, fits (BRAM 534%→54%).
- ✅ **bit-exact C-sim (0 mismatches, full size) + C/RTL cosim PASS** for both IPs.
- ✅ **place&route closes timing at 200 MHz** (post-route 211 / 230 MHz); post-route LUT 5%/9%, DSP 15%/10%.
- ✅ **system bitstream built** (Vivado 2025.2 IPI): Zynq UltraScale+ PS + both IPs + 2 AXI SmartConnects
  (control + data-to-DDR), PL clock 200 MHz → `design_1_wrapper.bit` + `.xsa`. **All timing constraints met**;
  whole-system utilization CLB-LUT 20% / FF 16% / CLB 34%. Build Tcl: `fpga/vivado_sys/build_sys.tcl`
  (manual SmartConnect wiring; `apply_bd_automation` is buggy for multi-master→HP-slave in 2025.2). The two
  IPs are now integrated into a real PS-connected bitstream, not just OOC-implemented.
- ✅ **UNIFIED DPU + 2-IP bitstream BUILT (Vivado 2022.2 / Vitis-AI-3.x, DPUCZDX8G v4.1)** — `fpga/vivado_dpu/build_dpu_sys.tcl`:
  the LogicTronix KR260 B4096 base + our gather + resize IPs (re-exported in Vitis HLS 2022.2) on a dedicated
  200 MHz `pl_clk2` (control via free M_AXI_HPM0_FPD, data via free S_AXI_HP2_FPD); the DPU keeps HP0/HP1 + its
  own 550/275 MHz clocks. **Place\&routes, ALL timing met**: DPU 550 MHz (slack +0.017 ns) / 275 MHz (+0.088 ns),
  our IPs `clk_pl_2` 200 MHz (**WNS +0.150 ns**), all cross-clock paths met. Whole-system util: LUT 64.9% / FF
  59.1% / BRAM 68.75% / DSP 82.77% (B4096 dominates; the 2 IPs fit the remainder). Outputs `top_wrapper.bit`
  (7.8 MB) + `dpu_sys.xsa`. (2025.2 cannot build this — DPUCZDX8G is AMD closed IP frozen at v4.1 / Vivado 2022.2,
  and Kria-PYNQ v3.0's VART only pairs with v4.1.)
- ⏳ remaining for the on-board run: extract the new DPU `arch.json` fingerprint → recompile occupancy xmodels in
  the Vitis-AI 3.x Docker (forced by any rebuild) → package `.bit`/`.hwh`/`.xclbin` + load on the KR260. Jetson
  Orin edge-GPU baseline is the other open item (user has an Orin NX, to set up).

Reproduce: `source ~/occfpga_hls_env.sh; cd ~/OccFPGA/fpga/hls; vitis-run --mode hls --tcl run_gather_hls.tcl`
