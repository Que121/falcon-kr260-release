# Real Vitis HLS synthesis on the K26/xczu5ev (KR260)

The two IPs use Xilinx headers (`ap_int.h`, `ap_fixed.h`) + pragmas, so they need **Vitis HLS** (the
open-source Bambu pass only synthesized header-free plain-C cores). This is the path to the real
`xck26-sfvc784-2LV-c` (xczu5ev) resource/timing/latency numbers for TAB7.

## Install (Windows desktop, free for Kria)

1. AMD Downloads → **Vivado (ML Edition)** or **Vitis**, a recent version (2023.2 / 2024.1 — all support
   xczu5ev). Get the **Windows Self-Extracting Web Installer** (`.exe`, ~300 MB).
2. Run it; **sign in with your AMD account** (you type your own password — your machine, your action).
3. Product: **Vivado** (includes Vitis HLS) — smaller than full Vitis.
4. Edition: **Vivado ML Standard** (free, no license). If `set_part` later rejects the part, the K26
   device isn't in the free edition → fall back to a Standard-supported ZU+ proxy (e.g. `xczu3eg`) or a
   free Kria license.
5. **Devices: tick "SoCs → Zynq UltraScale+ MPSoC"** (and Kria if listed) so the xczu5ev/xck26 device
   files install — otherwise `set_part` fails.
6. Install location: **`D:\Xilinx`** (C: is nearly full; D: has space). Web installer downloads only the
   selected components (~40–80 GB).

## Run (headless, no IDE)

```
cd fpga/hls
& "D:\Xilinx\Vitis_HLS\<ver>\bin\vitis_hls.bat" -f run_gather_hls.tcl
& "D:\Xilinx\Vitis_HLS\<ver>\bin\vitis_hls.bat" -f run_resize_hls.tcl
```
Reports land in `gather_hls/sol200_xck26/syn/report/` and `resize_hls/sol200_xck26/syn/report/`.

## Expected (and the point of doing it)

Full-size csynth will likely surface honest sizing facts the estimates/Bambu missed:
- **gather**: the `ZERO_BEV` + writeback touch 40000×64 BEV cells over AXI (~2.5M cycles at 1 B/cyc) —
  far above the 1.5 ms MAC-only estimate → fix = burst/widen the BEV init+write.
- **resize**: `tmp[64][200][100]` ≈ 25 Mb on-chip > the K26's ~5 Mb BRAM → fix = **row-fused** variant
  (compute one output row at a time so `tmp` shrinks to `[CTILE][WIN]`), keeping CTILE=64 and the WCET.

After fixes, re-synth → update TAB7 + §5.4/5.5 WCET with the measured `xck26` numbers (LUT/FF/DSP/BRAM,
fmax/slack, latency cycles). cosim (functional RTL==C) optional — the Python bit-exact checks already
cover correctness against the references.
