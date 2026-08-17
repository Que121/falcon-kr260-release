# HLS synthesis results (Bambu, open-source)

Both custom IPs synthesized with **Bambu / PandA** (Politecnico di Milano open-source HLS, `bambu-latest`
AppImage on the Pro6000), targeting `xc7z020` at a 5 ns (200 MHz) clock. No AMD login / Vitis required.
Header-free plain-C datapath kernels (`bev_gather_bambu.c`, `resize.c`).

| IP (datapath core) | est. fmax | DSPs | flip-flops | est. area | meets 200 MHz |
|---|---|---|---|---|---|
| gather (view transform, C=64 INT8 MAC) | **214.7 MHz** | 6 | 4603 | 4027 | ✅ (slack +0.34 ns) |
| resize (bilinear 2-tap, C=64) | **206.1 MHz** | 6 | 739 | 762 | ✅ (slack +0.15 ns) |

**Takeaways (real, citable):** both non-DPU datapath ops synthesize, **meet the 200 MHz target**, and are
**very small** (6 DSPs each, well within the XCK26's ~1248 DSPs / the budget left after the B4096 DPU). This
confirms the two custom IPs are not just designed-on-paper but **synthesizable at the target frequency with a
negligible resource footprint**, leaving ample fabric for the DPU.

**Honest caveats (state in the paper):**
- Bambu estimates on `xc7z020` (Zynq-7000), not Vivado/Vitis on the KR260's `xczu5ev` (UltraScale+); the
  final on-board `.bit` + cycle-accurate cosim still need Vivado/Vitis (pending toolchain access).
- Bambu's default schedule time-multiplexes (6 DSPs, not the 64-lane II=1 unroll of the Xilinx-pragma
  version), so its per-call cycle count differs from the by-construction WCET (§5) — the WCET claim rests on
  the *fixed iteration count* (input-invariance), which both schedules share; the absolute latency is a
  fmax×schedule detail to finalize in Vitis HLS.

## Reproduce
```
# Pro6000, Bambu AppImage in ~ ; run via the AppImage wrapper (sets up the bundled toolchain env):
cd ~/bambu_work
TMPDIR=~/bambu_tmp ~/bambu.AppImage --appimage-extract-and-run \
  ~/bambu_work/gather.c --top-fname=bev_gather --clock-period=5 -O3 --device-name=xc7z020-1clg484-VVD
# (kernels must be header-free plain C — the bundled clang-16 can't find Ubuntu 22.04 multiarch glibc headers)
```
