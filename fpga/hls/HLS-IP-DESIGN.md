# HLS view-transform IP — design, verification, WCET

The one stage of the camera→occupancy pipeline the DPU cannot run (data-dependent scatter), implemented
as a **custom HLS IP** with a **certifiable, input-invariant WCET**. This is the OccFPGA hardware contribution.

## Why it's needed
FlashOcc's LSS view transform (BEVPoolv2) lifts per-pixel features into a BEV grid by a **scatter-add**
with data-dependent indices — not a DPU-supported op. Naively it runs on the ARM (slow, jittery) or needs
custom PL. We harden it as an HLS gather. **Key enabling fact:** for a *fixed camera rig* the projection
indices (`ranks_depth/ranks_feat/ranks_bev`, `interval_*`) are **static constants** (functions only of
intrinsics/extrinsics/preprocessing). So the op reduces to a **fixed-iteration weighted scatter-add** —
the cycle count is a constant, independent of scene content.

## Operation (verified bit-exact vs CUDA bev_pool_v2)
For each occupied BEV pillar `p` (segmented by `interval_start/len`):
```
acc[0..C-1] = Σ over its points i:  depth[ranks_depth[i]] · feat[ranks_feat[i]][0..C-1]
bev[ranks_bev[p]] = requantize(acc · out_scale)
```
`fpga/bev_gather_verify.py` confirms both the dense scatter-add and the segmented (HLS) form reproduce
FlashOcc's CUDA `bev_pool_v2` to **max_abs_err 2.6e-5 / rel 1.4e-7** (float precision) on a real frame.

## Sizing (FlashOcc-R50 @256×704, Occ3D-nuScenes 6-cam, measured)
| qty | value |
|---|---|
| N_POINTS (fixed) | **302 558** |
| N_PILLAR | 21 853 |
| C (feature channels) | 64 |
| depth_len / feat_vecs | 371 712 / 4 224 |
| BEV grid | 200×200 (Dz=1) → 40 000 cells |
| MACs / frame | 19.36 M |

## WCET (the headline property)
Inner-loop iterations = **N_POINTS = 302 558, a constant**. With II=1 and the C=64 lanes unrolled:
```
cycles ≈ NUM_BEV(zero) + N_POINTS(accumulate) + N_PILLAR(write) ≈ 364 411
WCET = 1.82 ms @200MHz   /   1.21 ms @300MHz
```
**Input-invariant**: the cycle count is identical for the densest and emptiest scene — exactly the
certifiability the paper argues a GPU cannot give (whose LSS/scatter cost is data-dependent). This is the
hardware embodiment of leg-1 (data-independent compute) of the three-legged certifiability argument.

## Microarchitecture (`bev_gather.cpp`)
- **Static index tables** (`ranks_*`, `interval_*`) resident in BRAM/URAM — programmed once per rig.
- **Per-frame tensors** (`feat` INT8 from the DPU trunk+neck, `depth` Q0.8) streamed over AXI4 to/from DDR.
- **Segmented accumulation**: one C-wide accumulator per pillar, single write → no read-modify-write
  scatter hazard → inner point loop pipelines at **II=1**, C lanes unrolled (64 MACs/cycle).
- INT8 in / wide fixed accumulate / requantize to INT8 out → drops straight into the BEV DPU stage.

## Status
- ✅ Algorithm verified bit-exact (numpy, real frame).
- ✅ Synthesizable HLS C++ written (`bev_gather.{hpp,cpp}`), WCET analyzed by construction.
- ⏳ Synthesis (Vitis HLS 2022.1 → LUT/FF/BRAM/DSP/fmax + C/RTL cosim against the dumped golden vectors)
  pending the toolchain install.
- ⏳ Vivado block design (PS + this IP + DPU + AXI-DMA) → `.bit` → KR260 deploy.

## Next (once Vitis is installed)
1. `vitis_hls` C-sim with the golden vectors (export from `lss_dump.npz`), then C/RTL cosim for bit-exactness.
2. Synthesize → record resources + fmax; confirm II=1; report the cycle-accurate WCET.
3. Vivado integration + on-board end-to-end (camera trunk DPU → this IP → BEV DPU → occ head).

---

# IP #2 — bilinear resize (`resize.{hpp,cpp}`)

The on-board deployment attempt (`RESULTS-SO-FAR.md`) showed FlashOcc's FPN_LSS neck `nn.Upsample(bilinear,
align_corners=True)` ops are **not DPU-deployable** (bilinear → 10 CPU subgraphs + no board graph_runner;
nearest → `export_xmodel` fails). On the ARM they reintroduce OS jitter, defeating the WCET goal. So the
*second* required custom datapath IP: a deterministic bilinear resize.

- **Same enabling insight**: for fixed dims the source coords + interpolation weights are static
  (`y0/y1/wy`, `x0/x1/wx`) → a fixed-iteration **separable 2-tap weighted gather** (vertical then horizontal).
- **Verified bit-exact** vs torch `Upsample(bilinear, align_corners=True)`: **max_abs_err 1e-13**
  (`fpga/resize_verify.py`), both FlashOcc instances (25→100 ×4, 100→200 ×2, C=512).
- **WCET (by construction, input-invariant)**:
  - UP2 (100→200): 480k cycles → **2.4 ms @200MHz / 1.6 ms @300MHz**.
  - UP  (25→100): 100k cycles → **0.5 ms @200MHz**.
- Synthesizable HLS C++ written: separable, C-lane parallel (CTILE=64), INT8 in/out, AXI + static BRAM taps.

## The complete custom datapath (two IPs + DPU)
```
6× img → [trunk+neck+depthnet]DPU → [gather IP] → [BEV-backbone]DPU
       → [resize IP ×2 in the neck] interleaved with [BEV-conv]DPU → occ-head(tiny CPU softplus)
```
The DPU runs the dense convs; the **two custom HLS IPs (gather + resize)** cover exactly the ops the DPU
cannot (data-dependent scatter, bilinear resize) — each fixed-iteration → input-invariant WCET. Together
they make the *whole* camera→occupancy datapath WCET-certifiable on PL, which is the paper's central claim
and the author's own hardware contribution. (Both verified bit-exact; synthesis pending Vitis HLS 2022.1.)
