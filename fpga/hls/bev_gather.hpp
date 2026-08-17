#ifndef BEV_GATHER_HPP
#define BEV_GATHER_HPP
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>

// ============================================================================
// OccFPGA — HLS view-transform IP (LSS / BEVPoolv2 as a fixed-index gather)
//
// Replaces FlashOcc's data-dependent LSS scatter (which the DPU cannot do) with a
// precomputed-index weighted scatter-add. For a FIXED camera rig the projection
// indices (RANKS_*) are STATIC constants -> the op is a fixed-iteration gather:
//   for each occupied BEV pillar p:
//       acc[0..C-1] = sum over its points i of  depth[rank_depth[i]] * feat[rank_feat[i]][0..C-1]
//       bev[rank_bev[p]] = acc
// Fixed N_POINTS -> bounded cycles -> certifiable, INPUT-INVARIANT WCET.
//
// v2 (wide-word): the C=64 INT8 channels of a feature/BEV vector are packed into one 512-bit AXI word,
// so (a) the dense BEV writeback bursts one cell/cycle instead of one byte/cycle (the v1 ZERO_BEV was
// 2.56M cycles = 12.8 ms at 1 B/cyc; now ~NUM_BEV cycles), and (b) each point is a single wide feat
// read with the 64-lane MAC unrolled. Per-channel arithmetic is UNCHANGED from the bit-exact-verified
// scalar version (max_abs_err 2.6e-5 vs CUDA bev_pool_v2) -> packing is a pure memory-layout change.
//
// Sizing for FlashOcc-R50 @ 256x704, Occ3D-nuScenes 6-cam rig (measured, extract_lss.py):
//   N_POINTS = 302558 (fixed)   N_PILLAR = 21853   C = 64
//   depth_len = 6*88*16*44 = 371712   feat_vecs = 6*16*44 = 4224   BEV = 200*200 (Dz=1)
// ============================================================================

#ifdef HLS_SMALL              // tiny instance for fast C/RTL cosim (same structure)
#define N_POINTS   64
#define N_PILLAR   8
#define C          64
#define DEPTH_LEN  128
#define FEAT_VECS  16
#define NUM_BEV    16
#else
#define N_POINTS   302558
#define N_PILLAR   21853
#define C          64
#define DEPTH_LEN  371712
#define FEAT_VECS  4224
#define NUM_BEV    40000      // Dz*Dy*Dx = 1*200*200
#endif

typedef ap_int<8>           feat_t;     // INT8 context feature (one channel)
typedef ap_ufixed<8, 1>     depth_t;    // [0,1) depth distribution weight, Q0.8
typedef ap_fixed<24, 12>    acc_t;      // wide accumulator (per lane)
typedef ap_int<8>           bev_t;      // requantized BEV feature (one channel)
typedef ap_uint<32>         idx_t;
typedef ap_uint<8*C>        wide_t;     // 512-bit: C=64 packed INT8 channels (one feat/BEV vector)

void bev_gather(
    const wide_t  feat   [FEAT_VECS],      // per-frame: packed context features (AXI, 512-bit words)
    const depth_t depth  [DEPTH_LEN],      // per-frame: depth distribution (AXI)
    const idx_t   rank_depth   [N_POINTS], // static index tables (resident BRAM/URAM)
    const idx_t   rank_feat    [N_POINTS],
    const idx_t   rank_bev     [N_PILLAR], // one bev index per occupied pillar
    const idx_t   interval_start[N_PILLAR],
    const idx_t   interval_len  [N_PILLAR],
    const acc_t   out_scale,               // requant scale for the BEV stage
    wide_t        bev    [NUM_BEV]);        // output BEV feature (AXI, packed)

#endif
