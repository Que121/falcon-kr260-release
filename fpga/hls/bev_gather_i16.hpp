#ifndef BEV_GATHER_I16_HPP
#define BEV_GATHER_I16_HPP
#include <ap_int.h>
#include <ap_fixed.h>

// INT16-vt variant of the view-transform gather IP. Identical algorithm/WCET to bev_gather.cpp, but the
// BEV output is INT16 per channel (was INT8 clip@127). On-board decomp showed the gather's INT8 vt output
// clips the true vt (range ~250) at 127 -> -2.2 mIoU, and the step-1.0 resolution loses the bulk. INT16
// vt (range +-32767, finer step) removes both. feat input stays INT8 (it is DPU-INT8-sourced: widening it
// adds no information; verified fp_feat=2 vs 3 identical). Only the bev write path widens 512->1024 bit.

#ifdef HLS_SMALL
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
#define NUM_BEV    40000
#endif

typedef ap_int<8>           feat_t;       // INT8 context feature (one channel), from the DPU image neck
typedef ap_ufixed<8, 1>     depth_t;      // [0,2) depth weight, Q0.7 (1/128)
typedef ap_fixed<32, 16>    acc_t;        // wide accumulator (per lane) — widened for the INT16 out range
typedef ap_int<16>          bev_t;        // INT16 requantized BEV feature (one channel) — was ap_int<8>
typedef ap_uint<32>         idx_t;
typedef ap_uint<8*C>        feat_wide_t;  // 512-bit: C=64 packed INT8 feat channels (input)
typedef ap_uint<16*C>       bev_wide_t;   // 1024-bit: C=64 packed INT16 bev channels (output)

void bev_gather_i16(
    const feat_wide_t feat   [FEAT_VECS],
    const depth_t     depth  [DEPTH_LEN],
    const idx_t       rank_depth   [N_POINTS],
    const idx_t       rank_feat    [N_POINTS],
    const idx_t       rank_bev     [N_PILLAR],
    const idx_t       interval_start[N_PILLAR],
    const idx_t       interval_len  [N_PILLAR],
    const acc_t       out_scale,
    bev_wide_t        bev    [NUM_BEV]);

#endif
