#ifndef BEV_GATHER_I16DEP_HPP
#define BEV_GATHER_I16DEP_HPP
#include <ap_int.h>
#include <ap_fixed.h>

// INT16-vt DEPLOY variant of the gather IP: index tables over m_axi (DDR, host-loadable) like
// bev_gather_deploy.cpp, but BEV output is INT16 (was INT8 clip@127). Top kept named `bev_gather` so the
// existing vivado_gather/build.tcl block design (xilinx.com:hls:bev_gather:1.0, 6 m_axi gmem0..5) works
// unchanged except the bev port widens 512->1024 bit (SmartConnect adapts to the 128-bit HP0).

#define N_POINTS   302558
#define N_PILLAR   21853
#define C          64
#define DEPTH_LEN  371712
#define FEAT_VECS  4224
#define NUM_BEV    40000

typedef ap_int<8>           feat_t;
typedef ap_ufixed<8, 1>     depth_t;
typedef ap_fixed<32, 16>    acc_t;        // widened for the INT16 output range
typedef ap_int<16>          bev_t;        // INT16 BEV feature (was ap_int<8>)
typedef ap_uint<32>         idx_t;
typedef ap_uint<8*C>        feat_wide_t;  // 512-bit packed INT8 feat (input)
typedef ap_uint<16*C>       bev_wide_t;   // 1024-bit packed INT16 bev (output)

void bev_gather(
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
