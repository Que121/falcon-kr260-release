#include "bev_gather_i16.hpp"

// INT16-vt gather: same segmented weighted scatter-add as bev_gather.cpp (II=1, C lanes unrolled,
// input-invariant WCET) — only the requantize/write widens to INT16 (clamp +-32767, 16-bit lanes,
// 1024-bit wide bev word). feat input path unchanged (INT8, 512-bit).

void bev_gather_i16(
    const feat_wide_t feat   [FEAT_VECS],
    const depth_t     depth  [DEPTH_LEN],
    const idx_t       rank_depth   [N_POINTS],
    const idx_t       rank_feat    [N_POINTS],
    const idx_t       rank_bev     [N_PILLAR],
    const idx_t       interval_start[N_PILLAR],
    const idx_t       interval_len  [N_PILLAR],
    const acc_t       out_scale,
    bev_wide_t        bev    [NUM_BEV])
{
#pragma HLS INTERFACE m_axi     port=feat            offset=slave bundle=gmem0 depth=FEAT_VECS
#pragma HLS INTERFACE m_axi     port=depth           offset=slave bundle=gmem1 depth=DEPTH_LEN
#pragma HLS INTERFACE m_axi     port=bev             offset=slave bundle=gmem2 depth=NUM_BEV
#pragma HLS INTERFACE bram      port=rank_depth
#pragma HLS INTERFACE bram      port=rank_feat
#pragma HLS INTERFACE bram      port=rank_bev
#pragma HLS INTERFACE bram      port=interval_start
#pragma HLS INTERFACE bram      port=interval_len
#pragma HLS INTERFACE s_axilite port=out_scale
#pragma HLS INTERFACE s_axilite port=return

ZERO_BEV:
    for (int b = 0; b < NUM_BEV; b++) {
#pragma HLS PIPELINE II=1
        bev[b] = 0;
    }

PILLAR:
    for (int p = 0; p < N_PILLAR; p++) {
        acc_t acc[C];
#pragma HLS ARRAY_PARTITION variable=acc complete
    INIT_ACC:
        for (int c = 0; c < C; c++) {
#pragma HLS UNROLL
            acc[c] = 0;
        }

        const idx_t s   = interval_start[p];
        const idx_t len = interval_len[p];

    POINTS:
        for (idx_t k = 0; k < len; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=64 avg=14
            const idx_t i  = s + k;
            const depth_t    w  = depth[rank_depth[i]];
            const feat_wide_t fv = feat[rank_feat[i]];   // 512-bit: 64 INT8 channels
        MAC:
            for (int c = 0; c < C; c++) {
#pragma HLS UNROLL
                feat_t fc = (feat_t)fv.range(8*c + 7, 8*c);
                acc[c] += (acc_t)w * (acc_t)fc;
            }
        }

        const idx_t bidx = rank_bev[p];
        bev_wide_t O = 0;
    WRITE:
        for (int c = 0; c < C; c++) {
#pragma HLS UNROLL
            acc_t v = acc[c] * out_scale;
            if (v > (acc_t)32767)  v = 32767;       // INT16 clamp (was 127)
            if (v < (acc_t)-32768) v = -32768;
            O.range(16*c + 15, 16*c) = (ap_uint<16>)(bev_t)v;
        }
        bev[bidx] = O;
    }
    // Inner-loop iterations = NUM_BEV(zero) + N_POINTS(MAC) + per-pillar overhead -> input-invariant WCET.
}
