#include "bev_gather.hpp"

// HLS view-transform IP: precomputed-index weighted scatter-add (segmented form), wide-word.
// Segmented accumulation (one accumulator set per pillar, single wide write) avoids the read-modify-write
// hazard of a raw scatter, so the inner point loop pipelines at II=1 with C lanes unrolled.
// C=64 INT8 channels are packed per 512-bit AXI word -> BEV writeback bursts one cell/cycle; each point
// is one wide feat read. Same per-channel arithmetic as the bit-exact scalar version.

void bev_gather(
    const wide_t  feat   [FEAT_VECS],
    const depth_t depth  [DEPTH_LEN],
    const idx_t   rank_depth   [N_POINTS],
    const idx_t   rank_feat    [N_POINTS],
    const idx_t   rank_bev     [N_PILLAR],
    const idx_t   interval_start[N_PILLAR],
    const idx_t   interval_len  [N_PILLAR],
    const acc_t   out_scale,
    wide_t        bev    [NUM_BEV])
{
// ---- interfaces: per-frame tensors over AXI4 master to DDR; static index tables resident on-chip ----
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

    // 1) zero the dense BEV grid (one wide word per cell -> bursts; cells with no points stay empty)
ZERO_BEV:
    for (int b = 0; b < NUM_BEV; b++) {
#pragma HLS PIPELINE II=1
        bev[b] = 0;
    }

    // 2) segmented weighted accumulate: one pillar at a time, write its BEV cell once (wide)
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
#pragma HLS LOOP_TRIPCOUNT min=1 max=64 avg=14   // 302558 points / 21853 pillars ~= 13.8
            const idx_t i  = s + k;
            const depth_t w = depth[rank_depth[i]];     // depth distribution weight for this point
            const wide_t  fv = feat[rank_feat[i]];      // one wide read = C packed INT8 channels
        MAC:
            for (int c = 0; c < C; c++) {
#pragma HLS UNROLL
                feat_t fc = (feat_t)fv.range(8*c + 7, 8*c);
                acc[c] += (acc_t)w * (acc_t)fc;
            }
        }

        // requantize accumulators to INT8 and write the pillar's BEV cell once (wide)
        const idx_t bidx = rank_bev[p];
        wide_t O = 0;
    WRITE:
        for (int c = 0; c < C; c++) {
#pragma HLS UNROLL
            acc_t v = acc[c] * out_scale;
            if (v > (acc_t)127)  v = 127;
            if (v < (acc_t)-128) v = -128;
            O.range(8*c + 7, 8*c) = (ap_uint<8>)(bev_t)v;
        }
        bev[bidx] = O;
    }
    // Total inner-loop iterations = NUM_BEV (zero) + N_POINTS (MACs) + per-pillar overhead (fixed) ->
    // WCET INPUT-INVARIANT: identical cycle count regardless of scene content.
}
