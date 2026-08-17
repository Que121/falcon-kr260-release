#include "bev_gather.hpp"

// DEPLOYABLE variant of the view-transform gather IP for the on-board PYNQ run.
// Same arithmetic as bev_gather.cpp (bit-exact), but the five static index tables are exposed over
// m_axi (DDR, host-loadable from PYNQ) instead of resident bram. The bram tables do not fit on-chip at
// the real rig scale (rank_depth / rank_feat are ~1.2 MB each at N_POINTS = 302558). The gather already
// walks the tables in order -- pillars sequential in p, and points i = s + k strictly sequential over
// 0..N_POINTS for contiguous intervals -- so the m_axi reads are burst-friendly. Per-channel arithmetic
// is unchanged from the bit-exact-verified scalar version, so packing/placement is a pure memory-layout
// change. This is the form a PYNQ host needs: allocate DDR buffers, fill the tables once, set the
// pointers over s_axilite, and trigger the IP.

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
// ---- all per-frame tensors AND the static index tables over AXI4 master to DDR (host-loadable) ----
#pragma HLS INTERFACE m_axi     port=feat            offset=slave bundle=gmem0 depth=FEAT_VECS
#pragma HLS INTERFACE m_axi     port=depth           offset=slave bundle=gmem1 depth=DEPTH_LEN
#pragma HLS INTERFACE m_axi     port=bev             offset=slave bundle=gmem2 depth=NUM_BEV
#pragma HLS INTERFACE m_axi     port=rank_depth      offset=slave bundle=gmem3 depth=N_POINTS
#pragma HLS INTERFACE m_axi     port=rank_feat       offset=slave bundle=gmem4 depth=N_POINTS
#pragma HLS INTERFACE m_axi     port=rank_bev        offset=slave bundle=gmem5 depth=N_PILLAR
#pragma HLS INTERFACE m_axi     port=interval_start  offset=slave bundle=gmem5 depth=N_PILLAR
#pragma HLS INTERFACE m_axi     port=interval_len    offset=slave bundle=gmem5 depth=N_PILLAR
#pragma HLS INTERFACE s_axilite port=out_scale
#pragma HLS INTERFACE s_axilite port=return

    // 1) zero the dense BEV grid (one wide word per cell -> bursts; empty cells stay empty)
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
#pragma HLS LOOP_TRIPCOUNT min=1 max=64 avg=14
            const idx_t i  = s + k;
            const depth_t w = depth[rank_depth[i]];     // depth weight (indirect DDR gather)
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
    // inner-loop iterations = NUM_BEV (zero) + N_POINTS (MACs) + fixed per-pillar overhead ->
    // WCET INPUT-INVARIANT: identical cycle count regardless of scene content.
}
