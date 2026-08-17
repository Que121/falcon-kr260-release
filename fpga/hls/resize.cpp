#include "resize.hpp"

// Separable bilinear resize, wide-word (64 INT8 lanes / 512-bit AXI) + row-fused.
// Two 2-tap weighted gathers (vertical then horizontal); the intermediate is ONE output row.
// Fixed output size -> constant cycle count (input-invariant) -> bounded WCET.
// Cycle count = RZ_NTILE * RZ_HOUT * (RZ_WIN + RZ_WOUT) = 8*200*(100+200) = 480k (UP2), fixed.

void resize_bilinear(
    const wide_t  in   [RZ_NTILE][RZ_HIN][RZ_WIN],
    const rzidx_t y0   [RZ_HOUT], const rzidx_t y1[RZ_HOUT], const wgt_t wy[RZ_HOUT],
    const rzidx_t x0   [RZ_WOUT], const rzidx_t x1[RZ_WOUT], const wgt_t wx[RZ_WOUT],
    wide_t        out  [RZ_NTILE][RZ_HOUT][RZ_WOUT])
{
#pragma HLS INTERFACE m_axi     port=in   offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi     port=out  offset=slave bundle=gmem1
#pragma HLS INTERFACE bram      port=y0
#pragma HLS INTERFACE bram      port=y1
#pragma HLS INTERFACE bram      port=wy
#pragma HLS INTERFACE bram      port=x0
#pragma HLS INTERFACE bram      port=x1
#pragma HLS INTERFACE bram      port=wx
#pragma HLS INTERFACE s_axilite port=return

    // one output row's vertical-pass result, full precision, channel-partitioned for 64-lane parallelism
    rzacc_t vrow[RZ_LANES][RZ_WIN];
#pragma HLS ARRAY_PARTITION variable=vrow dim=1 complete

TILE:
    for (int t = 0; t < RZ_NTILE; t++) {
    OY:
        for (int oy = 0; oy < RZ_HOUT; oy++) {
            const rzidx_t r0 = y0[oy];
            const rzidx_t r1 = y1[oy];
            const wgt_t   wv = wy[oy];

            // ---- vertical pass for this output row -> vrow[lane][x] ----
        VX:
            for (int x = 0; x < RZ_WIN; x++) {
#pragma HLS PIPELINE II=1
                wide_t A = in[t][r0][x];
                wide_t B = in[t][r1][x];
            VL:
                for (int cl = 0; cl < RZ_LANES; cl++) {
#pragma HLS UNROLL
                    rzacc_t a = (rzacc_t)(ap_int<8>)A.range(8*cl + 7, 8*cl);
                    rzacc_t b = (rzacc_t)(ap_int<8>)B.range(8*cl + 7, 8*cl);
                    vrow[cl][x] = a + (rzacc_t)wv * (b - a);   // a*(1-wv) + b*wv
                }
            }

            // ---- horizontal pass -> out[t][oy][ox], requantize to INT8 ----
        HX:
            for (int ox = 0; ox < RZ_WOUT; ox++) {
#pragma HLS PIPELINE II=1
                const rzidx_t c0 = x0[ox];
                const rzidx_t c1 = x1[ox];
                const wgt_t   wh = wx[ox];
                wide_t O = 0;
            HL:
                for (int cl = 0; cl < RZ_LANES; cl++) {
#pragma HLS UNROLL
                    rzacc_t a = vrow[cl][c0];
                    rzacc_t b = vrow[cl][c1];
                    rzacc_t v = a + (rzacc_t)wh * (b - a);
                    if (v > (rzacc_t)127)  v = 127;
                    if (v < (rzacc_t)-128) v = -128;
                    O.range(8*cl + 7, 8*cl) = (ap_uint<8>)(ap_int<8>)v;
                }
                out[t][oy][ox] = O;
            }
        }
    }
    // iterations = RZ_NTILE*(RZ_HOUT*RZ_WIN + RZ_HOUT*RZ_WOUT) = 8*(200*100 + 200*200) = 480k -> WCET
    // input-invariant. Wide 512-bit AXI -> burst-friendly; vrow channel-partitioned -> 64-lane II=1.
}
