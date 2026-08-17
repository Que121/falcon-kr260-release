#include "resize.hpp"

// DEPLOYABLE variant of the bilinear-resize IP for the on-board PYNQ run.
// Same arithmetic as resize.cpp (bit-exact), but the static taps are exposed over m_axi (DDR,
// host-loadable from PYNQ) instead of a bram interface, and cached on-chip once at entry. The
// in/out activations are m_axi as before. This lets a PYNQ host allocate DDR buffers, fill them,
// set the pointers via s_axilite, and trigger the IP -- the form needed for an actual board run.
// (On-chip bram-resident tables are infeasible at the real rig scale; see the gather index tables.)

void resize_bilinear(
    const wide_t  in   [RZ_NTILE][RZ_HIN][RZ_WIN],
    const rzidx_t y0   [RZ_HOUT], const rzidx_t y1[RZ_HOUT], const wgt_t wy[RZ_HOUT],
    const rzidx_t x0   [RZ_WOUT], const rzidx_t x1[RZ_WOUT], const wgt_t wx[RZ_WOUT],
    wide_t        out  [RZ_NTILE][RZ_HOUT][RZ_WOUT])
{
#pragma HLS INTERFACE m_axi     port=in   offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi     port=out  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi     port=y0   offset=slave bundle=gmem2 depth=RZ_HOUT
#pragma HLS INTERFACE m_axi     port=y1   offset=slave bundle=gmem2 depth=RZ_HOUT
#pragma HLS INTERFACE m_axi     port=wy   offset=slave bundle=gmem2 depth=RZ_HOUT
#pragma HLS INTERFACE m_axi     port=x0   offset=slave bundle=gmem3 depth=RZ_WOUT
#pragma HLS INTERFACE m_axi     port=x1   offset=slave bundle=gmem3 depth=RZ_WOUT
#pragma HLS INTERFACE m_axi     port=wx   offset=slave bundle=gmem3 depth=RZ_WOUT
#pragma HLS INTERFACE s_axilite port=return

    // cache the static taps on-chip (small: HOUT/WOUT entries)
    rzidx_t ly0[RZ_HOUT], ly1[RZ_HOUT], lx0[RZ_WOUT], lx1[RZ_WOUT];
    wgt_t   lwy[RZ_HOUT], lwx[RZ_WOUT];
LOAD_Y:
    for (int i = 0; i < RZ_HOUT; i++) {
#pragma HLS PIPELINE II=1
        ly0[i] = y0[i]; ly1[i] = y1[i]; lwy[i] = wy[i];
    }
LOAD_X:
    for (int i = 0; i < RZ_WOUT; i++) {
#pragma HLS PIPELINE II=1
        lx0[i] = x0[i]; lx1[i] = x1[i]; lwx[i] = wx[i];
    }

    rzacc_t vrow[RZ_LANES][RZ_WIN];
#pragma HLS ARRAY_PARTITION variable=vrow dim=1 complete

TILE:
    for (int t = 0; t < RZ_NTILE; t++) {
    OY:
        for (int oy = 0; oy < RZ_HOUT; oy++) {
            const rzidx_t r0 = ly0[oy];
            const rzidx_t r1 = ly1[oy];
            const wgt_t   wv = lwy[oy];
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
                    vrow[cl][x] = a + (rzacc_t)wv * (b - a);
                }
            }
        HX:
            for (int ox = 0; ox < RZ_WOUT; ox++) {
#pragma HLS PIPELINE II=1
                const rzidx_t c0 = lx0[ox];
                const rzidx_t c1 = lx1[ox];
                const wgt_t   wh = lwx[ox];
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
}
