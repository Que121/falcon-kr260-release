#include "resize.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>

// C/RTL-sim testbench: proves the wide-word (512-bit packed) IP is bit-exact to the SCALAR algorithm
// (same ap_fixed arithmetic, computed independently/unpacked here). 0 mismatches => packing is lossless;
// combined with resize_verify.py (scalar == torch bilinear, 1e-13) this closes the bit-exact chain.

static signed char sin_[RZ_NTILE][RZ_LANES][RZ_HIN][RZ_WIN];

int main() {
    static wide_t  in [RZ_NTILE][RZ_HIN][RZ_WIN];
    static wide_t  out[RZ_NTILE][RZ_HOUT][RZ_WOUT];
    rzidx_t y0[RZ_HOUT], y1[RZ_HOUT], x0[RZ_WOUT], x1[RZ_WOUT];
    wgt_t   wy[RZ_HOUT], wx[RZ_WOUT];

    srand(7);
    for (int t = 0; t < RZ_NTILE; t++)
      for (int h = 0; h < RZ_HIN; h++)
        for (int x = 0; x < RZ_WIN; x++) {
            wide_t w = 0;
            for (int cl = 0; cl < RZ_LANES; cl++) {
                int v = (rand() % 255) - 127;          // INT8 in [-127,127]
                sin_[t][cl][h][x] = (signed char)v;
                w.range(8*cl + 7, 8*cl) = (ap_uint<8>)(ap_int<8>)v;
            }
            in[t][h][x] = w;
        }
    // align_corners=True upsampling taps (src = o*(IN-1)/(OUT-1))
    for (int oy = 0; oy < RZ_HOUT; oy++) {
        double s = (RZ_HOUT > 1) ? (double)oy * (RZ_HIN - 1) / (RZ_HOUT - 1) : 0;
        int i0 = (int)floor(s); int i1 = (i0 + 1 < RZ_HIN) ? i0 + 1 : RZ_HIN - 1;
        y0[oy] = i0; y1[oy] = i1; wy[oy] = (wgt_t)(s - i0);
    }
    for (int ox = 0; ox < RZ_WOUT; ox++) {
        double s = (RZ_WOUT > 1) ? (double)ox * (RZ_WIN - 1) / (RZ_WOUT - 1) : 0;
        int i0 = (int)floor(s); int i1 = (i0 + 1 < RZ_WIN) ? i0 + 1 : RZ_WIN - 1;
        x0[ox] = i0; x1[ox] = i1; wx[ox] = (wgt_t)(s - i0);
    }

    resize_bilinear(in, y0, y1, wy, x0, x1, wx, out);

    int errs = 0;
    for (int t = 0; t < RZ_NTILE; t++)
      for (int oy = 0; oy < RZ_HOUT; oy++)
        for (int ox = 0; ox < RZ_WOUT; ox++)
          for (int cl = 0; cl < RZ_LANES; cl++) {
            rzidx_t r0 = y0[oy], r1 = y1[oy]; wgt_t wv = wy[oy];
            rzidx_t c0 = x0[ox], c1 = x1[ox]; wgt_t wh = wx[ox];
            // vertical (full-precision) at the two needed columns, then horizontal -- same as the IP
            rzacc_t a0 = (rzacc_t)(ap_int<8>)sin_[t][cl][r0][c0];
            rzacc_t b0 = (rzacc_t)(ap_int<8>)sin_[t][cl][r1][c0];
            rzacc_t vA = a0 + (rzacc_t)wv * (b0 - a0);
            rzacc_t a1 = (rzacc_t)(ap_int<8>)sin_[t][cl][r0][c1];
            rzacc_t b1 = (rzacc_t)(ap_int<8>)sin_[t][cl][r1][c1];
            rzacc_t vB = a1 + (rzacc_t)wv * (b1 - a1);
            rzacc_t v  = vA + (rzacc_t)wh * (vB - vA);
            if (v > (rzacc_t)127)  v = 127;
            if (v < (rzacc_t)-128) v = -128;
            signed char gold = (signed char)(rz_t)v;
            signed char hwv  = (signed char)(ap_int<8>)out[t][oy][ox].range(8*cl + 7, 8*cl);
            if (hwv != gold) errs++;
        }
    printf("RESIZE_MISMATCHES=%d  (NTILE=%d HOUT=%d WOUT=%d LANES=%d)\n",
           errs, RZ_NTILE, RZ_HOUT, RZ_WOUT, RZ_LANES);
    return errs ? 1 : 0;
}
