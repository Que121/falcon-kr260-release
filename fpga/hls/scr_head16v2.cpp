/* FP32-parity SCR head engine v2: WEIGHT-CACHED. Reads all weights from DDR ONCE into on-chip URAM (the v1
 * bottleneck was re-reading 1.5MB/call x 51 calls = 76MB DDR, ~864ms). Internal pixel-tile loop (1x1 conv is
 * pointwise -> tiles independent) so ONE call processes the whole image. Larger NPT=512 amortizes the on-chip
 * weight reload. Same locked arithmetic (per-channel fixed-point multiplier requant + bias + int48 accum). */
#include <ap_int.h>
typedef ap_int<16> i16;
typedef ap_int<48> acc_t;
#define IC 512
#define NPT 256
#define OCP 32
#define ICU 16
#define MAXNL 3
static i16 wcache[MAXNL*IC*IC];
static void conv_layer(int woff, const i16 in[IC][NPT], i16 out[IC][NPT], int OUT_C,
                       const int *mult, const int *bias, int shift, int tn) {
  for (int ocb = 0; ocb < OUT_C/OCP; ocb++) {
    i16 wt[OCP][IC];
#pragma HLS ARRAY_PARTITION variable=wt complete dim=1
#pragma HLS ARRAY_PARTITION variable=wt cyclic factor=16 dim=2
    int M[OCP], B[OCP];
#pragma HLS ARRAY_PARTITION variable=M complete
#pragma HLS ARRAY_PARTITION variable=B complete
    for (int o = 0; o < OCP; o++) {
      M[o] = mult[ocb*OCP+o]; B[o] = bias[ocb*OCP+o];
      for (int i = 0; i < IC; i++) {
#pragma HLS PIPELINE II=1
        wt[o][i] = wcache[woff + (ocb*OCP+o)*IC + i];
      }
    }
    for (int p = 0; p < tn; p++) {
      acc_t acc[OCP];
#pragma HLS ARRAY_PARTITION variable=acc complete
      for (int o = 0; o < OCP; o++) acc[o] = 0;
      for (int ic = 0; ic < IC; ic += ICU) {
#pragma HLS PIPELINE II=1
        for (int o = 0; o < OCP; o++) {
#pragma HLS UNROLL
          acc_t ps = 0;
          for (int u = 0; u < ICU; u++) ps += (acc_t)in[ic+u][p]*(acc_t)wt[o][ic+u];
          acc[o] += ps;
        }
      }
      for (int o = 0; o < OCP; o++) {
        ap_int<72> r = ((ap_int<72>)acc[o]*(ap_int<72>)M[o]) >> shift;
        int v = (int)r + B[o];
        if (v < 0) v = 0; if (v > 32767) v = 32767;
        out[ocb*OCP+o][p] = (i16)v;
      }
    }
  }
}
void scr_head16v2(const i16 *wts, const i16 *in_ddr, i16 *out_ddr,
                  const int *mult, const int *bias, const int *shifts, int nl, int NP) {
#pragma HLS INTERFACE m_axi port=wts offset=slave bundle=gw depth=786432
#pragma HLS INTERFACE m_axi port=in_ddr offset=slave bundle=gi depth=3287040
#pragma HLS INTERFACE m_axi port=out_ddr offset=slave bundle=go depth=3287040
#pragma HLS INTERFACE m_axi port=mult offset=slave bundle=gp depth=1536
#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gp depth=1536
#pragma HLS INTERFACE m_axi port=shifts offset=slave bundle=gp depth=16
#pragma HLS INTERFACE s_axilite port=nl
#pragma HLS INTERFACE s_axilite port=NP
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS bind_storage variable=wcache type=ram_1p impl=uram
#pragma HLS array_reshape variable=wcache block factor=4 dim=1
  for (int i = 0; i < nl*IC*IC; i++) {
#pragma HLS PIPELINE II=1
    wcache[i] = wts[i];
  }
  i16 bufA[IC][NPT], bufB[IC][NPT];
#pragma HLS ARRAY_PARTITION variable=bufA cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=bufB cyclic factor=16 dim=1
  for (int t = 0; t < NP; t += NPT) {
    int tn = (NP - t < NPT) ? (NP - t) : NPT;
    for (int c = 0; c < IC; c++)
      for (int p = 0; p < tn; p++)
        bufA[c][p] = in_ddr[c*NP + t + p];
    int woff = 0, coff = 0;
    for (int l = 0; l < nl; l++) {
      if ((l&1) == 0) conv_layer(woff, bufA, bufB, IC, mult+coff, bias+coff, shifts[l], tn);
      else            conv_layer(woff, bufB, bufA, IC, mult+coff, bias+coff, shifts[l], tn);
      woff += IC*IC; coff += IC;
    }
    i16 (*last)[NPT] = ((nl&1) == 0) ? bufA : bufB;
    for (int c = 0; c < IC; c++)
      for (int p = 0; p < tn; p++)
        out_ddr[c*NP + t + p] = last[c][p];
  }
}
