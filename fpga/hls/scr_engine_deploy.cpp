/* Deployable SCR engine: m_axi DRAM for weights/input/output, time-multiplexed conv core (1 engine reused),
 * ping-pong on-chip features. Representative (NL layers, NP tile) to validate the deployable SCR-accel bitstream
 * flow (PS+AXI+P&R, render-IP-proven). Full 20-layer engine = the refinement. */
#include <ap_int.h>
typedef ap_int<16> i16;
typedef ap_int<32> i32;
#define IC 512
#define NP 64
#define OCP 32
#define ICU 16
#define NL 6
static void conv_layer(const i16 *wt_ddr, const i16 in[IC][NP], int IN_C, i16 out[IC][NP], int OUT_C, int shift) {
  for (int ocb = 0; ocb < OUT_C / OCP; ocb++) {
    i16 wt[OCP][IC];
#pragma HLS ARRAY_PARTITION variable=wt complete dim=1
#pragma HLS ARRAY_PARTITION variable=wt cyclic factor=16 dim=2
    for (int o = 0; o < OCP; o++) for (int i = 0; i < IN_C; i++) {
#pragma HLS PIPELINE II=1
      wt[o][i] = wt_ddr[(ocb * OCP + o) * IN_C + i];
    }
    for (int p = 0; p < NP; p++) {
      i32 acc[OCP];
#pragma HLS ARRAY_PARTITION variable=acc complete
      for (int o = 0; o < OCP; o++) acc[o] = 0;
      for (int ic = 0; ic < IN_C; ic += ICU) {
#pragma HLS PIPELINE II=1
        for (int o = 0; o < OCP; o++) {
#pragma HLS UNROLL
          i32 ps = 0;
          for (int u = 0; u < ICU; u++) ps += in[ic + u][p] * wt[o][ic + u];
          acc[o] += ps;
        }
      }
      for (int o = 0; o < OCP; o++) { i32 r = acc[o] >> shift; if (r < 0) r = 0; if (r > 32767) r = 32767; out[ocb * OCP + o][p] = (i16)r; }
    }
  }
}
void scr_engine_deploy(const i16 *wts, const i16 *in_ddr, i16 *out_ddr, int nl) {
#pragma HLS INTERFACE m_axi port=wts offset=slave bundle=gw depth=4000000
#pragma HLS INTERFACE m_axi port=in_ddr offset=slave bundle=gi depth=32768
#pragma HLS INTERFACE m_axi port=out_ddr offset=slave bundle=go depth=32768
#pragma HLS INTERFACE s_axilite port=nl
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS ALLOCATION instances=conv_layer limit=1 function
  i16 bufA[IC][NP], bufB[IC][NP];
#pragma HLS ARRAY_PARTITION variable=bufA cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=bufB cyclic factor=16 dim=1
  for (int c = 0; c < IC; c++) for (int p = 0; p < NP; p++) bufA[c][p] = in_ddr[c * NP + p];
  int woff = 0;
  for (int l = 0; l < NL; l++) {
    if ((l & 1) == 0) conv_layer(wts + woff, bufA, IC, bufB, IC, 8);
    else conv_layer(wts + woff, bufB, IC, bufA, IC, 8);
    woff += IC * IC;
  }
  for (int c = 0; c < IC; c++) for (int p = 0; p < NP; p++) out_ddr[c * NP + p] = ((NL & 1) == 0) ? bufA[c][p] : bufB[c][p];
}
