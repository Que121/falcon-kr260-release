/* S1 core: multi-layer on-chip dataflow - 2 conv layers chained with the intermediate features kept in BRAM
 * (NOT round-tripped to DRAM), II=1 systolic core, weights streamed per layer. Validates that the full 20-layer
 * SCR engine keeps features on-chip between layers (the key to compute-bound throughput). */
#include <ap_int.h>
typedef ap_int<16> i16;
typedef ap_int<32> i32;
#define IC 512
#define MID 512
#define OC 512
#define NP 64
#define OCP 32
#define ICU 16
static void conv_layer(const i16 *wt_ddr, const i16  in[IC][NP], int IN_C, i16 out[OC][NP], int OUT_C, int shift) {
  for (int ocb = 0; ocb < OUT_C / OCP; ocb++) {
    i16 wt[OCP][IC];
#pragma HLS ARRAY_PARTITION variable=wt complete dim=1
#pragma HLS ARRAY_PARTITION variable=wt cyclic factor=16 dim=2
    for (int o = 0; o < OCP; o++)
      for (int i = 0; i < IN_C; i++)
#pragma HLS PIPELINE II=1
        wt[o][i] = wt_ddr[(ocb * OCP + o) * IN_C + i];
    for (int p = 0; p < NP; p++) {
      i32 acc[OCP];
#pragma HLS ARRAY_PARTITION variable=acc complete
      for (int o = 0; o < OCP; o++) acc[o] = 0;
      for (int ic = 0; ic < IN_C; ic += ICU) {
#pragma HLS PIPELINE II=1
        for (int o = 0; o < OCP; o++) {
#pragma HLS UNROLL
          i32 ps = 0;
          for (int u = 0; u < ICU; u++) { ps += in[ic + u][p] * wt[o][ic + u]; }
          acc[o] += ps;
        }
      }
      for (int o = 0; o < OCP; o++) { i32 r = acc[o] >> shift; if (r < 0) r = 0; if (r > 32767) r = 32767; out[ocb * OCP + o][p] = (i16)r; }
    }
  }
}
void scr_2layer(const i16 *w1, const i16 *w2, const i16 ifm[IC][NP], i16 ofm[OC][NP]) {
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=g0 depth=262144
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=g1 depth=262144
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS ARRAY_PARTITION variable=ifm cyclic factor=16 dim=1
  i16 mid[MID][NP];   /* intermediate features stay ON-CHIP (BRAM) between layers */
#pragma HLS ARRAY_PARTITION variable=mid cyclic factor=16 dim=1
  conv_layer(w1, ifm, IC, mid, MID, 8);
  conv_layer(w2, mid, MID, ofm, OC, 8);
}
