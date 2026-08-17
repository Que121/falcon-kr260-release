/* P3 streaming conv layer: weights stream from DDR (m_axi) per OC-block into BRAM, REUSED across all NP pixels
 * (output-stationary -> weight bandwidth amortized, fits the ~19 GB/s DDR), II=1 systolic core (512 INT16 MAC).
 * Confirms the full-layer design with weight streaming fits + is compute-bound (not DDR-bound). */
#include <ap_int.h>
typedef ap_int<16> i16;
typedef ap_int<32> i32;
#define IC 512
#define OC 512
#define NP 64
#define OCP 32
#define ICU 16
void conv_stream(const i16 *wt_ddr, const i16 ifm[IC][NP], const i16 *bias_ddr, int shift, i16 ofm[OC][NP]) {
#pragma HLS INTERFACE m_axi port=wt_ddr offset=slave bundle=g depth=262144
#pragma HLS INTERFACE m_axi port=bias_ddr offset=slave bundle=g depth=512
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS ARRAY_PARTITION variable=ifm cyclic factor=16 dim=1
  for (int ocb = 0; ocb < OC / OCP; ocb++) {
    i16 wt[OCP][IC];
#pragma HLS ARRAY_PARTITION variable=wt complete dim=1
#pragma HLS ARRAY_PARTITION variable=wt cyclic factor=16 dim=2
    i16 bias[OCP];
#pragma HLS ARRAY_PARTITION variable=bias complete
    /* stream this OC-block's weights from DDR -> BRAM (once, reused across NP pixels) */
    for (int o = 0; o < OCP; o++)
      for (int i = 0; i < IC; i++)
#pragma HLS PIPELINE II=1
        wt[o][i] = wt_ddr[(ocb * OCP + o) * IC + i];
    for (int o = 0; o < OCP; o++) bias[o] = bias_ddr[ocb * OCP + o];
    /* II=1 output-stationary systolic over pixels */
    for (int p = 0; p < NP; p++) {
      i32 acc[OCP];
#pragma HLS ARRAY_PARTITION variable=acc complete
      for (int o = 0; o < OCP; o++) acc[o] = 0;
      for (int ic = 0; ic < IC; ic += ICU) {
#pragma HLS PIPELINE II=1
        for (int o = 0; o < OCP; o++) {
#pragma HLS UNROLL
          i32 ps = 0;
          for (int u = 0; u < ICU; u++) {
#pragma HLS UNROLL
            ps += ifm[ic + u][p] * wt[o][ic + u];
          }
          acc[o] += ps;
        }
      }
      for (int o = 0; o < OCP; o++) { i32 r = (acc[o] + bias[o]) >> shift; if (r < 0) r = 0; if (r > 32767) r = 32767; ofm[ocb * OCP + o][p] = (i16)r; }
    }
  }
}
