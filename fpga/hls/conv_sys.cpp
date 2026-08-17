/* P3 throughput core: output-stationary INT16 conv, OCP=32 output channels in parallel x ICU=16 input-channel
 * unroll = 512 MACs with 32 INDEPENDENT accumulators -> II=1 (no 512-deep adder-tree recurrence). Validates the
 * 512 MAC/cyc = 185 GMAC/s -> ~284ms full-SCR throughput. */
#include <ap_int.h>
typedef ap_int<16> i16;
typedef ap_int<32> i32;
#define IC 512
#define OCP 32
#define NP 64
#define ICU 16
void conv_sys(const i16 ifm[IC][NP], const i16 wt[OCP][IC], const i16 bias[OCP], int shift, i16 ofm[OCP][NP]) {
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS ARRAY_PARTITION variable=ifm cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=wt complete dim=1
#pragma HLS ARRAY_PARTITION variable=wt cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=bias complete
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=1
  for (int p = 0; p < NP; p++) {
    i32 acc[OCP];
#pragma HLS ARRAY_PARTITION variable=acc complete
    for (int oc = 0; oc < OCP; oc++) acc[oc] = 0;
    for (int ic = 0; ic < IC; ic += ICU) {
#pragma HLS PIPELINE II=1
      for (int oc = 0; oc < OCP; oc++) {
#pragma HLS UNROLL
        i32 ps = 0;
        for (int u = 0; u < ICU; u++) {
#pragma HLS UNROLL
          ps += ifm[ic + u][p] * wt[oc][ic + u];
        }
        acc[oc] += ps;
      }
    }
    for (int oc = 0; oc < OCP; oc++) { i32 r = (acc[oc] + bias[oc]) >> shift; if (r < 0) r = 0; if (r > 32767) r = 32767; ofm[oc][p] = (i16)r; }
  }
}
