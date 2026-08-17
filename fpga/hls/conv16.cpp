/* Representative INT16 conv MAC engine for the custom PL SCR accelerator (P2 feasibility). 1x1 conv tile
 * (the SCR heads are 1x1 512->512); UNROLL over input channels = U parallel INT16 MACs, pipelined II=1.
 * Synth gives DSP per U MACs -> scale to the full ~512-800 MAC engine; latency = 52.5 GMAC / (MACs*clk). */
#include <ap_int.h>
typedef ap_int<16> i16;
typedef ap_int<32> i32;
#define IC 512
#define OCT 8
#define NP 64
#define U 32
void conv1x1(const i16 ifm[IC][NP], const i16 wt[OCT][IC], const i16 bias[OCT], int shift, i16 ofm[OCT][NP]) {
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS ARRAY_PARTITION variable=ifm cyclic factor=32 dim=1
#pragma HLS ARRAY_PARTITION variable=wt cyclic factor=32 dim=2
  for (int oc = 0; oc < OCT; oc++) {
    for (int p = 0; p < NP; p++) {
#pragma HLS PIPELINE II=1
      i32 acc = bias[oc];
      for (int ic = 0; ic < IC; ic++) {
#pragma HLS UNROLL factor=U
        acc += ifm[ic][p] * wt[oc][ic];
      }
      i32 r = acc >> shift;
      if (r < 0) r = 0;            /* ReLU */
      if (r > 32767) r = 32767;
      ofm[oc][p] = (i16)r;
    }
  }
}
