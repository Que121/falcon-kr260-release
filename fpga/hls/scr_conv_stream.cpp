/* STREAMING SCR conv engine (1x1, 512->512), WIDE 512-bit m_axi. Root cause of v1/v2/narrow-stream slowness =
 * 16-bit m_axi used 1/8-1/32 of the 128-bit AXI bus + a handshake per i16. Fix: explicit ap_uint<512> datapath
 * (32 i16/transfer) -> 32x fewer DDR transactions. Weights-stationary 256-MAC array, pixels stream through
 * DATAFLOW (read/mac/write overlap), pixel-major. Locked arithmetic (per-channel fixed-point requant+bias+int48).
 * ARM chains layers (ping-pong DDR) + residuals. Driver buffers are the same bytes (reinterpreted wide). */
#include <hls_stream.h>
#include <ap_int.h>
typedef ap_int<16> i16;
typedef ap_int<48> acc_t;
typedef ap_uint<512> wide_t;   // 32 i16 per word
#define IC 512
#define OCP 16
#define ICU 16
#define WPP (IC/32)            // 16 wide words per pixel
static void read_px(const wide_t* in_ddr, hls::stream<wide_t>& s, int NW) {
  for (int i = 0; i < NW; i++) {
#pragma HLS PIPELINE II=1
    s.write(in_ddr[i]);
  }
}
static void mac_px(hls::stream<wide_t>& sin, hls::stream<wide_t>& sout, const i16 W[IC][IC],
                   const int M[IC], const int B[IC], int shift, int NP) {
  for (int p = 0; p < NP; p++) {
    i16 ipx[IC];
#pragma HLS ARRAY_PARTITION variable=ipx cyclic factor=16
    for (int w = 0; w < WPP; w++) {
#pragma HLS PIPELINE II=1
      wide_t ww = sin.read();
      for (int k = 0; k < 32; k++) ipx[w*32+k] = (i16)(ap_int<16>)ww.range(16*k+15, 16*k);
    }
    i16 opx[IC];
#pragma HLS ARRAY_PARTITION variable=opx cyclic factor=32
    for (int ocb = 0; ocb < IC/OCP; ocb++) {       // SIMPLE MAC = the HLS sweet spot (176ms); II=1 idioms backfire
      acc_t acc[OCP];
#pragma HLS ARRAY_PARTITION variable=acc complete
      for (int o = 0; o < OCP; o++) acc[o] = 0;
      for (int ic = 0; ic < IC; ic += ICU) {
#pragma HLS PIPELINE II=1
        for (int o = 0; o < OCP; o++) {
#pragma HLS UNROLL
          acc_t ps = 0;
          for (int u = 0; u < ICU; u++) ps += ipx[ic+u]*W[ocb*OCP+o][ic+u];
          acc[o] += ps;
        }
      }
      for (int o = 0; o < OCP; o++) {
        ap_int<72> r = ((ap_int<72>)acc[o]*(ap_int<72>)M[ocb*OCP+o]) >> shift;
        int v = (int)r + B[ocb*OCP+o]; if (v < 0) v = 0; if (v > 32767) v = 32767;
        opx[ocb*OCP+o] = (i16)v;
      }
    }
    for (int w = 0; w < WPP; w++) {
#pragma HLS PIPELINE II=1
      wide_t ww;
      for (int k = 0; k < 32; k++) ww.range(16*k+15, 16*k) = (ap_uint<16>)opx[w*32+k];
      sout.write(ww);
    }
  }
}
static void write_px(hls::stream<wide_t>& s, wide_t* out_ddr, int NW) {
  for (int i = 0; i < NW; i++) {
#pragma HLS PIPELINE II=1
    out_ddr[i] = s.read();
  }
}
static void stream_conv(const wide_t* in_ddr, wide_t* out_ddr, const i16 W[IC][IC],
                        const int M[IC], const int B[IC], int shift, int NP) {
#pragma HLS DATAFLOW
  hls::stream<wide_t> sin("sin"), sout("sout");
#pragma HLS STREAM variable=sin depth=32
#pragma HLS bind_storage variable=sin type=fifo impl=srl
#pragma HLS STREAM variable=sout depth=32
#pragma HLS bind_storage variable=sout type=fifo impl=srl
  read_px(in_ddr, sin, NP*WPP);
  mac_px(sin, sout, W, M, B, shift, NP);
  write_px(sout, out_ddr, NP*WPP);
}
void scr_conv_stream(const wide_t* in_ddr, wide_t* out_ddr, const i16* wts,
                     const int* mult, const int* bias, int shift, int NP) {
#pragma HLS INTERFACE m_axi port=in_ddr offset=slave bundle=gi depth=102720 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=out_ddr offset=slave bundle=go depth=102720 max_write_burst_length=64
#pragma HLS INTERFACE m_axi port=wts offset=slave bundle=gw depth=262144
#pragma HLS INTERFACE m_axi port=mult offset=slave bundle=gp depth=512
#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gp depth=512
#pragma HLS INTERFACE s_axilite port=shift
#pragma HLS INTERFACE s_axilite port=NP
#pragma HLS INTERFACE s_axilite port=return
  static i16 W[IC][IC];
#pragma HLS ARRAY_PARTITION variable=W cyclic factor=OCP dim=1
#pragma HLS ARRAY_PARTITION variable=W cyclic factor=ICU dim=2
  static int M[IC], B[IC];
#pragma HLS ARRAY_PARTITION variable=M cyclic factor=OCP
#pragma HLS ARRAY_PARTITION variable=B cyclic factor=OCP
  for (int o = 0; o < IC; o++) {
    for (int i = 0; i < IC; i++) {
#pragma HLS PIPELINE II=1
      W[o][i] = wts[o*IC + i];
    }
    M[o] = mult[o]; B[o] = bias[o];
  }
  stream_conv(in_ddr, out_ddr, W, M, B, shift, NP);
}
