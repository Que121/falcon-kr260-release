/* Vitis HLS render blend - on-chip (BRAM) tile data + pixel loop pipelined II=1 (the real rasterizer structure:
 * the binning streams the tile's per-pixel Gaussian lists into BRAM, then the blend pipelines). */
#define NPIX 1024
#define K 16
void render_blend(const unsigned char w[NPIX][K], const unsigned char col[NPIX][K][3], unsigned char out[NPIX][3]) {
#pragma HLS INTERFACE bram port=w
#pragma HLS INTERFACE bram port=col
#pragma HLS INTERFACE bram port=out
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS ARRAY_PARTITION variable=w dim=2 complete
#pragma HLS ARRAY_PARTITION variable=col dim=2 complete
  for (int p = 0; p < NPIX; p++) {
#pragma HLS PIPELINE II=1
    int C0=0,C1=0,C2=0,T=256;
    for (int k = 0; k < K; k++) {
#pragma HLS UNROLL
      int a=(w[p][k]*T)>>8; C0+=a*col[p][k][0]; C1+=a*col[p][k][1]; C2+=a*col[p][k][2]; T-=a;
    }
    out[p][0]=(unsigned char)(C0>>8); out[p][1]=(unsigned char)(C1>>8); out[p][2]=(unsigned char)(C2>>8);
  }
}
