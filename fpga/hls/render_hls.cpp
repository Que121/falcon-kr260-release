/* Vitis HLS 3DGS render blend core - real K26/xczu5ev part, 200 MHz, pixel loop pipelined II=1. */
#define NPIX 1024
#define K 16
void render_blend(const unsigned char w[NPIX][K],
                  const unsigned char col[NPIX][K][3],
                  unsigned char out[NPIX][3]) {
#pragma HLS INTERFACE m_axi port=w   offset=slave bundle=gmem0 depth=16384
#pragma HLS INTERFACE m_axi port=col offset=slave bundle=gmem1 depth=49152
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem2 depth=3072
#pragma HLS INTERFACE s_axilite port=return
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
