/* Deployable tiled 3DGS render IP: bursts a tile's per-pixel lists DRAM->BRAM, then pipelines the blend.
 * Self-contained m_axi (loadable as a PYNQ overlay). The memory architecture that makes the blend fast. */
#include <string.h>
#define NPIX 1024
#define K 16
void render_tile(const unsigned char* w_dram, const unsigned char* col_dram, unsigned char* out_dram) {
#pragma HLS INTERFACE m_axi port=w_dram   offset=slave bundle=gmem0 depth=16384
#pragma HLS INTERFACE m_axi port=col_dram offset=slave bundle=gmem1 depth=49152
#pragma HLS INTERFACE m_axi port=out_dram offset=slave bundle=gmem2 depth=3072
#pragma HLS INTERFACE s_axilite port=return
  static unsigned char w[NPIX][K], col[NPIX][K][3], out[NPIX][3];
#pragma HLS ARRAY_PARTITION variable=w dim=2 complete
#pragma HLS ARRAY_PARTITION variable=col dim=2 complete
  memcpy(&w[0][0], w_dram, NPIX*K);              /* burst load tile */
  memcpy(&col[0][0][0], col_dram, NPIX*K*3);
  for (int p = 0; p < NPIX; p++) {
#pragma HLS PIPELINE II=1
    int C0=0,C1=0,C2=0,T=256;
    for (int k = 0; k < K; k++) {
#pragma HLS UNROLL
      int a=(w[p][k]*T)>>8; C0+=a*col[p][k][0]; C1+=a*col[p][k][1]; C2+=a*col[p][k][2]; T-=a;
    }
    out[p][0]=(unsigned char)(C0>>8); out[p][1]=(unsigned char)(C1>>8); out[p][2]=(unsigned char)(C2>>8);
  }
  memcpy(out_dram, &out[0][0], NPIX*3);          /* burst store tile */
}
