/* Optimized deployable render IP: WIDE-BURST contiguous tile loads (m_axi, max_widen 512) into flat buffers,
 * then on-chip unpack into partitioned arrays for the pipelined render_fast2 compute. Fixes render_deploy's
 * per-tile load-bound bottleneck. */
#include <string.h>
#define TPIX 256
#define G    16
void render_deploy2(int ntiles,
                    const short* gdata, const unsigned char* wlut, const short* pix, unsigned char* out) {
#pragma HLS INTERFACE m_axi port=gdata offset=slave bundle=gmem0 depth=262144 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=wlut  offset=slave bundle=gmem1 depth=256
#pragma HLS INTERFACE m_axi port=pix   offset=slave bundle=gmem2 depth=819200 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=out   offset=slave bundle=gmem3 depth=1228800 max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=ntiles
#pragma HLS INTERFACE s_axilite port=return
  unsigned char lut[256];
#pragma HLS ARRAY_PARTITION variable=lut complete
  memcpy(lut, wlut, 256);
  int t, g, p;
  for (t = 0; t < ntiles; t++) {
    short gbuf[G*9], pbuf[TPIX*2];          /* flat, unpartitioned -> wide contiguous burst */
    unsigned char obuf[TPIX*3];
    memcpy(gbuf, gdata + t*G*9, sizeof(gbuf));
    memcpy(pbuf, pix + t*TPIX*2, sizeof(pbuf));
    short gx[G],gy[G],gia[G],gib[G],gic[G],gop[G],c0[G],c1[G],c2[G];
#pragma HLS ARRAY_PARTITION variable=gx complete
#pragma HLS ARRAY_PARTITION variable=gy complete
#pragma HLS ARRAY_PARTITION variable=gia complete
#pragma HLS ARRAY_PARTITION variable=gib complete
#pragma HLS ARRAY_PARTITION variable=gic complete
#pragma HLS ARRAY_PARTITION variable=gop complete
#pragma HLS ARRAY_PARTITION variable=c0 complete
#pragma HLS ARRAY_PARTITION variable=c1 complete
#pragma HLS ARRAY_PARTITION variable=c2 complete
    for (g = 0; g < G; g++) {
#pragma HLS UNROLL
      const short* q = gbuf + g*9;
      gx[g]=q[0]; gy[g]=q[1]; gia[g]=q[2]; gib[g]=q[3]; gic[g]=q[4]; gop[g]=q[5]; c0[g]=q[6]; c1[g]=q[7]; c2[g]=q[8];
    }
    for (p = 0; p < TPIX; p++) {
#pragma HLS PIPELINE II=1
      int C0=0,C1=0,C2=0,T=256; int u=pbuf[p*2], v=pbuf[p*2+1];
      for (g = 0; g < G; g++) {
#pragma HLS UNROLL
        int dx=u-gx[g], dy=v-gy[g];
        int d2=(gia[g]*dx*dx + 2*gib[g]*dx*dy + gic[g]*dy*dy) >> 8;
        int cov=(d2>=0 && d2<256) ? (int)lut[d2] : 0;
        int wt=(gop[g]*cov)>>8; int a=(wt*T)>>8;
        C0+=a*c0[g]; C1+=a*c1[g]; C2+=a*c2[g]; T-=a;
      }
      obuf[p*3]=(unsigned char)(C0>>8); obuf[p*3+1]=(unsigned char)(C1>>8); obuf[p*3+2]=(unsigned char)(C2>>8);
    }
    memcpy(out + t*TPIX*3, obuf, sizeof(obuf));
  }
}
