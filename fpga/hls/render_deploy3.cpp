/* Win-capable deployable render IP. Fixes the memory-bound deploy: (1) NO pix input - compute u,v on-chip from
 * tile+pixel index (saves 1.6 MB/frame); (2) uint32 RGBA output (wide store); (3) DATAFLOW overlaps load/render/
 * store across tiles -> render-bound. Only the lean per-tile Gaussian list crosses DRAM (288 B/tile). */
#include <string.h>
#define TPIX 256
#define G       16
#define TILES_X 54
#define TW      16

static void load_g(const short* src, short gbuf[G*9]) {
  memcpy(gbuf, src, G*9*sizeof(short));
}
static void render_tile(const short gbuf[G*9], const unsigned char lut[256], int t, unsigned int obuf[TPIX]) {
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
  int g, p;
  for (g = 0; g < G; g++) {
#pragma HLS UNROLL
    const short* q = gbuf + g*9;
    gx[g]=q[0]; gy[g]=q[1]; gia[g]=q[2]; gib[g]=q[3]; gic[g]=q[4]; gop[g]=q[5]; c0[g]=q[6]; c1[g]=q[7]; c2[g]=q[8];
  }
  int bu = (t % TILES_X) * TW, bv = (t / TILES_X) * TW;
  for (p = 0; p < TPIX; p++) {
#pragma HLS PIPELINE II=1
    int u = bu + (p & (TW-1)), v = bv + (p >> 4);
    int C0=0,C1=0,C2=0,T=256;
    for (g = 0; g < G; g++) {
#pragma HLS UNROLL
      int dx=u-gx[g], dy=v-gy[g];
      int d2=(gia[g]*dx*dx + 2*gib[g]*dx*dy + gic[g]*dy*dy) >> 8;
      int cov=(d2>=0 && d2<256) ? (int)lut[d2] : 0;
      int wt=(gop[g]*cov)>>8; int a=(wt*T)>>8;
      C0+=a*c0[g]; C1+=a*c1[g]; C2+=a*c2[g]; T-=a;
    }
    obuf[p] = ((unsigned)(C0>>8)) | (((unsigned)(C1>>8))<<8) | (((unsigned)(C2>>8))<<16) | (0xFFu<<24);
  }
}
static void store_o(const unsigned int obuf[TPIX], unsigned int* dst) {
  memcpy(dst, obuf, TPIX*sizeof(unsigned int));
}
void render_deploy3(int ntiles, const short* gdata, const unsigned char* wlut, unsigned int* out) {
#pragma HLS INTERFACE m_axi port=gdata offset=slave bundle=gmem0 depth=262144
#pragma HLS INTERFACE m_axi port=wlut  offset=slave bundle=gmem1 depth=256
#pragma HLS INTERFACE m_axi port=out   offset=slave bundle=gmem2 depth=409600
#pragma HLS INTERFACE s_axilite port=ntiles
#pragma HLS INTERFACE s_axilite port=return
  unsigned char lut[256];
#pragma HLS ARRAY_PARTITION variable=lut complete
  memcpy(lut, wlut, 256);
  int t;
  for (t = 0; t < ntiles; t++) {
#pragma HLS DATAFLOW
    short gbuf[G*9];
    unsigned int obuf[TPIX];
    load_g(gdata + t*G*9, gbuf);
    render_tile(gbuf, lut, t, obuf);
    store_o(obuf, out + t*TPIX);
  }
}
