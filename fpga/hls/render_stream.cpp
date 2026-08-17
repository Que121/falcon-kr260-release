/* Streaming 3DGS rasterizer (GSCore/KAIST-style): ONE pipelined datapath streams the tile's Gaussians per
 * pixel, with early-termination + shape-aware skip. Resources FIXED regardless of G (not unrolled) -> handles
 * large G (hi-fi map) on the FPGA. Tile's Gaussians (sorted front-to-back) sit in BRAM. */
#define TPIX 256
#define GMAX 512    /* large per-tile Gaussian budget (BRAM-held, NOT unrolled) */
void render_stream(int ng,
                   const short gx[GMAX], const short gy[GMAX],
                   const short gia[GMAX], const short gib[GMAX], const short gic[GMAX],
                   const unsigned char gop[GMAX], const unsigned char gcol[GMAX][3],
                   const unsigned char wlut[256],
                   const short px[TPIX], const short py[TPIX],
                   unsigned char out[TPIX][3]) {
#pragma HLS INTERFACE s_axilite port=return
  int p, g;
  for (p = 0; p < TPIX; p++) {
    int u = px[p], v = py[p];
    int C0=0,C1=0,C2=0,T=256;
    for (g = 0; g < ng; g++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=512 avg=24
      if (T < 4) break;                              /* early termination */
      int dx=u-gx[g], dy=v-gy[g];
      int d2=(gia[g]*dx*dx + 2*gib[g]*dx*dy + gic[g]*dy*dy) >> 8;
      if (d2 >= 0 && d2 < 256) {                     /* shape-aware skip */
        int wt=(gop[g]*(int)wlut[d2])>>8; int a=(wt*T)>>8;
        C0+=a*gcol[g][0]; C1+=a*gcol[g][1]; C2+=a*gcol[g][2]; T-=a;
      }
    }
    out[p][0]=(unsigned char)(C0>>8); out[p][1]=(unsigned char)(C1>>8); out[p][2]=(unsigned char)(C2>>8);
  }
}
