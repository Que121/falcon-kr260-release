/* FAST + LEAN + deployable rasterizer (the SCR-init-low-overlap EXPLOIT). The tile's G Gaussians (bounded small
 * by low overlap) sit on-chip (partitioned BRAM, loaded once/tile = lean DRAM). Inner G-loop UNROLLED, the
 * PIXEL loop pipelined II=1 -> ~1 cyc/px. Resolves the gather(load-bound)-vs-eval(overhead-bound) tension. */
#define TPIX 256   /* 16x16 tile */
#define G    16    /* bounded tile Gaussians (low overlap); coverage mask handles <G actual */
void render_fast2(const short gx[G], const short gy[G],
                 const short gia[G], const short gib[G], const short gic[G],
                 const unsigned char gop[G], const unsigned char gcol[G][3],
                 const unsigned char wlut[256],
                 const short px[TPIX], const short py[TPIX],
                 unsigned char out[TPIX][3]) {
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS ARRAY_PARTITION variable=gx complete
#pragma HLS ARRAY_PARTITION variable=gy complete
#pragma HLS ARRAY_PARTITION variable=gia complete
#pragma HLS ARRAY_PARTITION variable=gib complete
#pragma HLS ARRAY_PARTITION variable=gic complete
#pragma HLS ARRAY_PARTITION variable=gop complete
#pragma HLS ARRAY_PARTITION variable=gcol complete dim=1
#pragma HLS ARRAY_PARTITION variable=wlut complete
  int p, g;
  for (p = 0; p < TPIX; p++) {
#pragma HLS PIPELINE II=1
    int C0=0,C1=0,C2=0,T=256;
    int u = px[p], v = py[p];
    for (g = 0; g < G; g++) {
#pragma HLS UNROLL
      int dx = u-gx[g], dy = v-gy[g];
      int d2 = (gia[g]*dx*dx + 2*gib[g]*dx*dy + gic[g]*dy*dy) >> 8;
      int cov = (d2 >= 0 && d2 < 256) ? (int)wlut[d2] : 0;   /* gaussian weight, 0 if outside */
      int wt = (gop[g]*cov) >> 8;
      int a = (wt*T) >> 8;
      C0 += a*gcol[g][0]; C1 += a*gcol[g][1]; C2 += a*gcol[g][2]; T -= a;
    }
    out[p][0]=(unsigned char)(C0>>8); out[p][1]=(unsigned char)(C1>>8); out[p][2]=(unsigned char)(C2>>8);
  }
}
