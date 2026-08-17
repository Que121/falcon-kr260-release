/* Pixel-parallel streaming rasterizer: 16 pixel datapaths share the Gaussian stream (read once, broadcast).
 * Fixed resources w.r.t. G (16 datapaths, NOT G-unrolled). Renders any-G hi-fi map fast. */
#define TPIX 256
#define GMAX 512
#define PP   16
void render_stream_pp(int ng,
                      const short gx[GMAX], const short gy[GMAX],
                      const short gia[GMAX], const short gib[GMAX], const short gic[GMAX],
                      const unsigned char gop[GMAX], const unsigned char gcol[GMAX][3],
                      const unsigned char wlut[256],
                      const short px[TPIX], const short py[TPIX],
                      unsigned char out[TPIX][3]) {
#pragma HLS INTERFACE s_axilite port=return
  int pb, g, j;
  for (pb = 0; pb < TPIX; pb += PP) {
    int C0[PP],C1[PP],C2[PP],T[PP],U[PP],V[PP];
#pragma HLS ARRAY_PARTITION variable=C0 complete
#pragma HLS ARRAY_PARTITION variable=C1 complete
#pragma HLS ARRAY_PARTITION variable=C2 complete
#pragma HLS ARRAY_PARTITION variable=T complete
#pragma HLS ARRAY_PARTITION variable=U complete
#pragma HLS ARRAY_PARTITION variable=V complete
    for (j=0;j<PP;j++){ C0[j]=C1[j]=C2[j]=0; T[j]=256; U[j]=px[pb+j]; V[j]=py[pb+j]; }
    for (g = 0; g < ng; g++) {
#pragma HLS PIPELINE II=1
      short xx=gx[g],yy=gy[g],ia=gia[g],ib=gib[g],ic=gic[g]; unsigned char op=gop[g],c0=gcol[g][0],c1=gcol[g][1],c2=gcol[g][2];
      for (j=0;j<PP;j++){
#pragma HLS UNROLL
        int dx=U[j]-xx, dy=V[j]-yy;
        int d2=(ia*dx*dx + 2*ib*dx*dy + ic*dy*dy) >> 8;
        if (d2>=0 && d2<256 && T[j]>4){
          int wt=(op*(int)wlut[d2])>>8; int a=(wt*T[j])>>8;
          C0[j]+=a*c0; C1[j]+=a*c1; C2[j]+=a*c2; T[j]-=a;
        }
      }
    }
    for (j=0;j<PP;j++){ out[pb+j][0]=(unsigned char)(C0[j]>>8); out[pb+j][1]=(unsigned char)(C1[j]>>8); out[pb+j][2]=(unsigned char)(C2[j]>>8); }
  }
}
