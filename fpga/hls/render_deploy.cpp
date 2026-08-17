/* Deployable full-image 3DGS render IP (KR260 PYNQ overlay): loops over tiles, reads each tile's bounded
 * Gaussian list + pixel coords from DRAM (m_axi), runs the render_fast2 eval+blend, writes the image to DRAM.
 * Self-contained -> loadable as an overlay; the host (ARM) fills the DRAM buffers from the binning output. */
#include <string.h>
#define TPIX 256
#define G    16
void render_deploy(int ntiles,
                   const short* gdata,        /* [ntiles*G*9]  gx,gy,gia,gib,gic,gop,c0,c1,c2 per gaussian */
                   const unsigned char* wlut, /* [256] gaussian-weight LUT */
                   const short* pix,          /* [ntiles*TPIX*2]  u,v per pixel */
                   unsigned char* out) {      /* [ntiles*TPIX*3]  rgb per pixel */
#pragma HLS INTERFACE m_axi port=gdata offset=slave bundle=gmem0 depth=262144
#pragma HLS INTERFACE m_axi port=wlut  offset=slave bundle=gmem1 depth=256
#pragma HLS INTERFACE m_axi port=pix   offset=slave bundle=gmem2 depth=819200
#pragma HLS INTERFACE m_axi port=out   offset=slave bundle=gmem3 depth=1228800
#pragma HLS INTERFACE s_axilite port=ntiles
#pragma HLS INTERFACE s_axilite port=return
  unsigned char lut[256];
#pragma HLS ARRAY_PARTITION variable=lut complete
  memcpy(lut, wlut, 256);
  int t, g, p;
  for (t = 0; t < ntiles; t++) {
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
    short pu[TPIX], pv[TPIX];
    unsigned char o[TPIX][3];
    for (g = 0; g < G; g++) {
      const short* gp = gdata + (t*G+g)*9;
      gx[g]=gp[0]; gy[g]=gp[1]; gia[g]=gp[2]; gib[g]=gp[3]; gic[g]=gp[4];
      gop[g]=gp[5]; c0[g]=gp[6]; c1[g]=gp[7]; c2[g]=gp[8];
    }
    for (p = 0; p < TPIX; p++) { pu[p]=pix[(t*TPIX+p)*2]; pv[p]=pix[(t*TPIX+p)*2+1]; }
    for (p = 0; p < TPIX; p++) {
#pragma HLS PIPELINE II=1
      int C0=0,C1=0,C2=0,T=256; int u=pu[p], v=pv[p];
      for (g = 0; g < G; g++) {
#pragma HLS UNROLL
        int dx=u-gx[g], dy=v-gy[g];
        int d2=(gia[g]*dx*dx + 2*gib[g]*dx*dy + gic[g]*dy*dy) >> 8;
        int cov=(d2>=0 && d2<256) ? (int)lut[d2] : 0;
        int wt=(gop[g]*cov)>>8; int a=(wt*T)>>8;
        C0+=a*c0[g]; C1+=a*c1[g]; C2+=a*c2[g]; T-=a;
      }
      o[p][0]=(unsigned char)(C0>>8); o[p][1]=(unsigned char)(C1>>8); o[p][2]=(unsigned char)(C2>>8);
    }
    memcpy(out + t*TPIX*3, &o[0][0], TPIX*3);
  }
}
