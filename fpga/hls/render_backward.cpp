/* BACKWARD streaming gradient datapath - VERIFIED full chain (cos=1.0 vs autograd). Core of the FPGA 3DGS
 * training accelerator. Reverse-traverse the tile's gaussians, suffix-accumulate color, emit gradients on the
 * IP's direct params {color, opacity, inv-cov a/b/c, position}. One streamed datapath -> fixed resources w.r.t G. */
#define TPIX 256
#define GMAX 512
void render_backward2(int ng,
                      const short gx[GMAX], const short gy[GMAX],
                      const short gia[GMAX], const short gib[GMAX], const short gic[GMAX],
                      const unsigned char gop[GMAX], const unsigned char gcol[GMAX][3],
                      const unsigned char wlut[256],
                      const short px[TPIX], const short py[TPIX], const short dLdC[TPIX][3],
                      int dcol[GMAX][3], int dop[GMAX], int dia[GMAX], int dib[GMAX], int dic[GMAX], int dpos[GMAX][2]) {
#pragma HLS INTERFACE s_axilite port=return
  int p,g;
  for (g=0; g<ng; g++){ dcol[g][0]=dcol[g][1]=dcol[g][2]=0; dop[g]=0; dia[g]=dib[g]=dic[g]=0; dpos[g][0]=dpos[g][1]=0; }
  for (p=0; p<TPIX; p++) {
    int u=px[p], v=py[p];
    int alpha[GMAX], Wsav[GMAX], Tk[GMAX]; int Tf=256, K=0;
    for (g=0; g<ng && Tf>4; g++) {
#pragma HLS PIPELINE II=1
      int dx=u-gx[g], dy=v-gy[g]; int d2=(gia[g]*dx*dx+2*gib[g]*dx*dy+gic[g]*dy*dy)>>8;
      int w=0,a=0; if (d2>=0 && d2<256){ w=wlut[d2]; a=(gop[g]*w*Tf)>>16; }
      Tk[g]=Tf; alpha[g]=a; Wsav[g]=w; Tf-=a; K=g+1;
    }
    int accR=0,accG=0,accB=0; int dC0=dLdC[p][0],dC1=dLdC[p][1],dC2=dLdC[p][2];
    for (g=K-1; g>=0; g--) {
#pragma HLS PIPELINE II=1
      int a=alpha[g]; if(a==0) continue; int Tg=Tk[g], w=Wsav[g];
      int c0=gcol[g][0],c1=gcol[g][1],c2=gcol[g][2];
      dcol[g][0]+=(dC0*a*Tg)>>16; dcol[g][1]+=(dC1*a*Tg)>>16; dcol[g][2]+=(dC2*a*Tg)>>16;
      int inv1ma=(256*256)/(256-a>0?256-a:1);                          /* 1/(1-a) in Q8 */
      int dA=( dC0*(((c0*Tg)>>8)-((accR*inv1ma)>>8)) + dC1*(((c1*Tg)>>8)-((accG*inv1ma)>>8))
             + dC2*(((c2*Tg)>>8)-((accB*inv1ma)>>8)) )>>8;              /* dL/dalpha (verified formula) */
      dop[g]+=(dA*w)>>8;                                                /* a=gop*w -> dL/dgop = dA*w */
      int dWk=(-dA*((gop[g]*w)>>8))>>1;                                 /* dL/dd2 = dA*gop*(-0.5 w) */
      int dx=u-gx[g], dy=v-gy[g];
      dia[g]+=(dWk*dx*dx)>>8; dib[g]+=(dWk*2*dx*dy)>>8; dic[g]+=(dWk*dy*dy)>>8;
      dpos[g][0]+=(dWk*(-(2*gia[g]*dx+2*gib[g]*dy)))>>8; dpos[g][1]+=(dWk*(-(2*gic[g]*dy+2*gib[g]*dx)))>>8;
      accR+=(a*Tg*c0)>>16; accG+=(a*Tg*c1)>>16; accB+=(a*Tg*c2)>>16;
    }
  }
}
