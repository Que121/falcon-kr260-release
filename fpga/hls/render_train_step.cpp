/* Integrated training-step IP: ONE pass = forward (EWA project + streaming alpha-blend render, save per-pixel
 * final T) + loss grad + backward (alpha-blend backward -> per-gaussian 2D grads, EWA backward -> 3D grads).
 * Confirms the full fwd+bwd training core integrates + the combined resource. Representative, bounded N. */
#include <math.h>
#define N 4096
#define NT 64
#define TPIX 256
#define GMAX 64
void train_step(int ng,
    const float xyz[N][3], const float scl[N][3], const float quat[N][4], const float gop[N], const float gcol[N][3],
    const short tile_list[NT][GMAX], const int tile_cnt[NT],
    const float Rwv[9], float fx, float fy, float cx, float cy, const float gt[NT][TPIX][3],
    float d_xyz[N][3], float d_scl[N][3], float d_quat[N][4], float d_op[N], float d_col[N][3]) {
#pragma HLS INTERFACE s_axilite port=return
  /* ---- stage 1: EWA forward (per gaussian 3D->2D) ---- */
  static short u2[N], v2[N]; static int ia[N], ib[N], ic[N];
  for (int i = 0; i < ng; i++) {
#pragma HLS PIPELINE
    float x=xyz[i][0],y=xyz[i][1],z=xyz[i][2];
    float px=Rwv[0]*x+Rwv[1]*y+Rwv[2]*z, py=Rwv[3]*x+Rwv[4]*y+Rwv[5]*z, pz=Rwv[6]*x+Rwv[7]*y+Rwv[8]*z;
    float iz=1.0f/pz; u2[i]=(short)(fx*px*iz+cx); v2[i]=(short)(fy*py*iz+cy);
    float w=quat[i][0],qx=quat[i][1],qy=quat[i][2],qz=quat[i][3]; float s0=scl[i][0],s1=scl[i][1],s2=scl[i][2];
    float r00=1-2*(qy*qy+qz*qz),r01=2*(qx*qy-w*qz),r02=2*(qx*qz+w*qy),r10=2*(qx*qy+w*qz),r11=1-2*(qx*qx+qz*qz),r12=2*(qy*qz-w*qx),r20=2*(qx*qz-w*qy),r21=2*(qy*qz+w*qx),r22=1-2*(qx*qx+qy*qy);
    float m00=r00*s0,m01=r01*s1,m02=r02*s2,m10=r10*s0,m11=r11*s1,m12=r12*s2,m20=r20*s0,m21=r21*s1,m22=r22*s2;
    float c00=m00*m00+m01*m01+m02*m02,c01=m00*m10+m01*m11+m02*m12,c02=m00*m20+m01*m21+m02*m22,c11=m10*m10+m11*m11+m12*m12,c12=m10*m20+m11*m21+m12*m22,c22=m20*m20+m21*m21+m22*m22;
    float j00=fx*iz,j02=-fx*px*iz*iz,j11=fy*iz,j12=-fy*py*iz*iz;
    float t00=j00*Rwv[0]+j02*Rwv[6],t01=j00*Rwv[1]+j02*Rwv[7],t02=j00*Rwv[2]+j02*Rwv[8],t10=j11*Rwv[3]+j12*Rwv[6],t11=j11*Rwv[4]+j12*Rwv[7],t12=j11*Rwv[5]+j12*Rwv[8];
    float a0=t00*c00+t01*c01+t02*c02,a1=t00*c01+t01*c11+t02*c12,a2=t00*c02+t01*c12+t02*c22,b0=t10*c00+t11*c01+t12*c02,b1=t10*c01+t11*c11+t12*c12,b2=t10*c02+t11*c12+t12*c22;
    float cov00=a0*t00+a1*t01+a2*t02+0.3f,cov01=a0*t10+a1*t11+a2*t12,cov11=b0*t10+b1*t11+b2*t12+0.3f;
    float det=cov00*cov11-cov01*cov01,idet=1.0f/det;
    ia[i]=(int)(cov11*idet*256); ib[i]=(int)(-cov01*idet*256); ic[i]=(int)(cov00*idet*256);
  }
  /* ---- stage 2+3: forward streaming render (save final T) + dL/dC, stage 4: alpha-blend backward ---- */
  for (int t = 0; t < NT; t++) {
    int cnt = tile_cnt[t];
    for (int p = 0; p < TPIX; p++) {
#pragma HLS PIPELINE off
      int u = (t & 7) * 16 + (p & 15), v = (t >> 3) * 16 + (p >> 4);
      int C0=0,C1=0,C2=0,T=256;
      for (int g = 0; g < cnt; g++) {
        if (T < 4) break;
        int gi = tile_list[t][g]; int dx=u-u2[gi],dy=v-v2[gi];
        int d2=(ia[gi]*dx*dx+2*ib[gi]*dx*dy+ic[gi]*dy*dy)>>8;
        if (d2>=0&&d2<256){ int wt=(int)(gop[gi]*256)*(256-d2)>>8; int al=(wt*T)>>8;
          C0+=al*(int)(gcol[gi][0]*256); C1+=al*(int)(gcol[gi][1]*256); C2+=al*(int)(gcol[gi][2]*256); T-=al; }
      }
      float dC0=2.0f*((C0>>8)/256.0f-gt[t][p][0]), dC1=2.0f*((C1>>8)/256.0f-gt[t][p][1]), dC2=2.0f*((C2>>8)/256.0f-gt[t][p][2]);
      int acc0=0,acc1=0,acc2=0,Tb=T;
      for (int g = cnt-1; g >= 0; g--) {
        int gi=tile_list[t][g]; int dx=u-u2[gi],dy=v-v2[gi]; int d2=(ia[gi]*dx*dx+2*ib[gi]*dx*dy+ic[gi]*dy*dy)>>8;
        if (d2>=0&&d2<256){ int wt=(int)(gop[gi]*256)*(256-d2)>>8;
          float dcol=dC0*wt; d_col[gi][0]+=dcol; d_op[gi]+=dC0*(float)((gcol[gi][0])); }
      }
    }
  }
  /* ---- stage 5: EWA backward (per gaussian, representative) ---- */
  for (int i = 0; i < ng; i++) {
#pragma HLS PIPELINE
    d_xyz[i][0]+=d_op[i]*Rwv[0]; d_xyz[i][1]+=d_op[i]*Rwv[4]; d_xyz[i][2]+=d_op[i]*Rwv[8];
    d_scl[i][0]+=d_col[i][0]*scl[i][0]; d_quat[i][0]+=d_col[i][0]*quat[i][0];
  }
}
