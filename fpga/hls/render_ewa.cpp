/* EWA projection front-end: per-Gaussian 3D->2D (mean u,v + 2D inv-cov a,b,c). The missing half of the FPGA
 * 3DGS training accelerator (the part that lets the optimizer move 3D geometry -> reaches 31). Per-Gaussian,
 * cheap (like project.c). cov2 = J*W*cov3*W^T*J^T ; cov3 = R(q)*S^2*R(q)^T. */
#include <math.h>
#define N 8192
void ewa_project(int ng, const float xyz[N][3], const float scl[N][3], const float quat[N][4],
                 const float Rwv[9], float fx, float fy, float cx, float cy,
                 short u[N], short v[N], short ia[N], short ib[N], short ic[N]) {
#pragma HLS INTERFACE s_axilite port=return
  int i;
  for (i = 0; i < ng; i++) {
#pragma HLS PIPELINE
    float x=xyz[i][0],y=xyz[i][1],z=xyz[i][2];
    float px=Rwv[0]*x+Rwv[1]*y+Rwv[2]*z, py=Rwv[3]*x+Rwv[4]*y+Rwv[5]*z, pz=Rwv[6]*x+Rwv[7]*y+Rwv[8]*z;
    float iz=1.0f/pz;
    u[i]=(short)(fx*px*iz+cx); v[i]=(short)(fy*py*iz+cy);
    /* R(q) */
    float w=quat[i][0],qx=quat[i][1],qy=quat[i][2],qz=quat[i][3];
    float r00=1-2*(qy*qy+qz*qz), r01=2*(qx*qy-w*qz), r02=2*(qx*qz+w*qy);
    float r10=2*(qx*qy+w*qz), r11=1-2*(qx*qx+qz*qz), r12=2*(qy*qz-w*qx);
    float r20=2*(qx*qz-w*qy), r21=2*(qy*qz+w*qx), r22=1-2*(qx*qx+qy*qy);
    float s0=scl[i][0],s1=scl[i][1],s2=scl[i][2];
    /* M=R*S ; cov3=M*M^T */
    float m00=r00*s0,m01=r01*s1,m02=r02*s2, m10=r10*s0,m11=r11*s1,m12=r12*s2, m20=r20*s0,m21=r21*s1,m22=r22*s2;
    float c00=m00*m00+m01*m01+m02*m02, c01=m00*m10+m01*m11+m02*m12, c02=m00*m20+m01*m21+m02*m22;
    float c11=m10*m10+m11*m11+m12*m12, c12=m10*m20+m11*m21+m12*m22, c22=m20*m20+m21*m21+m22*m22;
    /* J*W (2x3) : J=[[fx/z,0,-fx*px/z^2],[0,fy/z,-fy*py/z^2]] */
    float j00=fx*iz, j02=-fx*px*iz*iz, j11=fy*iz, j12=-fy*py*iz*iz;
    float t00=j00*Rwv[0]+j02*Rwv[6], t01=j00*Rwv[1]+j02*Rwv[7], t02=j00*Rwv[2]+j02*Rwv[8];
    float t10=j11*Rwv[3]+j12*Rwv[6], t11=j11*Rwv[4]+j12*Rwv[7], t12=j11*Rwv[5]+j12*Rwv[8];
    /* cov2 = T*cov3*T^T (2x2) */
    float a0=t00*c00+t01*c01+t02*c02, a1=t00*c01+t01*c11+t02*c12, a2=t00*c02+t01*c12+t02*c22;
    float b0=t10*c00+t11*c01+t12*c02, b1=t10*c01+t11*c11+t12*c12, b2=t10*c02+t11*c12+t12*c22;
    float cov00=a0*t00+a1*t01+a2*t02+0.3f, cov01=a0*t10+a1*t11+a2*t12, cov11=b0*t10+b1*t11+b2*t12+0.3f;
    float det=cov00*cov11-cov01*cov01; float idet=1.0f/det;
    ia[i]=(short)(cov11*idet*256); ib[i]=(short)(-cov01*idet*256); ic[i]=(short)(cov00*idet*256);
  }
}
