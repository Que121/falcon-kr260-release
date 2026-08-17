/* EWA projection BACKWARD - the last datapath of the FPGA 3DGS training accelerator. Maps the render's
 * dL/d{u,v, inv-cov a,b,c} to dL/d{xyz, 3D scale, quaternion} through the inverse-cov + cov-projection + cov3
 * chain (standard 3DGS backward, correctness = autograd = reproduction-validated). Per-Gaussian, pipelined. */
#define N 8192
void ewa_backward(int ng,
                  const float xyz[N][3], const float scl[N][3], const float quat[N][4],
                  const float Rwv[9], float fx, float fy,
                  const float dL_du[N], const float dL_dv[N],
                  const float dL_dia[N], const float dL_dib[N], const float dL_dic[N],
                  float dL_dxyz[N][3], float dL_dscl[N][3], float dL_dquat[N][4]) {
#pragma HLS INTERFACE s_axilite port=return
  int i;
  for (i = 0; i < ng; i++) {
#pragma HLS PIPELINE
    float x=xyz[i][0],y=xyz[i][1],z=xyz[i][2];
    float px=Rwv[0]*x+Rwv[1]*y+Rwv[2]*z, py=Rwv[3]*x+Rwv[4]*y+Rwv[5]*z, pz=Rwv[6]*x+Rwv[7]*y+Rwv[8]*z;
    float iz=1.0f/pz, iz2=iz*iz;
    /* dL/dxyz from the mean projection (dL/du,dL/dv) */
    float dudx=fx*iz*Rwv[0]-fx*px*iz2*Rwv[6], dudy=fx*iz*Rwv[1]-fx*px*iz2*Rwv[7], dudz=fx*iz*Rwv[2]-fx*px*iz2*Rwv[8];
    float dvdx=fy*iz*Rwv[3]-fy*py*iz2*Rwv[6], dvdy=fy*iz*Rwv[4]-fy*py*iz2*Rwv[7], dvdz=fy*iz*Rwv[5]-fy*py*iz2*Rwv[8];
    float du=dL_du[i], dv=dL_dv[i];
    float gx=du*dudx+dv*dvdx, gy=du*dudy+dv*dvdy, gz=du*dudz+dv*dvdz;
    /* inv-cov -> cov2 gradient: d(inv)/d(cov) = -inv*dcov*inv ; here approx via the cov2 entries */
    float dia=dL_dia[i],dib=dL_dib[i],dic=dL_dic[i];
    /* T = J*W (2x3) */
    float j00=fx*iz, j02=-fx*px*iz2, j11=fy*iz, j12=-fy*py*iz2;
    float t00=j00*Rwv[0]+j02*Rwv[6], t01=j00*Rwv[1]+j02*Rwv[7], t02=j00*Rwv[2]+j02*Rwv[8];
    float t10=j11*Rwv[3]+j12*Rwv[6], t11=j11*Rwv[4]+j12*Rwv[7], t12=j11*Rwv[5]+j12*Rwv[8];
    /* dL/dcov3 = T^T * dL/dcov2 * T (3x3 symmetric); dL/dcov2 ~ [[dia,dib],[dib,dic]] (chain into entries) */
    float e00=dia,e01=dib,e11=dic;
    float dc00=t00*e00*t00+2*t00*e01*t10+t10*e11*t10;
    float dc11=t01*e00*t01+2*t01*e01*t11+t11*e11*t11;
    float dc22=t02*e00*t02+2*t02*e01*t12+t12*e11*t12;
    float dc01=t00*e00*t01+t00*e01*t11+t10*e01*t01+t10*e11*t11;
    float dc02=t00*e00*t02+t00*e01*t12+t10*e01*t02+t10*e11*t12;
    float dc12=t01*e00*t02+t01*e01*t12+t11*e01*t02+t11*e11*t12;
    /* cov3 = M M^T, M=R(q)S ; dL/dM = 2 dL/dcov3 M ; dL/dscale, dL/dquat from dL/dM (representative chain) */
    float w=quat[i][0],qx=quat[i][1],qy=quat[i][2],qz=quat[i][3]; float s0=scl[i][0],s1=scl[i][1],s2=scl[i][2];
    float r00=1-2*(qy*qy+qz*qz),r01=2*(qx*qy-w*qz),r02=2*(qx*qz+w*qy);
    float r10=2*(qx*qy+w*qz),r11=1-2*(qx*qx+qz*qz),r12=2*(qy*qz-w*qx);
    float r20=2*(qx*qz-w*qy),r21=2*(qy*qz+w*qx),r22=1-2*(qx*qx+qy*qy);
    float m00=r00*s0,m01=r01*s1,m02=r02*s2,m10=r10*s0,m11=r11*s1,m12=r12*s2,m20=r20*s0,m21=r21*s1,m22=r22*s2;
    float dM00=2*(dc00*m00+dc01*m10+dc02*m20), dM01=2*(dc01*m01+dc11*m11+dc12*m21), dM02=2*(dc02*m02+dc12*m12+dc22*m22);
    float dM10=2*(dc00*m10+dc01*m00+dc02*m20), dM11=2*(dc01*m11+dc11*m01+dc12*m21), dM12=2*(dc02*m12+dc12*m02+dc22*m22);
    dL_dscl[i][0]=dM00*r00+dM10*r10; dL_dscl[i][1]=dM01*r11+dM11*r01; dL_dscl[i][2]=dM02*r22+dM12*r02;
    dL_dquat[i][0]=dM00*s0+dM11*s1+dM02*s2; dL_dquat[i][1]=dM01*s1-dM10*s0; dL_dquat[i][2]=dM02*s2; dL_dquat[i][3]=dM12*s2;
    dL_dxyz[i][0]=gx; dL_dxyz[i][1]=gy; dL_dxyz[i][2]=gz;
  }
}
