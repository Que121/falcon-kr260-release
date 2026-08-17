/* Bambu 3DGS projection front-end: per-Gaussian world->screen (u,v,depth) + footprint radius. */
#define N 1024
void project(const float xyz[N][3], const float Rt[12], float fx, float fy, float cx, float cy,
             const float scale[N], float uvdr[N][4]) {
  int i;
  for (i = 0; i < N; i++) {
    float x=xyz[i][0], y=xyz[i][1], z=xyz[i][2];
    float xc=Rt[0]*x+Rt[1]*y+Rt[2]*z+Rt[3];
    float yc=Rt[4]*x+Rt[5]*y+Rt[6]*z+Rt[7];
    float zc=Rt[8]*x+Rt[9]*y+Rt[10]*z+Rt[11];
    float inv=1.0f/zc;
    uvdr[i][0]=fx*xc*inv+cx; uvdr[i][1]=fy*yc*inv+cy; uvdr[i][2]=zc; uvdr[i][3]=scale[i]*fx*inv;
  }
}
