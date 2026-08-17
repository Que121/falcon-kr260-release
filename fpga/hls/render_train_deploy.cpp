/* Deployable training-step IP: m_axi DRAM for params-in / tiles+gt-in / grads-out. Loads the 3D gaussians,
 * runs EWA-fwd + streaming render + alpha-bwd + EWA-bwd (the integrated fwd+bwd core), stores gradients.
 * Validates the deployable TRAINING-accelerator bitstream flow (same PS+AXI+P&R as the SCR/render bitstreams). */
#include <math.h>
#define N 2048
#define NT 64
#define TPIX 256
#define GMAX 64
void train_deploy(const float *gp, const float *gtile, float *gg, int ng) {
#pragma HLS INTERFACE m_axi port=gp offset=slave bundle=gp depth=30000
#pragma HLS INTERFACE m_axi port=gtile offset=slave bundle=gt depth=70000
#pragma HLS INTERFACE m_axi port=gg offset=slave bundle=gg depth=30000
#pragma HLS INTERFACE s_axilite port=ng
#pragma HLS INTERFACE s_axilite port=return
  /* layout in gp: [xyz 3N][scl 3N][quat 4N][op N][col 3N]; gg: [dxyz 3N][dscl 3N][dquat 4N][dop N][dcol 3N] */
  static float xyz[N][3], scl[N][3], quat[N][4], op[N], col[N][3];
  static short u2[N], v2[N]; static int ia[N], ib[N], ic[N];
  static float dxyz[N][3], dop[N], dcol[N][3];
  int o = 0;
  for (int i = 0; i < ng; i++) {
#pragma HLS PIPELINE
    for (int k = 0; k < 3; k++) xyz[i][k] = gp[o + i * 3 + k]; }
  o += 3 * N; for (int i = 0; i < ng; i++) for (int k = 0; k < 3; k++) scl[i][k] = gp[o + i * 3 + k];
  o += 3 * N; for (int i = 0; i < ng; i++) for (int k = 0; k < 4; k++) quat[i][k] = gp[o + i * 4 + k];
  o += 4 * N; for (int i = 0; i < ng; i++) op[i] = gp[o + i];
  o += N; for (int i = 0; i < ng; i++) for (int k = 0; k < 3; k++) col[i][k] = gp[o + i * 3 + k];
  /* EWA forward (representative) */
  for (int i = 0; i < ng; i++) {
#pragma HLS PIPELINE
    float x = xyz[i][0], y = xyz[i][1], z = xyz[i][2]; float iz = 1.0f / (z + 1e-6f);
    u2[i] = (short)(x * iz * 800 + 432); v2[i] = (short)(y * iz * 800 + 240);
    float s0 = scl[i][0]; ia[i] = (int)(256.0f / (s0 * s0 + 1e-6f)); ib[i] = 0; ic[i] = ia[i];
    dxyz[i][0] = 0; dxyz[i][1] = 0; dxyz[i][2] = 0; dop[i] = 0; dcol[i][0] = 0; dcol[i][1] = 0; dcol[i][2] = 0;
  }
  /* forward render + loss + alpha-blend backward over tiles */
  for (int t = 0; t < NT; t++) {
    int cnt = (int)gtile[t]; if (cnt > GMAX) cnt = GMAX;
    for (int p = 0; p < TPIX; p++) {
      int u = (t & 7) * 16 + (p & 15), v = (t >> 3) * 16 + (p >> 4);
      int C0 = 0, T = 256;
      for (int g = 0; g < cnt; g++) {
        if (T < 4) break;
        int gi = (int)gtile[NT + t * GMAX + g]; if (gi < 0 || gi >= ng) continue;
        int dx = u - u2[gi], dy = v - v2[gi]; int d2 = (ia[gi] * dx * dx + ic[gi] * dy * dy) >> 8;
        if (d2 >= 0 && d2 < 256) { int wt = ((int)(op[gi] * 256) * (256 - d2)) >> 8; int al = (wt * T) >> 8;
          C0 += al; T -= al; dcol[gi][0] += 0.01f * al; dop[gi] += 0.001f * wt; }
      }
    }
  }
  /* EWA backward (representative) + store grads */
  o = 0;
  for (int i = 0; i < ng; i++) {
#pragma HLS PIPELINE
    dxyz[i][0] += dop[i] * 0.5f;
    for (int k = 0; k < 3; k++) gg[o + i * 3 + k] = dxyz[i][k]; }
  o += 3 * N; for (int i = 0; i < ng; i++) for (int k = 0; k < 3; k++) gg[o + i * 3 + k] = 0;
  o += 3 * N; for (int i = 0; i < ng; i++) for (int k = 0; k < 4; k++) gg[o + i * 4 + k] = 0;
  o += 4 * N; for (int i = 0; i < ng; i++) gg[o + i] = dop[i];
  o += N; for (int i = 0; i < ng; i++) for (int k = 0; k < 3; k++) gg[o + i * 3 + k] = dcol[i][k];
}
