/* Multi-tile binning IP - the exact step that thrashed the ARM (3x3 spreading + per-tile top-K by depth).
 * On PL: each gaussian scatters to its 3x3 tiles; per tile keep the K smallest-depth (bounded insertion).
 * Output-stationary per-tile top-K = no global sort. Confirms the ARM bottleneck is cheap on the FPGA. */
#include <ap_int.h>
#define N 8192
#define NT 64
#define TXn 8
#define TYn 8
#define K 16
#define TW 16
void binning(int ng, const short gx[N], const short gy[N], const int gz[N], short slot[NT][K]) {
#pragma HLS INTERFACE s_axilite port=return
  int slotz[NT][K];
#pragma HLS ARRAY_PARTITION variable=slotz complete dim=2
#pragma HLS ARRAY_PARTITION variable=slot complete dim=2
  for (int t = 0; t < NT; t++)
    for (int k = 0; k < K; k++) { slot[t][k] = -1; slotz[t][k] = 0x7fffffff; }
  for (int i = 0; i < ng; i++) {
#pragma HLS PIPELINE II=1
    int tx = gx[i] / TW, ty = gy[i] / TW, z = gz[i];
    for (int ox = -1; ox <= 1; ox++)
      for (int oy = -1; oy <= 1; oy++) {
        int nx = tx + ox, ny = ty + oy;
        if (nx < 0 || nx >= TXn || ny < 0 || ny >= TYn) continue;
        int t = ny * TXn + nx;
        int maxk = 0, maxz = slotz[t][0];
        for (int k = 1; k < K; k++) {
#pragma HLS UNROLL
          if (slotz[t][k] > maxz) { maxz = slotz[t][k]; maxk = k; }
        }
        if (z < maxz) { slot[t][maxk] = (short)i; slotz[t][maxk] = z; }
      }
  }
}
