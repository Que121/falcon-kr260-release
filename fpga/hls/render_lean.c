/* Lean per-tile 3DGS rasterizer IP (xck26) — the SCR-init-low-overlap exploit + the deployable data model.
 * Reads the tile's LEAN Gaussian list (not materialized per-pixel lists): screen mean, inv-cov, opacity, color,
 * depth-sorted front-to-back. Each tile pixel evaluates the covering Gaussians: cov-distance -> LUT Gaussian
 * weight -> front-to-back alpha-blend. Lean DRAM (the G Gaussians) => compute-bound for low overlap. */
#define TPIX 256   /* 16x16 tile */
#define GMAX 128
void render_lean(int ng,
                 const short gx[GMAX], const short gy[GMAX],
                 const short gia[GMAX], const short gib[GMAX], const short gic[GMAX], /* inv-cov a,b,c Q8 */
                 const unsigned char gop[GMAX], const unsigned char gcol[GMAX][3],
                 const unsigned char wlut[256],  /* exp(-0.5 d2) LUT, Q0.8, d2 in [0,256) */
                 const short px[TPIX], const short py[TPIX],
                 unsigned char out[TPIX][3]) {
#pragma HLS INTERFACE s_axilite port=return
  int p, g;
  for (p = 0; p < TPIX; p++) {
    int C0 = 0, C1 = 0, C2 = 0, T = 256;
    int u = px[p], v = py[p];
    for (g = 0; g < ng; g++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=128 avg=12
      int dx = u - gx[g], dy = v - gy[g];
      int d2 = (gia[g]*dx*dx + 2*gib[g]*dx*dy + gic[g]*dy*dy) >> 8;  /* d^T Sigma^-1 d, Q8 */
      if (d2 >= 0 && d2 < 256 && T > 4) {
        int wt = (gop[g] * wlut[d2]) >> 8;     /* opacity * gaussian weight, Q0.8 */
        int a = (wt * T) >> 8;
        C0 += a*gcol[g][0]; C1 += a*gcol[g][1]; C2 += a*gcol[g][2]; T -= a;
      }
    }
    out[p][0]=(unsigned char)(C0>>8); out[p][1]=(unsigned char)(C1>>8); out[p][2]=(unsigned char)(C2>>8);
  }
}
