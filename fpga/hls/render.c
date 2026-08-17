/* Bambu 3DGS render blend datapath (no system headers, plain types).
 * Per-pixel front-to-back alpha compositing of K depth-sorted Gaussian contributions:
 *   C = sum_k  a_k * color_k ,  a_k = w_k * T ,  T <- T - a_k   (T = transmittance)
 * w_k (Q0.8) = opacity_k * gaussian2d(pixel - mean_k), precomputed by the projection/footprint front-end.
 * This is the dominant, GPU-divergent stage where the FPGA wins. NPIX pixels per call (one tile-batch);
 * the full image streams NPIX-batches. K is bounded (SCR-init -> alpha saturates by ~4-8, K=16 is safe).
 * Fixed-point Q0.8. Outer (pixel) loop is independent -> pipelines at II=1; inner K is the unrolled recurrence. */
#define NPIX 1024
#define K    16
void render_blend(const unsigned char w[NPIX][K],
                  const unsigned char col[NPIX][K][3],
                  unsigned char out[NPIX][3]) {
  int p, k;
  for (p = 0; p < NPIX; p++) {
    int C0 = 0, C1 = 0, C2 = 0, T = 256; /* T=1.0 in Q0.8 */
    for (k = 0; k < K; k++) {
      int a = (w[p][k] * T) >> 8;        /* a_k = w_k * T   (Q0.8) */
      C0 += a * col[p][k][0];
      C1 += a * col[p][k][1];
      C2 += a * col[p][k][2];
      T -= a;                            /* T <- T*(1-w_k) */
    }
    out[p][0] = (unsigned char)(C0 >> 8);
    out[p][1] = (unsigned char)(C1 >> 8);
    out[p][2] = (unsigned char)(C2 >> 8);
  }
}
