/* Bambu bounded-K depth sort: per pixel keep the K=16 nearest Gaussians by depth (front-to-back).
 * SCR-init low overlap (p99~47, alpha saturates ~4-8) -> a small K-buffer insertion, NOT a global sort. */
#define K 16
void ksort(int NG, const int depth[1024], const short id[1024],
           int out_depth[K], short out_id[K]) {
  int i, j;
  for (i = 0; i < K; i++) { out_depth[i] = 0x7fffffff; out_id[i] = -1; }
  for (i = 0; i < NG; i++) {
    int d = depth[i]; short di = id[i];
    for (j = K - 1; j >= 0; j--) {
      if (d < out_depth[j]) {
        if (j < K - 1) { out_depth[j + 1] = out_depth[j]; out_id[j + 1] = out_id[j]; }
        out_depth[j] = d; out_id[j] = di;
      } else { break; }
    }
  }
}
