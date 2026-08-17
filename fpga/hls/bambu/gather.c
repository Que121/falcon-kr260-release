/* Bambu gather datapath, no system headers (plain types). C=64 INT8 MACs/iter. */
#define C  64
#define NP 256
void bev_gather(const signed char feat[NP][C], const unsigned char depthw[NP], int bev_out[C]) {
  int acc[C]; int i, c;
  for (c = 0; c < C; c++) acc[c] = 0;
  for (i = 0; i < NP; i++) {
    unsigned char w = depthw[i];
    for (c = 0; c < C; c++) acc[c] += (int)w * (int)feat[i][c];
  }
  for (c = 0; c < C; c++) bev_out[c] = acc[c];
}
