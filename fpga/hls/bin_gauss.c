/* Binning front-end: assign projected Gaussians to screen tiles -> each tile's bounded Gaussian index list
 * (feeds render_fast2). SCR-init low overlap => bounded G/tile + center-tile assignment is a good approximation
 * (footprints are ~1-2 px). Output: tile_lists[t][0..count-1] = Gaussian indices, tile_counts[t]. */
#define NTILES 1590    /* ceil(854/16)=54 x ceil(480/16)=30 ~ 1590 */
#define G 16
#define TILES_X 54
void bin_gaussians(int ng, const short* su, const short* sv,
                   short tile_lists[NTILES][G], int tile_counts[NTILES]) {
#pragma HLS INTERFACE m_axi port=su offset=slave bundle=gmem0 depth=410000
#pragma HLS INTERFACE m_axi port=sv offset=slave bundle=gmem1 depth=410000
#pragma HLS INTERFACE bram port=tile_lists
#pragma HLS INTERFACE bram port=tile_counts
#pragma HLS INTERFACE s_axilite port=return
  int t, i;
  for (t = 0; t < NTILES; t++) {
#pragma HLS PIPELINE II=1
    tile_counts[t] = 0;
  }
  for (i = 0; i < ng; i++) {
#pragma HLS PIPELINE II=1
    int tx = su[i] >> 4, ty = sv[i] >> 4;     /* /16 = tile coords */
    int tt = ty * TILES_X + tx;
    if (tt >= 0 && tt < NTILES) {
      int c = tile_counts[tt];
      if (c < G) { tile_lists[tt][c] = (short)i; tile_counts[tt] = c + 1; }
    }
  }
}
