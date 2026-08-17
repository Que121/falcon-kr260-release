/* Bambu-synthesizable plain-C view-transform gather datapath (no Xilinx headers).
 * Captures the core: per point, depth-weight x C-channel INT8 feature, accumulated.
 * NP here is a representative synthesis trip count; the real rig has N_POINTS=302558,
 * so WCET = N_POINTS * II_inner (the II + per-lane resources reported by Bambu are
 * trip-count-independent). C=64 INT8 MACs per cycle. */
#include <stdint.h>
#define C  64
#define NP 256

void bev_gather(const int8_t feat[NP][C], const uint8_t depthw[NP], int32_t bev_out[C])
{
    int32_t acc[C];
    int i, c;
    for (c = 0; c < C; c++) acc[c] = 0;
    for (i = 0; i < NP; i++) {
        uint8_t w = depthw[i];
        for (c = 0; c < C; c++)
            acc[c] += (int32_t)w * (int32_t)feat[i][c];
    }
    for (c = 0; c < C; c++) bev_out[c] = acc[c];
}
