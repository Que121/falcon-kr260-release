#include "bev_gather.hpp"
#include <cstdio>
#include <cstdlib>

// C/RTL-sim testbench: proves the wide-word (512-bit packed) gather IP is bit-exact to the SCALAR
// segmented scatter-add (same ap_fixed accumulate + requant, computed unpacked here). 0 mismatches =>
// packing is lossless; with bev_gather_verify.py (scalar == CUDA bev_pool_v2, 2.6e-5) the chain closes.

static signed char sfeat[FEAT_VECS][C];
static signed char gbev[NUM_BEV][C];

int main() {
    static wide_t feat[FEAT_VECS];
    static depth_t depth[DEPTH_LEN];
    static idx_t  rank_depth[N_POINTS], rank_feat[N_POINTS];
    static idx_t  rank_bev[N_PILLAR], istart[N_PILLAR], ilen[N_PILLAR];
    static wide_t bev[NUM_BEV];

    srand(3);
    for (int v = 0; v < FEAT_VECS; v++) {
        wide_t w = 0;
        for (int c = 0; c < C; c++) {
            int x = (rand() % 255) - 127;
            sfeat[v][c] = (signed char)x;
            w.range(8*c + 7, 8*c) = (ap_uint<8>)(ap_int<8>)x;
        }
        feat[v] = w;
    }
    for (int i = 0; i < DEPTH_LEN; i++) depth[i] = (depth_t)((rand() % 256) / 256.0);

    // partition [0,N_POINTS) into N_PILLAR contiguous intervals; distinct BEV cell per pillar
    int per = N_POINTS / N_PILLAR, off = 0;
    for (int p = 0; p < N_PILLAR; p++) {
        istart[p] = off;
        int l = (p == N_PILLAR - 1) ? (N_POINTS - off) : per;
        ilen[p] = l; off += l;
        rank_bev[p] = p;                 // N_PILLAR < NUM_BEV -> distinct, no collisions
    }
    for (int i = 0; i < N_POINTS; i++) {
        rank_depth[i] = rand() % DEPTH_LEN;
        rank_feat[i]  = rand() % FEAT_VECS;
    }
    acc_t out_scale = (acc_t)0.05;

    bev_gather(feat, depth, rank_depth, rank_feat, rank_bev, istart, ilen, out_scale, bev);

    // scalar golden
    for (int b = 0; b < NUM_BEV; b++) for (int c = 0; c < C; c++) gbev[b][c] = 0;
    for (int p = 0; p < N_PILLAR; p++) {
        acc_t acc[C]; for (int c = 0; c < C; c++) acc[c] = 0;
        for (int k = 0; k < (int)ilen[p]; k++) {
            int i = istart[p] + k;
            depth_t w = depth[rank_depth[i]];
            int fv = rank_feat[i];
            for (int c = 0; c < C; c++) acc[c] += (acc_t)w * (acc_t)(feat_t)sfeat[fv][c];
        }
        int b = rank_bev[p];
        for (int c = 0; c < C; c++) {
            acc_t v = acc[c] * out_scale;
            if (v > (acc_t)127)  v = 127;
            if (v < (acc_t)-128) v = -128;
            gbev[b][c] = (signed char)(bev_t)v;
        }
    }

    int errs = 0;
    for (int b = 0; b < NUM_BEV; b++)
      for (int c = 0; c < C; c++) {
        signed char hv = (signed char)(ap_int<8>)bev[b].range(8*c + 7, 8*c);
        if (hv != gbev[b][c]) errs++;
      }
    printf("GATHER_MISMATCHES=%d  (N_POINTS=%d N_PILLAR=%d NUM_BEV=%d C=%d)\n",
           errs, N_POINTS, N_PILLAR, NUM_BEV, C);
    return errs ? 1 : 0;
}
