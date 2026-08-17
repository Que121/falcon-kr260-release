/* Bambu bilinear-resize horizontal 2-tap datapath, no system headers. C=64 lanes.
 * out[ox][c] = row[x0[ox]][c] + (wx[ox]*(row[x1[ox]][c]-row[x0[ox]][c]))>>8  (wx in Q8). */
#define C    64
#define WIN  100
#define WOUT 200
void resize_h(const int row[WIN][C], const unsigned short x0[WOUT], const unsigned short x1[WOUT],
              const unsigned short wx[WOUT], int out[WOUT][C]) {
  int ox, c;
  for (ox = 0; ox < WOUT; ox++) {
    int c0 = x0[ox], c1 = x1[ox], w = wx[ox];
    for (c = 0; c < C; c++) {
      int a = row[c0][c], b = row[c1][c];
      out[ox][c] = a + ((w * (b - a)) >> 8);
    }
  }
}
