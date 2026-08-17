#include <cstdio>
#include <cmath>
#include <ap_int.h>
typedef ap_int<16> i16;
typedef ap_uint<512> wide_t;
#define IC 512
#define NPT 128
void scr_conv_stream(const wide_t*, wide_t*, const i16*, const int*, const int*, int, int);
static short wbuf[3*IC*IC], ibuf[IC*NPT];
static i16 wts[3*IC*IC];
static i16 bufA[NPT*IC], bufB[NPT*IC];   // pixel-major p*IC+c (reinterpreted wide: 32 ch/word)
static int mult[3*IC], bias[3*IC], shifts[3];
static float ref[IC*NPT], sout;
static FILE* O(const char*n){ char path[256]; sprintf(path,"/home/ANON/emerge/hbin/%s",n); return fopen(path,"rb"); }
int main(){
  FILE*f;
  f=O("wts.bin"); fread(wbuf,2,3*IC*IC,f); fclose(f); for(int i=0;i<3*IC*IC;i++) wts[i]=wbuf[i];
  f=O("mult.bin"); fread(mult,4,3*IC,f); fclose(f);
  f=O("bias.bin"); fread(bias,4,3*IC,f); fclose(f);
  f=O("shifts.bin"); fread(shifts,4,3,f); fclose(f);
  f=O("input.bin"); fread(ibuf,2,IC*NPT,f); fclose(f);  // channel-major c*NPT+p
  f=O("ref.bin"); fread(ref,4,IC*NPT,f); fclose(f);
  f=O("sout.bin"); fread(&sout,4,1,f); fclose(f);
  for(int c=0;c<IC;c++) for(int p=0;p<NPT;p++) bufA[p*IC+c]=ibuf[c*NPT+p];  // -> pixel-major
  scr_conv_stream((const wide_t*)bufA,(wide_t*)bufB, wts+0*IC*IC, mult+0*IC, bias+0*IC, shifts[0], NPT);
  scr_conv_stream((const wide_t*)bufB,(wide_t*)bufA, wts+1*IC*IC, mult+1*IC, bias+1*IC, shifts[1], NPT);
  scr_conv_stream((const wide_t*)bufA,(wide_t*)bufB, wts+2*IC*IC, mult+2*IC, bias+2*IC, shifts[2], NPT);
  double dot=0,na=0,nb=0;
  for(int c=0;c<IC;c++) for(int p=0;p<NPT;p++){ float s=(float)((int)bufB[p*IC+c])*sout; float r=ref[c*NPT+p]; dot+=s*r; na+=(double)s*s; nb+=(double)r*r; }
  printf("WIDE STREAM res3 (3 calls) cosine vs FP32 = %.5f\n", dot/(sqrt(na)*sqrt(nb)+1e-9));
  return 0;
}
