#include <cstdio>
#include <cmath>
#include <ap_int.h>
typedef ap_int<16> i16;
#define IC 512
#define NPT 128
void scr_head16v2(const i16*, const i16*, i16*, const int*, const int*, const int*, int, int);
static short wbuf[3*IC*IC], ibuf[IC*NPT];
static i16 wts[3*IC*IC], input[IC*NPT], out[IC*NPT];
static int mult[3*IC], bias[3*IC], shifts[3];
static float ref[IC*NPT], sout;
static FILE* O(const char*n){ char path[256]; sprintf(path,"/home/ANON/emerge/hbin/%s",n); return fopen(path,"rb"); }
int main(){
  FILE*f;
  f=O("wts.bin"); fread(wbuf,2,3*IC*IC,f); fclose(f); for(int i=0;i<3*IC*IC;i++) wts[i]=wbuf[i];
  f=O("mult.bin"); fread(mult,4,3*IC,f); fclose(f);
  f=O("bias.bin"); fread(bias,4,3*IC,f); fclose(f);
  f=O("shifts.bin"); fread(shifts,4,3,f); fclose(f);
  f=O("input.bin"); fread(ibuf,2,IC*NPT,f); fclose(f); for(int i=0;i<IC*NPT;i++) input[i]=ibuf[i];
  f=O("ref.bin"); fread(ref,4,IC*NPT,f); fclose(f);
  f=O("sout.bin"); fread(&sout,4,1,f); fclose(f);
  scr_head16v2(wts, input, out, mult, bias, shifts, 3, NPT);   // NP=128 subset (1 internal tile)
  double dot=0,na=0,nb=0,maxd=0;
  for(int i=0;i<IC*NPT;i++){ float s=(float)((int)out[i])*sout; dot+=s*ref[i]; na+=(double)s*s; nb+=(double)ref[i]*ref[i]; double d=fabs(s-ref[i]); if(d>maxd)maxd=d; }
  printf("v2 HLS csim cosine vs FP32 = %.5f, maxd=%.4f\n", dot/(sqrt(na)*sqrt(nb)+1e-9), maxd);
  return 0;
}
