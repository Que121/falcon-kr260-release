#ifndef RESIZE_HPP
#define RESIZE_HPP
#include <ap_int.h>
#include <ap_fixed.h>

// ============================================================================
// OccFPGA — HLS bilinear-resize IP (the BEV-neck upsample the DPU cannot do)
//
// FlashOcc's FPN_LSS neck has two nn.Upsample(bilinear, align_corners=True) ops that the Vitis-AI
// DPU does not support natively. For a FIXED pipeline the source coords + interpolation weights depend
// ONLY on (H_in,W_in,H_out,W_out) -> they are STATIC -> the op is a fixed-iteration 2-tap separable
// gather. Fixed output size -> constant cycle count -> certifiable, INPUT-INVARIANT WCET on PL.
//
// v2 (wide-word, row-fused): the C=512 INT8 channels are packed 64-per-512-bit AXI word, so the AXI
// reads/writes are wide+burst-friendly and the 64 lanes compute in parallel; the intermediate is a
// SINGLE output row (vrow[LANES][WIN]) instead of the full plane -> fits BRAM (the v1 full-plane
// tmp[64][200][100] overflowed BRAM 534%). The per-channel arithmetic is UNCHANGED from the
// bit-exact-verified scalar version (max_abs_err 1e-13 vs torch align_corners), so packing is a pure
// memory-layout change: values identical.
//   UP  : 25x25  -> 100x100  (x4)   C=512
//   UP2 : 100x100-> 200x200  (x2)   C=512   (parameters below; set per instance at synthesis)
// ============================================================================

#ifdef HLS_SMALL                       // tiny instance for fast C/RTL cosim (same structure)
#define RZ_C        64
#define RZ_HIN      4
#define RZ_WIN      4
#define RZ_HOUT     8
#define RZ_WOUT     8
#define RZ_LANES    64
#else
#define RZ_C        512
#define RZ_HIN      100      // UP2 example (UP uses 25)
#define RZ_WIN      100
#define RZ_HOUT     200
#define RZ_WOUT     200
#define RZ_LANES    64                 // INT8 channels packed per 512-bit AXI word
#endif
#define RZ_NTILE    (RZ_C / RZ_LANES)  // wide-words spanning the channels

typedef ap_int<8>            rz_t;     // INT8 BEV-encoder activation
typedef ap_ufixed<9, 1>      wgt_t;    // bilinear weight in [0,1], Q0.9
typedef ap_fixed<20, 10>     rzacc_t;  // interpolation accumulator (per lane, full precision)
typedef ap_uint<16>          rzidx_t;
typedef ap_uint<8*RZ_LANES>  wide_t;   // 512-bit: 64 packed INT8 channels (one tile)

void resize_bilinear(
    const wide_t  in   [RZ_NTILE][RZ_HIN][RZ_WIN],   // packed input (AXI, 512-bit words)
    const rzidx_t y0   [RZ_HOUT], const rzidx_t y1[RZ_HOUT], const wgt_t wy[RZ_HOUT], // static V-taps (BRAM)
    const rzidx_t x0   [RZ_WOUT], const rzidx_t x1[RZ_WOUT], const wgt_t wx[RZ_WOUT], // static H-taps (BRAM)
    wide_t        out  [RZ_NTILE][RZ_HOUT][RZ_WOUT]); // packed output (AXI)

#endif
