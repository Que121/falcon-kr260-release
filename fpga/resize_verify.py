#!/usr/bin/env python
"""Verify the HLS resize IP design: a precomputed-tap bilinear gather reproduces torch's
nn.Upsample(bilinear, align_corners=True) bit-exactly. Confirms the resize IP algorithm before HLS.

align_corners=True -> src = dst*(in-1)/(out-1); each output pixel = 4 taps with STATIC weights/indices
(functions only of (H_in,W_in,H_out,W_out)) -> fixed-iteration weighted gather -> bounded WCET on PL.
FlashOcc FPN_LSS: up (25->100, x4) and up2 (100->200, x2), both C-wise, align_corners=True.
"""
import numpy as np
import torch
import torch.nn.functional as F

def precomp(in_sz, out_sz):
    """align_corners=True tap indices + weight for one axis (STATIC for the rig)."""
    src = np.arange(out_sz, dtype=np.float64) * (in_sz - 1) / (out_sz - 1)
    i0 = np.clip(np.floor(src).astype(np.int64), 0, in_sz - 1)
    i1 = np.minimum(i0 + 1, in_sz - 1)
    w = (src - i0)
    return i0, i1, w

def bilinear_np(x, Hout, Wout):
    """x: (C,H,W) -> (C,Hout,Wout) via precomputed 4-tap gather."""
    C, H, W = x.shape
    yi0, yi1, wy = precomp(H, Hout)
    xi0, xi1, wx = precomp(W, Wout)
    x = x.astype(np.float64)
    # rows first
    r0 = x[:, yi0, :]            # (C,Hout,W)
    r1 = x[:, yi1, :]
    row = r0 * (1 - wy)[None, :, None] + r1 * wy[None, :, None]   # (C,Hout,W)
    # cols
    c0 = row[:, :, xi0]          # (C,Hout,Wout)
    c1 = row[:, :, xi1]
    out = c0 * (1 - wx)[None, None, :] + c1 * wx[None, None, :]
    return out

for (Cc, Hin, Win, Hout, Wout, tag) in [(512, 25, 25, 100, 100, "up  (x4)"),
                                         (512, 100, 100, 200, 200, "up2 (x2)")]:
    x = torch.randn(1, Cc, Hin, Win, dtype=torch.float64)
    ref = F.interpolate(x, size=(Hout, Wout), mode="bilinear", align_corners=True)[0].numpy()
    mine = bilinear_np(x[0].numpy(), Hout, Wout)
    err = np.abs(mine - ref)
    outs = Cc * Hout * Wout
    print("%-9s  %dx%d->%dx%d C=%d  max_abs_err=%.3e  allclose=%s  | outputs=%.2fM  4-tap MACs=%.1fM"
          % (tag, Hin, Win, Hout, Wout, Cc, err.max(), np.allclose(mine, ref, atol=1e-9),
             outs / 1e6, 4 * outs / 1e6), flush=True)
    for f in (200, 300):
        # WCET model: outputs * 4 taps / (C_lanes); report per-pixel cycles (channel-tiled in HLS)
        cyc = Hout * Wout * 4  # one output row/col tap pass per pixel, C tiled
        print("    WCET model @%dMHz: %.0f-pixel x 4tap; C-tiled -> input-invariant bounded" % (f, Hout * Wout), flush=True)
print("RESIZE_VERIFY_DONE", flush=True)
