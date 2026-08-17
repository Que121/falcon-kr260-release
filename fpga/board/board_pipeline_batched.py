#!/usr/bin/env python3
"""Build-B step 5/6: FULL image->occ pipeline over N frames, batched-by-stage (amortize overlay swaps).

Per-frame the pipeline needs ~7 PL reconfigs (image-DPU, gather, bev-DPU x3, resize x2) = ~21s of
swapping. Batching by STAGE loads each overlay ONCE per chunk and loops all chunk frames through it,
so 6019 frames cost ~7 loads/chunk instead of 7/frame. Every conv on the B4096 DPU, every upsample on
the HLS resize IP, view-transform on the HLS gather IP -- NO numpy/host compute except softmax (depth)
and the Linear-Softplus-Linear predicter, which are off-DPU by FlashOcc design.

Phases per chunk (intermediates streamed to /home/ubuntu/buildB/work/):
  A image-DPU : img(6,3,256,704) -> (16,44,152)x6 -> softmax(depth)+feat   -> fd_%05d.npz
  B gather-IP : fd + ranks -> vt_out(64,200,200) int8                       -> vt_%05d.npy
  C1 bev-DPU  : vt -> bev sg1 feat0(100)+feat2(25)                          -> b1_%05d.npz
  C2 resize   : feat2 25->100                                               -> (held) merged in C3 feed
  C3 bev-DPU  : sg2 -> neck 100                                             -> b3_%05d.npy
  C4 resize   : 100->200                                                    -> b4_%05d.npy
  C5 bev-DPU  : sg3 -> conv_only(256,200,200)                               -> cv_%05d.npy
  D predicter : conv -> argmax occ(200,200,16) (ARM)                        -> OUT[i]
To keep it robust on the 4GB/16GB board we stream every stage to disk and process resize inline within
the bev-DPU phases by re-loading the resize overlay (so the swap pattern per chunk is:
  image | gather | bevDPU(sg1) | resize2510 | bevDPU(sg2) | resize100 | bevDPU(sg3) ).

  python3 board_pipeline_batched.py <frames_dir> <Nframes> <out_occ.npy> [chunk=512] [IMG_XM] [BEV_XM]
"""
import os, sys, time, glob, numpy as np, xir
os.environ["XILINX_XRT"] = "/usr"
import pynq.pl_server.embedded_device as _ed
_DX = os.path.join(os.path.dirname(_ed.__file__), "default.xclbin")
if os.path.exists(_DX):
    _B = open(_DX, "rb").read(); _ed._create_xclbin = lambda _m: _B

FR_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/buildB/io16"
N      = int(sys.argv[2]) if len(sys.argv) > 2 else 16
OUT    = sys.argv[3] if len(sys.argv) > 3 else "/home/ubuntu/buildB/board_argmax.npy"
CHUNK  = int(sys.argv[4]) if len(sys.argv) > 4 else 512
IMG_XM = sys.argv[5] if len(sys.argv) > 5 else "/home/ubuntu/flashocc/flashocc_r50_image.xmodel"
BEV_XM = sys.argv[6] if len(sys.argv) > 6 else "/home/ubuntu/bev_allpl/bev_cfg.xmodel"
GBIT   = "/home/ubuntu/gather_ovl_i16/gather_ovl_i16.bit"   # INT16-vt gather overlay
RZ25   = "/home/ubuntu/rovl_2510/rovl_2510.bit"
RZ100  = "/home/ubuntu/resize_ovl/resize_ovl.bit"
HEAD   = "/home/ubuntu/bev/predicter_head.npz"
WORK   = "/home/ubuntu/buildB/work"
WB = 64
GREG = {"feat":0x10, "depth":0x1c, "rank_depth":0x28, "rank_feat":0x34, "rank_bev":0x40,
        "interval_start":0x4c, "interval_len":0x58, "bev":0x6c}
GOUT_SCALE = 0x64
RZREG = {"in":0x10, "y0":0x1c, "y1":0x28, "wy":0x34, "x0":0x40, "x1":0x4c, "wx":0x58, "out":0x64}
AP = 0x00

def softmax(x, ax): e = np.exp(x - x.max(ax, keepdims=True)); return e / e.sum(ax, keepdims=True)
def t_fp(t): return t.get_attr("fix_point") if t.has_attr("fix_point") else None
def dpu_subs(xm):
    g = xir.Graph.deserialize(xm); return g, list(g.get_root_subgraph().toposort_child_subgraph())
def bilinear_taps(out_n, in_n):
    i0 = np.zeros(out_n, np.uint16); i1 = np.zeros(out_n, np.uint16); w = np.zeros(out_n, np.uint16)
    for o in range(out_n):
        s = o * (in_n - 1) / (out_n - 1) if out_n > 1 else 0.0
        f = int(np.floor(s)); i0[o] = f; i1[o] = min(f + 1, in_n - 1); w[o] = int(round((s - f) * 256)) & 0x1ff
    return i0, i1, w

def main():
    frames = sorted(glob.glob(os.path.join(FR_DIR, "frame_*.npz")))[:N]
    M = len(frames); os.makedirs(WORK, exist_ok=True)
    print("frames %d | chunk %d | IMG %s | BEV %s" % (M, CHUNK, os.path.basename(IMG_XM), os.path.basename(BEV_XM)), flush=True)
    h = np.load(HEAD); W0, b0, W2, b2 = h["0.weight"], h["0.bias"], h["2.weight"], h["2.bias"]
    out = np.empty((M, 200, 200, 16), np.uint8)
    t_all = time.time()

    for cs in range(0, M, CHUNK):
        ce = min(cs + CHUNK, M); idx = list(range(cs, ce))
        print("== chunk %d-%d ==" % (cs, ce - 1), flush=True); tC = time.time()

        # ---- A: image DPU ----
        import vart
        from pynq_dpu import DpuOverlay
        ov = DpuOverlay("dpu.bit")
        g, subs = dpu_subs(IMG_XM)
        dsub = [s for s in subs if s.has_attr("device") and s.get_attr("device") == "DPU"][0]
        r = vart.Runner.create_runner(dsub, "run")
        it = r.get_input_tensors()[0]; ots = r.get_output_tensors()
        idims = [int(x) for x in it.dims]; fpi = it.get_attr("fix_point")
        odims = [[int(x) for x in o.dims] for o in ots]; fpos = [o.get_attr("fix_point") for o in ots]
        # locate depth(88) + feat(64) heads: 1 output of 152ch (legacy) or 2 outputs (split)
        split2 = len(ots) == 2
        if split2:
            di = 0 if odims[0][-1] == 88 else 1; fi = 1 - di           # by channel count
        for i in idx:
            img = np.load(frames[i])["img"].astype(np.float32)        # (6,3,256,704)
            feat_o = np.empty((6, 16, 44, 64), np.float16); depth_o = np.empty((6, 88, 16, 44), np.float16)
            for c in range(6):
                xi = np.clip(np.round(img[c].transpose(1, 2, 0) * (2.0**fpi)), -128, 127).astype(np.int8)[None]
                outs = [np.empty(od, np.int8) for od in odims]
                jid = r.execute_async([np.ascontiguousarray(xi.reshape(idims), np.int8)], outs); r.wait(jid)
                if split2:
                    dlog = outs[di].reshape(16, 44, 88).astype(np.float32) * (2.0**-fpos[di])
                    feat_o[c] = outs[fi].reshape(16, 44, 64).astype(np.float32) * (2.0**-fpos[fi])
                    depth_o[c] = softmax(dlog, -1).transpose(2, 0, 1)
                else:
                    of = outs[0].reshape(16, 44, 152).astype(np.float32) * (2.0**-fpos[0])
                    depth_o[c] = softmax(of[..., :88], -1).transpose(2, 0, 1); feat_o[c] = of[..., 88:]
            np.savez(os.path.join(WORK, "fd_%05d.npz" % i), feat=feat_o, depth=depth_o)
        del r, ov
        print("  A image %.0fs" % (time.time() - tC), flush=True)

        # ---- B: gather IP ----
        from pynq import Overlay, allocate
        tB = time.time(); ol = Overlay(GBIT)
        gip = getattr(ol, [k for k in ol.ip_dict if "gather" in k.lower()][0].split('/')[-1])
        for i in idx:
            fr = np.load(frames[i]); fd = np.load(os.path.join(WORK, "fd_%05d.npz" % i))
            feat = fd["feat"].astype(np.float32).reshape(-1, 64); depth = fd["depth"].astype(np.float32).reshape(-1)
            rdep = fr["ranks_depth"].astype(np.uint32); rfea = fr["ranks_feat"].astype(np.uint32)
            ist = fr["interval_starts"].astype(np.uint32); iln = fr["interval_lengths"].astype(np.uint32)
            rbev = fr["ranks_bev"].astype(np.int64)[fr["interval_starts"].astype(np.int64)].astype(np.uint32)
            FV = feat.shape[0]; DL = depth.shape[0]; NP = rdep.shape[0]; NPI = ist.shape[0]
            bufs = {}
            def mk(shape, dt, src=None):
                b = allocate(shape=shape, dtype=dt)
                if src is not None: b[:] = src
                b.flush(); return b
            bufs["feat"] = mk((FV*WB,), np.uint8, np.clip(np.round(feat*4.0), -128, 127).astype(np.int8).ravel().view(np.uint8))
            bufs["depth"] = mk((DL,), np.uint8, np.clip(np.round(depth*128.0), 0, 255).astype(np.uint8))  # Q0.7
            bufs["rank_depth"] = mk((NP,), np.uint32, rdep); bufs["rank_feat"] = mk((NP,), np.uint32, rfea)
            bufs["rank_bev"] = mk((NPI,), np.uint32, rbev)
            bufs["interval_start"] = mk((NPI,), np.uint32, ist); bufs["interval_len"] = mk((NPI,), np.uint32, iln)
            bufs["bev"] = mk((40000*WB*2,), np.uint8)          # INT16 vt: 128 bytes/cell (was *WB INT8)
            for nm, reg in GREG.items():
                a = bufs[nm].device_address; gip.write(reg, a & 0xffffffff); gip.write(reg+4, (a >> 32) & 0xffffffff)
            gip.write(GOUT_SCALE, 2097152)                    # INT16 IP: out_scale=32 (fp_feat=2,fp_vt=7), acc_t ap_fixed<32,16> -> reg=32*2^16; vt reaches 250*128<32767 no-clip, res 1/128
            gip.write(AP, 1)
            while (gip.read(AP) & 0x2) == 0: pass
            bufs["bev"].invalidate()
            vt = (np.array(bufs["bev"]).view(np.int16).reshape(40000, WB).astype(np.float32) * (2.0**-7)).reshape(200, 200, 64)  # fp_vt=7
            np.save(os.path.join(WORK, "vt_%05d.npy" % i), vt.astype(np.float16))
            for b in bufs.values(): b.freebuffer()
        del ol
        print("  B gather %.0fs" % (time.time() - tB), flush=True)

        # ---- C: BEV all-PL (3 DPU + 2 resize), stage-batched within chunk ----
        g, bsubs = dpu_subs(BEV_XM)
        dev = lambda s: s.get_attr("device") if s.has_attr("device") else "?"
        in_t = list(bsubs[0].get_output_tensors())[0]; fp_bin = t_fp(in_t)
        # store: per-frame dict tensor_name->(int8 NHWC, fp); seed with vt input
        order = bsubs[1:]
        # we walk the subgraph list; DPU subs run under dpu.bit, CPU(upsample) under a resize overlay.
        # To batch swaps: iterate subgraphs; for each, load its overlay once, loop frames.
        st = [dict() for _ in idx]
        for k, i in enumerate(idx):
            vt = np.load(os.path.join(WORK, "vt_%05d.npy" % i)).astype(np.float32)[None]  # (1,200,200,64) value==fp0
            q = np.clip(np.round(vt * (2.0**fp_bin)), -128, 127).astype(np.int8)
            st[k][in_t.name] = (q, fp_bin)
        # liveness: tensor name -> last subgraph index (in `order`) that consumes it; free after.
        live_until = {in_t.name: -1}
        for si2, sg2 in enumerate(order):
            for t in sg2.get_input_tensors(): live_until[t.name] = si2
        tBEV = time.time()
        cur_overlay = None; dpu_ov = None
        for si, sg in enumerate(order):
            d = dev(sg)
            if d == "DPU":
                if cur_overlay != "dpu":
                    if dpu_ov is not None: del dpu_ov
                    dpu_ov = DpuOverlay("dpu.bit"); cur_overlay = "dpu"
                runner = vart.Runner.create_runner(sg, "run")
                its = runner.get_input_tensors(); ots = runner.get_output_tensors()
                for k in range(len(idx)):
                    ins = [np.ascontiguousarray(st[k][t.name][0].reshape([int(x) for x in t.dims]), np.int8) for t in its]
                    outs = [np.empty([int(x) for x in t.dims], np.int8) for t in ots]
                    jid = runner.execute_async(ins, outs); runner.wait(jid)
                    for j, t in enumerate(ots): st[k][t.name] = (outs[j].copy(), t.get_attr("fix_point"))
                del runner
            elif d == "CPU":
                itt = list(sg.get_input_tensors())[0]; ott = list(sg.get_output_tensors())[0]
                Hin = int(itt.dims[1]); Hout = int(ott.dims[1])
                if Hout > Hin:                                   # upsample on resize IP
                    bit = RZ25 if Hout == 100 else RZ100
                    if cur_overlay != bit:
                        ol = Overlay(bit); cur_overlay = bit
                        rip = None
                        for nm in ("resize_0", "resize_bilinear_0"):
                            rip = getattr(ol, nm, None)
                            if rip is not None: break
                        if rip is None: rip = getattr(ol, [k2 for k2 in ol.ip_dict if "resize" in k2.lower()][0].split('/')[-1])
                    Wout = int(ott.dims[2])
                    fpo2 = t_fp(ott) if t_fp(ott) is not None else None
                    s0 = st[0][itt.name][0]; _, Hi, Wi, C = s0.shape; NT = C // WB
                    # allocate IP buffers + taps ONCE per stage (constant across frames) -> ~8x speedup
                    ib = allocate(shape=(NT*Hi*Wi*WB,), dtype=np.uint8); ob = allocate(shape=(NT*Hout*Wout*WB,), dtype=np.uint8)
                    y0, y1, wy = bilinear_taps(Hout, Hi); x0, x1, wx = bilinear_taps(Wout, Wi); tb = {}
                    for nm, arr in (("y0",y0),("y1",y1),("wy",wy),("x0",x0),("x1",x1),("wx",wx)):
                        bb = allocate(shape=arr.shape, dtype=np.uint16); bb[:] = arr; bb.flush(); tb[nm] = bb
                    def sp(reg, bu): a = bu.device_address; rip.write(reg, a & 0xffffffff); rip.write(reg+4, (a >> 32) & 0xffffffff)
                    sp(RZREG["in"], ib); sp(RZREG["out"], ob)
                    for nm in ("y0","y1","wy","x0","x1","wx"): sp(RZREG[nm], tb[nm])
                    for k in range(len(idx)):
                        x, fpx = st[k][itt.name]
                        ib[:] = x[0].reshape(Hi, Wi, NT, WB).transpose(2, 0, 1, 3).astype(np.int8).view(np.uint8).ravel(); ib.flush()
                        rip.write(AP, 1)
                        while (rip.read(AP) & 0x2) == 0: pass
                        ob.invalidate()
                        o = np.array(ob).view(np.int8).reshape(NT, Hout, Wout, WB).transpose(1,2,0,3).reshape(1, Hout, Wout, C)
                        fpoo = fpo2 if fpo2 is not None else fpx
                        if fpoo != fpx:
                            o = np.clip(np.round(o.astype(np.float32) * (2.0**(fpoo-fpx))), -128, 127).astype(np.int8)
                        st[k][ott.name] = (np.ascontiguousarray(o), fpoo)
                    ib.freebuffer(); ob.freebuffer()
                    for bb in tb.values(): bb.freebuffer()
                else:                                            # trivial cast
                    for k in range(len(idx)): st[k][ott.name] = st[k][itt.name]
            # free tensors no longer needed by any later subgraph (bound RAM ~ chunk x live-set)
            for nm, lu in live_until.items():
                if lu == si:
                    for k in range(len(idx)): st[k].pop(nm, None)
        if dpu_ov is not None: del dpu_ov
        print("  C bev %.0fs" % (time.time() - tBEV), flush=True)

        # ---- D: predicter + argmax ----
        out_t = list(order[-1].get_output_tensors())[0]
        for k, i in enumerate(idx):
            arr, fp = st[k][out_t.name]
            conv = (arr[0].astype(np.float32) * (2.0**(-fp))).transpose(2, 0, 1)
            x = conv.transpose(2, 1, 0).reshape(-1, 256) @ W0.T + b0
            x = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
            out[i] = (x @ W2.T + b2).reshape(200, 200, 16, 18).argmax(-1).astype(np.uint8)
        # cleanup chunk intermediates
        for i in idx:
            for p in ("fd_%05d.npz", "vt_%05d.npy"):
                f = os.path.join(WORK, p % i)
                if os.path.exists(f): os.remove(f)
        print("  chunk done %.0fs (total %.0fs, %d/%d)" % (time.time()-tC, time.time()-t_all, ce, M), flush=True)

    np.save(OUT, out)
    print("DONE %d frames -> %s %s in %.0fs" % (M, OUT, out.shape, time.time()-t_all), flush=True)

if __name__ == "__main__":
    main()
