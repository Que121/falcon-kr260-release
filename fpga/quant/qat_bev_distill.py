#!/usr/bin/env python
"""QAT-distill the BEV stage to MATCH the DPU's INT8 (per-tensor power-of-2 8-bit activations).

Root cause (verified): heavy-tailed activations (max/mean up to 1000x) + DPU per-tensor-pow2-8bit
can't coexist -> conv cos 0.59. Fix: PACT learnable clamp + STE fake-quant in training, distilled
from the dumped FP32 occ logits, so weights adapt to the clamped+quantized regime. Then bake the
learned clamps as Hardtanh (Vitis respects it) and PTQ -> deploys to DPU at high accuracy.

Data: /scratch/ANON/flashocc_io_full/frame_%04d.npz  (vt_out (64,200,200) + occ_out logits).
Teacher = FP32 occ_out (dumped). Student = QAT-BEV(conv) -> trainable predicter -> occ logits.
Saves bev_stage_qatd_sd.pth (BEV weights + learned alphas) + predicter_qatd.npz.

  python qat_bev_distill.py --data <dir> --nframes 2000 --epochs 6 --out <dir>
"""
import os, sys, argparse, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bev_stage import BasicBlock, CustomResNet, FPN_LSS, BEVStage

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/scratch/ANON/flashocc_io_full")
ap.add_argument("--sd", default="/scratch/ANON/bev_stage_sd.pth")
ap.add_argument("--nframes", type=int, default=2000)
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--out", default="/scratch/ANON")
ap.add_argument("--clamp_init", type=float, default=12.0)
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device", dev, flush=True)

# ---- DPU-faithful fake quant: per-tensor power-of-2 8-bit, straight-through ----
class FQAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        qmax = 127.0
        mx = x.abs().max().clamp(min=1e-8)
        fp = torch.floor(torch.log2(qmax / mx))
        s = torch.pow(2.0, fp)
        return torch.clamp(torch.round(x * s), -128, 127) / s
    @staticmethod
    def backward(ctx, g): return g
def fq_w(w):                                   # per-output-channel pow2 8-bit (DPU weight scheme), STE
    qmax = 127.0
    mx = w.detach().abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
    fp = torch.floor(torch.log2(qmax / mx)); s = torch.pow(2.0, fp)
    wq = torch.clamp(torch.round(w * s), -128, 127) / s
    return w + (wq - w).detach()

class PACT(nn.Module):                          # ReLU + learnable clamp + act fake-quant
    def __init__(self, init): super().__init__(); self.a = nn.Parameter(torch.tensor(float(init)))
    def forward(self, x):
        x = torch.clamp(F.relu(x), max=self.a.abs())
        return FQAct.apply(x)

class QConv(nn.Module):                         # conv with fake-quant weights
    def __init__(self, c): super().__init__(); self.c = c
    def forward(self, x):
        return F.conv2d(x, fq_w(self.c.weight), self.c.bias, self.c.stride, self.c.padding)

def wrap(m, init):
    for n, ch in list(m.named_children()):
        if isinstance(ch, nn.ReLU): setattr(m, n, PACT(init))
        elif isinstance(ch, nn.Conv2d): setattr(m, n, QConv(ch))
        else: wrap(ch, init)

# ---- models ----
full_sd = torch.load(args.sd, map_location="cpu")
teacher = BEVStage(conv_only=True); teacher.load_state_dict({k: v for k, v in full_sd.items() if not k.startswith("predicter.")}, strict=False); teacher.eval().to(dev)
for p in teacher.parameters(): p.requires_grad = False
predT = BEVStage(conv_only=False); predT.load_state_dict(full_sd); predT = predT.predicter.eval().to(dev)
for p in predT.parameters(): p.requires_grad = False

student = BEVStage(conv_only=True); student.load_state_dict({k: v for k, v in full_sd.items() if not k.startswith("predicter.")}, strict=False)
wrap(student, args.clamp_init); student.train().to(dev)
predS = BEVStage(conv_only=False); predS.load_state_dict(full_sd); predS = predS.predicter.train().to(dev)

def occ_logits(conv, pred):                     # (B,256,200,200) -> (B,200,200,16,18)
    x = conv.permute(0, 3, 2, 1); x = pred(x)
    return x.view(x.shape[0], 200, 200, 16, 18)

frames = sorted(glob.glob(os.path.join(args.data, "frame_*.npz")))[:args.nframes]
print("preloading %d vt_out into RAM..." % len(frames), flush=True)
VT = np.empty((len(frames), 64, 200, 200), np.float16)
for i, f in enumerate(frames):
    VT[i] = np.load(f)["vt_out_0"][0].astype(np.float16)
    if (i + 1) % 500 == 0: print("  loaded %d" % (i + 1), flush=True)
N = len(frames)
opt = torch.optim.Adam(list(student.parameters()) + list(predS.parameters()), lr=2e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * (N // args.bs + 1))

for ep in range(args.epochs):
    perm = np.random.permutation(N); tot = 0.0; nb = 0
    for i in range(0, N, args.bs):
        idx = perm[i:i + args.bs]
        vt = torch.from_numpy(VT[idx].astype(np.float32)).to(dev)
        with torch.no_grad():
            tconv = teacher(vt); tlog = occ_logits(tconv, predT)
            tprob = F.softmax(tlog, dim=-1)
        sconv = student(vt); slog = occ_logits(sconv, predS)
        loss = F.kl_div(F.log_softmax(slog, dim=-1), tprob, reduction="batchmean") + 0.1 * F.mse_loss(sconv, tconv.clamp(max=20))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        tot += float(loss); nb += 1
        if nb % 25 == 0: print("ep %d %d/%d loss %.4f" % (ep, i, len(frames), tot / nb), flush=True)
    print("== epoch %d mean loss %.4f  alphas %s" % (ep, tot / nb,
          [round(float(p.a.abs()), 1) for p in student.modules() if isinstance(p, PACT)][:8]), flush=True)

# map QAT weights back to a CLEAN BEVStage state_dict (strip the QConv ".c" wrapper); alphas separate.
clean = {}
for k, v in student.state_dict().items():
    if k.endswith(".a"):                          # PACT alpha -> stored separately
        continue
    ck = k.replace(".c.weight", ".weight").replace(".c.bias", ".bias")
    clean[ck] = v.detach().cpu()
alphas = [float(p.a.abs()) for p in student.modules() if isinstance(p, PACT)]
torch.save({"clean_sd": clean, "alphas": alphas}, os.path.join(args.out, "bev_stage_qatd_sd.pth"))
np.savez(os.path.join(args.out, "predicter_qatd.npz"),
         **{k: v.detach().cpu().numpy() for k, v in predS.state_dict().items()})
print("QATD_DONE alphas(first8)=%s saved bev_stage_qatd_sd.pth + predicter_qatd.npz" % alphas[:8], flush=True)
