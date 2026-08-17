#!/usr/bin/env python
"""QAT-distill the IMAGE path (backbone+FPN+depth_net) to match DPU INT8. Same recipe as the BEV QAT:
PACT learnable clamp + STE per-tensor-pow2-8bit act FQ + per-channel-pow2 weight FQ, distilled from the
FP32 teacher's depth distribution (softmax over 88 bins) + context feat (64ch). The depth softmax is the
seam the gather consumes, so matching it (not raw logits) is what counts.

Data: buildB_io_full frame_%04d.npz  ['img' (6,3,256,704) f16 normalized backbone input].
  python qat_image_distill.py --data <dir> --sd image_full_sd.pth --nframes 1500 --epochs 6 --out <dir>
"""
import os, sys, argparse, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_full_model import ImageFull

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/scratch/ANON/buildB_io_full")
ap.add_argument("--sd", default="/scratch/ANON/image_full_sd.pth")
ap.add_argument("--nframes", type=int, default=1500)
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--bs", type=int, default=6)   # 6 cams flattened per frame -> effective images = bs*6
ap.add_argument("--out", default="/scratch/ANON")
ap.add_argument("--clamp_init", type=float, default=8.0)
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, flush=True)

class FQAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        mx = x.abs().max().clamp(min=1e-8); s = torch.pow(2.0, torch.floor(torch.log2(127.0 / mx)))
        return torch.clamp(torch.round(x * s), -128, 127) / s
    @staticmethod
    def backward(ctx, g): return g
def fq_w(w):
    mx = w.detach().abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
    s = torch.pow(2.0, torch.floor(torch.log2(127.0 / mx)))
    return w + (torch.clamp(torch.round(w * s), -128, 127) / s - w).detach()
class PACT(nn.Module):
    def __init__(self, init): super().__init__(); self.a = nn.Parameter(torch.tensor(float(init)))
    def forward(self, x): return FQAct.apply(torch.clamp(F.relu(x), max=self.a.abs()))
class QConv(nn.Module):
    def __init__(self, c): super().__init__(); self.c = c
    def forward(self, x): return F.conv2d(x, fq_w(self.c.weight), self.c.bias, self.c.stride, self.c.padding)
def wrap(m, init):
    for n, ch in list(m.named_children()):
        if isinstance(ch, nn.ReLU): setattr(m, n, PACT(init))
        elif isinstance(ch, nn.Conv2d): setattr(m, n, QConv(ch))
        else: wrap(ch, init)

sd = torch.load(args.sd, map_location="cpu")
teacher = ImageFull(); teacher.load_state_dict(sd); teacher.eval().to(dev)
for p in teacher.parameters(): p.requires_grad = False
student = ImageFull(); student.load_state_dict(sd); wrap(student, args.clamp_init); student.train().to(dev)

frames = sorted(glob.glob(os.path.join(args.data, "frame_*.npz")))[:args.nframes]
print("preloading %d img into RAM..." % len(frames), flush=True)
IMG = np.empty((len(frames), 6, 3, 256, 704), np.float16)
for i, f in enumerate(frames):
    IMG[i] = np.load(f)["img"].astype(np.float16)
    if (i + 1) % 300 == 0: print("  loaded %d" % (i + 1), flush=True)
N = len(frames)
opt = torch.optim.Adam(student.parameters(), lr=2e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * (N // args.bs + 1))

def split(y):  # (B,152,16,44) -> depth softmax (B,88,...) , feat (B,64,...)
    return F.softmax(y[:, :88], dim=1), y[:, 88:]

for ep in range(args.epochs):
    perm = np.random.permutation(N); tot = 0.0; nb = 0
    for i in range(0, N, args.bs):
        idx = perm[i:i + args.bs]
        img = torch.from_numpy(IMG[idx].astype(np.float32)).reshape(-1, 3, 256, 704).to(dev)  # (bs*6,3,256,704)
        with torch.no_grad():
            td, tf = split(teacher(img))
        sout = FQAct.apply(student(img))   # KEY: fake-quant the depth_net OUTPUT (the deployed bottleneck)
        sd_, sf = split(sout)
        loss = F.kl_div((sd_ + 1e-8).log(), td, reduction="batchmean") + 0.5 * F.mse_loss(sf, tf)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        tot += float(loss); nb += 1
        if nb % 30 == 0: print("ep %d %d/%d loss %.4f" % (ep, i, N, tot / nb), flush=True)
    print("== epoch %d mean loss %.4f alphas %s" % (ep, tot / nb,
          [round(float(p.a.abs()), 1) for p in student.modules() if isinstance(p, PACT)][:8]), flush=True)

clean = {}
for k, v in student.state_dict().items():
    if k.endswith(".a"): continue
    clean[k.replace(".c.weight", ".weight").replace(".c.bias", ".bias")] = v.detach().cpu()
alphas = [float(p.a.abs()) for p in student.modules() if isinstance(p, PACT)]
torch.save({"clean_sd": clean, "alphas": alphas}, os.path.join(args.out, "image_full_qatd_sd.pth"))
print("QAT_IMG_DONE alphas(first8)=%s" % alphas[:8], flush=True)
