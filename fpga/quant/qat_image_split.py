#!/usr/bin/env python
"""QAT-distill the SPLIT image head (depth_conv + feat_conv) -> separate INT8 fix_points + per-output
fake-quant. Distill from the original FP32 ImageFull teacher (depth-softmax KL + feat MSE). Env USE_BN=1.
  python qat_image_split.py --data <dir> --sd image_full_sd.pth --nframes 3000 --epochs 14 --out <dir>
"""
import os, sys, argparse, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_full_model import ImageFull
from image_full_split import ImageSplit, init_from_full

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/scratch/ANON/buildB_io_full")
ap.add_argument("--sd", default="/scratch/ANON/image_full_sd.pth")
ap.add_argument("--nframes", type=int, default=3000); ap.add_argument("--epochs", type=int, default=14)
ap.add_argument("--bs", type=int, default=4); ap.add_argument("--out", default="/scratch/ANON")
ap.add_argument("--tag", default="split")
args = ap.parse_args()
USE_BN = os.environ.get("USE_BN", "0") == "1"
dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, "USE_BN", USE_BN, flush=True)

class FQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        mx = x.abs().max().clamp(min=1e-8); s = torch.pow(2.0, torch.floor(torch.log2(127.0 / mx)))
        return torch.clamp(torch.round(x * s), -128, 127) / s
    @staticmethod
    def backward(ctx, g): return g
def fqw(w):
    mx = w.detach().abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8); s = torch.pow(2.0, torch.floor(torch.log2(127.0 / mx)))
    return w + (torch.clamp(torch.round(w * s), -128, 127) / s - w).detach()
class PACT(nn.Module):
    def __init__(self, a=8.0): super().__init__(); self.a = nn.Parameter(torch.tensor(float(a)))
    def forward(self, x): return FQ.apply(torch.clamp(F.relu(x), max=self.a.abs()))
class QConv(nn.Module):
    def __init__(self, c): super().__init__(); self.c = c
    def forward(self, x): return F.conv2d(x, fqw(self.c.weight), self.c.bias, self.c.stride, self.c.padding)
def wrap(m):
    for n, ch in list(m.named_children()):
        if isinstance(ch, nn.ReLU): setattr(m, n, PACT())
        elif isinstance(ch, nn.Conv2d): setattr(m, n, QConv(ch))
        else: wrap(ch)

full_sd = torch.load(args.sd, map_location="cpu")
teacher = ImageFull(); teacher.load_state_dict(full_sd); teacher.eval().to(dev)
for p in teacher.parameters(): p.requires_grad = False
student = ImageSplit(use_bn=USE_BN); init_from_full(student, full_sd)
wrap(student); student.train().to(dev)
# KEY: fake-quant the FPN conv OUTPUTS too (no ReLU -> PACT never touched them; they're the FPN-INT8
# 0.862 limiter). Hooks apply FQ (STE) in the forward so the QAT trains weights robust to FPN INT8.
if os.environ.get("FPN_FQ", "1") == "1":
    def _fqhook(mod, inp, out): return FQ.apply(out)
    for _nm in ("lat0", "lat1", "fpn0"):
        getattr(student, _nm).register_forward_hook(_fqhook)
    print("FPN output fake-quant hooks: lat0,lat1,fpn0", flush=True)

frames = sorted(glob.glob(os.path.join(args.data, "frame_*.npz")))[:args.nframes]
print("preloading %d img..." % len(frames), flush=True)
IMG = np.empty((len(frames), 6, 3, 256, 704), np.float16)
for i, f in enumerate(frames):
    IMG[i] = np.load(f)["img"].astype(np.float16)
    if (i + 1) % 500 == 0: print("  %d" % (i + 1), flush=True)
N = len(frames)
opt = torch.optim.Adam(student.parameters(), lr=2e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * (N // args.bs + 1))

for ep in range(args.epochs):
    perm = np.random.permutation(N); tot = 0.0; nb = 0
    for i in range(0, N, args.bs):
        img = torch.from_numpy(IMG[perm[i:i+args.bs]].astype(np.float32)).reshape(-1, 3, 256, 704).to(dev)
        with torch.no_grad():
            ty = teacher(img); td = F.softmax(ty[:, :88], 1); tf = ty[:, 88:]
        sdl, sf = student(img)                       # depth logits (B,88), feat (B,64)
        sdl = FQ.apply(sdl); sf = FQ.apply(sf)       # per-output fake-quant (separate scales)
        sd_ = F.softmax(sdl, 1)
        loss = F.kl_div((sd_ + 1e-8).log(), td, reduction="batchmean") + 0.5 * F.mse_loss(sf, tf)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        tot += float(loss); nb += 1
        if nb % 40 == 0: print("ep %d %d/%d loss %.4f" % (ep, i, N, tot/nb), flush=True)
    print("== ep %d loss %.4f" % (ep, tot/nb), flush=True)

clean = {}
for k, v in student.state_dict().items():
    if k.endswith(".a"): continue
    clean[k.replace(".c.weight", ".weight").replace(".c.bias", ".bias")] = v.detach().cpu()
alphas = [float(p.a.abs()) for p in student.modules() if isinstance(p, PACT)]
torch.save({"clean_sd": clean, "alphas": alphas, "use_bn": USE_BN}, os.path.join(args.out, "image_split_%s_sd.pth" % args.tag))
print("QAT_SPLIT_DONE tag=%s use_bn=%s" % (args.tag, USE_BN), flush=True)
