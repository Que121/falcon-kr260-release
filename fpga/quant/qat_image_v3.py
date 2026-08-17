#!/usr/bin/env python
"""QAT-distill IMAGE path FAITHFUL to the DPUCZDX8G deployment (the v1/v2 QAT plateaued at 0.949 because
it trained against an EASIER quant than it deploys to). Fixes:
  (1) PER-TENSOR weight fake-quant (DPU is per-tensor, NOT per-channel as v1 used).
  (2) FQ EVERY conv output (incl the no-ReLU FPN convs lat0/lat1/fpn0 + depth_net) — the DPU quantizes
      all of them; v1 only FQ'd the final output.
  (3) drop PACT (image is NOT heavy-tailed; check_featrange: feat absmax 16, p99 8.5) -> plain per-tensor
      pow2 act FQ after ReLU; (4) more epochs + cosine LR.
Distills depth-softmax(88) KL + feat(64) MSE from the FP teacher (= real FlashOcc, verify_reimpl cos 1.0).
  python qat_image_v3.py --data <dir> --sd image_full_sd.pth --nframes 2000 --epochs 24 --out <dir>
"""
import os, sys, argparse, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_full_model import ImageFull

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/scratch/ANON/buildB_io_full")
ap.add_argument("--sd", default="/scratch/ANON/image_full_sd.pth")
ap.add_argument("--nframes", type=int, default=2000)
ap.add_argument("--epochs", type=int, default=24)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--out", default="/scratch/ANON")
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, flush=True)

class FQAct(torch.autograd.Function):   # per-tensor pow2 INT8, STE
    @staticmethod
    def forward(ctx, x):
        mx = x.abs().max().clamp(min=1e-8); s = torch.pow(2.0, torch.floor(torch.log2(127.0 / mx)))
        return torch.clamp(torch.round(x * s), -128, 127) / s
    @staticmethod
    def backward(ctx, g): return g
def fq_w(w):   # PER-TENSOR pow2 weight FQ (matches DPUCZDX8G; v1 wrongly used per-channel)
    mx = w.detach().abs().max().clamp(min=1e-8)
    s = torch.pow(2.0, torch.floor(torch.log2(127.0 / mx)))
    return w + (torch.clamp(torch.round(w * s), -128, 127) / s - w).detach()
class QConv(nn.Module):
    def __init__(self, c, fq_out): super().__init__(); self.c = c; self.fq_out = fq_out
    def forward(self, x):
        y = F.conv2d(x, fq_w(self.c.weight), self.c.bias, self.c.stride, self.c.padding)
        return FQAct.apply(y) if self.fq_out else y     # FQ no-ReLU conv outputs; ReLU convs FQ'd by ReLUQ
class ReLUQ(nn.Module):
    def forward(self, x): return FQAct.apply(F.relu(x))
def wrap(m):
    for n, ch in list(m.named_children()):
        if isinstance(ch, nn.ReLU): setattr(m, n, ReLUQ())                       # relu -> relu + FQ
        elif isinstance(ch, nn.Conv2d): setattr(m, n, QConv(ch, fq_out=False))   # conv (FQ done by following ReLUQ)
        else: wrap(ch)

sd = torch.load(args.sd, map_location="cpu")
teacher = ImageFull(); teacher.load_state_dict(sd); teacher.eval().to(dev)
for p in teacher.parameters(): p.requires_grad = False
student = ImageFull(); student.load_state_dict(sd); wrap(student)
# the 4 no-ReLU convs (lat0/lat1/fpn0/depth_net) are NOT followed by ReLU -> FQ their outputs explicitly
for nm in ["lat0", "lat1", "fpn0"]:
    setattr(student, nm, QConv(getattr(student, nm).c, fq_out=True))
student.depth_net = QConv(student.depth_net.c, fq_out=False)   # depth_net out FQ'd by the final FQAct below
student.train().to(dev)

frames = sorted(glob.glob(os.path.join(args.data, "frame_*.npz")))[:args.nframes]
print("preloading %d img..." % len(frames), flush=True)
IMG = np.empty((len(frames), 6, 3, 256, 704), np.float16)
for i, f in enumerate(frames):
    IMG[i] = np.load(f)["img"].astype(np.float16)
    if (i + 1) % 400 == 0: print("  loaded %d" % (i + 1), flush=True)
N = len(frames)
opt = torch.optim.Adam(student.parameters(), lr=2e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * (N // args.bs + 1))
def split(y): return F.softmax(y[:, :88], dim=1), y[:, 88:]

for ep in range(args.epochs):
    perm = np.random.permutation(N); tot = 0.0; nb = 0
    for i in range(0, N, args.bs):
        idx = perm[i:i + args.bs]
        img = torch.from_numpy(IMG[idx].astype(np.float32)).reshape(-1, 3, 256, 704).to(dev)
        with torch.no_grad(): td, tf = split(teacher(img))
        sout = FQAct.apply(student(img))
        sd_, sf = split(sout)
        loss = F.kl_div((sd_ + 1e-8).log(), td, reduction="batchmean") + 0.5 * F.mse_loss(sf, tf)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        tot += float(loss); nb += 1
    # eval depth-softmax cos + argmax on the last batch
    with torch.no_grad():
        dcos = float((sd_.flatten() @ td.flatten()) / (sd_.norm() * td.norm() + 1e-9))
        am = float((sd_.argmax(1) == td.argmax(1)).float().mean())
    print("== ep %d loss %.4f | depth-softmax cos %.4f argmax %.3f" % (ep, tot / nb, dcos, am), flush=True)

clean = {}
for k, v in student.state_dict().items():
    clean[k.replace(".c.weight", ".weight").replace(".c.bias", ".bias")] = v.detach().cpu()
torch.save({"clean_sd": clean}, os.path.join(args.out, "image_full_qatv3_sd.pth"))
print("QATV3_DONE -> image_full_qatv3_sd.pth", flush=True)
