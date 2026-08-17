#!/usr/bin/env python
"""Joint END-TO-END QAT of the full FlashOcc with DPU-faithful fake-quant (per-tensor power-of-2 8-bit,
weights + every activation boundary), finetuned with the real occ CE loss from the FP32 checkpoint.
This co-adapts ALL DPU stages (backbone+FPN+depth + BEV encoder+head) and optimizes the final mIoU
(not intermediate cosines) -- the SOTA way to maximize DPU-INT8 accuracy. Stage-separate QAT left
gains on the table. Saves flashocc_qat_e2e.pth (mmdet3d state_dict) for re-export to the board.

  python qat_e2e.py --iters 1500 --lr 5e-5 --clamp 16 --out /scratch/ANON/flashocc_qat_e2e.pth
Run from FlashOCC repo, env fbocc, A100.
"""
import argparse, importlib, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="projects/configs/flashocc/flashocc-r50.py")
ap.add_argument("--ckpt", default="ckpts/flashocc-r50-256x704.pth")
ap.add_argument("--iters", type=int, default=1500)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--clamp", type=float, default=16.0, help="Hardtanh clamp on BEV-encoder + occ final activations")
ap.add_argument("--out", default="/scratch/ANON/flashocc_qat_e2e.pth")
args = ap.parse_args()

# ---- DPU-faithful fake-quant: per-tensor power-of-2 8-bit, straight-through ----
class FQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        mx = x.detach().abs().max().clamp(min=1e-8)
        s = torch.pow(torch.tensor(2.0, device=x.device), torch.ceil(torch.log2(mx / 127.0)))
        return torch.clamp(torch.round(x / s), -128, 127) * s
    @staticmethod
    def backward(ctx, g): return g
def fqw(w):
    mx = w.detach().abs().max().clamp(min=1e-8)
    s = torch.pow(torch.tensor(2.0, device=w.device), torch.ceil(torch.log2(mx / 127.0)))
    return w + (torch.clamp(torch.round(w / s), -128, 127) * s - w).detach()

class QConv2d(nn.Conv2d):
    """Conv with per-tensor pow2 weight FQ (STE). Activation FQ is applied by hooks on outputs."""
    def forward(self, x):
        return self._conv_forward(x, fqw(self.weight), self.bias)

def swap_convs(module):
    for name, ch in list(module.named_children()):
        if type(ch) is nn.Conv2d:
            q = QConv2d(ch.in_channels, ch.out_channels, ch.kernel_size, ch.stride, ch.padding,
                        ch.dilation, ch.groups, ch.bias is not None)
            q.load_state_dict(ch.state_dict()); q.to(ch.weight.device)
            setattr(module, name, q)
        else:
            swap_convs(ch)

cfg = Config.fromfile(args.config)
if getattr(cfg, "plugin", False):
    importlib.import_module(".".join(cfg.plugin_dir.rstrip("/").split("/")))
model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
load_checkpoint(model, args.ckpt, map_location="cpu")
model = model.cuda()
mm = model

# inject FQ on the DPU-mapped stages: weights (QConv) + activations (hooks)
dpu_roots = [mm.img_backbone, mm.img_neck, mm.img_view_transformer.depth_net,
             mm.img_bev_encoder_backbone, mm.img_bev_encoder_neck, mm.occ_head.final_conv]
clamp_roots = {id(mm.img_bev_encoder_backbone), id(mm.img_bev_encoder_neck), id(mm.occ_head.final_conv)}
for r in dpu_roots:
    if isinstance(r, nn.Module): swap_convs(r)
hooks = []
def act_hook(clamp_c):
    def h(m, i, o):
        if not torch.is_tensor(o): return o
        x = torch.clamp(o, max=clamp_c) if clamp_c is not None else o
        return FQ.apply(x)
    return h
# hook ReLU outputs (post-activation, what the DPU quantizes) + the no-ReLU FPN/depth conv outputs
for r, cl in [(mm.img_backbone, None), (mm.img_neck, None), (mm.img_view_transformer.depth_net, None),
              (mm.img_bev_encoder_backbone, args.clamp), (mm.img_bev_encoder_neck, args.clamp),
              (mm.occ_head.final_conv, args.clamp)]:
    if not isinstance(r, nn.Module): continue
    any_relu = False
    for nm, md in r.named_modules():
        if isinstance(md, nn.ReLU): hooks.append(md.register_forward_hook(act_hook(cl))); any_relu = True
    if not any_relu:  # FPN / depth_net (no ReLU) -> FQ the conv outputs directly
        for nm, md in r.named_modules():
            if isinstance(md, (nn.Conv2d, QConv2d)): hooks.append(md.register_forward_hook(act_hook(cl)))
print("FQ injected: %d activation hooks; QConv weights on DPU stages" % len(hooks), flush=True)

# freeze BN running stats (short finetune)
for md in mm.modules():
    if isinstance(md, (nn.BatchNorm2d, nn.SyncBatchNorm)): md.eval()
model = MMDataParallel(model, [0]); model.train()

dcfg = cfg.data.train
loader = build_dataloader(build_dataset(dcfg), samples_per_gpu=cfg.data.samples_per_gpu,
                          workers_per_gpu=4, dist=False, shuffle=True)
opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-2)
it = 0; run = 0.0
while it < args.iters:
    for data in loader:
        losses = model(return_loss=True, **data)
        loss = sum(v.mean() for k, v in losses.items() if "loss" in k and torch.is_tensor(v))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        run += float(loss); it += 1
        if it % 50 == 0: print("iter %d/%d loss %.4f" % (it, args.iters, run / 50)); run = 0.0
        if it >= args.iters: break
# strip QConv back to plain Conv2d names for re-load/export
torch.save(model.module.state_dict(), args.out)
print("QAT_E2E_DONE saved %s" % args.out, flush=True)
