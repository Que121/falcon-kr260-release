#!/usr/bin/env python
"""Build-B step 2 (v2): INT8 image path with Vitis-AI fast_finetune (AdaQuant-style advanced PTQ).

Plain PTQ gave depth-argmax match 67% / exp-depth MAE 4.9 bins -- the depth_net logits are too
sensitive to per-layer rounding. fast_finetune optimizes weights/biases against the calib set to
cut that error before export. Mount ~/03_OccFPGA_Work as /work; reuses calib.npy + image_full_sd.pth.
"""
import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision
from pytorch_nndct.apis import torch_quantizer

WORK = "/work/occfpga_image"; QREF = "/work/occfpga_quant"

class ImageFull(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50(weights=None) if hasattr(torchvision.models, "ResNet50_Weights") else torchvision.models.resnet50(pretrained=False)
        self.conv1, self.bn1, self.relu, self.maxpool = r.conv1, r.bn1, r.relu, r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.lat0 = nn.Conv2d(1024, 256, 1); self.lat1 = nn.Conv2d(2048, 256, 1)
        self.fpn0 = nn.Conv2d(256, 256, 3, padding=1); self.depth_net = nn.Conv2d(256, 152, 1)
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x); x = self.layer2(x)
        c3 = self.layer3(x); c4 = self.layer4(c3)
        l0 = self.lat0(c3); l1 = self.lat1(c4)
        l0 = l0 + F.interpolate(l1, size=l0.shape[-2:], mode="nearest")
        return self.depth_net(self.fpn0(l0))

model = ImageFull()
model.load_state_dict(torch.load(os.path.join(WORK, "image_full_sd.pth"), map_location="cpu"))
model.eval()
calib = np.load(os.path.join(QREF, "calib.npy")).astype(np.float32)
test_in = torch.from_numpy(np.load(os.path.join(QREF, "test_input.npy")))
dummy = torch.randn(1, 3, 256, 704)
qdir = os.path.join(WORK, "quantize_result_ft")
import sys as _sys
NSAMP = int(_sys.argv[1]) if len(_sys.argv) > 1 else len(calib)   # reduce calib to bound CPU FT time
calib = calib[:NSAMP]
batches = [torch.from_numpy(calib[i:i + 4]) for i in range(0, len(calib), 4)]

def evaluate(qmodel, loader):
    qmodel.eval()
    with torch.no_grad():
        for b in loader: qmodel(b)
    return 0.0

# pass 1: calib + fast finetune
quantizer = torch_quantizer("calib", model, (dummy,), output_dir=qdir)
qm = quantizer.quant_model
print("fast_finetune start (%d batches)..." % len(batches), flush=True)
quantizer.fast_finetune(evaluate, (qm, batches))
print("fast_finetune done; calibrating stats...", flush=True)
with torch.no_grad():
    for b in batches: qm(b)
quantizer.export_quant_config()

# pass 2: test (reloads finetuned params) + export
quantizer = torch_quantizer("test", model, (dummy,), output_dir=qdir)
qm = quantizer.quant_model
quantizer.load_ft_param()
with torch.no_grad():
    out = qm(test_in).numpy()
np.save(os.path.join(WORK, "ref_int8ft_depthnet.npy"), out)
ref = np.load(os.path.join(WORK, "ref_fp32_depthnet.npy"))
def sm(x, ax=1):
    e = np.exp(x - x.max(ax, keepdims=True)); return e / e.sum(ax, keepdims=True)
def cos(a, b): return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
df, dq = sm(ref[:, :88]), sm(out[:, :88])
am = float((df.argmax(1) == dq.argmax(1)).mean())
mae = float(np.abs((df * np.arange(88)[None, :, None, None]).sum(1) - (dq * np.arange(88)[None, :, None, None]).sum(1)).mean())
print("FT INT8 vs FP32  logit-cos %.4f | softmax-cos %.4f | feat-cos %.4f | argmax %.3f | exp-MAE %.2f" % (
    cos(ref[:, :88], out[:, :88]), cos(df, dq), cos(ref[:, 88:], out[:, 88:]), am, mae), flush=True)
quantizer.export_xmodel(deploy_check=False, output_dir=qdir)
print("XMODEL_EXPORTED_FT")
for f in sorted(os.listdir(qdir)): print("  ", f)
