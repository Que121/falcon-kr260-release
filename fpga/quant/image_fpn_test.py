#!/usr/bin/env python
"""Decisive test: is the image depth limited by the depth_net head INT8, or the upstream FPN INT8?
Quantize backbone+FPN (output fpn0, 256ch) on the DPU; then run depth_net in FP32 on the INT8 fpn0
(off-DPU, FlashOcc-consistent). Compare depth-softmax vs FP32. If >> on-DPU-depthnet (0.949), the
off-DPU head is the win. Mount 03_OccFPGA_Work as /work."""
import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision
from pytorch_nndct.apis import torch_quantizer
W = "/work/occfpga_image"; Q = "/work/occfpga_quant"

class ImageFPN(nn.Module):   # backbone+FPN -> fpn0 (256,16,44)
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.conv1, self.bn1, self.relu, self.maxpool = r.conv1, r.bn1, r.relu, r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.lat0 = nn.Conv2d(1024, 256, 1); self.lat1 = nn.Conv2d(2048, 256, 1)
        self.fpn0 = nn.Conv2d(256, 256, 3, padding=1)
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x); x = self.layer2(x); c3 = self.layer3(x); c4 = self.layer4(c3)
        l0 = self.lat0(c3) + F.interpolate(self.lat1(c4), size=self.lat0(c3).shape[-2:], mode="nearest")
        return self.fpn0(l0)

full = torch.load(os.path.join(W, "image_full_sd.pth"), map_location="cpu")
m = ImageFPN(); m.load_state_dict({k: v for k, v in full.items() if not k.startswith("depth_net")}, strict=False); m.eval()
calib = np.load(os.path.join(Q, "calib.npy")).astype(np.float32)[:64]
test_in = torch.from_numpy(np.load(os.path.join(Q, "test_input.npy")))
qdir = os.path.join(W, "quantize_result_fpn")
q = torch_quantizer("calib", m, (torch.randn(1, 3, 256, 704),), output_dir=qdir)
qm = q.quant_model
with torch.no_grad():
    for i in range(0, 64, 8): qm(torch.from_numpy(calib[i:i + 8]))
q.export_quant_config()
q = torch_quantizer("test", m, (torch.randn(1, 3, 256, 704),), output_dir=qdir)
qm = q.quant_model
with torch.no_grad():
    fpn_int8 = qm(test_in)                  # INT8-sim fpn0
    fpn_fp32 = m(test_in)                    # FP32 fpn0
dn_w = full["depth_net.weight"][:88]; dn_b = full["depth_net.bias"][:88]  # FP32 depth_net (depth part)
def depthsoft(fpn): return F.softmax(F.conv2d(fpn, dn_w, dn_b), dim=1).numpy()
with torch.no_grad():
    d_from_int8 = depthsoft(fpn_int8); d_from_fp32 = depthsoft(fpn_fp32)
def cos(a, b): return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
print("FPN-INT8 cos vs FP32: %.4f" % cos(fpn_int8.numpy(), fpn_fp32.numpy()), flush=True)
print("OFF-DPU FP32-depthnet(INT8-fpn) depth-softmax cos vs FP32: %.4f  (on-DPU depthnet was 0.949)"
      % cos(d_from_int8, d_from_fp32), flush=True)
print("  argmax match %.3f" % float((d_from_int8.argmax(1) == d_from_fp32.argmax(1)).mean()), flush=True)
