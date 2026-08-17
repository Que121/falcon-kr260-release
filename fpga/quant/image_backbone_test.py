#!/usr/bin/env python
"""Match the #7 INT8 scheme: backbone on DPU (INT8), FPN+depth_net in FP32 (off-DPU, like #7 which never
quantized them). Quantize ImageBackbone (->C3,C4); run FP32 FPN+depth_net on the INT8 C3/C4; measure
depth-softmax cos vs FP32. If ~0.99 (vs full-INT8 0.949), this matches #7. Mount 03_OccFPGA_Work as /work."""
import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision
from pytorch_nndct.apis import torch_quantizer
W = "/work/occfpga_image"; Q = "/work/occfpga_quant"

class ImageBackbone(nn.Module):   # image -> C3 (1024@/16), C4 (2048@/32)
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.conv1, self.bn1, self.relu, self.maxpool = r.conv1, r.bn1, r.relu, r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x); x = self.layer2(x); c3 = self.layer3(x); c4 = self.layer4(c3)
        return c3, c4

full = torch.load(os.path.join(W, "image_full_sd.pth"), map_location="cpu")
m = ImageBackbone()
m.load_state_dict({k: v for k, v in full.items() if k.split(".")[0] in
                   ("conv1","bn1","layer1","layer2","layer3","layer4")}, strict=False); m.eval()
ti = torch.from_numpy(np.load(os.path.join(Q, "test_input.npy")))
ref = np.load(os.path.join(W, "ref_fp32_depthnet.npy"))   # FP32 depth_net out (1,152,16,44)
qdir = os.path.join(W, "quantize_result_bb")
q = torch_quantizer("calib", m, (torch.randn(1, 3, 256, 704),), output_dir=qdir)
qm = q.quant_model
calib = np.load(os.path.join(Q, "calib.npy")).astype(np.float32)[:64]
with torch.no_grad():
    for i in range(0, 64, 8): qm(torch.from_numpy(calib[i:i + 8]))
q.export_quant_config()
q = torch_quantizer("test", m, (torch.randn(1, 3, 256, 704),), output_dir=qdir)
qm = q.quant_model
with torch.no_grad():
    c3, c4 = qm(ti)                                    # INT8-sim C3/C4
# FP32 FPN + depth_net on the INT8 C3/C4 (the #7 scheme)
def fpn_depth(c3, c4):
    l0 = F.conv2d(c3, full["lat0.weight"], full["lat0.bias"])
    l1 = F.conv2d(c4, full["lat1.weight"], full["lat1.bias"])
    l0 = l0 + F.interpolate(l1, size=l0.shape[-2:], mode="nearest")
    h = F.conv2d(l0, full["fpn0.weight"], full["fpn0.bias"], padding=1)
    out = F.conv2d(h, full["depth_net.weight"], full["depth_net.bias"])
    return out
with torch.no_grad():
    out_int8bb = fpn_depth(c3, c4).numpy()
def sm(x): e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def cos(a, b): return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
df, dq = sm(ref[:, :88]), sm(out_int8bb[:, :88]); am = float((df.argmax(1) == dq.argmax(1)).mean())
print("#7-SCHEME (DPU backbone INT8 + FP32 FPN+depth): depth-softmax cos %.4f | feat cos %.4f | argmax %.3f"
      % (cos(df, dq), cos(ref[:, 88:], out_int8bb[:, 88:]), am), flush=True)
print("  (full-INT8 deployed was depth-softmax 0.949; #7 framework INT8 = 31.6 mIoU)", flush=True)
