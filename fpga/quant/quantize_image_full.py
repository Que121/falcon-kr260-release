#!/usr/bin/env python
"""Build-B step 2: quantize FlashOcc-R50 FULL image path to INT8 (inside Vitis-AI container).

Mount ~/03_OccFPGA_Work as /work. Reuses /work/occfpga_quant/calib.npy (128,3,256,704) and
test_input.npy; reloads /work/occfpga_image/image_full_sd.pth. Emits quantize_result/ (xmodel
for vai_c_xir) + ref_int8_depthnet.npy + the INT8-sim vs FP32 cosine.
"""
import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision
from pytorch_nndct.apis import torch_quantizer

WORK = "/work/occfpga_image"
QREF = "/work/occfpga_quant"

class ImageFull(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50(weights=None) if hasattr(torchvision.models, "ResNet50_Weights") else torchvision.models.resnet50(pretrained=False)
        self.conv1, self.bn1, self.relu, self.maxpool = r.conv1, r.bn1, r.relu, r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.lat0 = nn.Conv2d(1024, 256, 1)
        self.lat1 = nn.Conv2d(2048, 256, 1)
        self.fpn0 = nn.Conv2d(256, 256, 3, padding=1)
        self.depth_net = nn.Conv2d(256, 152, 1)
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

calib = np.load(os.path.join(QREF, "calib.npy"))
test_in = torch.from_numpy(np.load(os.path.join(QREF, "test_input.npy")))
dummy = torch.randn(1, 3, 256, 704)
qdir = os.path.join(WORK, "quantize_result")

quantizer = torch_quantizer("calib", model, (dummy,), output_dir=qdir)
qm = quantizer.quant_model
with torch.no_grad():
    bs = 8
    for i in range(0, len(calib), bs):
        qm(torch.from_numpy(calib[i:i + bs]))
        print("calib %d/%d" % (min(i + bs, len(calib)), len(calib)), flush=True)
quantizer.export_quant_config()

quantizer = torch_quantizer("test", model, (dummy,), output_dir=qdir)
qm = quantizer.quant_model
with torch.no_grad():
    out = qm(test_in).numpy()
np.save(os.path.join(WORK, "ref_int8_depthnet.npy"), out)
ref = np.load(os.path.join(WORK, "ref_fp32_depthnet.npy"))
def cos(a, b): return float(np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
print("INT8-sim vs FP32  ALL cos %.5f | depth(88) %.5f | feat(64) %.5f" % (
    cos(ref, out), cos(ref[:, :88], out[:, :88]), cos(ref[:, 88:], out[:, 88:])), flush=True)
quantizer.export_xmodel(deploy_check=False, output_dir=qdir)
print("XMODEL_EXPORTED")
for f in sorted(os.listdir(qdir)):
    print("  ", f)
