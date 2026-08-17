#!/usr/bin/env python
"""Build-B step 2: assemble FlashOcc-R50's FULL image path as a plain-torch model for Vitis-AI.

image (1,3,256,704) -> ResNet50 backbone (C3 1024@/16, C4 2048@/32)
                    -> CustomFPN(in=[1024,2048],out=256,out_ids=[0])  [lateral 1x1 + nearest top-down + 3x3]
                    -> depth_net Conv2d(256->152)  [88 depth-logits + 64 context/feat, pre-softmax]
Output = (1,152,16,44).  Softmax over the first 88 + INT8 seam-quant happen on the ARM/board side
(matches Build-B step1 reconciliation: gather takes post-softmax depth Q0.7 + feat ap_int8).

Runs on Pro6000 (ANONPROJ_310). Reuses occfpga_quant/calib.npy (128,3,256,704). Emits to occfpga_image/:
  image_full_sd.pth  - state_dict in the plain-torch naming the quantizer reloads
  ref_fp32_depthnet.npy - FP32 (1,152,16,44) for INT8 cosine sanity
"""
import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision

OUT  = "/home/ANON/03_OccFPGA_Work/occfpga_image"
QDIR = "/home/ANON/03_OccFPGA_Work/occfpga_quant"
CKPT = "/home/ANON/01_Projects/FlashOCC/ckpts/flashocc-r50-256x704.pth"

class ImageFull(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.conv1, self.bn1, self.relu, self.maxpool = r.conv1, r.bn1, r.relu, r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.lat0 = nn.Conv2d(1024, 256, 1)          # img_neck.lateral_convs.0 (C3)
        self.lat1 = nn.Conv2d(2048, 256, 1)          # img_neck.lateral_convs.1 (C4)
        self.fpn0 = nn.Conv2d(256, 256, 3, padding=1)  # img_neck.fpn_convs.0
        self.depth_net = nn.Conv2d(256, 152, 1)      # img_view_transformer.depth_net (88+64)
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x); x = self.layer2(x)
        c3 = self.layer3(x)                          # 1024, 16x44
        c4 = self.layer4(c3)                         # 2048, 8x22
        l0 = self.lat0(c3)                           # 256, 16x44
        l1 = self.lat1(c4)                           # 256, 8x22
        l0 = l0 + F.interpolate(l1, size=l0.shape[-2:], mode="nearest")
        return self.depth_net(self.fpn0(l0))         # 152, 16x44

os.makedirs(OUT, exist_ok=True)
m = ImageFull()
# backbone: reuse the already-extracted torchvision trunk sd (conv1..layer4)
bb = torch.load(os.path.join(QDIR, "resnet50_flashocc_sd.pth"), map_location="cpu")
miss, unexp = m.load_state_dict(bb, strict=False)
miss = [k for k in miss if not k.startswith(("lat0", "lat1", "fpn0", "depth_net"))]
assert not miss, ("backbone missing: %s" % miss[:6])
# neck + depth_net: straight from the flashocc ckpt
sd = torch.load(CKPT, map_location="cpu"); sd = sd.get("state_dict", sd)
def cp(dst_w, dst_b, src):
    getattr(m, dst_w.split('.')[0]).weight.data.copy_(sd[src + ".weight"])
    getattr(m, dst_w.split('.')[0]).bias.data.copy_(sd[src + ".bias"])
cp("lat0.weight", "lat0.bias", "img_neck.lateral_convs.0.conv")
cp("lat1.weight", "lat1.bias", "img_neck.lateral_convs.1.conv")
cp("fpn0.weight", "fpn0.bias", "img_neck.fpn_convs.0.conv")
m.depth_net.weight.data.copy_(sd["img_view_transformer.depth_net.weight"])
m.depth_net.bias.data.copy_(sd["img_view_transformer.depth_net.bias"])
m.eval()
torch.save(m.state_dict(), os.path.join(OUT, "image_full_sd.pth"))

test_in = np.load(os.path.join(QDIR, "test_input.npy"))    # (1,3,256,704)
with torch.no_grad():
    ref = m(torch.from_numpy(test_in)).numpy()
np.save(os.path.join(OUT, "ref_fp32_depthnet.npy"), ref)
print("image_full_sd.pth + ref_fp32_depthnet.npy", ref.shape,
      "depth-logit|mean", float(np.abs(ref[:, :88]).mean()),
      "feat|mean", float(np.abs(ref[:, 88:]).mean()))
print("DONE")
