#!/usr/bin/env python
"""FlashOcc-R50 image path with a SPLIT, INT8-friendly depth head:
   backbone(C3+C4) -> CustomFPN -> [optional BN] -> depth_conv(256->88) + feat_conv(256->64).
The two convs get SEPARATE DPU output fix_points (depth logits no longer share a scale with feat),
which is the key INT8 fix for the 88-bin depth distribution. FP32-equivalent at init (sliced from the
original depth_net 256->152). Env USE_BN=1 adds a BatchNorm before the heads (normalizes depth input).
Returns (depth_logits (B,88,h,w), feat (B,64,h,w)).
"""
import os, torch, torch.nn as nn, torch.nn.functional as F, torchvision

class ImageSplit(nn.Module):
    def __init__(self, use_bn=False):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.conv1, self.bn1, self.relu, self.maxpool = r.conv1, r.bn1, r.relu, r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.lat0 = nn.Conv2d(1024, 256, 1); self.lat1 = nn.Conv2d(2048, 256, 1)
        self.fpn0 = nn.Conv2d(256, 256, 3, padding=1)
        self.use_bn = use_bn
        if use_bn:
            self.head_bn = nn.BatchNorm2d(256)
        self.depth_conv = nn.Conv2d(256, 88, 1)
        self.feat_conv = nn.Conv2d(256, 64, 1)
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x); x = self.layer2(x)
        c3 = self.layer3(x); c4 = self.layer4(c3)
        l0 = self.lat0(c3); l1 = self.lat1(c4)
        l0 = l0 + F.interpolate(l1, size=l0.shape[-2:], mode="nearest")
        h = self.fpn0(l0)
        if self.use_bn: h = self.head_bn(h)
        return self.depth_conv(h), self.feat_conv(h)

def init_from_full(m, full_sd):
    """Load backbone+FPN from the ImageFull sd; init split heads by slicing the original depth_net (256->152)."""
    bbfpn = {k: v for k, v in full_sd.items() if not k.startswith(("depth_net", "head_bn"))}
    m.load_state_dict(bbfpn, strict=False)
    dn_w = full_sd["depth_net.weight"]; dn_b = full_sd["depth_net.bias"]   # (152,256,1,1),(152,)
    m.depth_conv.weight.data.copy_(dn_w[:88]); m.depth_conv.bias.data.copy_(dn_b[:88])
    m.feat_conv.weight.data.copy_(dn_w[88:]); m.feat_conv.bias.data.copy_(dn_b[88:])
    return m
