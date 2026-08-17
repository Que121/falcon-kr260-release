#!/usr/bin/env python
"""Standalone FlashOcc-R50 image path (backbone C3+C4 + CustomFPN + depth_net) -> (1,152,16,44).
Importable for QAT. Weight names match extract_image_full.py."""
import torch, torch.nn as nn, torch.nn.functional as F, torchvision

class ImageFull(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
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
