"""ResNet-50 Siamese encoder used by AWDA."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones.resnet_backbone import ResNet50Backbone


class PyramidPooling(nn.Module):
    def __init__(self, in_channels=2048, bins=(1, 2, 3, 6)):
        super().__init__()
        out_channels = in_channels // len(bins)
        self.stages = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(size), nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)) for size in bins
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + out_channels * len(bins), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, features):
        size = features.shape[-2:]
        pooled = [F.interpolate(stage(features), size=size, mode="bilinear", align_corners=False)
                  for stage in self.stages]
        return self.bottleneck(torch.cat([features, *pooled], dim=1))


class EncoderResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weight_path = "models/backbones/pretrained/3x3resnet50-imagenet.pth" if pretrained else None
        self.backbone = ResNet50Backbone(pretrained=weight_path)
        self.pooling = PyramidPooling()

    def forward(self, image_a, image_b):
        return self.pooling(torch.abs(self.backbone(image_a) - self.backbone(image_b)))

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_module_params(self):
        return self.pooling.parameters()
