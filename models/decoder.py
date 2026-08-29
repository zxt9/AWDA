"""Segmentation and domain-classification decoder heads."""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, features, alpha):
        ctx.alpha = alpha
        return features.view_as(features)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.alpha * gradient, None


class Decoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(conv_in_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=upscale, mode="bilinear", align_corners=False),
            nn.Conv2d(32, num_classes, kernel_size=1, bias=False),
        )
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, features, reverse_alpha=None):
        if reverse_alpha is not None:
            features = GradientReversal.apply(features, reverse_alpha)
        return self.head(features)
