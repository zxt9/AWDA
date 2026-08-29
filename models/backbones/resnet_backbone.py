"""Deep-stem dilated ResNet-50 backbone used by AWDA."""

import math
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, inputs):
        residual = inputs
        output = self.relu(self.bn1(self.conv1(inputs)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        if self.downsample is not None:
            residual = self.downsample(inputs)
        return self.relu(output + residual)


class ResNet50Backbone(nn.Module):
    """ResNet-50 with a three-convolution stem and output stride 8."""

    def __init__(self, pretrained=None):
        super().__init__()
        self.in_channels = 128
        self.prefix = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)),
            ("bn1", nn.BatchNorm2d(64)),
            ("relu1", nn.ReLU(inplace=False)),
            ("conv2", nn.Conv2d(64, 64, 3, padding=1, bias=False)),
            ("bn2", nn.BatchNorm2d(64)),
            ("relu2", nn.ReLU(inplace=False)),
            ("conv3", nn.Conv2d(64, 128, 3, padding=1, bias=False)),
            ("bn3", nn.BatchNorm2d(128)),
            ("relu3", nn.ReLU(inplace=False)),
        ]))
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks=3)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=2)
        self._initialize()
        self._set_output_stride_8()
        if pretrained is not None:
            self._load_pretrained(pretrained)

    def _make_layer(self, channels, blocks, stride=1):
        out_channels = channels * Bottleneck.expansion
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [Bottleneck(self.in_channels, channels, stride, downsample)]
        self.in_channels = out_channels
        layers.extend(Bottleneck(self.in_channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    @staticmethod
    def _dilate(module, dilation):
        if not isinstance(module, nn.Conv2d):
            return
        if module.stride == (2, 2):
            module.stride = (1, 1)
            if module.kernel_size == (3, 3):
                module.dilation = module.padding = (dilation // 2, dilation // 2)
        elif module.kernel_size == (3, 3):
            module.dilation = module.padding = (dilation, dilation)

    def _set_output_stride_8(self):
        self.layer3.apply(lambda module: self._dilate(module, 2))
        self.layer4.apply(lambda module: self._dilate(module, 4))

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                variance = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / variance))
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _load_pretrained(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Pretrained ResNet-50 weights not found: {path}. "
                "Run models/backbones/get_resnet50_pretrained_model.sh first."
            )
        weights = torch.load(path, map_location="cpu")
        if "state_dict" in weights:
            weights = weights["state_dict"]
        current = self.state_dict()
        compatible = {}
        for key, value in weights.items():
            candidate = f"prefix.{key}" if f"prefix.{key}" in current else key
            if candidate in current and current[candidate].shape == value.shape:
                compatible[candidate] = value
        self.load_state_dict(compatible, strict=False)

    def forward(self, inputs):
        output = self.maxpool(self.prefix(inputs))
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        return self.layer4(output)
