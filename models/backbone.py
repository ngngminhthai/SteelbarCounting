# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""

from collections import OrderedDict

from torch import nn
import torchvision

import models.vgg_ as models


def _build_torchvision_resnet50(pretrained=True):
    # Support both old and new torchvision APIs.
    try:
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        return torchvision.models.resnet50(weights=weights)
    except AttributeError:
        return torchvision.models.resnet50(pretrained=pretrained)


def _build_vgg16(pretrained=True):
    return models.vgg16(pretrained=pretrained)


def _build_vgg16_bn(pretrained=True):
    return models.vgg16_bn(pretrained=pretrained)


BACKBONE_REGISTRY = OrderedDict(
    [
        (
            "resnet50",
            {
                "family": "resnet",
                "builder": _build_torchvision_resnet50,
                # Use layer1/layer2/layer3 for C3/C4/C5 in decoder.
                "pyramid_channels": (256, 512, 1024),
            },
        ),
        (
            "vgg16_bn",
            {
                "family": "vgg",
                "builder": _build_vgg16_bn,
                "pyramid_channels": (256, 512, 512),
            },
        ),
        (
            "vgg16",
            {
                "family": "vgg",
                "builder": _build_vgg16,
                "pyramid_channels": (256, 512, 512),
            },
        ),
    ]
)


def get_supported_backbones():
    return tuple(BACKBONE_REGISTRY.keys())


class BackboneBaseVGG(nn.Module):
    def __init__(self, backbone: nn.Module, name: str, return_interm_layers: bool, pyramid_channels):
        super().__init__()
        features = list(backbone.features.children())
        if return_interm_layers:
            if name == "vgg16_bn":
                self.body1 = nn.Sequential(*features[:13])
                self.body2 = nn.Sequential(*features[13:23])
                self.body3 = nn.Sequential(*features[23:33])
                self.body4 = nn.Sequential(*features[33:43])
            else:
                self.body1 = nn.Sequential(*features[:9])
                self.body2 = nn.Sequential(*features[9:16])
                self.body3 = nn.Sequential(*features[16:23])
                self.body4 = nn.Sequential(*features[23:30])
        else:
            if name == "vgg16_bn":
                self.body = nn.Sequential(*features[:44])  # 16x down-sample
            elif name == "vgg16":
                self.body = nn.Sequential(*features[:30])  # 16x down-sample
        self.num_channels = pyramid_channels[0]
        self.pyramid_channels = pyramid_channels
        self.return_interm_layers = return_interm_layers

    def forward(self, x):
        out = []
        if self.return_interm_layers:
            xs = x
            for layer in [self.body1, self.body2, self.body3, self.body4]:
                xs = layer(xs)
                out.append(xs)
        else:
            xs = self.body(x)
            out.append(xs)
        return out


class BackboneVGG(BackboneBaseVGG):
    def __init__(self, name: str, return_interm_layers: bool):
        config = BACKBONE_REGISTRY[name]
        backbone = config["builder"](pretrained=True)
        super().__init__(
            backbone=backbone,
            name=name,
            return_interm_layers=return_interm_layers,
            pyramid_channels=config["pyramid_channels"],
        )


class BackboneResNet(nn.Module):
    def __init__(self, name: str, return_interm_layers: bool):
        super().__init__()
        config = BACKBONE_REGISTRY[name]
        self.backbone = config["builder"](pretrained=True)
        self.return_interm_layers = return_interm_layers
        self.num_channels = config["pyramid_channels"][0]
        self.pyramid_channels = config["pyramid_channels"]

    def forward(self, x):
        out = []

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        out.append(x)

        x = self.backbone.layer1(x)
        out.append(x)
        x = self.backbone.layer2(x)
        out.append(x)
        x = self.backbone.layer3(x)
        out.append(x)

        if self.return_interm_layers:
            return out
        return [out[-1]]


def build_backbone(args):
    name = args.backbone.lower()
    if name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unsupported backbone '{args.backbone}'. "
            f"Available: {', '.join(get_supported_backbones())}"
        )

    family = BACKBONE_REGISTRY[name]["family"]
    if family == "vgg":
        return BackboneVGG(name, True)
    if family == "resnet":
        return BackboneResNet(name, True)

    raise ValueError(f"Backbone family '{family}' is not implemented.")
