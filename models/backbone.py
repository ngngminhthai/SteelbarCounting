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


def _build_convnext_tiny(pretrained=True):
    try:
        weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        return torchvision.models.convnext_tiny(weights=weights)
    except AttributeError:
        return torchvision.models.convnext_tiny(pretrained=pretrained)


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
        (
            "convnext_tiny",
            {
                "family": "convnext",
                "builder": _build_convnext_tiny,
                # Map to match ResNet spatial scales:
                #   C3 = 96 ch  (stride 4,  same scale as ResNet layer1)
                #   C4 = 192 ch (stride 8)
                #   C5 = 384 ch (stride 16)
                "pyramid_channels": (96, 192, 384),
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


class BackboneConvNeXt(nn.Module):
    """ConvNeXt backbone that returns 4 feature maps to match the ResNet interface.

    torchvision ConvNeXt-Tiny features layout (8 children, 0-indexed):
      [0] stem            -> 96 ch,  stride 4
      [1] stage1          -> 96 ch,  stride 4
      [2] downsample      -> 192 ch, stride 8
      [3] stage2          -> 192 ch, stride 8
      [4] downsample      -> 384 ch, stride 16
      [5] stage3          -> 384 ch, stride 16
      [6] downsample      -> 768 ch, stride 32
      [7] stage4          -> 768 ch, stride 32

    Output mirrors the ResNet spatial-scale convention used by p2pnet.py:
      out[0]: stem only           96 ch, stride 4  (unused placeholder)
      out[1]: stem+stage1         96 ch, stride 4  -> C3 (matches ResNet layer1 scale)
      out[2]: +ds2+stage2        192 ch, stride 8  -> C4
      out[3]: +ds3+stage3        384 ch, stride 16 -> C5
    pyramid_channels = (96, 192, 384)
    This keeps fpn_out[1] at stride 8 (32x32 for 256px input), matching the
    AnchorPoints(pyramid_levels=[3]) stride-8 grid used in the head.
    """

    def __init__(self, name: str):
        super().__init__()
        config = BACKBONE_REGISTRY[name]
        backbone = config["builder"](pretrained=True)
        feats = list(backbone.features.children())
        self.stem = feats[0]                        # stem only,        96 ch, stride 4
        self.stage1 = feats[1]                      # stage1,           96 ch, stride 4
        self.stage2 = nn.Sequential(*feats[2:4])    # ds2 + stage2,    192 ch, stride 8
        self.stage3 = nn.Sequential(*feats[4:6])    # ds3 + stage3,    384 ch, stride 16
        self.pyramid_channels = config["pyramid_channels"]  # (96, 192, 384)
        self.num_channels = self.pyramid_channels[0]

    def forward(self, x):
        x0 = self.stem(x)      # 96 ch, stride 4  (placeholder at index 0)
        x1 = self.stage1(x0)   # 96 ch, stride 4  -> C3
        x2 = self.stage2(x1)   # 192 ch, stride 8  -> C4
        x3 = self.stage3(x2)   # 384 ch, stride 16 -> C5
        # p2pnet.py indexes [1],[2],[3] as C3,C4,C5 for the FPN decoder
        return [x0, x1, x2, x3]


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
    if family == "convnext":
        return BackboneConvNeXt(name)

    raise ValueError(f"Backbone family '{family}' is not implemented.")
