"""
Model architecture for PulmoGuard.

Uses an ImageNet-pretrained EfficientNet-B0 from torchvision with a
replaced classification head that includes an explicit Dropout layer.
That dropout layer is deliberately never disabled at inference time when
running Monte Carlo Dropout (see uncertainty.py) - it is the mechanism
that turns a single deterministic network into an implicit ensemble.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_model(
    backbone: str = "efficientnet_b0",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout_p: float = 0.3,
) -> nn.Module:
    """Build a classification model with a dropout-equipped head.

    Only efficientnet_b0 is implemented (sufficient accuracy/speed tradeoff
    for a single T4 session), but the function is structured so additional
    backbones can be added without touching calling code.
    """
    if backbone != "efficientnet_b0":
        raise ValueError(
            f"Unsupported backbone '{backbone}'. Only 'efficientnet_b0' is "
            f"currently implemented. Add support in model.py::build_model."
        )

    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.efficientnet_b0(weights=weights)

    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(p=dropout_p, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return net


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters - useful for sanity-checking
    that the backbone is not accidentally frozen or fully retrained."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
