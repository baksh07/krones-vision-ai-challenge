"""Shared segmentation-model construction."""

from __future__ import annotations

from typing import Protocol

import segmentation_models_pytorch as smp
from torch import nn


class ModelConfig(Protocol):
    arch: str
    encoder_name: str
    encoder_weights: str


def build_model(
    cfg: ModelConfig,
    n_classes: int,
    *,
    pretrained: bool = True,
) -> nn.Module:
    """Build a supported segmentation model for training or inference."""
    architectures = {
        "unetplusplus": smp.UnetPlusPlus,
        "unet": smp.Unet,
        "fpn": smp.FPN,
    }
    try:
        architecture = architectures[cfg.arch]
    except KeyError as exc:
        supported = ", ".join(architectures)
        raise ValueError(
            f"Unsupported architecture {cfg.arch!r}; choose one of: {supported}"
        ) from exc

    return architecture(
        encoder_name=cfg.encoder_name,
        encoder_weights=cfg.encoder_weights if pretrained else None,
        in_channels=3,
        classes=n_classes,
    )
