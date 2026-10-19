"""Convert segmentation logits into auditable bottle-reuse predictions."""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

Trigger = tuple[int, int]


@dataclass(frozen=True)
class PredictionBatch:
    """Binary predictions, channel probabilities, and threshold triggers."""

    predictions: NDArray[np.uint8]
    max_probabilities: NDArray[np.float32]
    triggers: tuple[tuple[Trigger, ...], ...]


@torch.no_grad()
def predict_from_logits(
    logits: torch.Tensor,
    channel_thresholds: Sequence[int | None],
    *,
    probability_threshold: float = 0.5,
    area_scale: float = 1.0,
    aggregated_channels: Collection[int] = (),
) -> PredictionBatch:
    """Apply connected-component area rules to segmentation logits.

    Trigger areas are returned in original-image pixels and are rounded down
    to match the historical diagnostic output.
    """
    if logits.ndim != 4:
        raise ValueError(f"Expected BxCxHxW logits, got shape {tuple(logits.shape)}")
    if logits.shape[1] != len(channel_thresholds):
        raise ValueError(
            f"Received {logits.shape[1]} logit channels but "
            f"{len(channel_thresholds)} thresholds"
        )
    if not math.isfinite(probability_threshold) or not 0 <= probability_threshold <= 1:
        raise ValueError("probability_threshold must be finite and between 0 and 1")
    if not math.isfinite(area_scale) or area_scale <= 0:
        raise ValueError("area_scale must be finite and greater than zero")
    if any(threshold is not None and threshold < 0 for threshold in channel_thresholds):
        raise ValueError("channel thresholds must be non-negative or None")
    if any(index < 0 or index >= logits.shape[1] for index in aggregated_channels):
        raise ValueError("aggregated channel index is outside the logit channel range")

    probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    binary_masks = (probabilities > probability_threshold).astype(np.uint8)
    batch_size, channel_count, _, _ = binary_masks.shape
    predictions = np.zeros(batch_size, dtype=np.uint8)
    max_probabilities = probabilities.reshape(batch_size, channel_count, -1).max(axis=2)
    batch_triggers: list[tuple[Trigger, ...]] = []

    for batch_index in range(batch_size):
        triggers: list[Trigger] = []
        for channel_index, threshold in enumerate(channel_thresholds):
            if threshold is None:
                continue

            mask = binary_masks[batch_index, channel_index]
            if not mask.any():
                continue

            component_count, _, stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
            if component_count <= 1:
                continue

            component_areas = stats[1:, cv2.CC_STAT_AREA]
            if channel_index in aggregated_channels:
                area = int(component_areas.sum())
            else:
                area = int(component_areas.max())

            scaled_area = area * area_scale
            if scaled_area > threshold:
                predictions[batch_index] = 1
                triggers.append((channel_index, int(scaled_area)))

        batch_triggers.append(tuple(triggers))

    return PredictionBatch(
        predictions=predictions,
        max_probabilities=max_probabilities,
        triggers=tuple(batch_triggers),
    )
