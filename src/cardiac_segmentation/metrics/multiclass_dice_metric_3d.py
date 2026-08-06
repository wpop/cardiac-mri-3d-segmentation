from math import isfinite
from typing import Final

import torch
from torch import Tensor

from cardiac_segmentation.metrics.multiclass_dice_metric_result import (
    MulticlassDiceMetricResult,
)

_LOGITS_DIMENSION_COUNT: Final[int] = 5
_TARGET_DIMENSION_COUNT: Final[int] = 4


class MulticlassDiceMetric3D:
    """Accumulate multiclass Dice statistics for 3D segmentation logits."""

    def __init__(
        self,
        num_classes: int,
        include_background: bool = False,
        smooth: float = 1e-5,
    ) -> None:
        """Initialize class count, background policy, and smoothing value."""
        self._validate_num_classes(num_classes)
        self._validate_include_background(include_background)
        self._validate_smooth(smooth)

        self._num_classes = num_classes
        self._include_background = include_background
        self._smooth = smooth
        self.reset()

    @property
    def volume_count(self) -> int:
        """Return the number of accumulated target volumes."""
        return self._volume_count

    def reset(self) -> None:
        """Reset accumulated confusion counts and volume count."""
        self._confusion_matrix = torch.zeros(
            (
                self._num_classes,
                self._num_classes,
            ),
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        self._volume_count = 0

    def update(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> None:
        """Accumulate confusion counts from raw logits and integer targets."""
        self._validate_inputs(
            logits=logits,
            target=target,
        )

        with torch.no_grad():
            prediction = torch.argmax(
                logits,
                dim=1,
            )
            encoded_pairs = (
                target.reshape(-1) * self._num_classes
                + prediction.reshape(-1)
            )
            confusion_matrix = torch.bincount(
                encoded_pairs,
                minlength=self._num_classes * self._num_classes,
            ).reshape(
                self._num_classes,
                self._num_classes,
            )
            self._confusion_matrix += confusion_matrix.to(
                device=torch.device("cpu"),
                dtype=torch.float64,
            )
            self._volume_count += int(logits.shape[0])

    def compute(self) -> MulticlassDiceMetricResult:
        """Return accumulated Dice values for configured classes."""
        if self._volume_count == 0:
            raise RuntimeError("Cannot compute Dice before at least one update.")

        intersection = torch.diag(self._confusion_matrix)
        target_count = torch.sum(
            self._confusion_matrix,
            dim=1,
        )
        prediction_count = torch.sum(
            self._confusion_matrix,
            dim=0,
        )
        class_dice = (
            (2.0 * intersection + self._smooth)
            / (target_count + prediction_count + self._smooth)
        )

        if self._include_background:
            included_class_indices = tuple(range(self._num_classes))
        else:
            included_class_indices = tuple(range(1, self._num_classes))

        included_dice = class_dice[
            torch.tensor(
                included_class_indices,
                dtype=torch.int64,
            )
        ]
        per_class_dice = tuple(float(value) for value in included_dice.tolist())
        mean_dice = float(torch.mean(included_dice).item())

        return MulticlassDiceMetricResult(
            included_class_indices=included_class_indices,
            per_class_dice=per_class_dice,
            mean_dice=mean_dice,
            volume_count=self._volume_count,
        )

    def _validate_inputs(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> None:
        """Validate logits and target tensors before accumulation."""
        if logits.ndim != _LOGITS_DIMENSION_COUNT:
            raise ValueError("Logits must be five-dimensional [B, C, D, H, W].")

        if target.ndim != _TARGET_DIMENSION_COUNT:
            raise ValueError("Target must be four-dimensional [B, D, H, W].")

        batch_size = int(logits.shape[0])

        if batch_size <= 0:
            raise ValueError("Logits batch dimension must be positive.")

        if target.shape[0] != batch_size:
            raise ValueError("Logits and target batch dimensions must match.")

        if tuple(logits.shape[2:]) != tuple(target.shape[1:]):
            raise ValueError("Logits and target spatial dimensions must match.")

        if any(int(dimension) <= 0 for dimension in logits.shape[2:]):
            raise ValueError("Logits spatial dimensions must be positive.")

        if int(logits.shape[1]) != self._num_classes:
            raise ValueError(
                f"Logits channel count must equal num_classes: "
                f"{int(logits.shape[1])} != {self._num_classes}."
            )

        if not logits.is_floating_point():
            raise TypeError("Logits must use a floating-point dtype.")

        if target.dtype != torch.int64:
            raise TypeError("Target must have dtype torch.int64.")

        if logits.device != target.device:
            raise ValueError("Logits and target must be on the same device.")

        if not bool(torch.isfinite(logits).all().item()):
            raise ValueError("Logits contain non-finite values.")

        if not bool(((target >= 0) & (target < self._num_classes)).all().item()):
            raise ValueError(
                "Target labels must be within [0, num_classes - 1]."
            )

    @staticmethod
    def _validate_num_classes(
        num_classes: int,
    ) -> None:
        """Validate the number of segmentation classes."""
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 1:
            raise ValueError("Number of classes must be an integer greater than one.")

    @staticmethod
    def _validate_include_background(
        include_background: bool,
    ) -> None:
        """Validate whether background contributes to Dice."""
        if not isinstance(include_background, bool):
            raise TypeError("include_background must be a boolean.")

    @staticmethod
    def _validate_smooth(
        smooth: float,
    ) -> None:
        """Validate the positive Dice smoothing value."""
        if not isfinite(smooth) or smooth <= 0.0:
            raise ValueError("Dice smooth value must be finite and strictly positive.")
