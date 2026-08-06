from math import isfinite
from typing import Final, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

_LOGITS_DIMENSION_COUNT: Final[int] = 5
_TARGET_DIMENSION_COUNT: Final[int] = 4


class SoftDiceLoss3D(nn.Module):
    """Calculate differentiable multiclass Soft Dice loss for 3D volumes."""

    def __init__(
        self,
        num_classes: int,
        include_background: bool = False,
        smooth: float = 1e-5,
    ) -> None:
        """Initialize class count and Dice smoothing policy."""
        super().__init__()
        self._validate_num_classes(num_classes)
        self._validate_include_background(include_background)
        self._validate_smooth(smooth)

        self._num_classes = num_classes
        self._include_background = include_background
        self._smooth = smooth

    def forward(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Return scalar Soft Dice loss from raw logits and integer targets."""
        self._validate_inputs(
            logits=logits,
            target=target,
        )
        probabilities = torch.softmax(
            logits,
            dim=1,
        )
        one_hot_target = functional.one_hot(
            target,
            num_classes=self._num_classes,
        ).permute(0, 4, 1, 2, 3)
        target_probabilities = one_hot_target.to(
            device=logits.device,
            dtype=logits.dtype,
        )

        if not self._include_background:
            probabilities = probabilities[:, 1:]
            target_probabilities = target_probabilities[:, 1:]

        reduction_dimensions = (
            0,
            2,
            3,
            4,
        )
        intersection = torch.sum(
            probabilities * target_probabilities,
            dim=reduction_dimensions,
        )
        prediction_sum = torch.sum(
            probabilities,
            dim=reduction_dimensions,
        )
        target_sum = torch.sum(
            target_probabilities,
            dim=reduction_dimensions,
        )
        class_dice = (
            (2.0 * intersection + self._smooth)
            / (prediction_sum + target_sum + self._smooth)
        )

        return cast(Tensor, 1.0 - torch.mean(class_dice))

    def _validate_inputs(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> None:
        """Validate logits and target tensors before Dice computation."""
        if logits.ndim != _LOGITS_DIMENSION_COUNT:
            raise ValueError("Logits must be five-dimensional [B, C, D, H, W].")

        if target.ndim != _TARGET_DIMENSION_COUNT:
            raise ValueError("Target must be four-dimensional [B, D, H, W].")

        if logits.shape[0] != target.shape[0]:
            raise ValueError("Logits and target batch dimensions must match.")

        if tuple(logits.shape[2:]) != tuple(target.shape[1:]):
            raise ValueError("Logits and target spatial dimensions must match.")

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
