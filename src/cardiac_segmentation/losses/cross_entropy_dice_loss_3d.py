from math import isfinite
from typing import cast

from torch import Tensor, nn

from cardiac_segmentation.losses.soft_dice_loss_3d import SoftDiceLoss3D


class CrossEntropyDiceLoss3D(nn.Module):
    """Combine multiclass cross-entropy and Soft Dice losses for 3D segmentation."""

    def __init__(
        self,
        num_classes: int,
        cross_entropy_weight: float = 0.5,
        dice_weight: float = 0.5,
        include_background_in_dice: bool = False,
        dice_smooth: float = 1e-5,
    ) -> None:
        """Initialize weighted cross-entropy and Dice loss components."""
        super().__init__()
        self._validate_num_classes(num_classes)
        self._validate_weights(
            cross_entropy_weight=cross_entropy_weight,
            dice_weight=dice_weight,
        )
        self._validate_include_background(include_background_in_dice)
        self._validate_dice_smooth(dice_smooth)

        self._cross_entropy_weight = cross_entropy_weight
        self._dice_weight = dice_weight
        self._cross_entropy_loss = nn.CrossEntropyLoss()
        self._soft_dice_loss = SoftDiceLoss3D(
            num_classes=num_classes,
            include_background=include_background_in_dice,
            smooth=dice_smooth,
        )

    def forward(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Return normalized weighted cross-entropy plus Dice loss."""
        cross_entropy_loss = cast(
            Tensor,
            self._cross_entropy_loss(
                logits,
                target,
            ),
        )
        dice_loss = self._soft_dice_loss(
            logits,
            target,
        )

        return cast(
            Tensor,
            (
                self._cross_entropy_weight * cross_entropy_loss
                + self._dice_weight * dice_loss
            )
            / (self._cross_entropy_weight + self._dice_weight),
        )

    @staticmethod
    def _validate_num_classes(
        num_classes: int,
    ) -> None:
        """Validate the number of segmentation classes."""
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 1:
            raise ValueError("Number of classes must be an integer greater than one.")

    @staticmethod
    def _validate_weights(
        cross_entropy_weight: float,
        dice_weight: float,
    ) -> None:
        """Validate non-negative component weights."""
        if not isfinite(cross_entropy_weight) or cross_entropy_weight < 0.0:
            raise ValueError("Cross-entropy weight must be finite and non-negative.")

        if not isfinite(dice_weight) or dice_weight < 0.0:
            raise ValueError("Dice weight must be finite and non-negative.")

        if cross_entropy_weight == 0.0 and dice_weight == 0.0:
            raise ValueError("At least one loss weight must be strictly positive.")

    @staticmethod
    def _validate_include_background(
        include_background_in_dice: bool,
    ) -> None:
        """Validate whether background contributes to Dice."""
        if not isinstance(include_background_in_dice, bool):
            raise TypeError("include_background_in_dice must be a boolean.")

    @staticmethod
    def _validate_dice_smooth(
        dice_smooth: float,
    ) -> None:
        """Validate the positive Dice smoothing value."""
        if not isfinite(dice_smooth) or dice_smooth <= 0.0:
            raise ValueError("Dice smooth value must be finite and strictly positive.")
