from dataclasses import dataclass
from math import isfinite
from pathlib import Path

_EXPECTED_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class SegmentationTrainingCheckpoint:
    """Store validated metadata from a segmentation training checkpoint."""

    checkpoint_path: Path
    format_version: int
    epoch_number: int
    training_average_loss: float
    validation_average_loss: float
    training_mean_dice: float
    validation_mean_dice: float
    included_class_indices: tuple[int, ...]
    validation_per_class_dice: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate checkpoint metadata without retaining model or optimizer state."""
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint path must exist and be a regular file: {self.checkpoint_path}"
            )

        if self.format_version != _EXPECTED_FORMAT_VERSION:
            raise ValueError(
                f"Checkpoint format version must be {_EXPECTED_FORMAT_VERSION}."
            )

        if (
            isinstance(self.epoch_number, bool)
            or not isinstance(self.epoch_number, int)
            or self.epoch_number <= 0
        ):
            raise ValueError("Checkpoint epoch number must be a positive integer.")

        self._validate_loss(
            self.training_average_loss,
            name="Training average loss",
        )
        self._validate_loss(
            self.validation_average_loss,
            name="Validation average loss",
        )
        self._validate_dice(
            self.training_mean_dice,
            name="Training mean Dice",
        )
        self._validate_dice(
            self.validation_mean_dice,
            name="Validation mean Dice",
        )
        self._validate_included_class_indices()
        self._validate_validation_per_class_dice()

    @staticmethod
    def _validate_loss(
        value: float,
        *,
        name: str,
    ) -> None:
        """Validate a finite non-negative loss value."""
        if not isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")

    @staticmethod
    def _validate_dice(
        value: float,
        *,
        name: str,
    ) -> None:
        """Validate a finite Dice value inside the normalized range."""
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and inside [0.0, 1.0].")

    def _validate_included_class_indices(self) -> None:
        """Validate foreground class indices represented in the checkpoint."""
        if not self.included_class_indices:
            raise ValueError("Included class indices must not be empty.")

        if any(
            isinstance(class_index, bool)
            or not isinstance(class_index, int)
            or class_index < 0
            for class_index in self.included_class_indices
        ):
            raise ValueError("Included class indices must be non-negative integers.")

        if len(set(self.included_class_indices)) != len(self.included_class_indices):
            raise ValueError("Included class indices must be unique.")

    def _validate_validation_per_class_dice(self) -> None:
        """Validate per-class Dice values against the included classes."""
        if len(self.validation_per_class_dice) != len(self.included_class_indices):
            raise ValueError(
                "Validation per-class Dice count must match included class count."
            )

        for dice_value in self.validation_per_class_dice:
            self._validate_dice(
                dice_value,
                name="Validation per-class Dice",
            )
