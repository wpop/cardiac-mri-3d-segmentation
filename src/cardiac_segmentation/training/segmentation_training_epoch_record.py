from dataclasses import dataclass
from math import isfinite

from cardiac_segmentation.training.segmentation_epoch_result import (
    SegmentationEpochResult,
)


@dataclass(frozen=True, slots=True)
class SegmentationTrainingEpochRecord:
    """Store training and validation results for one completed epoch."""

    epoch_number: int
    training_result: SegmentationEpochResult
    validation_result: SegmentationEpochResult
    training_duration_seconds: float
    validation_duration_seconds: float

    def __post_init__(self) -> None:
        """Validate epoch identity and timing values."""
        if (
            isinstance(self.epoch_number, bool)
            or not isinstance(self.epoch_number, int)
            or self.epoch_number <= 0
        ):
            raise ValueError("Epoch number must be a positive integer.")

        if not isfinite(self.training_duration_seconds) or self.training_duration_seconds < 0.0:
            raise ValueError("Training duration must be finite and non-negative.")

        if not isfinite(self.validation_duration_seconds) or self.validation_duration_seconds < 0.0:
            raise ValueError("Validation duration must be finite and non-negative.")
