from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from cardiac_segmentation.training.segmentation_training_epoch_record import (
    SegmentationTrainingEpochRecord,
)


@dataclass(frozen=True, slots=True)
class ResumedSegmentationTrainingHistory:
    """Store training history produced after resuming from a checkpoint."""

    resumed_from_epoch_number: int
    epoch_records: tuple[SegmentationTrainingEpochRecord, ...]
    best_epoch_number: int
    best_validation_mean_dice: float
    best_validation_average_loss: float
    checkpoint_path: Path

    def __post_init__(self) -> None:
        """Validate resumed epoch sequence, best epoch, metrics, and checkpoint."""
        if (
            isinstance(self.resumed_from_epoch_number, bool)
            or not isinstance(self.resumed_from_epoch_number, int)
            or self.resumed_from_epoch_number <= 0
        ):
            raise ValueError("Resumed-from epoch number must be a positive integer.")

        if not self.epoch_records:
            raise ValueError("Resumed training history must contain new epoch records.")

        epoch_numbers = tuple(record.epoch_number for record in self.epoch_records)
        expected_epoch_numbers = tuple(
            range(
                self.resumed_from_epoch_number + 1,
                self.resumed_from_epoch_number + 1 + len(self.epoch_records),
            )
        )

        if epoch_numbers != expected_epoch_numbers:
            raise ValueError(
                "Resumed epoch numbers must be consecutive and begin after the "
                "resumed checkpoint epoch."
            )

        if self.best_epoch_number not in {
            self.resumed_from_epoch_number,
            *epoch_numbers,
        }:
            raise ValueError(
                "Best epoch number must be the resumed checkpoint epoch or a new epoch."
            )

        if not isfinite(self.best_validation_mean_dice) or not (
            0.0 <= self.best_validation_mean_dice <= 1.0
        ):
            raise ValueError("Best validation mean Dice must be finite and inside [0.0, 1.0].")

        if (
            not isfinite(self.best_validation_average_loss)
            or self.best_validation_average_loss < 0.0
        ):
            raise ValueError("Best validation average loss must be finite and non-negative.")

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint path must exist and be a regular file: {self.checkpoint_path}"
            )

    @property
    def final_record(self) -> SegmentationTrainingEpochRecord:
        """Return the final completed resumed epoch record."""
        return self.epoch_records[-1]

    @property
    def completed_epoch_count(self) -> int:
        """Return the number of newly completed epochs."""
        return len(self.epoch_records)

    @property
    def final_epoch_number(self) -> int:
        """Return the final epoch number reached by resumed training."""
        return self.final_record.epoch_number
