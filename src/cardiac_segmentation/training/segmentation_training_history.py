from dataclasses import dataclass
from pathlib import Path

from cardiac_segmentation.training.segmentation_training_epoch_record import (
    SegmentationTrainingEpochRecord,
)


@dataclass(frozen=True, slots=True)
class SegmentationTrainingHistory:
    """Store multi-epoch training history and the best checkpoint path."""

    epoch_records: tuple[SegmentationTrainingEpochRecord, ...]
    best_epoch_number: int
    checkpoint_path: Path

    def __post_init__(self) -> None:
        """Validate epoch sequence, best epoch, and checkpoint file."""
        if not self.epoch_records:
            raise ValueError("Training history must contain at least one epoch record.")

        epoch_numbers = tuple(record.epoch_number for record in self.epoch_records)
        expected_epoch_numbers = tuple(range(1, len(self.epoch_records) + 1))

        if epoch_numbers != expected_epoch_numbers:
            raise ValueError("Epoch numbers must be consecutive and start from 1.")

        if self.best_epoch_number not in epoch_numbers:
            raise ValueError("Best epoch number must refer to an existing epoch record.")

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint path must exist and be a regular file: {self.checkpoint_path}"
            )

    @property
    def final_record(self) -> SegmentationTrainingEpochRecord:
        """Return the final completed epoch record."""
        return self.epoch_records[-1]

    @property
    def best_record(self) -> SegmentationTrainingEpochRecord:
        """Return the record selected by the best-checkpoint policy."""
        return next(
            record
            for record in self.epoch_records
            if record.epoch_number == self.best_epoch_number
        )

    @property
    def epoch_count(self) -> int:
        """Return the number of completed epochs."""
        return len(self.epoch_records)
