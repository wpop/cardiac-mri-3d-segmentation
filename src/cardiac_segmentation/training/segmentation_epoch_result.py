from dataclasses import dataclass
from math import isfinite

from cardiac_segmentation.metrics import MulticlassDiceMetricResult


@dataclass(frozen=True, slots=True)
class SegmentationEpochResult:
    """Store aggregated segmentation loss and Dice values for one epoch."""

    average_loss: float
    dice_result: MulticlassDiceMetricResult
    batch_count: int
    volume_count: int

    def __post_init__(self) -> None:
        """Validate epoch aggregate values."""
        if not isfinite(self.average_loss) or self.average_loss < 0.0:
            raise ValueError("Average loss must be finite and non-negative.")

        if (
            isinstance(self.batch_count, bool)
            or not isinstance(self.batch_count, int)
            or self.batch_count <= 0
        ):
            raise ValueError("Batch count must be a positive integer.")

        if (
            isinstance(self.volume_count, bool)
            or not isinstance(self.volume_count, int)
            or self.volume_count <= 0
        ):
            raise ValueError("Volume count must be a positive integer.")

        if self.volume_count != self.dice_result.volume_count:
            raise ValueError(
                "Epoch volume count must equal the Dice result volume count."
            )
