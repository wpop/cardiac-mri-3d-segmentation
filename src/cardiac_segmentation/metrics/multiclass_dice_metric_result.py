from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class MulticlassDiceMetricResult:
    """Store accumulated multiclass Dice results for included classes."""

    included_class_indices: tuple[int, ...]
    per_class_dice: tuple[float, ...]
    mean_dice: float
    volume_count: int

    def __post_init__(self) -> None:
        """Validate reported class indices, Dice values, and volume count."""
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

        if len(self.per_class_dice) != len(self.included_class_indices):
            raise ValueError(
                "Per-class Dice values must match the number of included classes."
            )

        if any(not self._is_valid_dice_value(dice_value) for dice_value in self.per_class_dice):
            raise ValueError("Per-class Dice values must be finite values inside [0.0, 1.0].")

        if not self._is_valid_dice_value(self.mean_dice):
            raise ValueError("Mean Dice must be a finite value inside [0.0, 1.0].")

        if (
            isinstance(self.volume_count, bool)
            or not isinstance(self.volume_count, int)
            or self.volume_count <= 0
        ):
            raise ValueError("Volume count must be a positive integer.")

    @staticmethod
    def _is_valid_dice_value(
        value: float,
    ) -> bool:
        """Return whether a value is a finite Dice coefficient."""
        return isfinite(value) and 0.0 <= value <= 1.0
