from dataclasses import dataclass
from math import isfinite
from typing import Final

_SPATIAL_DIMENSION_COUNT: Final[int] = 3
_MINIMUM_PERCENTILE: Final[float] = 0.0
_MAXIMUM_PERCENTILE: Final[float] = 100.0


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Store spatial and intensity preprocessing parameters."""

    target_spacing_mm: tuple[float, float, float]
    target_shape: tuple[int, int, int]
    intensity_lower_percentile: float
    intensity_upper_percentile: float
    normalize_nonzero_only: bool

    def __post_init__(self) -> None:
        """Validate spatial dimensions and intensity normalization settings."""
        if len(self.target_spacing_mm) != _SPATIAL_DIMENSION_COUNT:
            raise ValueError(
                "Target spacing must contain exactly three values."
            )

        if len(self.target_shape) != _SPATIAL_DIMENSION_COUNT:
            raise ValueError(
                "Target shape must contain exactly three dimensions."
            )

        if any(
            not self._is_positive_finite_number(spacing)
            for spacing in self.target_spacing_mm
        ):
            raise ValueError(
                "Target spacing values must be finite and positive."
            )

        if any(
            not self._is_positive_integer(dimension)
            for dimension in self.target_shape
        ):
            raise ValueError(
                "Target shape dimensions must be positive integers."
            )

        if not isfinite(self.intensity_lower_percentile):
            raise ValueError(
                "Lower intensity percentile must be finite."
            )

        if not isfinite(self.intensity_upper_percentile):
            raise ValueError(
                "Upper intensity percentile must be finite."
            )

        if not (
            _MINIMUM_PERCENTILE
            <= self.intensity_lower_percentile
            < self.intensity_upper_percentile
            <= _MAXIMUM_PERCENTILE
        ):
            raise ValueError(
                "Intensity percentiles must satisfy "
                "0 <= lower < upper <= 100."
            )

    @staticmethod
    def _is_positive_finite_number(value: object) -> bool:
        """Return whether a runtime value is a finite positive number."""
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and isfinite(value)
            and value > 0.0
        )

    @staticmethod
    def _is_positive_integer(value: object) -> bool:
        """Return whether a runtime value is a positive integer."""
        return (
            not isinstance(value, bool)
            and isinstance(value, int)
            and value > 0
        )
