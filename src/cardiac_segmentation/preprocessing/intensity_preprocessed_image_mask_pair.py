from dataclasses import dataclass
from math import isfinite
from typing import Final

import numpy as np
from numpy.typing import NDArray

from cardiac_segmentation.preprocessing.center_cropped_padded_image_mask_pair import (
    CenterCroppedPaddedImageMaskPair,
)

_SPATIAL_DIMENSION_COUNT: Final[int] = 3


@dataclass(frozen=True, slots=True)
class IntensityPreprocessedImageMaskPair:
    """Store an image-mask pair after MRI clipping and z-score normalization."""

    source_pair: CenterCroppedPaddedImageMaskPair
    image_data: NDArray[np.float32]
    mask_data: NDArray[np.int64]
    lower_clip_value: float
    upper_clip_value: float
    normalization_mean: float
    normalization_standard_deviation: float
    normalize_nonzero_only: bool

    def __post_init__(self) -> None:
        """Validate intensity-preprocessed arrays and recorded statistics."""
        self._validate_arrays()
        self._validate_statistics()
        self._validate_mask_matches_source()

    def _validate_arrays(self) -> None:
        """Validate array dimensionality, dtypes, shapes, and finite values."""
        if self.image_data.ndim != _SPATIAL_DIMENSION_COUNT:
            raise ValueError(
                "Intensity-preprocessed image array must be exactly 3D, but "
                f"received shape {self.image_data.shape}."
            )

        if self.mask_data.ndim != _SPATIAL_DIMENSION_COUNT:
            raise ValueError(
                "Intensity-preprocessed mask array must be exactly 3D, but "
                f"received shape {self.mask_data.shape}."
            )

        if self.image_data.dtype != np.dtype(np.float32):
            raise TypeError("Intensity-preprocessed image array must have dtype float32.")

        if self.mask_data.dtype != np.dtype(np.int64):
            raise TypeError("Intensity-preprocessed mask array must have dtype int64.")

        if self.image_data.shape != self.source_pair.shape:
            raise ValueError(
                "Intensity-preprocessed image shape must match the source pair "
                f"shape {self.source_pair.shape}."
            )

        if self.mask_data.shape != self.source_pair.shape:
            raise ValueError(
                "Intensity-preprocessed mask shape must match the source pair "
                f"shape {self.source_pair.shape}."
            )

        if not bool(np.isfinite(self.image_data).all()):
            raise ValueError(
                "Intensity-preprocessed image array contains non-finite values."
            )

        if not bool(np.isfinite(self.mask_data).all()):
            raise ValueError(
                "Intensity-preprocessed mask array contains non-finite values."
            )

    def _validate_statistics(self) -> None:
        """Validate clipping and normalization statistics."""
        statistics = (
            self.lower_clip_value,
            self.upper_clip_value,
            self.normalization_mean,
            self.normalization_standard_deviation,
        )

        if any(not isfinite(value) for value in statistics):
            raise ValueError(
                "Intensity preprocessing statistics must contain only finite values."
            )

        if self.lower_clip_value > self.upper_clip_value:
            raise ValueError("Lower clip value must not exceed upper clip value.")

        if self.normalization_standard_deviation <= 0.0:
            raise ValueError(
                "Normalization standard deviation must be greater than zero."
            )

        if not isinstance(self.normalize_nonzero_only, bool):
            raise TypeError("normalize_nonzero_only must be a boolean.")

    def _validate_mask_matches_source(self) -> None:
        """Validate that segmentation labels were not changed."""
        if not np.array_equal(self.mask_data, self.source_pair.mask_data):
            raise ValueError(
                "Intensity preprocessing must not change source mask labels."
            )
