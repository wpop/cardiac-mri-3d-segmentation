from dataclasses import dataclass
from math import isfinite
from typing import Final

import numpy as np
from numpy.typing import NDArray

from cardiac_segmentation.data.nifti_volume_metadata import AffineMatrix
from cardiac_segmentation.preprocessing.resampled_image_mask_pair import (
    ResampledImageMaskPair,
)

_SPATIAL_DIMENSION_COUNT: Final[int] = 3
_AFFINE_MATRIX_SHAPE: Final[tuple[int, int]] = (4, 4)
_VALID_ORIENTATION_CODES: Final[frozenset[str]] = frozenset(
    {"L", "R", "A", "P", "I", "S"}
)


@dataclass(frozen=True, slots=True)
class CenterCroppedPaddedImageMaskPair:
    """Store a spacing-resampled image-mask pair after centered crop and padding."""

    source_pair: ResampledImageMaskPair
    image_data: NDArray[np.float32]
    mask_data: NDArray[np.int64]
    shape: tuple[int, int, int]
    voxel_spacing: tuple[float, float, float]
    orientation: tuple[str, str, str]
    affine: AffineMatrix
    crop_start: tuple[int, int, int]
    crop_end: tuple[int, int, int]
    padding_before: tuple[int, int, int]
    padding_after: tuple[int, int, int]

    def __post_init__(self) -> None:
        """Validate cropped, padded arrays and their spatial metadata."""
        self._validate_arrays()
        self._validate_shape()
        self._validate_spacing()
        self._validate_orientation()
        self._validate_affine()
        self._validate_crop_and_padding()

    def _validate_arrays(self) -> None:
        """Validate array dimensionality, dtypes, matching shapes, and values."""
        if self.image_data.ndim != _SPATIAL_DIMENSION_COUNT:
            raise ValueError(
                f"Cropped and padded image array must be exactly 3D, but "
                f"received shape {self.image_data.shape}."
            )

        if self.mask_data.ndim != _SPATIAL_DIMENSION_COUNT:
            raise ValueError(
                f"Cropped and padded mask array must be exactly 3D, but "
                f"received shape {self.mask_data.shape}."
            )

        if self.image_data.dtype != np.dtype(np.float32):
            raise TypeError("Cropped and padded image array must have dtype float32.")

        if self.mask_data.dtype != np.dtype(np.int64):
            raise TypeError("Cropped and padded mask array must have dtype int64.")

        if self.image_data.shape != self.mask_data.shape:
            raise ValueError(
                "Cropped and padded image and mask arrays must have identical "
                f"shapes: image {self.image_data.shape}, mask {self.mask_data.shape}."
            )

        if not bool(np.isfinite(self.image_data).all()):
            raise ValueError("Cropped and padded image array contains non-finite values.")

        if not bool(np.isfinite(self.mask_data).all()):
            raise ValueError("Cropped and padded mask array contains non-finite values.")

    def _validate_shape(self) -> None:
        """Validate declared output shape and match it to stored arrays."""
        if len(self.shape) != _SPATIAL_DIMENSION_COUNT:
            raise ValueError(
                "Cropped and padded shape must contain exactly three dimensions."
            )

        if any(not self._is_positive_integer(dimension) for dimension in self.shape):
            raise ValueError(
                "Cropped and padded shape dimensions must be positive integers."
            )

        array_shape = tuple(int(value) for value in self.image_data.shape)

        if array_shape != self.shape:
            raise ValueError(
                f"Cropped and padded array shape {array_shape} does not match "
                f"declared shape {self.shape}."
            )

    def _validate_spacing(self) -> None:
        """Validate finite positive voxel spacing."""
        if len(self.voxel_spacing) != _SPATIAL_DIMENSION_COUNT:
            raise ValueError("Cropped and padded spacing must contain three values.")

        if any(
            not self._is_positive_finite_number(spacing)
            for spacing in self.voxel_spacing
        ):
            raise ValueError(
                "Cropped and padded spacing values must be finite and positive."
            )

    def _validate_orientation(self) -> None:
        """Validate anatomical axis orientation codes."""
        if len(self.orientation) != _SPATIAL_DIMENSION_COUNT:
            raise ValueError("Cropped and padded orientation must contain three codes.")

        if any(code not in _VALID_ORIENTATION_CODES for code in self.orientation):
            raise ValueError(
                "Cropped and padded orientation contains an unsupported anatomical "
                "axis code."
            )

    def _validate_affine(self) -> None:
        """Validate the 4x4 finite affine matrix."""
        affine_array = np.asarray(self.affine, dtype=np.float64)

        if affine_array.shape != _AFFINE_MATRIX_SHAPE:
            raise ValueError(
                f"Cropped and padded affine matrix must have shape "
                f"{_AFFINE_MATRIX_SHAPE}."
            )

        if not bool(np.isfinite(affine_array).all()):
            raise ValueError(
                "Cropped and padded affine matrix contains non-finite values."
            )

    def _validate_crop_and_padding(self) -> None:
        """Validate crop and padding tuples against source and output shapes."""
        transform_tuples = (
            self.crop_start,
            self.crop_end,
            self.padding_before,
            self.padding_after,
        )

        for transform_tuple in transform_tuples:
            if len(transform_tuple) != _SPATIAL_DIMENSION_COUNT:
                raise ValueError(
                    "Crop and padding tuples must contain exactly three values."
                )

            if any(
                not self._is_non_negative_integer(value)
                for value in transform_tuple
            ):
                raise ValueError(
                    "Crop and padding tuples must contain non-negative integers."
                )

        for axis in range(_SPATIAL_DIMENSION_COUNT):
            if self.crop_start[axis] > self.crop_end[axis]:
                raise ValueError("Crop start must not be greater than crop end.")

            if self.crop_end[axis] > self.source_pair.shape[axis]:
                raise ValueError("Crop end must not exceed the source shape.")

            cropped_size = self.crop_end[axis] - self.crop_start[axis]
            reconstructed_size = (
                cropped_size
                + self.padding_before[axis]
                + self.padding_after[axis]
            )

            if reconstructed_size != self.shape[axis]:
                raise ValueError(
                    "Crop and padding dimensions must reconstruct the output "
                    f"shape at axis {axis}."
                )

    @staticmethod
    def _is_non_negative_integer(value: object) -> bool:
        """Return whether a runtime value is a non-negative integer."""
        return (
            not isinstance(value, bool)
            and isinstance(value, int)
            and value >= 0
        )

    @staticmethod
    def _is_positive_integer(value: object) -> bool:
        """Return whether a runtime value is a positive integer."""
        return (
            not isinstance(value, bool)
            and isinstance(value, int)
            and value > 0
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
