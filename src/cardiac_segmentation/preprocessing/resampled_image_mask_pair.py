from dataclasses import dataclass
from math import isfinite
from typing import Final

import numpy as np
from numpy.typing import NDArray

from cardiac_segmentation.data.nifti_volume_metadata import AffineMatrix
from cardiac_segmentation.preprocessing.nifti_image_mask_pair import (
    NiftiImageMaskPair,
)

_SPATIAL_DIMENSION_COUNT: Final[int] = 3
_AFFINE_MATRIX_SHAPE: Final[tuple[int, int]] = (4, 4)
_VALID_ORIENTATION_CODES: Final[frozenset[str]] = frozenset(
    {"L", "R", "A", "P", "I", "S"}
)


@dataclass(frozen=True, slots=True)
class ResampledImageMaskPair:
    """Store an in-memory image-mask pair after spacing-only resampling."""

    source_pair: NiftiImageMaskPair
    image_data: NDArray[np.float32]
    mask_data: NDArray[np.int64]
    shape: tuple[int, int, int]
    voxel_spacing: tuple[float, float, float]
    orientation: tuple[str, str, str]
    affine: AffineMatrix

    def __post_init__(self) -> None:
        """Validate resampled arrays and spatial metadata."""
        self._validate_arrays()
        self._validate_shape()
        self._validate_spacing()
        self._validate_orientation()
        self._validate_affine()

    def _validate_arrays(self) -> None:
        """Validate dimensionality, dtypes, matching shapes, and finite values."""
        if self.image_data.ndim != _SPATIAL_DIMENSION_COUNT:
            raise ValueError(
                f"Resampled image array must be exactly 3D, but received "
                f"shape {self.image_data.shape}."
            )

        if self.mask_data.ndim != _SPATIAL_DIMENSION_COUNT:
            raise ValueError(
                f"Resampled mask array must be exactly 3D, but received "
                f"shape {self.mask_data.shape}."
            )

        if self.image_data.dtype != np.dtype(np.float32):
            raise TypeError("Resampled image array must have dtype float32.")

        if self.mask_data.dtype != np.dtype(np.int64):
            raise TypeError("Resampled mask array must have dtype int64.")

        if self.image_data.shape != self.mask_data.shape:
            raise ValueError(
                "Resampled image and mask arrays must have identical shapes: "
                f"image {self.image_data.shape}, mask {self.mask_data.shape}."
            )

        if not bool(np.isfinite(self.image_data).all()):
            raise ValueError("Resampled image array contains non-finite values.")

        if not bool(np.isfinite(self.mask_data).all()):
            raise ValueError("Resampled mask array contains non-finite values.")

    def _validate_shape(self) -> None:
        """Validate declared output shape and match it to the arrays."""
        if any(dimension <= 0 for dimension in self.shape):
            raise ValueError("Resampled shape dimensions must be positive.")

        array_shape = tuple(int(value) for value in self.image_data.shape)

        if array_shape != self.shape:
            raise ValueError(
                f"Resampled array shape {array_shape} does not match "
                f"declared shape {self.shape}."
            )

    def _validate_spacing(self) -> None:
        """Validate finite positive voxel spacing."""
        if any(
            not isfinite(spacing) or spacing <= 0.0
            for spacing in self.voxel_spacing
        ):
            raise ValueError(
                "Resampled voxel spacing values must be finite and positive."
            )

    def _validate_orientation(self) -> None:
        """Validate anatomical axis orientation codes."""
        if any(code not in _VALID_ORIENTATION_CODES for code in self.orientation):
            raise ValueError(
                "Resampled orientation contains an unsupported anatomical axis code."
            )

    def _validate_affine(self) -> None:
        """Validate the 4x4 finite affine matrix."""
        affine_array = np.asarray(self.affine, dtype=np.float64)

        if affine_array.shape != _AFFINE_MATRIX_SHAPE:
            raise ValueError(
                f"Resampled affine matrix must have shape {_AFFINE_MATRIX_SHAPE}."
            )

        if not bool(np.isfinite(affine_array).all()):
            raise ValueError("Resampled affine matrix contains non-finite values.")
