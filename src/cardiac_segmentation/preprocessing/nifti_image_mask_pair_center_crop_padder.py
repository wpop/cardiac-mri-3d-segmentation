from collections.abc import Callable
from typing import Final, cast

import nibabel as nib
import numpy as np
from nibabel import affines
from numpy.typing import NDArray

from cardiac_segmentation.data.nifti_volume_metadata import AffineMatrix
from cardiac_segmentation.preprocessing.center_cropped_padded_image_mask_pair import (
    CenterCroppedPaddedImageMaskPair,
)
from cardiac_segmentation.preprocessing.resampled_image_mask_pair import (
    ResampledImageMaskPair,
)

_SPATIAL_DIMENSION_COUNT: Final[int] = 3
_AFFINE_MATRIX_SHAPE: Final[tuple[int, int]] = (4, 4)
_SPACING_ABSOLUTE_TOLERANCE: Final[float] = 1e-5


class NiftiImageMaskPairCenterCropPadder:
    """Center crop and zero-pad a spacing-resampled image-mask pair."""

    def __init__(
        self,
        target_shape: tuple[int, int, int],
        expected_labels: tuple[int, ...],
    ) -> None:
        """Initialize fixed output shape and label validation rules."""
        self._validate_target_shape(target_shape)
        self._validate_expected_labels(expected_labels)

        self._target_shape = target_shape
        self._expected_labels = frozenset(expected_labels)

    def transform(
        self,
        pair: ResampledImageMaskPair,
    ) -> CenterCroppedPaddedImageMaskPair:
        """Return a centered crop and zero-padded copy at the target shape."""
        crop_start, crop_end, padding_before, padding_after = (
            self._calculate_transform_parameters(pair.shape)
        )
        padding_width = (
            (padding_before[0], padding_after[0]),
            (padding_before[1], padding_after[1]),
            (padding_before[2], padding_after[2]),
        )
        image_data = self._crop_and_pad_image(
            pair=pair,
            crop_start=crop_start,
            crop_end=crop_end,
            padding_width=padding_width,
        )
        mask_data = self._crop_and_pad_mask(
            pair=pair,
            crop_start=crop_start,
            crop_end=crop_end,
            padding_width=padding_width,
        )
        target_affine = self._calculate_target_affine(
            pair=pair,
            crop_start=crop_start,
            padding_before=padding_before,
        )

        self._validate_transformed_arrays(
            image_data=image_data,
            mask_data=mask_data,
        )
        self._validate_target_affine(
            pair=pair,
            target_affine=target_affine,
        )

        return CenterCroppedPaddedImageMaskPair(
            source_pair=pair,
            image_data=image_data,
            mask_data=mask_data,
            shape=self._target_shape,
            voxel_spacing=pair.voxel_spacing,
            orientation=pair.orientation,
            affine=self._to_affine_matrix(target_affine),
            crop_start=crop_start,
            crop_end=crop_end,
            padding_before=padding_before,
            padding_after=padding_after,
        )

    def _calculate_transform_parameters(
        self,
        source_shape: tuple[int, int, int],
    ) -> tuple[
        tuple[int, int, int],
        tuple[int, int, int],
        tuple[int, int, int],
        tuple[int, int, int],
    ]:
        """Calculate centered crop bounds and padding widths for every axis."""
        axis_parameters = tuple(
            self._calculate_axis_parameters(
                source_size=source_shape[axis],
                target_size=self._target_shape[axis],
            )
            for axis in range(_SPATIAL_DIMENSION_COUNT)
        )

        return (
            (
                axis_parameters[0][0],
                axis_parameters[1][0],
                axis_parameters[2][0],
            ),
            (
                axis_parameters[0][1],
                axis_parameters[1][1],
                axis_parameters[2][1],
            ),
            (
                axis_parameters[0][2],
                axis_parameters[1][2],
                axis_parameters[2][2],
            ),
            (
                axis_parameters[0][3],
                axis_parameters[1][3],
                axis_parameters[2][3],
            ),
        )

    def _calculate_axis_parameters(
        self,
        source_size: int,
        target_size: int,
    ) -> tuple[int, int, int, int]:
        """Calculate crop and padding parameters for one spatial axis."""
        if source_size > target_size:
            crop_start = (source_size - target_size) // 2
            crop_end = crop_start + target_size

            return (
                crop_start,
                crop_end,
                0,
                0,
            )

        total_padding = target_size - source_size
        padding_before = total_padding // 2

        return (
            0,
            source_size,
            padding_before,
            total_padding - padding_before,
        )

    def _crop_and_pad_image(
        self,
        pair: ResampledImageMaskPair,
        crop_start: tuple[int, int, int],
        crop_end: tuple[int, int, int],
        padding_width: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    ) -> NDArray[np.float32]:
        """Apply the crop and zero-padding to MRI data."""
        cropped_data = pair.image_data[
            crop_start[0] : crop_end[0],
            crop_start[1] : crop_end[1],
            crop_start[2] : crop_end[2],
        ]
        padded_data = np.pad(
            cropped_data,
            pad_width=padding_width,
            mode="constant",
            constant_values=0.0,
        )

        return np.ascontiguousarray(
            padded_data,
            dtype=np.float32,
        )

    def _crop_and_pad_mask(
        self,
        pair: ResampledImageMaskPair,
        crop_start: tuple[int, int, int],
        crop_end: tuple[int, int, int],
        padding_width: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    ) -> NDArray[np.int64]:
        """Apply the crop and background-label padding to mask data."""
        cropped_data = pair.mask_data[
            crop_start[0] : crop_end[0],
            crop_start[1] : crop_end[1],
            crop_start[2] : crop_end[2],
        ]
        padded_data = np.pad(
            cropped_data,
            pad_width=padding_width,
            mode="constant",
            constant_values=0,
        )

        return np.ascontiguousarray(
            padded_data,
            dtype=np.int64,
        )

    def _calculate_target_affine(
        self,
        pair: ResampledImageMaskPair,
        crop_start: tuple[int, int, int],
        padding_before: tuple[int, int, int],
    ) -> NDArray[np.float64]:
        """Update affine translation for the crop and padding index offset."""
        source_affine = np.asarray(
            pair.affine,
            dtype=np.float64,
        )
        target_affine = source_affine.copy()
        index_offset = (
            np.asarray(crop_start, dtype=np.float64)
            - np.asarray(padding_before, dtype=np.float64)
        )
        target_affine[:3, 3] = (
            source_affine[:3, 3]
            + source_affine[:3, :3] @ index_offset
        )

        return target_affine

    def _validate_transformed_arrays(
        self,
        image_data: NDArray[np.float32],
        mask_data: NDArray[np.int64],
    ) -> None:
        """Validate transformed arrays before building the result object."""
        if tuple(int(value) for value in image_data.shape) != self._target_shape:
            raise ValueError("Cropped and padded image shape does not match target shape.")

        if tuple(int(value) for value in mask_data.shape) != self._target_shape:
            raise ValueError("Cropped and padded mask shape does not match target shape.")

        if not bool(np.isfinite(image_data).all()):
            raise ValueError("Cropped and padded image contains non-finite values.")

        if not bool(np.isfinite(mask_data).all()):
            raise ValueError("Cropped and padded mask contains non-finite values.")

        unexpected_labels = tuple(
            int(label)
            for label in np.unique(mask_data)
            if int(label) not in self._expected_labels
        )

        if unexpected_labels:
            raise ValueError(
                "Cropped and padded mask contains labels outside the expected "
                f"set: {unexpected_labels}."
            )

    def _validate_target_affine(
        self,
        pair: ResampledImageMaskPair,
        target_affine: NDArray[np.float64],
    ) -> None:
        """Validate spacing, orientation, and finite values of the output affine."""
        if not bool(np.isfinite(target_affine).all()):
            raise ValueError("Cropped and padded affine contains non-finite values.")

        voxel_sizes = cast(
            Callable[[NDArray[np.float64]], NDArray[np.float64]],
            affines.voxel_sizes,
        )
        output_spacing = voxel_sizes(target_affine)

        if not np.allclose(
            output_spacing,
            np.asarray(pair.voxel_spacing, dtype=np.float64),
            atol=_SPACING_ABSOLUTE_TOLERANCE,
            rtol=0.0,
        ):
            raise ValueError("Cropped and padded affine changed voxel spacing.")

        aff2axcodes = cast(
            Callable[[NDArray[np.float64]], tuple[str | None, ...]],
            nib.aff2axcodes,
        )
        output_orientation = aff2axcodes(target_affine)

        if output_orientation != pair.orientation:
            raise ValueError("Cropped and padded affine changed orientation.")

    def _to_affine_matrix(
        self,
        affine_array: NDArray[np.float64],
    ) -> AffineMatrix:
        """Convert a finite 4x4 affine array into immutable metadata."""
        if affine_array.shape != _AFFINE_MATRIX_SHAPE:
            raise ValueError(
                f"Cropped and padded affine matrix must have shape "
                f"{_AFFINE_MATRIX_SHAPE}."
            )

        if not bool(np.isfinite(affine_array).all()):
            raise ValueError(
                "Cropped and padded affine matrix contains non-finite values."
            )

        return cast(
            AffineMatrix,
            tuple(
                tuple(float(value) for value in row)
                for row in affine_array
            ),
        )

    def _validate_target_shape(
        self,
        target_shape: tuple[int, int, int],
    ) -> None:
        """Validate the fixed target shape."""
        if len(target_shape) != _SPATIAL_DIMENSION_COUNT:
            raise ValueError("Target shape must contain exactly three dimensions.")

        if any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in target_shape
        ):
            raise ValueError("Target shape dimensions must be positive integers.")

    def _validate_expected_labels(
        self,
        expected_labels: tuple[int, ...],
    ) -> None:
        """Validate the configured segmentation labels."""
        if not expected_labels:
            raise ValueError("Expected labels must not be empty.")

        if any(
            isinstance(label, bool)
            or not isinstance(label, int)
            or label < 0
            for label in expected_labels
        ):
            raise ValueError("Expected labels must be non-negative integers.")

        if len(set(expected_labels)) != len(expected_labels):
            raise ValueError("Expected labels must be unique.")

        if 0 not in expected_labels:
            raise ValueError("Expected labels must include background label 0.")
