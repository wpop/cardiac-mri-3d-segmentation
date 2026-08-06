from collections.abc import Callable
from math import isfinite
from typing import Final, cast

import nibabel as nib
import numpy as np
from nibabel import affines
from nibabel.nifti1 import Nifti1Image
from nibabel.processing import resample_from_to
from numpy.typing import NDArray

from cardiac_segmentation.data.nifti_volume_metadata import AffineMatrix
from cardiac_segmentation.preprocessing.nifti_image_mask_pair import (
    NiftiImageMaskPair,
)
from cardiac_segmentation.preprocessing.resampled_image_mask_pair import (
    ResampledImageMaskPair,
)

_SPATIAL_DIMENSION_COUNT: Final[int] = 3
_AFFINE_MATRIX_SHAPE: Final[tuple[int, int]] = (4, 4)
_SPACING_ABSOLUTE_TOLERANCE: Final[float] = 1e-5


class NiftiImageMaskPairResampler:
    """Resample an already loaded NIfTI image-mask pair to target spacing."""

    def __init__(
        self,
        target_spacing_mm: tuple[float, float, float],
        expected_labels: tuple[int, ...],
    ) -> None:
        """Initialize spacing and label validation rules."""
        self._validate_target_spacing(target_spacing_mm)
        self._validate_expected_labels(expected_labels)

        self._target_spacing_mm = target_spacing_mm
        self._expected_labels = frozenset(expected_labels)

    def resample(
        self,
        pair: NiftiImageMaskPair,
    ) -> ResampledImageMaskPair:
        """Return an in-memory pair resampled to the configured voxel spacing."""
        target_shape = self._calculate_target_shape(pair)
        source_affine = np.asarray(
            pair.image_metadata.affine,
            dtype=np.float64,
        )
        rescale_affine = cast(
            Callable[
                [
                    NDArray[np.float64],
                    tuple[int, int, int],
                    tuple[float, float, float],
                    tuple[int, int, int],
                ],
                NDArray[np.float64],
            ],
            affines.rescale_affine,
        )
        target_affine = rescale_affine(
            source_affine,
            pair.image_metadata.shape,
            self._target_spacing_mm,
            target_shape,
        )
        self._align_world_space_center(
            source_affine=source_affine,
            source_shape=pair.image_metadata.shape,
            target_affine=target_affine,
            target_shape=target_shape,
        )

        image_data = self._resample_image(
            image_data=pair.image_data,
            source_affine=source_affine,
            target_shape=target_shape,
            target_affine=target_affine,
        )
        mask_data = self._resample_mask(
            mask_data=pair.mask_data,
            source_affine=source_affine,
            target_shape=target_shape,
            target_affine=target_affine,
        )
        voxel_spacing = self._calculate_output_spacing(target_affine)
        self._validate_output_spacing(voxel_spacing)
        orientation = self._calculate_output_orientation(target_affine)

        return ResampledImageMaskPair(
            source_pair=pair,
            image_data=image_data,
            mask_data=mask_data,
            shape=target_shape,
            voxel_spacing=voxel_spacing,
            orientation=orientation,
            affine=self._to_affine_matrix(target_affine),
        )

    def _calculate_target_shape(
        self,
        pair: NiftiImageMaskPair,
    ) -> tuple[int, int, int]:
        """Calculate the spacing-derived target shape independently per axis."""
        return (
            self._calculate_axis_shape(pair, axis=0),
            self._calculate_axis_shape(pair, axis=1),
            self._calculate_axis_shape(pair, axis=2),
        )

    def _calculate_axis_shape(
        self,
        pair: NiftiImageMaskPair,
        axis: int,
    ) -> int:
        """Calculate one target dimension from source shape and spacing."""
        return max(
            1,
            round(
                pair.image_metadata.shape[axis]
                * pair.image_metadata.voxel_spacing[axis]
                / self._target_spacing_mm[axis]
            ),
        )

    def _resample_image(
        self,
        image_data: NDArray[np.float32],
        source_affine: NDArray[np.float64],
        target_shape: tuple[int, int, int],
        target_affine: NDArray[np.float64],
    ) -> NDArray[np.float32]:
        """Resample MRI intensities with linear interpolation."""
        source_image = self._create_nifti_image(
            image_data,
            source_affine,
            np.dtype(np.float32),
        )
        resampled_image = self._resample_from_to(
            source_image,
            (target_shape, target_affine),
            1,
            "constant",
            0.0,
        )

        return np.asarray(
            resampled_image.dataobj,
            dtype=np.float32,
        )

    def _resample_mask(
        self,
        mask_data: NDArray[np.int64],
        source_affine: NDArray[np.float64],
        target_shape: tuple[int, int, int],
        target_affine: NDArray[np.float64],
    ) -> NDArray[np.int64]:
        """Resample mask labels with nearest-neighbor interpolation."""
        source_image = self._create_nifti_image(
            mask_data,
            source_affine,
            np.dtype(np.int64),
        )
        resampled_image = self._resample_from_to(
            source_image,
            (target_shape, target_affine),
            0,
            "constant",
            0.0,
        )
        raw_data = np.asarray(resampled_image.dataobj)

        if not bool(np.isfinite(raw_data).all()):
            raise ValueError("Resampled mask contains non-finite values.")

        rounded_data = np.rint(raw_data)

        if not bool(np.equal(raw_data, rounded_data).all()):
            raise ValueError("Resampled mask contains fractional labels.")

        integer_data = cast(
            NDArray[np.int64],
            rounded_data.astype(
                np.int64,
                copy=False,
            ),
        )
        unexpected_labels = tuple(
            int(label)
            for label in np.unique(integer_data)
            if int(label) not in self._expected_labels
        )

        if unexpected_labels:
            raise ValueError(
                "Resampled mask contains labels outside the expected set: "
                f"{unexpected_labels}."
            )

        return integer_data

    def _calculate_output_spacing(
        self,
        target_affine: NDArray[np.float64],
    ) -> tuple[float, float, float]:
        """Calculate voxel spacing from the target affine matrix."""
        voxel_sizes = cast(
            Callable[[NDArray[np.float64]], NDArray[np.float64]],
            affines.voxel_sizes,
        )
        spacing = voxel_sizes(target_affine)

        return (
            float(spacing[0]),
            float(spacing[1]),
            float(spacing[2]),
        )

    def _validate_output_spacing(
        self,
        voxel_spacing: tuple[float, float, float],
    ) -> None:
        """Validate the realized output spacing against the requested spacing."""
        if not np.allclose(
            np.asarray(voxel_spacing, dtype=np.float64),
            np.asarray(self._target_spacing_mm, dtype=np.float64),
            atol=_SPACING_ABSOLUTE_TOLERANCE,
            rtol=0.0,
        ):
            raise ValueError(
                "Resampled voxel spacing does not match the requested target "
                f"spacing: {voxel_spacing} != {self._target_spacing_mm}."
            )

    def _calculate_output_orientation(
        self,
        target_affine: NDArray[np.float64],
    ) -> tuple[str, str, str]:
        """Derive anatomical axis codes from the target affine matrix."""
        aff2axcodes = cast(
            Callable[[NDArray[np.float64]], tuple[str | None, ...]],
            nib.aff2axcodes,
        )
        raw_orientation = aff2axcodes(target_affine)

        if len(raw_orientation) != _SPATIAL_DIMENSION_COUNT:
            raise ValueError("Resampled affine did not produce three orientation codes.")

        first_code = raw_orientation[0]
        second_code = raw_orientation[1]
        third_code = raw_orientation[2]

        if first_code is None or second_code is None or third_code is None:
            raise ValueError("Unable to determine resampled orientation.")

        return (
            first_code,
            second_code,
            third_code,
        )

    def _align_world_space_center(
        self,
        source_affine: NDArray[np.float64],
        source_shape: tuple[int, int, int],
        target_affine: NDArray[np.float64],
        target_shape: tuple[int, int, int],
    ) -> None:
        """Translate the target affine to preserve the continuous volume center."""
        source_center = self._calculate_world_space_center(
            shape=source_shape,
            affine_matrix=source_affine,
        )
        target_center = self._calculate_world_space_center(
            shape=target_shape,
            affine_matrix=target_affine,
        )
        target_affine[:3, 3] += source_center - target_center

    def _calculate_world_space_center(
        self,
        shape: tuple[int, int, int],
        affine_matrix: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate the world-space center of a voxel grid."""
        apply_affine = cast(
            Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
            affines.apply_affine,
        )
        center_voxel = (np.asarray(shape, dtype=np.float64) - 1.0) / 2.0

        return apply_affine(
            affine_matrix,
            center_voxel,
        )

    def _create_nifti_image(
        self,
        data: NDArray[np.generic],
        affine_matrix: NDArray[np.float64],
        data_type: np.dtype[np.generic],
    ) -> Nifti1Image:
        """Construct a temporary in-memory NIfTI image."""
        image_factory = cast(
            Callable[
                [
                    NDArray[np.generic],
                    NDArray[np.float64],
                    object,
                    object,
                    object,
                    np.dtype[np.generic],
                ],
                Nifti1Image,
            ],
            Nifti1Image,
        )

        return image_factory(
            data,
            affine_matrix,
            None,
            None,
            None,
            data_type,
        )

    def _resample_from_to(
        self,
        source_image: Nifti1Image,
        target: tuple[tuple[int, int, int], NDArray[np.float64]],
        order: int,
        mode: str,
        background_value: float,
    ) -> Nifti1Image:
        """Call NiBabel's resampling helper through a narrow typed wrapper."""
        resample = cast(
            Callable[
                [
                    Nifti1Image,
                    tuple[tuple[int, int, int], NDArray[np.float64]],
                    int,
                    str,
                    float,
                ],
                Nifti1Image,
            ],
            resample_from_to,
        )

        return resample(
            source_image,
            target,
            order,
            mode,
            background_value,
        )

    def _to_affine_matrix(
        self,
        affine_array: NDArray[np.float64],
    ) -> AffineMatrix:
        """Convert a finite 4x4 affine array into immutable metadata."""
        if affine_array.shape != _AFFINE_MATRIX_SHAPE:
            raise ValueError(
                f"Resampled affine matrix must have shape {_AFFINE_MATRIX_SHAPE}."
            )

        if not bool(np.isfinite(affine_array).all()):
            raise ValueError("Resampled affine matrix contains non-finite values.")

        return cast(
            AffineMatrix,
            tuple(
                tuple(float(value) for value in row)
                for row in affine_array
            ),
        )

    def _validate_target_spacing(
        self,
        target_spacing_mm: tuple[float, float, float],
    ) -> None:
        """Validate the requested target spacing."""
        if any(
            not isfinite(spacing) or spacing <= 0.0
            for spacing in target_spacing_mm
        ):
            raise ValueError("Target spacing values must be finite and positive.")

    def _validate_expected_labels(
        self,
        expected_labels: tuple[int, ...],
    ) -> None:
        """Validate the configured segmentation labels."""
        if not expected_labels:
            raise ValueError("Expected labels must not be empty.")

        if any(label < 0 for label in expected_labels):
            raise ValueError("Expected labels must be non-negative integers.")

        if len(set(expected_labels)) != len(expected_labels):
            raise ValueError("Expected labels must be unique.")
