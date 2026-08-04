from collections.abc import Callable
from pathlib import Path
from typing import Final, cast

import nibabel as nib
import numpy as np
from nibabel.nifti1 import Nifti1Image
from numpy.typing import NDArray

from cardiac_segmentation.data.nifti_volume_metadata import (
    AffineMatrix,
    NiftiVolumeMetadata,
)

_SPATIAL_DIMENSION_COUNT: Final[int] = 3
_AFFINE_MATRIX_SHAPE: Final[tuple[int, int]] = (4, 4)


class NiftiMetadataReader:
    """Extract spatial and numerical metadata from one real 3D NIfTI file."""

    def read(self, file_path: Path) -> NiftiVolumeMetadata:
        """Load one NIfTI volume and return validated metadata."""
        resolved_path = file_path.expanduser().resolve(strict=False)

        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"NIfTI file does not exist: {resolved_path}"
            )

        image = cast(
            Nifti1Image,
            nib.load(str(resolved_path)),
        )

        shape = self._read_shape(
            tuple(int(value) for value in image.shape),
            resolved_path,
        )
        voxel_spacing = self._read_spacing(
            self._read_header_zooms(image),
            resolved_path,
        )

        affine_array = np.asarray(
            image.affine,
            dtype=np.float64,
        )

        orientation = self._read_orientation(
            affine_array,
            resolved_path,
        )
        affine = self._read_affine(
            affine_array,
            resolved_path,
        )

        data = np.asarray(image.dataobj)
        intensity_min, intensity_max, has_only_finite_values = (
            self._read_intensity_statistics(
                data,
                resolved_path,
            )
        )

        return NiftiVolumeMetadata(
            file_path=resolved_path,
            shape=shape,
            voxel_spacing=voxel_spacing,
            orientation=orientation,
            affine=affine,
            data_type=self._read_header_data_type(image),
            intensity_min=intensity_min,
            intensity_max=intensity_max,
            has_only_finite_values=has_only_finite_values,
        )

    def _read_header_zooms(
        self,
        image: Nifti1Image,
    ) -> tuple[float, ...]:
        """Read voxel spacing through the partially typed NiBabel header API."""
        get_zooms = cast(
            Callable[[], tuple[float, ...]],
            image.header.get_zooms,
        )

        return tuple(float(value) for value in get_zooms())

    def _read_header_data_type(
        self,
        image: Nifti1Image,
    ) -> str:
        """Read the stored data type through the partially typed NiBabel API."""
        get_data_dtype = cast(
            Callable[[], object],
            image.header.get_data_dtype,
        )

        return str(get_data_dtype())

    def _read_shape(
        self,
        raw_shape: tuple[int, ...],
        file_path: Path,
    ) -> tuple[int, int, int]:
        """Validate and return the three spatial dimensions of a phase volume."""
        if len(raw_shape) != _SPATIAL_DIMENSION_COUNT:
            raise ValueError(
                f"Expected a 3D NIfTI volume, but received shape "
                f"{raw_shape} from {file_path}."
            )

        return (
            raw_shape[0],
            raw_shape[1],
            raw_shape[2],
        )

    def _read_spacing(
        self,
        raw_spacing: tuple[float, ...],
        file_path: Path,
    ) -> tuple[float, float, float]:
        """Validate and return voxel spacing for the three spatial axes."""
        if len(raw_spacing) < _SPATIAL_DIMENSION_COUNT:
            raise ValueError(
                f"NIfTI header does not contain three spatial spacing "
                f"values: {file_path}"
            )

        spacing = (
            raw_spacing[0],
            raw_spacing[1],
            raw_spacing[2],
        )

        if any(value <= 0.0 for value in spacing):
            raise ValueError(
                f"NIfTI voxel spacing must be positive: {file_path}"
            )

        return spacing

    def _read_orientation(
        self,
        affine: NDArray[np.float64],
        file_path: Path,
    ) -> tuple[str, str, str]:
        """Derive anatomical axis codes from the NIfTI affine matrix."""
        aff2axcodes = cast(
            Callable[
                [NDArray[np.float64]],
                tuple[str | None, ...],
            ],
            nib.aff2axcodes,
        )
        raw_orientation = aff2axcodes(affine)

        if len(raw_orientation) != _SPATIAL_DIMENSION_COUNT:
            raise ValueError(
                f"Expected three NIfTI orientation codes: {file_path}"
            )

        first_code = raw_orientation[0]
        second_code = raw_orientation[1]
        third_code = raw_orientation[2]

        if (
            first_code is None
            or second_code is None
            or third_code is None
        ):
            raise ValueError(
                f"Unable to determine NIfTI orientation: {file_path}"
            )

        return (
            first_code,
            second_code,
            third_code,
        )

    def _read_affine(
        self,
        affine: NDArray[np.float64],
        file_path: Path,
    ) -> AffineMatrix:
        """Convert a finite 4x4 affine array into immutable metadata."""
        if affine.shape != _AFFINE_MATRIX_SHAPE:
            raise ValueError(
                f"NIfTI affine matrix must have shape "
                f"{_AFFINE_MATRIX_SHAPE}: {file_path}"
            )

        if not np.isfinite(affine).all():
            raise ValueError(
                f"NIfTI affine matrix contains non-finite values: {file_path}"
            )

        return cast(
            AffineMatrix,
            tuple(
                tuple(float(value) for value in row)
                for row in affine
            ),
        )

    def _read_intensity_statistics(
        self,
        data: NDArray[np.generic],
        file_path: Path,
    ) -> tuple[float, float, bool]:
        """Calculate the finite-value status and intensity range."""
        if data.size == 0:
            raise ValueError(
                f"NIfTI volume is empty: {file_path}"
            )

        finite_mask = np.isfinite(data)
        has_only_finite_values = bool(finite_mask.all())
        finite_values = data[finite_mask]

        if finite_values.size == 0:
            raise ValueError(
                f"NIfTI volume contains no finite values: {file_path}"
            )

        return (
            float(finite_values.min()),
            float(finite_values.max()),
            has_only_finite_values,
        )
