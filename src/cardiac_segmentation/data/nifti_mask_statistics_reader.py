from pathlib import Path
from typing import Final, cast

import nibabel as nib
import numpy as np
from nibabel.nifti1 import Nifti1Image

from cardiac_segmentation.data.nifti_mask_statistics import (
    NiftiMaskStatistics,
)

_SPATIAL_DIMENSION_COUNT: Final[int] = 3


class NiftiMaskStatisticsReader:
    """Extract class-label statistics from one three-dimensional NIfTI mask."""

    def read(self, file_path: Path) -> NiftiMaskStatistics:
        """Load one NIfTI mask and calculate voxel counts for its labels."""
        resolved_path = file_path.expanduser().resolve(strict=False)

        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"NIfTI mask file does not exist: {resolved_path}"
            )

        image = cast(
            Nifti1Image,
            nib.load(str(resolved_path)),
        )
        data = np.asarray(image.dataobj)

        self._validate_data(
            data=data,
            file_path=resolved_path,
        )

        rounded_data = np.rint(data)

        if not bool(np.equal(data, rounded_data).all()):
            raise ValueError(
                f"NIfTI mask contains non-integer labels: {resolved_path}"
            )

        integer_data = rounded_data.astype(
            np.int64,
            copy=False,
        )

        labels, voxel_counts = np.unique(
            integer_data,
            return_counts=True,
        )

        label_voxel_counts = tuple(
            (int(label), int(voxel_count))
            for label, voxel_count in zip(
                labels,
                voxel_counts,
                strict=True,
            )
        )

        return NiftiMaskStatistics(
            file_path=resolved_path,
            label_voxel_counts=label_voxel_counts,
            total_voxel_count=int(integer_data.size),
        )

    def _validate_data(
        self,
        data: np.ndarray,
        file_path: Path,
    ) -> None:
        """Validate mask dimensionality, size, and finite numerical values."""
        if data.ndim != _SPATIAL_DIMENSION_COUNT:
            raise ValueError(
                f"Expected a 3D NIfTI mask, but received shape "
                f"{data.shape} from {file_path}."
            )

        if data.size == 0:
            raise ValueError(
                f"NIfTI mask is empty: {file_path}"
            )

        if not bool(np.isfinite(data).all()):
            raise ValueError(
                f"NIfTI mask contains non-finite values: {file_path}"
            )
