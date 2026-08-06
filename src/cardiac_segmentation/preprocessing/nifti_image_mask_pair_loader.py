from pathlib import Path
from typing import Final, cast

import nibabel as nib
import numpy as np
from nibabel.nifti1 import Nifti1Image
from numpy.typing import NDArray

from cardiac_segmentation.data.nifti_geometry_validator import (
    NiftiGeometryValidator,
)
from cardiac_segmentation.data.nifti_mask_label_validator import (
    NiftiMaskLabelValidator,
)
from cardiac_segmentation.data.nifti_mask_statistics_reader import (
    NiftiMaskStatisticsReader,
)
from cardiac_segmentation.data.nifti_metadata_reader import NiftiMetadataReader
from cardiac_segmentation.preprocessing.nifti_image_mask_pair import (
    NiftiImageMaskPair,
)

_SPATIAL_DIMENSION_COUNT: Final[int] = 3


class NiftiImageMaskPairLoader:
    """Load and validate one real 3D NIfTI MRI-mask pair."""

    def __init__(
        self,
        expected_labels: tuple[int, ...],
        affine_absolute_tolerance: float,
        require_finite_intensities: bool,
    ) -> None:
        """Initialize validation rules for loading one image-mask pair."""
        self._require_finite_intensities = require_finite_intensities
        self._metadata_reader = NiftiMetadataReader()
        self._geometry_validator = NiftiGeometryValidator(
            absolute_tolerance=affine_absolute_tolerance,
        )
        self._mask_statistics_reader = NiftiMaskStatisticsReader()
        self._mask_label_validator = NiftiMaskLabelValidator(
            expected_labels=expected_labels,
        )

    def load(
        self,
        image_path: Path,
        mask_path: Path,
    ) -> NiftiImageMaskPair:
        """Load one image-mask pair after metadata, geometry, and label checks."""
        resolved_image_path = image_path.expanduser().resolve(strict=False)
        resolved_mask_path = mask_path.expanduser().resolve(strict=False)

        image_metadata = self._metadata_reader.read(resolved_image_path)
        mask_metadata = self._metadata_reader.read(resolved_mask_path)

        self._geometry_validator.validate_pair(
            image_metadata=image_metadata,
            mask_metadata=mask_metadata,
        )

        mask_statistics = self._mask_statistics_reader.read(resolved_mask_path)
        self._mask_label_validator.validate(mask_statistics)

        image_data = self._load_image(resolved_image_path)
        mask_data = self._load_mask(resolved_mask_path)

        return NiftiImageMaskPair(
            image_path=resolved_image_path,
            mask_path=resolved_mask_path,
            image_data=image_data,
            mask_data=mask_data,
            image_metadata=image_metadata,
            mask_metadata=mask_metadata,
        )

    def _load_image(
        self,
        file_path: Path,
    ) -> NDArray[np.float32]:
        """Load one MRI volume as float32 without further preprocessing."""
        image = cast(
            Nifti1Image,
            nib.load(str(file_path)),
        )
        data = np.asarray(
            image.dataobj,
            dtype=np.float32,
        )

        if data.ndim != _SPATIAL_DIMENSION_COUNT:
            raise ValueError(
                f"Expected a 3D NIfTI image, but received shape "
                f"{data.shape} from {file_path}."
            )

        if self._require_finite_intensities and not bool(np.isfinite(data).all()):
            raise ValueError(
                f"NIfTI image contains non-finite intensity values: {file_path}"
            )

        return data

    def _load_mask(
        self,
        file_path: Path,
    ) -> NDArray[np.int64]:
        """Load one segmentation mask, verify integer labels, and cast to int64."""
        image = cast(
            Nifti1Image,
            nib.load(str(file_path)),
        )
        raw_data = np.asarray(image.dataobj)

        if raw_data.ndim != _SPATIAL_DIMENSION_COUNT:
            raise ValueError(
                f"Expected a 3D NIfTI mask, but received shape "
                f"{raw_data.shape} from {file_path}."
            )

        if not bool(np.isfinite(raw_data).all()):
            raise ValueError(
                f"NIfTI mask contains non-finite values: {file_path}"
            )

        rounded_data = np.rint(raw_data)

        if not bool(np.equal(raw_data, rounded_data).all()):
            raise ValueError(
                f"NIfTI mask contains non-integer values: {file_path}"
            )

        return cast(
            NDArray[np.int64],
            rounded_data.astype(
                np.int64,
                copy=False,
            ),
        )
