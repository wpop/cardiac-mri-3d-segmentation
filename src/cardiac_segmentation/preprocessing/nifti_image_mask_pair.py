from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from cardiac_segmentation.data.nifti_volume_metadata import NiftiVolumeMetadata

_SPATIAL_DIMENSION_COUNT: Final[int] = 3


@dataclass(frozen=True, slots=True)
class NiftiImageMaskPair:
    """Store one loaded 3D NIfTI MRI volume and its segmentation mask."""

    image_path: Path
    mask_path: Path
    image_data: NDArray[np.float32]
    mask_data: NDArray[np.int64]
    image_metadata: NiftiVolumeMetadata
    mask_metadata: NiftiVolumeMetadata

    def __post_init__(self) -> None:
        """Validate file paths, array data, and metadata consistency."""
        image_path = self.image_path.expanduser().resolve(strict=False)
        mask_path = self.mask_path.expanduser().resolve(strict=False)

        self._validate_paths(
            image_path=image_path,
            mask_path=mask_path,
        )
        self._validate_arrays(
            image_path=image_path,
            mask_path=mask_path,
        )
        self._validate_shapes_against_metadata(
            image_path=image_path,
            mask_path=mask_path,
        )
        self._validate_metadata_paths(
            image_path=image_path,
            mask_path=mask_path,
        )

    def _validate_paths(
        self,
        image_path: Path,
        mask_path: Path,
    ) -> None:
        """Validate that image and mask files exist and are distinct."""
        if not image_path.is_file():
            raise FileNotFoundError(
                f"NIfTI image file does not exist: {image_path}"
            )

        if not mask_path.is_file():
            raise FileNotFoundError(
                f"NIfTI mask file does not exist: {mask_path}"
            )

        if image_path == mask_path:
            raise ValueError(
                "NIfTI image and mask paths must refer to different files."
            )

    def _validate_arrays(
        self,
        image_path: Path,
        mask_path: Path,
    ) -> None:
        """Validate array dimensionality, dtypes, shape, and finite values."""
        if self.image_data.ndim != _SPATIAL_DIMENSION_COUNT:
            raise ValueError(
                f"NIfTI image array must be exactly 3D, but received "
                f"shape {self.image_data.shape} from {image_path}."
            )

        if self.mask_data.ndim != _SPATIAL_DIMENSION_COUNT:
            raise ValueError(
                f"NIfTI mask array must be exactly 3D, but received "
                f"shape {self.mask_data.shape} from {mask_path}."
            )

        if self.image_data.dtype != np.dtype(np.float32):
            raise TypeError(
                f"NIfTI image array must have dtype float32: {image_path}"
            )

        if self.mask_data.dtype != np.dtype(np.int64):
            raise TypeError(
                f"NIfTI mask array must have dtype int64: {mask_path}"
            )

        if self.image_data.shape != self.mask_data.shape:
            raise ValueError(
                "NIfTI image and mask arrays must have identical shapes: "
                f"image {image_path} has {self.image_data.shape}, while "
                f"mask {mask_path} has {self.mask_data.shape}."
            )

        if not bool(np.isfinite(self.mask_data).all()):
            raise ValueError(
                f"NIfTI mask array contains non-finite values: {mask_path}"
            )

    def _validate_shapes_against_metadata(
        self,
        image_path: Path,
        mask_path: Path,
    ) -> None:
        """Validate loaded array shapes against their metadata records."""
        image_shape = tuple(int(value) for value in self.image_data.shape)
        mask_shape = tuple(int(value) for value in self.mask_data.shape)

        if image_shape != self.image_metadata.shape:
            raise ValueError(
                f"NIfTI image array shape {image_shape} does not match "
                f"metadata shape {self.image_metadata.shape}: {image_path}"
            )

        if mask_shape != self.mask_metadata.shape:
            raise ValueError(
                f"NIfTI mask array shape {mask_shape} does not match "
                f"metadata shape {self.mask_metadata.shape}: {mask_path}"
            )

    def _validate_metadata_paths(
        self,
        image_path: Path,
        mask_path: Path,
    ) -> None:
        """Validate metadata records point back to the loaded files."""
        if self.image_metadata.file_path.resolve(strict=False) != image_path:
            raise ValueError(
                "NIfTI image metadata file path does not match the image path: "
                f"{self.image_metadata.file_path} != {image_path}."
            )

        if self.mask_metadata.file_path.resolve(strict=False) != mask_path:
            raise ValueError(
                "NIfTI mask metadata file path does not match the mask path: "
                f"{self.mask_metadata.file_path} != {mask_path}."
            )
