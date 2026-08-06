from math import ceil
from pathlib import Path
from typing import Final, cast

import nibabel as nib
import numpy as np
from nibabel.nifti1 import Nifti1Image
from numpy.typing import NDArray

from cardiac_segmentation.data.acdc_indexer import AcdcDatasetIndexer
from cardiac_segmentation.data.acdc_info_parser import AcdcInfoParser
from cardiac_segmentation.data.acdc_patient_case import AcdcPatientCase
from cardiac_segmentation.data.nifti_metadata_reader import (
    NiftiMetadataReader,
)
from cardiac_segmentation.preprocessing.acdc_phase_preprocessing_profile import (
    AcdcPhasePreprocessingProfile,
)

_SPATIAL_DIMENSION_COUNT: Final[int] = 3
_PERCENTILES: Final[tuple[float, ...]] = (
    1.0,
    5.0,
    50.0,
    95.0,
    99.0,
)


class AcdcPreprocessingProfiler:
    """Measure spatial and intensity properties needed for preprocessing."""

    def __init__(
        self,
        dataset_root: Path,
        candidate_spacing: tuple[float, float, float],
    ) -> None:
        """Initialize the profiler with the dataset and candidate spacing."""
        if any(spacing <= 0.0 for spacing in candidate_spacing):
            raise ValueError("Candidate voxel spacing must be positive.")

        self._candidate_spacing = candidate_spacing
        self._indexer = AcdcDatasetIndexer(
            dataset_root=dataset_root,
            info_parser=AcdcInfoParser(),
        )
        self._metadata_reader = NiftiMetadataReader()

    def profile(self) -> tuple[AcdcPhasePreprocessingProfile, ...]:
        """Profile every real ACDC ED and ES image-mask pair."""
        records: list[AcdcPhasePreprocessingProfile] = []

        for patient_case in self._indexer.index():
            records.extend(
                (
                    self._profile_phase(
                        patient_case=patient_case,
                        phase_name="ED",
                        image_path=patient_case.ed_image_path,
                        mask_path=patient_case.ed_mask_path,
                    ),
                    self._profile_phase(
                        patient_case=patient_case,
                        phase_name="ES",
                        image_path=patient_case.es_image_path,
                        mask_path=patient_case.es_mask_path,
                    ),
                )
            )

        return tuple(records)

    def _profile_phase(
        self,
        patient_case: AcdcPatientCase,
        phase_name: str,
        image_path: Path,
        mask_path: Path,
    ) -> AcdcPhasePreprocessingProfile:
        """Measure spatial and intensity properties for one cardiac phase."""
        metadata = self._metadata_reader.read(image_path)
        image_data = self._load_image(image_path)
        mask_data = self._load_mask(mask_path)

        self._validate_matching_shapes(
            image_data=image_data,
            mask_data=mask_data,
            expected_shape=metadata.shape,
            image_path=image_path,
            mask_path=mask_path,
        )

        (
            foreground_bbox_shape,
            foreground_bbox_center_offset_voxels,
        ) = self._calculate_foreground_bbox_geometry(
            mask_data=mask_data,
            mask_path=mask_path,
        )
        intensity_percentiles = self._calculate_intensity_percentiles(
            image_data=image_data,
            image_path=image_path,
        )

        foreground_bbox_size_mm = (
            foreground_bbox_shape[0] * metadata.voxel_spacing[0],
            foreground_bbox_shape[1] * metadata.voxel_spacing[1],
            foreground_bbox_shape[2] * metadata.voxel_spacing[2],
        )
        foreground_bbox_center_offset_mm = (
            foreground_bbox_center_offset_voxels[0]
            * metadata.voxel_spacing[0],
            foreground_bbox_center_offset_voxels[1]
            * metadata.voxel_spacing[1],
            foreground_bbox_center_offset_voxels[2]
            * metadata.voxel_spacing[2],
        )
        candidate_resampled_bbox_shape = self._calculate_resampled_shape(
            shape=foreground_bbox_shape,
            source_spacing=metadata.voxel_spacing,
        )

        return AcdcPhasePreprocessingProfile(
            patient_id=patient_case.patient_id,
            split_name=patient_case.split_name,
            phase_name=phase_name,
            original_shape=metadata.shape,
            voxel_spacing=metadata.voxel_spacing,
            foreground_bbox_shape=foreground_bbox_shape,
            foreground_bbox_size_mm=foreground_bbox_size_mm,
            foreground_bbox_center_offset_mm=foreground_bbox_center_offset_mm,
            candidate_resampled_shape=self._calculate_resampled_shape(
                shape=metadata.shape,
                source_spacing=metadata.voxel_spacing,
            ),
            candidate_resampled_bbox_shape=candidate_resampled_bbox_shape,
            candidate_centered_crop_min_shape=(
                self._calculate_centered_crop_min_shape(
                    bbox_size_mm=foreground_bbox_size_mm,
                    center_offset_mm=foreground_bbox_center_offset_mm,
                )
            ),
            nonzero_intensity_voxel_count=int(
                np.count_nonzero(image_data)
            ),
            intensity_p01=intensity_percentiles[0],
            intensity_p05=intensity_percentiles[1],
            intensity_p50=intensity_percentiles[2],
            intensity_p95=intensity_percentiles[3],
            intensity_p99=intensity_percentiles[4],
        )

    def _load_image(
        self,
        file_path: Path,
    ) -> NDArray[np.float32]:
        """Load one MRI volume as a finite float32 array."""
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
                f"Expected a 3D MRI volume: {file_path}"
            )

        if not bool(np.isfinite(data).all()):
            raise ValueError(
                f"MRI volume contains non-finite values: {file_path}"
            )

        return data

    def _load_mask(
        self,
        file_path: Path,
    ) -> NDArray[np.int64]:
        """Load one segmentation mask as an integer array."""
        image = cast(
            Nifti1Image,
            nib.load(str(file_path)),
        )
        raw_data = np.asarray(image.dataobj)

        if raw_data.ndim != _SPATIAL_DIMENSION_COUNT:
            raise ValueError(
                f"Expected a 3D segmentation mask: {file_path}"
            )

        if not bool(np.isfinite(raw_data).all()):
            raise ValueError(
                f"Segmentation mask contains non-finite values: {file_path}"
            )

        rounded_data = np.rint(raw_data)

        if not bool(np.equal(raw_data, rounded_data).all()):
            raise ValueError(
                f"Segmentation mask contains non-integer labels: {file_path}"
            )

        return cast(
            NDArray[np.int64],
            rounded_data.astype(
                np.int64,
                copy=False,
            ),
        )

    def _validate_matching_shapes(
        self,
        image_data: NDArray[np.float32],
        mask_data: NDArray[np.int64],
        expected_shape: tuple[int, int, int],
        image_path: Path,
        mask_path: Path,
    ) -> None:
        """Validate array shapes against each other and NIfTI metadata."""
        image_shape = tuple(int(value) for value in image_data.shape)
        mask_shape = tuple(int(value) for value in mask_data.shape)

        if image_shape != expected_shape:
            raise ValueError(
                f"MRI array shape does not match metadata: {image_path}"
            )

        if mask_shape != expected_shape:
            raise ValueError(
                f"Mask shape does not match MRI shape: {mask_path}"
            )

    def _calculate_foreground_bbox_geometry(
        self,
        mask_data: NDArray[np.int64],
        mask_path: Path,
    ) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
        """Calculate foreground box shape and offset from volume center."""
        foreground_coordinates = np.argwhere(mask_data > 0)

        if foreground_coordinates.size == 0:
            raise ValueError(
                f"Segmentation mask contains no foreground: {mask_path}"
            )

        minimum_coordinates = foreground_coordinates.min(axis=0)
        maximum_coordinates = foreground_coordinates.max(axis=0)
        volume_center = (np.asarray(mask_data.shape, dtype=np.float64) - 1.0)
        volume_center /= 2.0
        bbox_center = (
            minimum_coordinates.astype(np.float64)
            + maximum_coordinates.astype(np.float64)
        )
        bbox_center /= 2.0
        center_offset = bbox_center - volume_center

        return (
            (
                int(maximum_coordinates[0] - minimum_coordinates[0] + 1),
                int(maximum_coordinates[1] - minimum_coordinates[1] + 1),
                int(maximum_coordinates[2] - minimum_coordinates[2] + 1),
            ),
            (
                float(center_offset[0]),
                float(center_offset[1]),
                float(center_offset[2]),
            ),
        )

    def _calculate_intensity_percentiles(
        self,
        image_data: NDArray[np.float32],
        image_path: Path,
    ) -> tuple[float, float, float, float, float]:
        """Calculate percentiles using finite non-zero MRI voxels."""
        nonzero_values = image_data[image_data != 0.0]

        if nonzero_values.size == 0:
            raise ValueError(
                f"MRI volume contains no non-zero intensities: {image_path}"
            )

        values = np.percentile(
            nonzero_values,
            _PERCENTILES,
        )

        return (
            float(values[0]),
            float(values[1]),
            float(values[2]),
            float(values[3]),
            float(values[4]),
        )

    def _calculate_resampled_shape(
        self,
        shape: tuple[int, int, int],
        source_spacing: tuple[float, float, float],
    ) -> tuple[int, int, int]:
        """Estimate shape after resampling to the candidate voxel spacing."""
        return (
            max(
                1,
                round(
                    shape[0]
                    * source_spacing[0]
                    / self._candidate_spacing[0]
                ),
            ),
            max(
                1,
                round(
                    shape[1]
                    * source_spacing[1]
                    / self._candidate_spacing[1]
                ),
            ),
            max(
                1,
                round(
                    shape[2]
                    * source_spacing[2]
                    / self._candidate_spacing[2]
                ),
            ),
        )

    def _calculate_centered_crop_min_shape(
        self,
        bbox_size_mm: tuple[float, float, float],
        center_offset_mm: tuple[float, float, float],
    ) -> tuple[int, int, int]:
        """Calculate the smallest candidate-space crop centered on volume."""
        return (
            max(
                1,
                ceil(
                    (
                        bbox_size_mm[0]
                        + 2.0 * abs(center_offset_mm[0])
                    )
                    / self._candidate_spacing[0]
                ),
            ),
            max(
                1,
                ceil(
                    (
                        bbox_size_mm[1]
                        + 2.0 * abs(center_offset_mm[1])
                    )
                    / self._candidate_spacing[1]
                ),
            ),
            max(
                1,
                ceil(
                    (
                        bbox_size_mm[2]
                        + 2.0 * abs(center_offset_mm[2])
                    )
                    / self._candidate_spacing[2]
                ),
            ),
        )
