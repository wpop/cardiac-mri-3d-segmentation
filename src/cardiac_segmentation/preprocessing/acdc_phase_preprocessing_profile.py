from dataclasses import dataclass
from math import isfinite
from typing import Final

_VALID_SPLIT_NAMES: Final[frozenset[str]] = frozenset(
    {"training", "testing"}
)
_VALID_PHASE_NAMES: Final[frozenset[str]] = frozenset(
    {"ED", "ES"}
)


@dataclass(frozen=True, slots=True)
class AcdcPhasePreprocessingProfile:
    """Store preprocessing measurements for one ACDC cardiac phase."""

    patient_id: str
    split_name: str
    phase_name: str
    original_shape: tuple[int, int, int]
    voxel_spacing: tuple[float, float, float]
    foreground_bbox_shape: tuple[int, int, int]
    foreground_bbox_size_mm: tuple[float, float, float]
    foreground_bbox_center_offset_mm: tuple[float, float, float]
    candidate_resampled_shape: tuple[int, int, int]
    candidate_resampled_bbox_shape: tuple[int, int, int]
    candidate_centered_crop_min_shape: tuple[int, int, int]
    nonzero_intensity_voxel_count: int
    intensity_p01: float
    intensity_p05: float
    intensity_p50: float
    intensity_p95: float
    intensity_p99: float

    def __post_init__(self) -> None:
        """Validate identifiers, dimensions, spacing, and percentiles."""
        if not self.patient_id.strip():
            raise ValueError("Patient identifier must not be empty.")

        if self.split_name not in _VALID_SPLIT_NAMES:
            raise ValueError(
                "Dataset split must be either 'training' or 'testing'."
            )

        if self.phase_name not in _VALID_PHASE_NAMES:
            raise ValueError("Cardiac phase must be either 'ED' or 'ES'.")

        dimension_groups = (
            self.original_shape,
            self.foreground_bbox_shape,
            self.candidate_resampled_shape,
            self.candidate_resampled_bbox_shape,
            self.candidate_centered_crop_min_shape,
        )

        if any(
            dimension <= 0
            for dimensions in dimension_groups
            for dimension in dimensions
        ):
            raise ValueError(
                "All preprocessing profile dimensions must be positive."
            )

        if any(spacing <= 0.0 for spacing in self.voxel_spacing):
            raise ValueError("Voxel spacing must be positive.")

        if any(
            not isfinite(offset)
            for offset in self.foreground_bbox_center_offset_mm
        ):
            raise ValueError(
                "Foreground center offsets must contain finite values."
            )

        if any(
            bbox_dimension > volume_dimension
            for bbox_dimension, volume_dimension in zip(
                self.foreground_bbox_shape,
                self.original_shape,
                strict=True,
            )
        ):
            raise ValueError(
                "Foreground bounding box must fit inside the original volume."
            )

        if any(
            bbox_dimension > volume_dimension
            for bbox_dimension, volume_dimension in zip(
                self.candidate_resampled_bbox_shape,
                self.candidate_resampled_shape,
                strict=True,
            )
        ):
            raise ValueError(
                "Resampled bounding box must fit inside the resampled volume."
            )

        if any(
            crop_dimension < bbox_dimension
            for crop_dimension, bbox_dimension in zip(
                self.candidate_centered_crop_min_shape,
                self.candidate_resampled_bbox_shape,
                strict=True,
            )
        ):
            raise ValueError(
                "Centered crop must not be smaller than the foreground box."
            )

        if self.nonzero_intensity_voxel_count <= 0:
            raise ValueError(
                "Non-zero intensity voxel count must be positive."
            )

        percentiles = (
            self.intensity_p01,
            self.intensity_p05,
            self.intensity_p50,
            self.intensity_p95,
            self.intensity_p99,
        )

        if percentiles != tuple(sorted(percentiles)):
            raise ValueError(
                "MRI intensity percentiles must be ordered."
            )
