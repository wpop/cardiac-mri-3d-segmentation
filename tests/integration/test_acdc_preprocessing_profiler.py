import math
from pathlib import Path
from typing import Final

import pytest

from cardiac_segmentation.config.loader import AppConfigLoader
from cardiac_segmentation.preprocessing import AcdcPreprocessingProfiler

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_CONFIG_PATH: Final[Path] = Path("configs/data.yaml")
_CANDIDATE_SPACING: Final[tuple[float, float, float]] = (
    1.5,
    1.5,
    5.0,
)
_EXPECTED_RECORD_COUNT: Final[int] = 300


@pytest.mark.acdc
@pytest.mark.integration
def test_profile_all_real_acdc_phases_for_preprocessing() -> None:
    """Profile spatial and intensity properties of all real ACDC phases."""
    config = AppConfigLoader(
        project_root=_PROJECT_ROOT
    ).load(_CONFIG_PATH)

    records = AcdcPreprocessingProfiler(
        dataset_root=config.dataset.root_dir,
        candidate_spacing=_CANDIDATE_SPACING,
    ).profile()

    assert len(records) == _EXPECTED_RECORD_COUNT
    assert {
        record.phase_name
        for record in records
    } == {"ED", "ES"}

    for record in records:
        assert all(
            dimension > 0
            for dimension in record.foreground_bbox_shape
        )
        assert all(
            dimension > 0
            for dimension in record.candidate_resampled_shape
        )
        assert all(
            dimension > 0
            for dimension in record.candidate_resampled_bbox_shape
        )
        assert all(
            crop_dimension >= bbox_dimension
            for crop_dimension, bbox_dimension in zip(
                record.candidate_centered_crop_min_shape,
                record.candidate_resampled_bbox_shape,
                strict=True,
            )
        )
        assert all(
            math.isfinite(offset)
            for offset in record.foreground_bbox_center_offset_mm
        )
        assert (
            record.intensity_p01
            <= record.intensity_p05
            <= record.intensity_p50
            <= record.intensity_p95
            <= record.intensity_p99
        )
