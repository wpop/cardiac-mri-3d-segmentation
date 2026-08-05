from pathlib import Path
from typing import Final

import pytest

from cardiac_segmentation.config.loader import AppConfigLoader
from cardiac_segmentation.data import AcdcDatasetInspector

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_CONFIG_PATH: Final[Path] = Path("configs/data.yaml")
_EXPECTED_PATIENT_COUNT: Final[int] = 150
_EXPECTED_PHASE_COUNT: Final[int] = 300
_EXPECTED_TRAINING_PATIENT_COUNT: Final[int] = 100
_EXPECTED_TESTING_PATIENT_COUNT: Final[int] = 50
_EXPECTED_LABELS: Final[tuple[int, ...]] = (0, 1, 2, 3)


@pytest.mark.acdc
@pytest.mark.integration
def test_inspect_complete_real_acdc_dataset() -> None:
    """Inspect all real ACDC patients and verify dataset-level results."""
    config = AppConfigLoader(
        project_root=_PROJECT_ROOT
    ).load(_CONFIG_PATH)

    report = AcdcDatasetInspector(config).inspect()

    assert report.dataset_name == "ACDC"
    assert report.patient_count == _EXPECTED_PATIENT_COUNT
    assert report.phase_count == _EXPECTED_PHASE_COUNT
    assert report.patient_count_for_split(
        "training"
    ) == _EXPECTED_TRAINING_PATIENT_COUNT
    assert report.patient_count_for_split(
        "testing"
    ) == _EXPECTED_TESTING_PATIENT_COUNT
    assert report.observed_labels == _EXPECTED_LABELS

    for patient_record in report.patient_records:
        assert patient_record.phase("ED").phase_name == "ED"
        assert patient_record.phase("ES").phase_name == "ES"

        for phase_record in patient_record.phase_records:
            assert (
                phase_record.image_metadata.has_only_finite_values
                is True
            )
            assert (
                phase_record.mask_metadata.has_only_finite_values
                is True
            )
