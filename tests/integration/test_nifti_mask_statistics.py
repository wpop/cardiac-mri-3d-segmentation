from math import prod
from pathlib import Path
from typing import Final

import pytest

from cardiac_segmentation.data import (
    AcdcDatasetIndexer,
    AcdcInfoParser,
    NiftiMaskLabelValidator,
    NiftiMaskStatisticsReader,
    NiftiMetadataReader,
)

_DATASET_ROOT: Final = Path("data/raw/acdc")
_EXPECTED_LABELS: Final[tuple[int, ...]] = (0, 1, 2, 3)
_EXPECTED_PATIENT_COUNT: Final[int] = 150
_MASKS_PER_PATIENT: Final[int] = 2
_EXPECTED_MASK_COUNT: Final[int] = (
    _EXPECTED_PATIENT_COUNT * _MASKS_PER_PATIENT
)


@pytest.mark.acdc
@pytest.mark.integration
def test_read_and_validate_all_real_acdc_mask_labels() -> None:
    """Validate labels and voxel counts for all real ACDC ED and ES masks."""
    indexer = AcdcDatasetIndexer(
        dataset_root=_DATASET_ROOT,
        info_parser=AcdcInfoParser(),
    )
    statistics_reader = NiftiMaskStatisticsReader()
    metadata_reader = NiftiMetadataReader()
    label_validator = NiftiMaskLabelValidator(
        expected_labels=_EXPECTED_LABELS
    )

    patient_cases = indexer.index()
    observed_dataset_labels: set[int] = set()
    inspected_mask_count = 0

    for patient_case in patient_cases:
        mask_paths = (
            patient_case.ed_mask_path,
            patient_case.es_mask_path,
        )

        for mask_path in mask_paths:
            statistics = statistics_reader.read(mask_path)
            metadata = metadata_reader.read(mask_path)

            label_validator.validate(statistics)

            assert statistics.file_path == mask_path.resolve()
            assert statistics.total_voxel_count == prod(metadata.shape)

            observed_dataset_labels.update(statistics.labels)
            inspected_mask_count += 1

    assert len(patient_cases) == _EXPECTED_PATIENT_COUNT
    assert inspected_mask_count == _EXPECTED_MASK_COUNT
    assert observed_dataset_labels == set(_EXPECTED_LABELS)
