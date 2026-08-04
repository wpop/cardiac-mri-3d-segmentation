from pathlib import Path

import pytest

from cardiac_segmentation.data import (
    AcdcDatasetIndexer,
    AcdcInfoParser,
    NiftiGeometryValidator,
    NiftiMetadataReader,
)

_DATASET_ROOT = Path("data/raw/acdc")
_EXPECTED_PATIENT_COUNT = 150
_PHASE_PAIRS_PER_PATIENT = 2
_EXPECTED_PAIR_COUNT = (
    _EXPECTED_PATIENT_COUNT * _PHASE_PAIRS_PER_PATIENT
)


@pytest.mark.acdc
@pytest.mark.integration
def test_validate_all_real_acdc_image_mask_geometries() -> None:
    """Validate geometry for every real ACDC ED and ES image-mask pair."""
    indexer = AcdcDatasetIndexer(
        dataset_root=_DATASET_ROOT,
        info_parser=AcdcInfoParser(),
    )
    metadata_reader = NiftiMetadataReader()
    geometry_validator = NiftiGeometryValidator()

    patient_cases = indexer.index()
    validated_pair_count = 0

    for patient_case in patient_cases:
        image_mask_pairs = (
            (
                patient_case.ed_image_path,
                patient_case.ed_mask_path,
            ),
            (
                patient_case.es_image_path,
                patient_case.es_mask_path,
            ),
        )

        for image_path, mask_path in image_mask_pairs:
            image_metadata = metadata_reader.read(image_path)
            mask_metadata = metadata_reader.read(mask_path)

            geometry_validator.validate_pair(
                image_metadata=image_metadata,
                mask_metadata=mask_metadata,
            )
            validated_pair_count += 1

    assert len(patient_cases) == _EXPECTED_PATIENT_COUNT
    assert validated_pair_count == _EXPECTED_PAIR_COUNT
