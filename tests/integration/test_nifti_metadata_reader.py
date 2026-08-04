from pathlib import Path

import pytest

from cardiac_segmentation.data import (
    AcdcDatasetIndexer,
    AcdcInfoParser,
    NiftiMetadataReader,
)

_DATASET_ROOT = Path("data/raw/acdc")


@pytest.mark.acdc
@pytest.mark.integration
def test_read_all_real_acdc_phase_volume_metadata() -> None:
    """Verify metadata from all real ACDC ED and ES images and masks."""
    indexer = AcdcDatasetIndexer(
        dataset_root=_DATASET_ROOT,
        info_parser=AcdcInfoParser(),
    )
    reader = NiftiMetadataReader()
    patient_cases = indexer.index()
    inspected_file_count = 0

    for patient_case in patient_cases:
        phase_paths = (
            patient_case.ed_image_path,
            patient_case.ed_mask_path,
            patient_case.es_image_path,
            patient_case.es_mask_path,
        )

        for phase_path in phase_paths:
            metadata = reader.read(phase_path)

            assert metadata.file_path == phase_path.resolve()
            assert all(dimension > 0 for dimension in metadata.shape)
            assert metadata.orientation == ("L", "P", "S")
            assert 0.703 <= metadata.voxel_spacing[0] <= 1.954
            assert 0.703 <= metadata.voxel_spacing[1] <= 1.954
            assert 5.0 <= metadata.voxel_spacing[2] <= 10.0
            assert metadata.has_only_finite_values is True
            assert metadata.intensity_min <= metadata.intensity_max

            if phase_path.name.endswith("_gt.nii.gz"):
                assert metadata.intensity_min >= 0.0
                assert metadata.intensity_max <= 3.0

            inspected_file_count += 1

    assert inspected_file_count == 600
