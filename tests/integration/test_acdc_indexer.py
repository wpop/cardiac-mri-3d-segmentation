from collections import Counter
from pathlib import Path

import pytest

from cardiac_segmentation.data import AcdcDatasetIndexer, AcdcInfoParser

_DATASET_ROOT = Path("data/raw/acdc")


@pytest.mark.acdc
@pytest.mark.integration
def test_index_real_acdc_dataset() -> None:
    """Verify all real ACDC patients and their required files are indexed."""
    indexer = AcdcDatasetIndexer(
        dataset_root=_DATASET_ROOT,
        info_parser=AcdcInfoParser(),
    )

    patient_cases = indexer.index()

    assert len(patient_cases) == 150
    assert len({patient_case.patient_id for patient_case in patient_cases}) == 150

    split_counts = Counter(
        patient_case.split_name for patient_case in patient_cases
    )

    assert split_counts == {
        "training": 100,
        "testing": 50,
    }

    assert patient_cases[0].patient_id == "patient001"
    assert patient_cases[-1].patient_id == "patient150"

    for patient_case in patient_cases:
        assert patient_case.patient_dir.is_dir()
        assert patient_case.cine_path.is_file()
        assert patient_case.ed_image_path.is_file()
        assert patient_case.ed_mask_path.is_file()
        assert patient_case.es_image_path.is_file()
        assert patient_case.es_mask_path.is_file()
