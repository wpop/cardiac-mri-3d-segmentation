from collections import Counter
from pathlib import Path

import pytest

from cardiac_segmentation.data import AcdcInfoParser

_DATASET_ROOT = Path("data/raw/acdc")


@pytest.mark.acdc
@pytest.mark.integration
def test_parse_real_patient001_info() -> None:
    """Verify metadata parsed from the real patient001 Info.cfg file."""
    parser = AcdcInfoParser()
    info_path = _DATASET_ROOT / "training" / "patient001" / "Info.cfg"

    patient_info = parser.parse(info_path)

    assert patient_info.patient_id == "patient001"
    assert patient_info.ed_frame == 1
    assert patient_info.es_frame == 12
    assert patient_info.clinical_group == "DCM"
    assert patient_info.height_cm == pytest.approx(184.0)
    assert patient_info.frame_count == 30
    assert patient_info.weight_kg == pytest.approx(95.0)


@pytest.mark.acdc
@pytest.mark.integration
def test_parse_all_real_acdc_patient_info_files() -> None:
    """Verify all 150 real ACDC Info.cfg files and clinical-group counts."""
    parser = AcdcInfoParser()

    info_paths = sorted(
        path
        for split_name in ("training", "testing")
        for path in (_DATASET_ROOT / split_name).glob("patient???/Info.cfg")
    )

    patient_infos = [parser.parse(path) for path in info_paths]

    assert len(patient_infos) == 150
    assert len({info.patient_id for info in patient_infos}) == 150

    group_counts = Counter(
        info.clinical_group for info in patient_infos
    )

    assert group_counts == {
        "DCM": 30,
        "HCM": 30,
        "MINF": 30,
        "NOR": 30,
        "RV": 30,
    }

    assert min(info.ed_frame for info in patient_infos) == 1
    assert max(info.ed_frame for info in patient_infos) == 4
    assert min(info.es_frame for info in patient_infos) == 6
    assert max(info.es_frame for info in patient_infos) == 16
    assert min(info.frame_count for info in patient_infos) == 12
    assert max(info.frame_count for info in patient_infos) == 35
