import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Final, cast

import pytest

from cardiac_segmentation.config.loader import AppConfigLoader
from cardiac_segmentation.data import AcdcDatasetInspectionRunner

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_CONFIG_PATH: Final[Path] = Path("configs/data.yaml")
_EXPECTED_PATIENT_COUNT: Final[int] = 150
_EXPECTED_PHASE_COUNT: Final[int] = 300


@pytest.mark.acdc
@pytest.mark.integration
def test_run_real_acdc_inspection_and_write_reports(
    tmp_path: Path,
) -> None:
    """Inspect the real ACDC dataset and write valid JSON and CSV reports."""
    config = AppConfigLoader(
        project_root=_PROJECT_ROOT
    ).load(_CONFIG_PATH)

    test_config = replace(
        config,
        inspection=replace(
            config.inspection,
            output_dir=tmp_path,
        ),
    )

    report = AcdcDatasetInspectionRunner(test_config).run()

    assert report.patient_count == _EXPECTED_PATIENT_COUNT
    assert report.phase_count == _EXPECTED_PHASE_COUNT
    assert test_config.inspection.report_path.is_file()
    assert test_config.inspection.summary_path.is_file()

    raw_payload: object = json.loads(
        test_config.inspection.report_path.read_text(
            encoding="utf-8"
        )
    )

    assert isinstance(raw_payload, dict)
    payload = cast(dict[str, object], raw_payload)

    assert payload["dataset_name"] == "ACDC"
    assert payload["patient_count"] == _EXPECTED_PATIENT_COUNT
    assert payload["phase_count"] == _EXPECTED_PHASE_COUNT
    assert payload["expected_labels"] == [0, 1, 2, 3]
    assert payload["observed_labels"] == [0, 1, 2, 3]

    patients = payload["patients"]

    assert isinstance(patients, list)
    assert len(patients) == _EXPECTED_PATIENT_COUNT

    with test_config.inspection.summary_path.open(
        encoding="utf-8",
        newline="",
    ) as summary_file:
        rows = list(csv.DictReader(summary_file))

    assert len(rows) == _EXPECTED_PHASE_COUNT
    assert {
        row["phase_name"]
        for row in rows
    } == {"ED", "ES"}
    assert "label_0_voxel_count" in rows[0]
    assert "label_3_voxel_count" in rows[0]
