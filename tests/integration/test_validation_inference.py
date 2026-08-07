import csv
import json
from dataclasses import replace
from math import isfinite
from pathlib import Path
from typing import Any, Final

import torch

from cardiac_segmentation.config import (
    AppConfigLoader,
    PatientLevelTrainingConfigLoader,
    ValidationInferenceConfig,
)
from cardiac_segmentation.evaluation import ValidationInferenceExperiment
from cardiac_segmentation.training import PatientLevelTrainingExperiment

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_APP_CONFIG_PATH: Final[Path] = Path("configs/data.yaml")
_TRAINING_CONFIG_PATH: Final[Path] = Path("configs/patient_level_training.yaml")
_PATIENT_COUNT: Final[int] = 4
_VALIDATION_FRACTION: Final[float] = 0.25
_VALIDATION_PATIENT_COUNT: Final[int] = 1
_VALIDATION_VOLUME_COUNT: Final[int] = 2


def test_validation_inference_uses_real_acdc_validation_split(
    tmp_path: Path,
) -> None:
    """Train a tiny real checkpoint, then run validation inference on real data."""
    app_config = AppConfigLoader(
        project_root=_PROJECT_ROOT,
    ).load(_APP_CONFIG_PATH)
    training_config = PatientLevelTrainingConfigLoader(
        project_root=_PROJECT_ROOT,
    ).load(_TRAINING_CONFIG_PATH)
    test_training_config = replace(
        training_config,
        patient_count=_PATIENT_COUNT,
        validation_fraction=_VALIDATION_FRACTION,
        epoch_count=1,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        base_channels=2,
        device="cpu",
        checkpoint_path=tmp_path / "validation_inference_checkpoint.pt",
    )
    training_experiment = PatientLevelTrainingExperiment(
        app_config=app_config,
        training_config=test_training_config,
    )
    training_history = training_experiment.run()

    assert training_history.checkpoint_path.is_file()

    inference_config = ValidationInferenceConfig(
        patient_count=_PATIENT_COUNT,
        validation_fraction=_VALIDATION_FRACTION,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        random_seed=test_training_config.random_seed,
        base_channels=2,
        device="cpu",
        checkpoint_path=test_training_config.checkpoint_path,
        csv_report_path=tmp_path / "validation_inference.csv",
        json_report_path=tmp_path / "validation_inference_summary.json",
        visualization_dir=tmp_path / "visualizations",
    )
    experiment = ValidationInferenceExperiment(
        app_config=app_config,
        inference_config=inference_config,
    )

    assert experiment.device == torch.device("cpu")
    assert len(experiment.validation_patient_ids) == _VALIDATION_PATIENT_COUNT
    assert experiment.validation_volume_count == _VALIDATION_VOLUME_COUNT

    report = experiment.run()

    assert report.checkpoint_epoch_number == training_history.best_epoch_number
    assert len(report.case_results) == _VALIDATION_VOLUME_COUNT

    for result in report.case_results:
        assert result.patient_id
        assert result.volume_id
        _assert_normalized_dice(result.rv_dice)
        _assert_normalized_dice(result.myocardium_dice)
        _assert_normalized_dice(result.lv_dice)
        _assert_normalized_dice(result.mean_foreground_dice)

    _assert_csv_matches_contract(inference_config.csv_report_path)
    _assert_json_matches_contract(inference_config.json_report_path)
    _assert_visualizations_match_contract(report.visualization_paths)


def _assert_normalized_dice(
    dice_value: float,
) -> None:
    """Verify Dice is finite and inside [0, 1]."""
    assert isfinite(dice_value)
    assert 0.0 <= dice_value <= 1.0


def _assert_csv_matches_contract(
    csv_report_path: Path,
) -> None:
    """Verify CSV exists and contains exactly two data rows."""
    assert csv_report_path.is_file()

    with csv_report_path.open(encoding="utf-8", newline="") as csv_file:
        rows = tuple(csv.DictReader(csv_file))

    assert len(rows) == _VALIDATION_VOLUME_COUNT


def _assert_json_matches_contract(
    json_report_path: Path,
) -> None:
    """Verify JSON exists and contains valid aggregate statistics."""
    assert json_report_path.is_file()
    summary = json.loads(json_report_path.read_text(encoding="utf-8"))

    assert summary["validation_patient_count"] == _VALIDATION_PATIENT_COUNT
    assert summary["validation_volume_count"] == _VALIDATION_VOLUME_COUNT
    _assert_metric_summary(summary["mean_foreground_dice"])

    per_class = summary["per_class"]
    _assert_metric_summary(per_class["rv"])
    _assert_metric_summary(per_class["myocardium"])
    _assert_metric_summary(per_class["lv"])
    assert summary["worst_volume_identifier"]
    assert summary["middle_volume_identifier"]
    assert summary["best_volume_identifier"]


def _assert_metric_summary(
    metric_summary: dict[str, Any],
) -> None:
    """Verify an aggregate metric summary has normalized finite values."""
    assert set(metric_summary) == {"mean", "median", "minimum", "maximum"}

    for value in metric_summary.values():
        _assert_normalized_dice(value)


def _assert_visualizations_match_contract(
    visualization_paths: tuple[Path, Path, Path],
) -> None:
    """Verify exactly three non-empty PNG visualizations were written."""
    assert len(visualization_paths) == 3
    assert len(set(visualization_paths)) == 3

    for path in visualization_paths:
        assert path.is_file()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0
