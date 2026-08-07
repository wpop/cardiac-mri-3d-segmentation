from dataclasses import replace
from math import isfinite
from pathlib import Path
from typing import Any, Final, cast

import torch

from cardiac_segmentation.config import (
    AppConfigLoader,
    PatientLevelTrainingConfigLoader,
)
from cardiac_segmentation.training import (
    PatientLevelTrainingExperiment,
    SegmentationTrainingHistory,
)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_APP_CONFIG_PATH: Final[Path] = Path("configs/data.yaml")
_TRAINING_CONFIG_PATH: Final[Path] = Path("configs/patient_level_training.yaml")
_EPOCH_COUNT: Final[int] = 1
_TRAINING_PATIENT_COUNT: Final[int] = 3
_VALIDATION_PATIENT_COUNT: Final[int] = 1
_TRAINING_VOLUME_COUNT: Final[int] = 6
_VALIDATION_VOLUME_COUNT: Final[int] = 2


def test_patient_level_training_experiment_runs_on_real_acdc_patients(
    tmp_path: Path,
) -> None:
    """Run one CPU epoch using a real deterministic patient-level split."""
    app_config = AppConfigLoader(
        project_root=_PROJECT_ROOT,
    ).load(_APP_CONFIG_PATH)
    training_config = PatientLevelTrainingConfigLoader(
        project_root=_PROJECT_ROOT,
    ).load(_TRAINING_CONFIG_PATH)
    test_config = replace(
        training_config,
        patient_count=4,
        validation_fraction=0.25,
        epoch_count=_EPOCH_COUNT,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        base_channels=2,
        device="cpu",
        checkpoint_path=tmp_path / "patient_level_training.pt",
    )

    experiment = PatientLevelTrainingExperiment(
        app_config=app_config,
        training_config=test_config,
    )

    assert experiment.device == torch.device("cpu")
    assert len(experiment.training_patient_ids) == _TRAINING_PATIENT_COUNT
    assert len(experiment.validation_patient_ids) == _VALIDATION_PATIENT_COUNT
    assert not set(experiment.training_patient_ids) & set(experiment.validation_patient_ids)
    assert experiment.training_volume_count == _TRAINING_VOLUME_COUNT
    assert experiment.validation_volume_count == _VALIDATION_VOLUME_COUNT

    history = experiment.run()

    _assert_history_matches_contract(history)
    _assert_checkpoint_matches_contract(
        checkpoint_path=test_config.checkpoint_path,
        history=history,
    )
    assert not (tmp_path / ".patient_level_training.pt.tmp").exists()


def _assert_history_matches_contract(
    history: SegmentationTrainingHistory,
) -> None:
    """Verify the completed epoch, metrics, timings, and checkpoint path."""
    assert history.epoch_count == _EPOCH_COUNT
    assert history.best_epoch_number == 1
    assert history.checkpoint_path.is_file()
    record = history.final_record

    assert record.training_result.volume_count == _TRAINING_VOLUME_COUNT
    assert record.validation_result.volume_count == _VALIDATION_VOLUME_COUNT
    assert isfinite(record.training_result.average_loss)
    assert isfinite(record.validation_result.average_loss)
    assert record.training_result.average_loss > 0.0
    assert record.validation_result.average_loss > 0.0
    _assert_dice_values_match_contract(record.training_result.dice_result.per_class_dice)
    _assert_dice_values_match_contract(record.validation_result.dice_result.per_class_dice)
    assert isfinite(record.training_result.dice_result.mean_dice)
    assert isfinite(record.validation_result.dice_result.mean_dice)
    assert 0.0 <= record.training_result.dice_result.mean_dice <= 1.0
    assert 0.0 <= record.validation_result.dice_result.mean_dice <= 1.0
    assert isfinite(record.training_duration_seconds)
    assert isfinite(record.validation_duration_seconds)
    assert record.training_duration_seconds >= 0.0
    assert record.validation_duration_seconds >= 0.0


def _assert_dice_values_match_contract(
    dice_values: tuple[float, ...],
) -> None:
    """Verify every Dice value is finite and normalized."""
    assert len(dice_values) == 3

    for dice_value in dice_values:
        assert isfinite(dice_value)
        assert 0.0 <= dice_value <= 1.0


def _assert_checkpoint_matches_contract(
    checkpoint_path: Path,
    history: SegmentationTrainingHistory,
) -> None:
    """Verify the saved checkpoint payload."""
    assert checkpoint_path.is_file()
    checkpoint = cast(
        dict[str, Any],
        torch.load(
            checkpoint_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        ),
    )

    assert checkpoint["epoch_number"] == history.best_epoch_number
    assert checkpoint["model_state_dict"]
    assert checkpoint["optimizer_state_dict"]
