from math import isfinite
from pathlib import Path
from typing import Any, Final, cast

import torch

from cardiac_segmentation.config import AppConfigLoader
from cardiac_segmentation.config.patient_level_resume_training_config import (
    PatientLevelResumeTrainingConfig,
)
from cardiac_segmentation.config.patient_level_training_config import (
    PatientLevelTrainingConfig,
)
from cardiac_segmentation.training import (
    PatientLevelResumeTrainingExperiment,
    PatientLevelTrainingExperiment,
    ResumedSegmentationTrainingHistory,
    SegmentationTrainingEpochRecord,
)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_APP_CONFIG_PATH: Final[Path] = Path("configs/data.yaml")
_PATIENT_COUNT: Final[int] = 4
_VALIDATION_FRACTION: Final[float] = 0.25
_BATCH_SIZE: Final[int] = 1
_NUM_WORKERS: Final[int] = 0
_PIN_MEMORY: Final[bool] = False
_RANDOM_SEED: Final[int] = 42
_BASE_CHANNELS: Final[int] = 2
_LEARNING_RATE: Final[float] = 0.001
_WEIGHT_DECAY: Final[float] = 0.00001
_DEVICE: Final[str] = "cpu"
_TRAINING_VOLUME_COUNT: Final[int] = 6
_VALIDATION_VOLUME_COUNT: Final[int] = 2


def test_resume_segmentation_training_from_real_checkpoint(
    tmp_path: Path,
) -> None:
    """Train one real epoch, resume for one epoch, and validate checkpointing."""
    app_config = AppConfigLoader(
        project_root=_PROJECT_ROOT,
    ).load(_APP_CONFIG_PATH)
    initial_config = PatientLevelTrainingConfig(
        patient_count=_PATIENT_COUNT,
        validation_fraction=_VALIDATION_FRACTION,
        epoch_count=1,
        batch_size=_BATCH_SIZE,
        num_workers=_NUM_WORKERS,
        pin_memory=_PIN_MEMORY,
        random_seed=_RANDOM_SEED,
        base_channels=_BASE_CHANNELS,
        learning_rate=_LEARNING_RATE,
        weight_decay=_WEIGHT_DECAY,
        device=_DEVICE,
        checkpoint_path=tmp_path / "initial.pt",
    )
    initial_history = PatientLevelTrainingExperiment(
        app_config=app_config,
        training_config=initial_config,
    ).run()
    initial_checkpoint = _load_checkpoint(initial_config.checkpoint_path)
    resume_config = PatientLevelResumeTrainingConfig(
        patient_count=_PATIENT_COUNT,
        validation_fraction=_VALIDATION_FRACTION,
        final_epoch_number=2,
        batch_size=_BATCH_SIZE,
        num_workers=_NUM_WORKERS,
        pin_memory=_PIN_MEMORY,
        random_seed=_RANDOM_SEED,
        base_channels=_BASE_CHANNELS,
        learning_rate=_LEARNING_RATE,
        weight_decay=_WEIGHT_DECAY,
        device=_DEVICE,
        resume_checkpoint_path=initial_config.checkpoint_path,
        checkpoint_path=tmp_path / "resumed.pt",
    )
    resume_experiment = PatientLevelResumeTrainingExperiment(
        app_config=app_config,
        resume_config=resume_config,
    )
    callback_records: list[SegmentationTrainingEpochRecord] = []

    history = resume_experiment.run(epoch_callback=callback_records.append)

    assert initial_history.best_epoch_number == 1
    assert initial_checkpoint["epoch_number"] == 1
    assert len(history.epoch_records) == 1
    assert len(callback_records) == 1
    assert callback_records[0].epoch_number == 2
    assert history.resumed_from_epoch_number == 1
    assert history.final_epoch_number == 2
    assert not set(resume_experiment.training_patient_ids) & set(
        resume_experiment.validation_patient_ids
    )
    assert resume_experiment.training_volume_count == _TRAINING_VOLUME_COUNT
    assert resume_experiment.validation_volume_count == _VALIDATION_VOLUME_COUNT

    _assert_history_matches_contract(history)
    _assert_checkpoint_matches_contract(
        checkpoint_path=resume_config.checkpoint_path,
        history=history,
    )
    assert not (tmp_path / ".resumed.pt.tmp").exists()


def _assert_history_matches_contract(
    history: ResumedSegmentationTrainingHistory,
) -> None:
    """Verify resumed epoch metrics, timings, and best epoch."""
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
    assert history.best_epoch_number in {1, 2}


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
    history: ResumedSegmentationTrainingHistory,
) -> None:
    """Verify the resumed checkpoint payload."""
    assert checkpoint_path.is_file()
    checkpoint = _load_checkpoint(checkpoint_path)

    assert checkpoint["epoch_number"] == history.best_epoch_number
    assert checkpoint["model_state_dict"]
    assert checkpoint["optimizer_state_dict"]


def _load_checkpoint(
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Load a checkpoint on CPU for validation."""
    return cast(
        dict[str, Any],
        torch.load(
            checkpoint_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        ),
    )
