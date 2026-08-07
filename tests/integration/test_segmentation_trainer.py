import random
from pathlib import Path
from typing import Any, Final, cast

import pytest
import torch

from cardiac_segmentation.config.app_config import AppConfig
from cardiac_segmentation.config.loader import AppConfigLoader
from cardiac_segmentation.data import (
    AcdcDataLoaderFactory,
    AcdcDataLoaders,
    AcdcDatasetIndexer,
    AcdcInfoParser,
    AcdcPatientCase,
)
from cardiac_segmentation.losses import CrossEntropyDiceLoss3D
from cardiac_segmentation.models import CompactUNet3D
from cardiac_segmentation.training import (
    EarlyStoppingMonitor,
    SegmentationEpochRunner,
    SegmentationTrainer,
    SegmentationTrainingEpochRecord,
    SegmentationTrainingHistory,
)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_CONFIG_PATH: Final[Path] = Path("configs/data.yaml")
_RANDOM_SEED: Final[int] = 42
_VALIDATION_FRACTION: Final[float] = 0.5
_BATCH_SIZE: Final[int] = 1
_NUM_WORKERS: Final[int] = 0
_PIN_MEMORY: Final[bool] = False
_TEST_BASE_CHANNELS: Final[int] = 2
_LEARNING_RATE: Final[float] = 1e-3
_WEIGHT_DECAY: Final[float] = 1e-5
_EPOCH_COUNT: Final[int] = 2


@pytest.mark.acdc
@pytest.mark.integration
def test_segmentation_trainer_records_history_and_saves_best_checkpoint(
    tmp_path: Path,
) -> None:
    """Train for two real-data epochs and save the best checkpoint."""
    random.seed(_RANDOM_SEED)
    torch.manual_seed(_RANDOM_SEED)
    config = AppConfigLoader(
        project_root=_PROJECT_ROOT,
    ).load(_CONFIG_PATH)
    data_loaders = _create_data_loaders(config)
    model = CompactUNet3D(
        in_channels=1,
        num_classes=len(config.validation.expected_labels),
        base_channels=_TEST_BASE_CHANNELS,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=_LEARNING_RATE,
        weight_decay=_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8,
    )
    early_stopping_monitor = EarlyStoppingMonitor(
        patience=20,
        minimum_improvement=0.001,
    )
    epoch_runner = SegmentationEpochRunner(
        model=model,
        loss_function=CrossEntropyDiceLoss3D(
            num_classes=len(config.validation.expected_labels),
            cross_entropy_weight=0.5,
            dice_weight=0.5,
            include_background_in_dice=False,
        ),
        optimizer=optimizer,
        num_classes=len(config.validation.expected_labels),
        device=torch.device("cpu"),
        include_background_in_dice=False,
    )
    trainer = SegmentationTrainer(
        model=model,
        optimizer=optimizer,
        epoch_runner=epoch_runner,
        scheduler=scheduler,
        early_stopping_monitor=early_stopping_monitor,
    )
    checkpoint_path = tmp_path / "best_segmentation_checkpoint.pt"

    history = trainer.fit(
        training_loader=data_loaders.training_loader,
        validation_loader=data_loaders.validation_loader,
        epoch_count=_EPOCH_COUNT,
        checkpoint_path=checkpoint_path,
    )

    _assert_history_matches_contract(history)
    _assert_checkpoint_matches_contract(
        checkpoint_path=checkpoint_path,
        history=history,
    )
    _assert_best_policy_matches_history(history)
    assert not (tmp_path / ".best_segmentation_checkpoint.pt.tmp").exists()


def _create_data_loaders(
    config: AppConfig,
) -> AcdcDataLoaders:
    """Create real ACDC train and validation DataLoaders from two patients."""
    patient_cases = AcdcDatasetIndexer(
        dataset_root=config.dataset.root_dir,
        info_parser=AcdcInfoParser(),
    ).index()
    selected_cases = tuple(
        patient_case
        for patient_case in patient_cases
        if patient_case.split_name == "training"
    )[:2]

    return AcdcDataLoaderFactory(
        preprocessing_config=config.preprocessing,
        validation_config=config.validation,
        validation_fraction=_VALIDATION_FRACTION,
        random_seed=_RANDOM_SEED,
        batch_size=_BATCH_SIZE,
        num_workers=_NUM_WORKERS,
        pin_memory=_PIN_MEMORY,
    ).create(selected_cases)


def _assert_history_matches_contract(
    history: SegmentationTrainingHistory,
) -> None:
    """Verify training history records, final record, and best record."""
    assert history.epoch_count == _EPOCH_COUNT
    assert tuple(record.epoch_number for record in history.epoch_records) == (1, 2)
    assert history.final_record.epoch_number == 2
    assert history.best_epoch_number in {1, 2}
    assert history.best_record.epoch_number == history.best_epoch_number
    assert history.checkpoint_path.is_file()

    for record in history.epoch_records:
        _assert_epoch_record_matches_contract(record)


def _assert_epoch_record_matches_contract(
    record: SegmentationTrainingEpochRecord,
) -> None:
    """Verify one training epoch history record."""
    assert record.training_result.batch_count == 2
    assert record.training_result.volume_count == 2
    assert record.validation_result.batch_count == 2
    assert record.validation_result.volume_count == 2
    assert record.training_result.average_loss > 0.0
    assert record.validation_result.average_loss > 0.0
    assert all(
        0.0 <= value <= 1.0
        for value in record.training_result.dice_result.per_class_dice
    )
    assert all(
        0.0 <= value <= 1.0
        for value in record.validation_result.dice_result.per_class_dice
    )
    assert 0.0 <= record.training_result.dice_result.mean_dice <= 1.0
    assert 0.0 <= record.validation_result.dice_result.mean_dice <= 1.0
    assert record.training_duration_seconds >= 0.0
    assert record.validation_duration_seconds >= 0.0
    assert record.learning_rate == pytest.approx(_LEARNING_RATE)
    assert record.learning_rate_changed is False
    assert record.early_stopping_triggered is False


def _assert_checkpoint_matches_contract(
    checkpoint_path: Path,
    history: SegmentationTrainingHistory,
) -> None:
    """Verify the saved checkpoint payload."""
    checkpoint = cast(
        dict[str, Any],
        torch.load(
            checkpoint_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        ),
    )
    expected_keys = {
        "format_version",
        "epoch_number",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "early_stopping_state_dict",
        "training_average_loss",
        "validation_average_loss",
        "training_mean_dice",
        "validation_mean_dice",
        "included_class_indices",
        "validation_per_class_dice",
    }

    assert set(checkpoint) == expected_keys
    assert checkpoint["format_version"] == 1
    assert checkpoint["epoch_number"] == history.best_epoch_number
    assert checkpoint["model_state_dict"]
    assert checkpoint["optimizer_state_dict"]
    assert checkpoint["validation_mean_dice"] == pytest.approx(
        history.best_record.validation_result.dice_result.mean_dice
    )
    assert checkpoint["validation_average_loss"] == pytest.approx(
        history.best_record.validation_result.average_loss
    )
    assert checkpoint["included_class_indices"] == (1, 2, 3)
    assert len(checkpoint["validation_per_class_dice"]) == 3


def _assert_best_policy_matches_history(
    history: SegmentationTrainingHistory,
) -> None:
    """Verify best epoch selection against validation Dice and loss."""
    expected_best_record = history.epoch_records[0]

    for candidate_record in history.epoch_records[1:]:
        candidate_dice = candidate_record.validation_result.dice_result.mean_dice
        best_dice = expected_best_record.validation_result.dice_result.mean_dice

        if candidate_dice > best_dice or (
            candidate_dice == best_dice
            and candidate_record.validation_result.average_loss
            < expected_best_record.validation_result.average_loss
        ):
            expected_best_record = candidate_record

    assert history.best_epoch_number == expected_best_record.epoch_number


def _patient_ids(
    patient_cases: tuple[AcdcPatientCase, ...],
) -> tuple[str, ...]:
    """Return patient identifiers for reporting-friendly debugging."""
    return tuple(patient_case.patient_id for patient_case in patient_cases)
