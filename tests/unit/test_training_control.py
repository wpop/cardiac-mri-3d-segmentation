from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from cardiac_segmentation.metrics import MulticlassDiceMetricResult
from cardiac_segmentation.training import (
    EarlyStoppingMonitor,
    SegmentationEpochResult,
    SegmentationTrainer,
    SegmentationTrainingCheckpoint,
)
from cardiac_segmentation.training.segmentation_training_checkpoint_loader import (
    SegmentationTrainingCheckpointLoader,
)


class _ScalarMetricEpochRunner:
    """Return deterministic epoch results from scalar validation Dice values."""

    def __init__(
        self,
        validation_mean_dice_values: tuple[float, ...],
    ) -> None:
        self._validation_mean_dice_values = validation_mean_dice_values
        self._validation_epoch_index = 0

    def train_epoch(
        self,
        _training_loader: DataLoader[dict[str, Tensor | str]],
    ) -> SegmentationEpochResult:
        """Return a fixed training result."""
        return _epoch_result(mean_dice=0.25)

    def validate_epoch(
        self,
        _validation_loader: DataLoader[dict[str, Tensor | str]],
    ) -> SegmentationEpochResult:
        """Return the next configured validation result."""
        mean_dice = self._validation_mean_dice_values[self._validation_epoch_index]
        self._validation_epoch_index += 1
        return _epoch_result(mean_dice=mean_dice)


def test_scheduler_reduces_learning_rate_after_configured_plateau(
    tmp_path: Path,
) -> None:
    """Verify ReduceLROnPlateau lowers LR after validation Dice plateaus."""
    trainer = _create_trainer(
        validation_mean_dice_values=(0.5, 0.5, 0.5),
        lr_scheduler_patience=1,
        early_stopping_patience=10,
    )

    history = trainer.fit(
        training_loader=_empty_loader(),
        validation_loader=_empty_loader(),
        epoch_count=3,
        checkpoint_path=tmp_path / "plateau.pt",
    )

    assert tuple(record.learning_rate for record in history.epoch_records) == pytest.approx(
        (0.001, 0.001, 0.0005)
    )
    assert tuple(record.learning_rate_changed for record in history.epoch_records) == (
        False,
        False,
        True,
    )


def test_early_stopping_does_not_trigger_while_meaningful_improvement_occurs() -> None:
    """Verify patience resets when improvements exceed the configured minimum."""
    monitor = EarlyStoppingMonitor(
        patience=2,
        minimum_improvement=0.001,
    )

    should_stop = [monitor.step(metric) for metric in (0.50, 0.502, 0.504)]

    assert should_stop == [False, False, False]
    assert monitor.epochs_without_improvement == 0


def test_early_stopping_triggers_after_configured_patience() -> None:
    """Verify early stopping triggers after repeated non-meaningful changes."""
    monitor = EarlyStoppingMonitor(
        patience=2,
        minimum_improvement=0.001,
    )

    should_stop = [monitor.step(metric) for metric in (0.50, 0.5005, 0.5007)]

    assert should_stop == [False, False, True]
    assert monitor.epochs_without_improvement == 2


def test_early_stopping_state_can_be_serialized_and_restored() -> None:
    """Verify early-stopping state round-trips through its state dict."""
    monitor = EarlyStoppingMonitor(
        patience=2,
        minimum_improvement=0.001,
    )
    monitor.step(0.50)
    monitor.step(0.5005)
    restored_monitor = EarlyStoppingMonitor(
        patience=99,
        minimum_improvement=0.0,
    )

    restored_monitor.load_state_dict(monitor.state_dict())

    assert restored_monitor.state_dict() == monitor.state_dict()
    assert restored_monitor.step(0.5007) is True


def test_scheduler_state_is_checkpointed_and_restored(
    tmp_path: Path,
) -> None:
    """Verify scheduler state is saved and loaded with a best checkpoint."""
    trainer = _create_trainer(
        validation_mean_dice_values=(0.5, 0.5, 0.5, 0.6),
        lr_scheduler_patience=1,
        early_stopping_patience=10,
    )
    checkpoint_path = tmp_path / "scheduler_state.pt"

    trainer.fit(
        training_loader=_empty_loader(),
        validation_loader=_empty_loader(),
        epoch_count=4,
        checkpoint_path=checkpoint_path,
    )
    checkpoint = _load_checkpoint(checkpoint_path)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )

    loaded_checkpoint = SegmentationTrainingCheckpointLoader().load_into(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
        scheduler=scheduler,
    )

    assert "scheduler_state_dict" in checkpoint
    assert "early_stopping_state_dict" in checkpoint
    assert loaded_checkpoint.scheduler_state_dict == checkpoint["scheduler_state_dict"]
    assert loaded_checkpoint.early_stopping_state_dict == checkpoint[
        "early_stopping_state_dict"
    ]
    assert scheduler.state_dict() == checkpoint["scheduler_state_dict"]


def test_older_checkpoint_without_control_fields_remains_loadable(
    tmp_path: Path,
) -> None:
    """Verify historical checkpoints without scheduler fields still load."""
    trainer = _create_trainer(
        validation_mean_dice_values=(0.5,),
        lr_scheduler_patience=1,
        early_stopping_patience=10,
    )
    checkpoint_path = tmp_path / "new_checkpoint.pt"
    old_checkpoint_path = tmp_path / "old_checkpoint.pt"
    trainer.fit(
        training_loader=_empty_loader(),
        validation_loader=_empty_loader(),
        epoch_count=1,
        checkpoint_path=checkpoint_path,
    )
    checkpoint = _load_checkpoint(checkpoint_path)
    checkpoint.pop("scheduler_state_dict")
    checkpoint.pop("early_stopping_state_dict")
    torch.save(checkpoint, old_checkpoint_path)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )

    loaded_checkpoint = SegmentationTrainingCheckpointLoader().load_into(
        checkpoint_path=old_checkpoint_path,
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
        scheduler=scheduler,
    )

    assert loaded_checkpoint.scheduler_state_dict is None
    assert loaded_checkpoint.early_stopping_state_dict is None


def test_resume_fit_restores_control_state_from_new_checkpoint(
    tmp_path: Path,
) -> None:
    """Verify resumed training honors saved scheduler and early-stopping state."""
    checkpoint_path = tmp_path / "checkpoint.pt"
    resumed_checkpoint_path = tmp_path / "resumed.pt"
    torch.save({"checkpoint": "placeholder"}, checkpoint_path)
    scheduler_state = _scheduler_state_after_metrics(
        metrics=(0.7, 0.699),
        lr_scheduler_patience=1,
    )
    checkpoint = _checkpoint(
        checkpoint_path=checkpoint_path,
        epoch_number=5,
        validation_mean_dice=0.7,
        scheduler_state_dict=scheduler_state,
        early_stopping_state_dict={
            "patience": 2,
            "minimum_improvement": 0.001,
            "best_metric": 0.7,
            "epochs_without_improvement": 1,
        },
    )
    trainer = _create_trainer(
        validation_mean_dice_values=(0.698, 0.697),
        lr_scheduler_patience=1,
        early_stopping_patience=99,
    )

    history = trainer.resume_fit(
        training_loader=_empty_loader(),
        validation_loader=_empty_loader(),
        checkpoint=checkpoint,
        final_epoch_number=7,
        checkpoint_path=resumed_checkpoint_path,
    )

    assert tuple(record.epoch_number for record in history.epoch_records) == (6,)
    assert history.final_epoch_number == 6
    assert history.final_record.early_stopping_triggered is True
    assert history.final_record.learning_rate_changed is True
    assert history.final_record.learning_rate == pytest.approx(0.0005)


def test_resume_fit_seeds_control_state_from_old_checkpoint_metadata(
    tmp_path: Path,
) -> None:
    """Verify old checkpoints seed scheduler and early stopping from metadata."""
    checkpoint_path = tmp_path / "old_checkpoint.pt"
    resumed_checkpoint_path = tmp_path / "old_resumed.pt"
    torch.save({"checkpoint": "placeholder"}, checkpoint_path)
    checkpoint = _checkpoint(
        checkpoint_path=checkpoint_path,
        epoch_number=5,
        validation_mean_dice=0.7,
        scheduler_state_dict=None,
        early_stopping_state_dict=None,
    )
    trainer = _create_trainer(
        validation_mean_dice_values=(0.699, 0.698),
        lr_scheduler_patience=0,
        early_stopping_patience=1,
    )

    history = trainer.resume_fit(
        training_loader=_empty_loader(),
        validation_loader=_empty_loader(),
        checkpoint=checkpoint,
        final_epoch_number=7,
        checkpoint_path=resumed_checkpoint_path,
    )

    assert tuple(record.epoch_number for record in history.epoch_records) == (6,)
    assert history.final_epoch_number == 6
    assert history.final_record.early_stopping_triggered is True
    assert history.final_record.learning_rate_changed is True
    assert history.final_record.learning_rate == pytest.approx(0.0005)


def _create_trainer(
    *,
    validation_mean_dice_values: tuple[float, ...],
    lr_scheduler_patience: int,
    early_stopping_patience: int,
) -> SegmentationTrainer:
    """Create a trainer with deterministic scalar validation metrics."""
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=lr_scheduler_patience,
    )
    early_stopping_monitor = EarlyStoppingMonitor(
        patience=early_stopping_patience,
        minimum_improvement=0.001,
    )

    return SegmentationTrainer(
        model=model,
        optimizer=optimizer,
        epoch_runner=cast(
            Any,
            _ScalarMetricEpochRunner(validation_mean_dice_values),
        ),
        scheduler=scheduler,
        early_stopping_monitor=early_stopping_monitor,
    )


def _scheduler_state_after_metrics(
    *,
    metrics: tuple[float, ...],
    lr_scheduler_patience: int,
) -> dict[str, Any]:
    """Build scheduler state after stepping through deterministic metrics."""
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=lr_scheduler_patience,
    )

    for metric in metrics:
        scheduler.step(metric)

    return cast(dict[str, Any], scheduler.state_dict())


def _checkpoint(
    *,
    checkpoint_path: Path,
    epoch_number: int,
    validation_mean_dice: float,
    scheduler_state_dict: dict[str, Any] | None,
    early_stopping_state_dict: dict[str, Any] | None,
) -> SegmentationTrainingCheckpoint:
    """Build checkpoint metadata for resumed trainer control tests."""
    return SegmentationTrainingCheckpoint(
        checkpoint_path=checkpoint_path,
        format_version=1,
        epoch_number=epoch_number,
        training_average_loss=1.0,
        validation_average_loss=1.0,
        training_mean_dice=0.25,
        validation_mean_dice=validation_mean_dice,
        included_class_indices=(1, 2, 3),
        validation_per_class_dice=(
            validation_mean_dice,
            validation_mean_dice,
            validation_mean_dice,
        ),
        scheduler_state_dict=scheduler_state_dict,
        early_stopping_state_dict=early_stopping_state_dict,
    )


def _epoch_result(
    *,
    mean_dice: float,
) -> SegmentationEpochResult:
    """Build a deterministic epoch result for control-flow tests."""
    return SegmentationEpochResult(
        average_loss=1.0,
        dice_result=MulticlassDiceMetricResult(
            included_class_indices=(1, 2, 3),
            per_class_dice=(mean_dice, mean_dice, mean_dice),
            mean_dice=mean_dice,
            volume_count=1,
        ),
        batch_count=1,
        volume_count=1,
    )


def _empty_loader() -> DataLoader[dict[str, Tensor | str]]:
    """Return a typed placeholder DataLoader ignored by the fake runner."""
    return cast(DataLoader[dict[str, Tensor | str]], object())


def _load_checkpoint(
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Load a checkpoint mapping from disk."""
    return cast(
        dict[str, Any],
        torch.load(
            checkpoint_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        ),
    )
