import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from cardiac_segmentation.training.early_stopping_monitor import (
    EarlyStoppingMonitor,
)
from cardiac_segmentation.training.resumed_segmentation_training_history import (
    ResumedSegmentationTrainingHistory,
)
from cardiac_segmentation.training.segmentation_epoch_runner import (
    SegmentationEpochRunner,
)
from cardiac_segmentation.training.segmentation_training_checkpoint import (
    SegmentationTrainingCheckpoint,
)
from cardiac_segmentation.training.segmentation_training_epoch_record import (
    SegmentationTrainingEpochRecord,
)
from cardiac_segmentation.training.segmentation_training_history import (
    SegmentationTrainingHistory,
)


class SegmentationTrainer:
    """Coordinate multi-epoch segmentation training and best-checkpoint saving."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch_runner: SegmentationEpochRunner,
        scheduler: ReduceLROnPlateau,
        early_stopping_monitor: EarlyStoppingMonitor,
    ) -> None:
        """Initialize model, optimizer, scheduler, stop monitor, and epoch runner."""
        self._validate_model_and_optimizer(
            model=model,
            optimizer=optimizer,
        )

        self._model = model
        self._optimizer = optimizer
        self._epoch_runner = epoch_runner
        self._scheduler = scheduler
        self._early_stopping_monitor = early_stopping_monitor

    def fit(
        self,
        training_loader: DataLoader[dict[str, Tensor | str]],
        validation_loader: DataLoader[dict[str, Tensor | str]],
        epoch_count: int,
        checkpoint_path: Path,
        epoch_callback: Callable[[SegmentationTrainingEpochRecord], None] | None = None,
    ) -> SegmentationTrainingHistory:
        """Run training for a fixed number of epochs and save the best checkpoint."""
        self._validate_epoch_count(epoch_count)
        resolved_checkpoint_path = self._prepare_checkpoint_path(checkpoint_path)
        epoch_records: list[SegmentationTrainingEpochRecord] = []
        best_record: SegmentationTrainingEpochRecord | None = None

        for epoch_number in range(1, epoch_count + 1):
            training_start_time = time.perf_counter()
            training_result = self._epoch_runner.train_epoch(training_loader)
            training_duration_seconds = time.perf_counter() - training_start_time

            validation_start_time = time.perf_counter()
            validation_result = self._epoch_runner.validate_epoch(validation_loader)
            validation_duration_seconds = time.perf_counter() - validation_start_time
            validation_mean_dice = validation_result.dice_result.mean_dice
            learning_rate_before_scheduler = self._get_learning_rate()
            self._scheduler.step(validation_mean_dice)
            learning_rate = self._get_learning_rate()
            learning_rate_changed = learning_rate != learning_rate_before_scheduler
            early_stopping_triggered = self._early_stopping_monitor.step(
                validation_mean_dice
            )
            epoch_record = SegmentationTrainingEpochRecord(
                epoch_number=epoch_number,
                training_result=training_result,
                validation_result=validation_result,
                training_duration_seconds=training_duration_seconds,
                validation_duration_seconds=validation_duration_seconds,
                learning_rate=learning_rate,
                learning_rate_changed=learning_rate_changed,
                early_stopping_triggered=early_stopping_triggered,
            )
            epoch_records.append(epoch_record)

            if epoch_callback is not None:
                epoch_callback(epoch_record)

            if best_record is None or self._is_better_record(
                candidate_record=epoch_record,
                best_record=best_record,
            ):
                best_record = epoch_record
                self._save_checkpoint(
                    epoch_record=epoch_record,
                    checkpoint_path=resolved_checkpoint_path,
                )

            if early_stopping_triggered:
                break

        if best_record is None:
            raise RuntimeError("Training did not produce any epoch records.")

        return SegmentationTrainingHistory(
            epoch_records=tuple(epoch_records),
            best_epoch_number=best_record.epoch_number,
            checkpoint_path=resolved_checkpoint_path,
        )

    def resume_fit(  # noqa: PLR0913
        self,
        training_loader: DataLoader[dict[str, Tensor | str]],
        validation_loader: DataLoader[dict[str, Tensor | str]],
        checkpoint: SegmentationTrainingCheckpoint,
        final_epoch_number: int,
        checkpoint_path: Path,
        epoch_callback: Callable[[SegmentationTrainingEpochRecord], None] | None = None,
    ) -> ResumedSegmentationTrainingHistory:
        """Resume training from a loaded checkpoint until a final epoch number."""
        self._validate_resume_final_epoch_number(
            final_epoch_number=final_epoch_number,
            checkpoint=checkpoint,
        )
        resolved_checkpoint_path = self._prepare_checkpoint_path(checkpoint_path)

        if resolved_checkpoint_path != checkpoint.checkpoint_path:
            self._copy_checkpoint(
                source_path=checkpoint.checkpoint_path,
                destination_path=resolved_checkpoint_path,
            )

        epoch_records: list[SegmentationTrainingEpochRecord] = []
        best_epoch_number = checkpoint.epoch_number
        best_validation_mean_dice = checkpoint.validation_mean_dice
        best_validation_average_loss = checkpoint.validation_average_loss
        self._restore_control_state(checkpoint)

        for epoch_number in range(checkpoint.epoch_number + 1, final_epoch_number + 1):
            training_start_time = time.perf_counter()
            training_result = self._epoch_runner.train_epoch(training_loader)
            training_duration_seconds = time.perf_counter() - training_start_time

            validation_start_time = time.perf_counter()
            validation_result = self._epoch_runner.validate_epoch(validation_loader)
            validation_duration_seconds = time.perf_counter() - validation_start_time
            validation_mean_dice = validation_result.dice_result.mean_dice
            learning_rate_before_scheduler = self._get_learning_rate()
            self._scheduler.step(validation_mean_dice)
            learning_rate = self._get_learning_rate()
            learning_rate_changed = learning_rate != learning_rate_before_scheduler
            early_stopping_triggered = self._early_stopping_monitor.step(
                validation_mean_dice
            )
            epoch_record = SegmentationTrainingEpochRecord(
                epoch_number=epoch_number,
                training_result=training_result,
                validation_result=validation_result,
                training_duration_seconds=training_duration_seconds,
                validation_duration_seconds=validation_duration_seconds,
                learning_rate=learning_rate,
                learning_rate_changed=learning_rate_changed,
                early_stopping_triggered=early_stopping_triggered,
            )
            epoch_records.append(epoch_record)

            if epoch_callback is not None:
                epoch_callback(epoch_record)

            if self._is_better_than_checkpoint_metadata(
                candidate_record=epoch_record,
                best_validation_mean_dice=best_validation_mean_dice,
                best_validation_average_loss=best_validation_average_loss,
            ):
                best_epoch_number = epoch_record.epoch_number
                best_validation_mean_dice = epoch_record.validation_result.dice_result.mean_dice
                best_validation_average_loss = epoch_record.validation_result.average_loss
                self._save_checkpoint(
                    epoch_record=epoch_record,
                    checkpoint_path=resolved_checkpoint_path,
                )

            if early_stopping_triggered:
                break

        return ResumedSegmentationTrainingHistory(
            resumed_from_epoch_number=checkpoint.epoch_number,
            epoch_records=tuple(epoch_records),
            best_epoch_number=best_epoch_number,
            best_validation_mean_dice=best_validation_mean_dice,
            best_validation_average_loss=best_validation_average_loss,
            checkpoint_path=resolved_checkpoint_path,
        )

    @staticmethod
    def _validate_epoch_count(
        epoch_count: int,
    ) -> None:
        """Validate fixed training epoch count."""
        if isinstance(epoch_count, bool) or not isinstance(epoch_count, int) or epoch_count <= 0:
            raise ValueError("Epoch count must be a positive integer.")

    @staticmethod
    def _prepare_checkpoint_path(
        checkpoint_path: Path,
    ) -> Path:
        """Validate and prepare the best-checkpoint destination."""
        resolved_checkpoint_path = checkpoint_path.expanduser().resolve(strict=False)

        if resolved_checkpoint_path.is_dir():
            raise IsADirectoryError(
                f"Checkpoint path must not be an existing directory: {resolved_checkpoint_path}"
            )

        resolved_checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        return resolved_checkpoint_path

    @staticmethod
    def _validate_resume_final_epoch_number(
        final_epoch_number: int,
        checkpoint: SegmentationTrainingCheckpoint,
    ) -> None:
        """Validate the requested final epoch for resumed training."""
        if (
            isinstance(final_epoch_number, bool)
            or not isinstance(final_epoch_number, int)
        ):
            raise TypeError("Final epoch number must be an integer.")

        if final_epoch_number <= checkpoint.epoch_number:
            raise ValueError(
                "Final epoch number must be greater than the checkpoint epoch number."
            )

    @staticmethod
    def _is_better_record(
        candidate_record: SegmentationTrainingEpochRecord,
        best_record: SegmentationTrainingEpochRecord,
    ) -> bool:
        """Return whether the candidate should replace the current best record."""
        candidate_dice = candidate_record.validation_result.dice_result.mean_dice
        best_dice = best_record.validation_result.dice_result.mean_dice

        if candidate_dice > best_dice:
            return True

        if candidate_dice < best_dice:
            return False

        return (
            candidate_record.validation_result.average_loss
            < best_record.validation_result.average_loss
        )

    @staticmethod
    def _is_better_than_checkpoint_metadata(
        candidate_record: SegmentationTrainingEpochRecord,
        best_validation_mean_dice: float,
        best_validation_average_loss: float,
    ) -> bool:
        """Return whether a record improves over checkpoint metadata."""
        candidate_dice = candidate_record.validation_result.dice_result.mean_dice

        if candidate_dice > best_validation_mean_dice:
            return True

        if candidate_dice < best_validation_mean_dice:
            return False

        return candidate_record.validation_result.average_loss < best_validation_average_loss

    @staticmethod
    def _copy_checkpoint(
        source_path: Path,
        destination_path: Path,
    ) -> None:
        """Safely copy a checkpoint to a destination using a temporary file."""
        temporary_path = destination_path.with_name(
            f".{destination_path.name}.tmp"
        )

        try:
            shutil.copyfile(
                source_path,
                temporary_path,
            )
            os.replace(
                temporary_path,
                destination_path,
            )
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _save_checkpoint(
        self,
        epoch_record: SegmentationTrainingEpochRecord,
        checkpoint_path: Path,
    ) -> None:
        """Safely save the current best checkpoint."""
        temporary_path = checkpoint_path.with_name(
            f".{checkpoint_path.name}.tmp"
        )
        checkpoint_payload = self._build_checkpoint_payload(epoch_record)

        try:
            torch.save(
                checkpoint_payload,
                temporary_path,
            )
            os.replace(
                temporary_path,
                checkpoint_path,
            )
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _build_checkpoint_payload(
        self,
        epoch_record: SegmentationTrainingEpochRecord,
    ) -> dict[str, Any]:
        """Build the serializable checkpoint payload."""
        validation_dice_result = epoch_record.validation_result.dice_result

        return {
            "format_version": 1,
            "epoch_number": epoch_record.epoch_number,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "scheduler_state_dict": self._scheduler_state_dict(),
            "early_stopping_state_dict": self._early_stopping_monitor.state_dict(),
            "training_average_loss": epoch_record.training_result.average_loss,
            "validation_average_loss": epoch_record.validation_result.average_loss,
            "training_mean_dice": epoch_record.training_result.dice_result.mean_dice,
            "validation_mean_dice": validation_dice_result.mean_dice,
            "included_class_indices": validation_dice_result.included_class_indices,
            "validation_per_class_dice": validation_dice_result.per_class_dice,
        }

    def _restore_control_state(
        self,
        checkpoint: SegmentationTrainingCheckpoint,
    ) -> None:
        """Restore scheduler and early-stopping state or seed from old metadata."""
        if checkpoint.scheduler_state_dict is not None:
            scheduler_load_state_dict = cast(
                Callable[[dict[str, Any]], None],
                self._scheduler.load_state_dict,
            )
            scheduler_load_state_dict(checkpoint.scheduler_state_dict)
        else:
            self._scheduler.step(checkpoint.validation_mean_dice)

        if checkpoint.early_stopping_state_dict is not None:
            self._early_stopping_monitor.load_state_dict(
                checkpoint.early_stopping_state_dict
            )
        elif self._early_stopping_monitor.best_metric is None:
            self._early_stopping_monitor.best_metric = checkpoint.validation_mean_dice

    def _get_learning_rate(self) -> float:
        """Return the single learning rate used by all optimizer parameter groups."""
        learning_rates = {
            float(parameter_group["lr"])
            for parameter_group in self._optimizer.param_groups
        }

        if len(learning_rates) != 1:
            raise ValueError("All optimizer parameter groups must share one learning rate.")

        return next(iter(learning_rates))

    def _scheduler_state_dict(self) -> dict[str, Any]:
        """Return the scheduler state through a typed boundary."""
        scheduler_state_dict = cast(
            Callable[[], dict[str, Any]],
            self._scheduler.state_dict,
        )

        return scheduler_state_dict()

    @staticmethod
    def _validate_model_and_optimizer(
        model: nn.Module,
        optimizer: Optimizer,
    ) -> None:
        """Validate trainable model parameters and optimizer ownership."""
        model_parameters = tuple(model.parameters())
        trainable_model_parameters = tuple(
            parameter
            for parameter in model_parameters
            if parameter.requires_grad
        )

        if not trainable_model_parameters:
            raise ValueError("Model must expose at least one trainable parameter.")

        model_parameter_ids = {id(parameter) for parameter in model_parameters}
        optimizer_parameters = tuple(
            parameter
            for parameter_group in optimizer.param_groups
            for parameter in parameter_group["params"]
        )

        if not optimizer_parameters:
            raise ValueError("Optimizer must contain at least one parameter.")

        if any(id(parameter) not in model_parameter_ids for parameter in optimizer_parameters):
            raise ValueError("Optimizer parameters must belong to the supplied model.")
