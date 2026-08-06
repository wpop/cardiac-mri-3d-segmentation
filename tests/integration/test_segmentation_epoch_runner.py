import random
from collections.abc import Sized
from pathlib import Path
from typing import Final

import pytest
import torch
from torch import Tensor

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
from cardiac_segmentation.training import SegmentationEpochResult, SegmentationEpochRunner

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


@pytest.mark.acdc
@pytest.mark.integration
def test_segmentation_epoch_runner_trains_and_validates_on_real_acdc_volumes() -> None:
    """Run one training and one validation epoch on real ACDC volumes."""
    random.seed(_RANDOM_SEED)
    torch.manual_seed(_RANDOM_SEED)
    config = AppConfigLoader(
        project_root=_PROJECT_ROOT,
    ).load(_CONFIG_PATH)
    selected_cases = _select_first_training_cases(config)
    data_loaders = AcdcDataLoaderFactory(
        preprocessing_config=config.preprocessing,
        validation_config=config.validation,
        validation_fraction=_VALIDATION_FRACTION,
        random_seed=_RANDOM_SEED,
        batch_size=_BATCH_SIZE,
        num_workers=_NUM_WORKERS,
        pin_memory=_PIN_MEMORY,
    ).create(selected_cases)
    device = torch.device("cpu")
    model = CompactUNet3D(
        in_channels=1,
        num_classes=len(config.validation.expected_labels),
        base_channels=_TEST_BASE_CHANNELS,
    ).to(device)
    loss_function = CrossEntropyDiceLoss3D(
        num_classes=len(config.validation.expected_labels),
        cross_entropy_weight=0.5,
        dice_weight=0.5,
        include_background_in_dice=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=_LEARNING_RATE,
        weight_decay=_WEIGHT_DECAY,
    )
    runner = SegmentationEpochRunner(
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        num_classes=len(config.validation.expected_labels),
        device=device,
        include_background_in_dice=False,
    )
    parameters_before_training = _clone_trainable_parameters(model)

    training_result = runner.train_epoch(data_loaders.training_loader)

    _assert_model_mode(
        model=model,
        expected_training=True,
    )
    _assert_epoch_result_matches_contract(training_result)
    assert _parameters_changed(
        before_parameters=parameters_before_training,
        after_parameters=_clone_trainable_parameters(model),
    )
    _assert_parameters_are_finite(model)
    parameters_after_training = _clone_all_parameters(model)
    gradients_after_training = _clone_existing_gradients(model)

    validation_result = runner.validate_epoch(data_loaders.validation_loader)

    _assert_model_mode(
        model=model,
        expected_training=False,
    )
    _assert_epoch_result_matches_contract(validation_result)
    _assert_parameters_unchanged(
        before_parameters=parameters_after_training,
        after_parameters=_clone_all_parameters(model),
    )
    _assert_gradients_unchanged(
        before_gradients=gradients_after_training,
        after_gradients=_clone_existing_gradients(model),
    )
    _assert_parameters_are_finite(model)
    _assert_data_loader_split_matches_contract(data_loaders)


def _select_first_training_cases(
    config: AppConfig,
) -> tuple[AcdcPatientCase, ...]:
    """Return the first two real training patient cases."""
    patient_cases = AcdcDatasetIndexer(
        dataset_root=config.dataset.root_dir,
        info_parser=AcdcInfoParser(),
    ).index()

    return tuple(
        patient_case
        for patient_case in patient_cases
        if patient_case.split_name == "training"
    )[:2]


def _assert_epoch_result_matches_contract(
    result: SegmentationEpochResult,
) -> None:
    """Verify one epoch's aggregate loss and Dice values."""
    assert result.batch_count == 2
    assert result.volume_count == 2
    assert result.dice_result.volume_count == 2
    assert result.average_loss > 0.0
    assert result.dice_result.included_class_indices == (1, 2, 3)
    assert all(0.0 <= dice_value <= 1.0 for dice_value in result.dice_result.per_class_dice)
    assert 0.0 <= result.dice_result.mean_dice <= 1.0


def _assert_data_loader_split_matches_contract(
    data_loaders: AcdcDataLoaders,
) -> None:
    """Verify deterministic train/validation split sizes."""
    training_dataset = _require_sized(data_loaders.training_loader.dataset)
    validation_dataset = _require_sized(data_loaders.validation_loader.dataset)

    assert len(data_loaders.patient_split.training_cases) == 1
    assert len(data_loaders.patient_split.validation_cases) == 1
    assert len(training_dataset) == 2
    assert len(validation_dataset) == 2


def _clone_trainable_parameters(
    model: CompactUNet3D,
) -> tuple[Tensor, ...]:
    """Clone trainable model parameters."""
    return tuple(
        parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _clone_all_parameters(
    model: CompactUNet3D,
) -> tuple[Tensor, ...]:
    """Clone all model parameters."""
    return tuple(parameter.detach().clone() for parameter in model.parameters())


def _clone_existing_gradients(
    model: CompactUNet3D,
) -> tuple[Tensor | None, ...]:
    """Clone current model gradients while preserving missing-gradient positions."""
    return tuple(
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in model.parameters()
    )


def _parameters_changed(
    before_parameters: tuple[Tensor, ...],
    after_parameters: tuple[Tensor, ...],
) -> bool:
    """Return whether any parameter changed."""
    return any(
        not torch.equal(before_parameter, after_parameter)
        for before_parameter, after_parameter in zip(
            before_parameters,
            after_parameters,
            strict=True,
        )
    )


def _assert_parameters_unchanged(
    before_parameters: tuple[Tensor, ...],
    after_parameters: tuple[Tensor, ...],
) -> None:
    """Verify every parameter is unchanged."""
    assert len(before_parameters) == len(after_parameters)
    assert all(
        torch.equal(before_parameter, after_parameter)
        for before_parameter, after_parameter in zip(
            before_parameters,
            after_parameters,
            strict=True,
        )
    )


def _assert_gradients_unchanged(
    before_gradients: tuple[Tensor | None, ...],
    after_gradients: tuple[Tensor | None, ...],
) -> None:
    """Verify validation did not create or change parameter gradients."""
    assert len(before_gradients) == len(after_gradients)

    for before_gradient, after_gradient in zip(
        before_gradients,
        after_gradients,
        strict=True,
    ):
        if before_gradient is None:
            assert after_gradient is None
        else:
            assert after_gradient is not None
            assert torch.equal(before_gradient, after_gradient)


def _assert_parameters_are_finite(
    model: CompactUNet3D,
) -> None:
    """Verify every model parameter is finite."""
    assert all(bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters())


def _assert_model_mode(
    model: CompactUNet3D,
    *,
    expected_training: bool,
) -> None:
    """Verify the model training/evaluation mode."""
    assert model.training is expected_training


def _require_sized(
    value: object,
) -> Sized:
    """Return a value that supports len()."""
    if not isinstance(value, Sized):
        raise TypeError("Value must be sized.")

    return value
