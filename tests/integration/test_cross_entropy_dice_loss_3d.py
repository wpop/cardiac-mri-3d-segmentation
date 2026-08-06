from pathlib import Path
from typing import Final

import pytest
import torch
from torch import Tensor
from torch.nn import functional

from cardiac_segmentation.config.app_config import AppConfig
from cardiac_segmentation.config.loader import AppConfigLoader
from cardiac_segmentation.data import (
    AcdcDatasetIndexer,
    AcdcInfoParser,
    AcdcPatientCase,
    AcdcSegmentationDataset,
)
from cardiac_segmentation.losses import CrossEntropyDiceLoss3D, SoftDiceLoss3D
from cardiac_segmentation.models import CompactUNet3D

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_CONFIG_PATH: Final[Path] = Path("configs/data.yaml")
_TEST_BASE_CHANNELS: Final[int] = 2
_FLOAT32_ABSOLUTE_TOLERANCE: Final[float] = 1e-5


@pytest.mark.acdc
@pytest.mark.integration
def test_cross_entropy_dice_loss_3d_backpropagates_from_real_acdc_ed_volume() -> None:
    """Calculate multiclass segmentation losses for one real ACDC ED volume."""
    config = AppConfigLoader(
        project_root=_PROJECT_ROOT,
    ).load(_CONFIG_PATH)
    training_case = _select_training_case(config)
    image_tensor, target = _load_real_ed_tensors(
        config=config,
        training_case=training_case,
    )
    model = CompactUNet3D(
        in_channels=1,
        num_classes=len(config.validation.expected_labels),
        base_channels=_TEST_BASE_CHANNELS,
    )
    model.eval()

    with torch.no_grad():
        model_output = model(image_tensor)

    logits = model_output.detach().requires_grad_(True)
    soft_dice_loss = SoftDiceLoss3D(
        num_classes=len(config.validation.expected_labels),
        include_background=False,
    )
    combined_loss_function = CrossEntropyDiceLoss3D(
        num_classes=len(config.validation.expected_labels),
        cross_entropy_weight=0.5,
        dice_weight=0.5,
        include_background_in_dice=False,
    )

    _assert_logits_and_target_match_contract(
        logits=logits,
        target=target,
        config=config,
    )
    dice_loss = soft_dice_loss(
        logits,
        target,
    )
    combined_loss = combined_loss_function(
        logits,
        target,
    )
    cross_entropy_loss = functional.cross_entropy(
        logits,
        target,
    )
    expected_combined_loss = (
        0.5 * cross_entropy_loss
        + 0.5 * soft_dice_loss(
            logits,
            target,
        )
    ) / 1.0

    _assert_loss_values_match_contract(
        dice_loss=dice_loss,
        cross_entropy_loss=cross_entropy_loss,
        combined_loss=combined_loss,
        expected_combined_loss=expected_combined_loss,
    )
    combined_loss.backward()
    _assert_logits_gradient_matches_contract(logits)


def _select_training_case(
    config: AppConfig,
) -> AcdcPatientCase:
    """Return the first real training patient case."""
    patient_cases = AcdcDatasetIndexer(
        dataset_root=config.dataset.root_dir,
        info_parser=AcdcInfoParser(),
    ).index()

    return next(
        patient_case
        for patient_case in patient_cases
        if patient_case.split_name == "training"
    )


def _load_real_ed_tensors(
    config: AppConfig,
    training_case: AcdcPatientCase,
) -> tuple[Tensor, Tensor]:
    """Load one real ED image and mask from the Dataset."""
    dataset = AcdcSegmentationDataset(
        patient_cases=(training_case,),
        preprocessing_config=config.preprocessing,
        validation_config=config.validation,
    )
    ed_item = dataset[0]
    image_tensor = _require_tensor(ed_item["image"]).unsqueeze(0)
    target = _require_tensor(ed_item["mask"]).unsqueeze(0)

    return image_tensor, target


def _assert_logits_and_target_match_contract(
    logits: Tensor,
    target: Tensor,
    config: AppConfig,
) -> None:
    """Verify logits and target tensor shape, dtype, values, and labels."""
    target_shape = config.preprocessing.target_shape
    expected_logits_shape = (
        1,
        len(config.validation.expected_labels),
        target_shape[2],
        target_shape[1],
        target_shape[0],
    )
    expected_target_shape = (
        1,
        target_shape[2],
        target_shape[1],
        target_shape[0],
    )

    assert tuple(logits.shape) == expected_logits_shape
    assert tuple(target.shape) == expected_target_shape
    assert logits.dtype == torch.float32
    assert target.dtype == torch.int64
    assert bool(torch.isfinite(logits).all().item())
    assert bool(torch.isfinite(target).all().item())
    assert {int(label) for label in torch.unique(target).tolist()}.issubset(
        set(config.validation.expected_labels)
    )


def _assert_loss_values_match_contract(
    dice_loss: Tensor,
    cross_entropy_loss: Tensor,
    combined_loss: Tensor,
    expected_combined_loss: Tensor,
) -> None:
    """Verify scalar loss values and weighted-combination arithmetic."""
    assert dice_loss.ndim == 0
    assert bool(torch.isfinite(dice_loss).item())
    assert 0.0 <= float(dice_loss.item()) <= 1.0
    assert combined_loss.ndim == 0
    assert bool(torch.isfinite(combined_loss).item())
    assert float(combined_loss.item()) > 0.0
    assert torch.isclose(
        combined_loss,
        expected_combined_loss,
        atol=_FLOAT32_ABSOLUTE_TOLERANCE,
        rtol=0.0,
    )
    assert bool(torch.isfinite(cross_entropy_loss).item())


def _assert_logits_gradient_matches_contract(
    logits: Tensor,
) -> None:
    """Verify loss backward pass produced finite non-zero logits gradients."""
    gradient = logits.grad

    assert gradient is not None
    assert tuple(gradient.shape) == tuple(logits.shape)
    assert bool(torch.isfinite(gradient).all().item())
    assert bool(torch.any(gradient != 0.0).item())


def _require_tensor(
    value: Tensor | str,
) -> Tensor:
    """Return a Dataset item value as a tensor."""
    if not isinstance(value, Tensor):
        raise TypeError("Dataset item value must be a tensor.")

    return value
