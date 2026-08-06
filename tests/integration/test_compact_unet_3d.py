from pathlib import Path
from typing import Final

import pytest
import torch
from torch import Tensor

from cardiac_segmentation.config.app_config import AppConfig
from cardiac_segmentation.config.loader import AppConfigLoader
from cardiac_segmentation.data import (
    AcdcDatasetIndexer,
    AcdcInfoParser,
    AcdcPatientCase,
    AcdcSegmentationDataset,
)
from cardiac_segmentation.models import CompactUNet3D

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_CONFIG_PATH: Final[Path] = Path("configs/data.yaml")
_TEST_BASE_CHANNELS: Final[int] = 4


@pytest.mark.acdc
@pytest.mark.integration
def test_compact_unet_3d_forwards_one_real_acdc_ed_volume() -> None:
    """Run CompactUNet3D inference on one real preprocessed ACDC ED volume."""
    config = AppConfigLoader(
        project_root=_PROJECT_ROOT,
    ).load(_CONFIG_PATH)
    training_case = _select_training_case(config)
    dataset = AcdcSegmentationDataset(
        patient_cases=(training_case,),
        preprocessing_config=config.preprocessing,
        validation_config=config.validation,
    )
    ed_item = dataset[0]
    image_tensor = _require_tensor(ed_item["image"])
    input_tensor = image_tensor.unsqueeze(0)
    model = CompactUNet3D(
        in_channels=1,
        num_classes=len(config.validation.expected_labels),
        base_channels=_TEST_BASE_CHANNELS,
    )
    model.eval()

    _assert_input_tensor_matches_contract(input_tensor)

    with torch.inference_mode():
        logits = model(input_tensor)

    _assert_logits_match_contract(
        logits=logits,
        input_tensor=input_tensor,
        config=config,
    )
    _assert_model_has_trainable_parameters(model)


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


def _assert_input_tensor_matches_contract(
    input_tensor: Tensor,
) -> None:
    """Verify model input tensor properties before inference."""
    assert input_tensor.dtype == torch.float32
    assert input_tensor.is_contiguous()
    assert bool(torch.isfinite(input_tensor).all().item())


def _assert_logits_match_contract(
    logits: Tensor,
    input_tensor: Tensor,
    config: AppConfig,
) -> None:
    """Verify raw model logits satisfy the segmentation shape contract."""
    target_shape = config.preprocessing.target_shape
    expected_logits_shape = (
        1,
        len(config.validation.expected_labels),
        target_shape[2],
        target_shape[1],
        target_shape[0],
    )

    assert tuple(logits.shape) == expected_logits_shape
    assert logits.dtype == torch.float32
    assert logits.is_contiguous()
    assert bool(torch.isfinite(logits).all().item())
    assert tuple(logits.shape[2:]) == tuple(input_tensor.shape[2:])
    assert logits.shape[1] == len(config.validation.expected_labels)


def _assert_model_has_trainable_parameters(
    model: CompactUNet3D,
) -> None:
    """Verify the model exposes trainable parameters."""
    trainable_parameters = tuple(
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert trainable_parameters
    assert all(parameter.requires_grad for parameter in trainable_parameters)


def _require_tensor(
    value: Tensor | str,
) -> Tensor:
    """Return a Dataset item value as a tensor."""
    if not isinstance(value, Tensor):
        raise TypeError("Dataset item value must be a tensor.")

    return value
