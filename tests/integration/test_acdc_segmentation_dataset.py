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

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_CONFIG_PATH: Final[Path] = Path("configs/data.yaml")
_FLOAT32_ABSOLUTE_TOLERANCE: Final[float] = 5e-5


@pytest.mark.acdc
@pytest.mark.integration
def test_acdc_segmentation_dataset_returns_real_ed_and_es_tensors() -> None:
    """Load one real training patient through the PyTorch Dataset interface."""
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
    es_item = dataset[1]
    negative_index_item = dataset[-1]

    assert len(dataset) == 2
    _assert_item_matches_phase(
        item=ed_item,
        patient_case=training_case,
        phase_name="ED",
        config=config,
    )
    _assert_item_matches_phase(
        item=es_item,
        patient_case=training_case,
        phase_name="ES",
        config=config,
    )
    assert negative_index_item["phase_name"] == "ES"
    assert negative_index_item["image_path"] == str(training_case.es_image_path)
    assert negative_index_item["mask_path"] == str(training_case.es_mask_path)

    with pytest.raises(IndexError):
        _ = dataset[2]

    with pytest.raises(IndexError):
        _ = dataset[-3]


def _select_training_case(config: AppConfig) -> AcdcPatientCase:
    """Return the first real training patient case from the configured dataset."""
    patient_cases = AcdcDatasetIndexer(
        dataset_root=config.dataset.root_dir,
        info_parser=AcdcInfoParser(),
    ).index()

    return next(
        patient_case
        for patient_case in patient_cases
        if patient_case.split_name == "training"
    )


def _assert_item_matches_phase(
    item: dict[str, Tensor | str],
    patient_case: AcdcPatientCase,
    phase_name: str,
    config: AppConfig,
) -> None:
    """Verify tensor values and metadata for one dataset item."""
    image_tensor = _require_tensor(item["image"])
    mask_tensor = _require_tensor(item["mask"])
    image_path, mask_path = _get_phase_paths(
        patient_case=patient_case,
        phase_name=phase_name,
    )
    target_shape = config.preprocessing.target_shape
    expected_image_shape = (
        1,
        target_shape[2],
        target_shape[1],
        target_shape[0],
    )
    expected_mask_shape = (
        target_shape[2],
        target_shape[1],
        target_shape[0],
    )

    assert item["patient_id"] == patient_case.patient_id
    assert item["split_name"] == "training"
    assert item["phase_name"] == phase_name
    assert item["image_path"] == str(image_path)
    assert item["mask_path"] == str(mask_path)
    assert tuple(image_tensor.shape) == expected_image_shape
    assert tuple(mask_tensor.shape) == expected_mask_shape
    assert image_tensor.dtype == torch.float32
    assert mask_tensor.dtype == torch.int64
    assert image_tensor.is_contiguous()
    assert mask_tensor.is_contiguous()
    assert bool(torch.isfinite(image_tensor).all().item())
    assert bool(torch.isfinite(mask_tensor).all().item())
    assert {int(label) for label in torch.unique(mask_tensor).tolist()}.issubset(
        set(config.validation.expected_labels)
    )

    if config.preprocessing.normalize_nonzero_only:
        _assert_nonzero_only_normalization(image_tensor)


def _assert_nonzero_only_normalization(
    image_tensor: Tensor,
) -> None:
    """Verify configured nonzero-only normalization behavior."""
    zero_voxels = image_tensor == 0.0
    nonzero_voxels = image_tensor != 0.0

    assert bool(zero_voxels.any().item())
    assert bool(torch.equal(image_tensor[zero_voxels], torch.zeros_like(image_tensor[zero_voxels])))

    normalized_values = image_tensor[nonzero_voxels]
    assert torch.isclose(
        torch.mean(normalized_values),
        torch.tensor(0.0, dtype=torch.float32),
        atol=_FLOAT32_ABSOLUTE_TOLERANCE,
        rtol=0.0,
    )
    assert torch.isclose(
        torch.std(
            normalized_values,
            unbiased=False,
        ),
        torch.tensor(1.0, dtype=torch.float32),
        atol=_FLOAT32_ABSOLUTE_TOLERANCE,
        rtol=0.0,
    )


def _require_tensor(value: Tensor | str) -> Tensor:
    """Return a dataset item value as a tensor."""
    if not isinstance(value, Tensor):
        raise TypeError("Dataset item value must be a tensor.")

    return value


def _get_phase_paths(
    patient_case: AcdcPatientCase,
    phase_name: str,
) -> tuple[Path, Path]:
    """Return image and mask paths for the requested cardiac phase."""
    if phase_name == "ED":
        return (
            patient_case.ed_image_path,
            patient_case.ed_mask_path,
        )

    if phase_name == "ES":
        return (
            patient_case.es_image_path,
            patient_case.es_mask_path,
        )

    raise ValueError(f"Unsupported cardiac phase name: {phase_name}")
