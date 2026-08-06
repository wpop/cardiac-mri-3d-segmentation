from collections.abc import Sequence, Sized
from pathlib import Path
from typing import Final

import pytest
import torch
from torch import Tensor
from torch.utils.data import SequentialSampler

from cardiac_segmentation.config.app_config import AppConfig
from cardiac_segmentation.config.loader import AppConfigLoader
from cardiac_segmentation.data import (
    AcdcDataLoaderFactory,
    AcdcDataLoaders,
    AcdcDatasetIndexer,
    AcdcInfoParser,
    AcdcPatientCase,
)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_CONFIG_PATH: Final[Path] = Path("configs/data.yaml")
_VALIDATION_FRACTION: Final[float] = 0.25
_RANDOM_SEED: Final[int] = 42
_BATCH_SIZE: Final[int] = 1
_NUM_WORKERS: Final[int] = 0
_PIN_MEMORY: Final[bool] = False


@pytest.mark.acdc
@pytest.mark.integration
def test_acdc_data_loader_factory_splits_training_patients_and_loads_batches() -> None:
    """Create deterministic real ACDC train/validation DataLoaders."""
    config = AppConfigLoader(
        project_root=_PROJECT_ROOT,
    ).load(_CONFIG_PATH)
    selected_cases = _select_first_training_cases(config)
    factory = AcdcDataLoaderFactory(
        preprocessing_config=config.preprocessing,
        validation_config=config.validation,
        validation_fraction=_VALIDATION_FRACTION,
        random_seed=_RANDOM_SEED,
        batch_size=_BATCH_SIZE,
        num_workers=_NUM_WORKERS,
        pin_memory=_PIN_MEMORY,
    )

    data_loaders = factory.create(selected_cases)
    repeated_data_loaders = factory.create(selected_cases)

    _assert_patient_split(
        data_loaders=data_loaders,
        repeated_data_loaders=repeated_data_loaders,
        selected_cases=selected_cases,
    )
    _assert_loader_datasets(data_loaders)
    _assert_validation_loader_uses_sequential_sampling(data_loaders)
    _assert_batch_matches_policy(
        batch=next(iter(data_loaders.training_loader)),
        allowed_patient_ids=_patient_ids(data_loaders.patient_split.training_cases),
        config=config,
    )
    _assert_batch_matches_policy(
        batch=next(iter(data_loaders.validation_loader)),
        allowed_patient_ids=_patient_ids(data_loaders.patient_split.validation_cases),
        config=config,
    )


def _select_first_training_cases(
    config: AppConfig,
) -> tuple[AcdcPatientCase, ...]:
    """Return the first four real training patient cases."""
    patient_cases = AcdcDatasetIndexer(
        dataset_root=config.dataset.root_dir,
        info_parser=AcdcInfoParser(),
    ).index()

    return tuple(
        patient_case
        for patient_case in patient_cases
        if patient_case.split_name == "training"
    )[:4]


def _assert_patient_split(
    data_loaders: AcdcDataLoaders,
    repeated_data_loaders: AcdcDataLoaders,
    selected_cases: tuple[AcdcPatientCase, ...],
) -> None:
    """Verify deterministic patient-level split membership."""
    training_patient_ids = _patient_ids(data_loaders.patient_split.training_cases)
    validation_patient_ids = _patient_ids(data_loaders.patient_split.validation_cases)
    repeated_training_patient_ids = _patient_ids(
        repeated_data_loaders.patient_split.training_cases
    )
    repeated_validation_patient_ids = _patient_ids(
        repeated_data_loaders.patient_split.validation_cases
    )

    assert len(data_loaders.patient_split.training_cases) == 3
    assert len(data_loaders.patient_split.validation_cases) == 1
    assert training_patient_ids.isdisjoint(validation_patient_ids)
    assert training_patient_ids | validation_patient_ids == _patient_ids(selected_cases)
    assert training_patient_ids == repeated_training_patient_ids
    assert validation_patient_ids == repeated_validation_patient_ids


def _assert_loader_datasets(
    data_loaders: AcdcDataLoaders,
) -> None:
    """Verify DataLoader dataset lengths."""
    training_dataset = _require_sized(data_loaders.training_loader.dataset)
    validation_dataset = _require_sized(data_loaders.validation_loader.dataset)

    assert len(training_dataset) == 6
    assert len(validation_dataset) == 2


def _assert_validation_loader_uses_sequential_sampling(
    data_loaders: AcdcDataLoaders,
) -> None:
    """Verify validation loading is not randomly sampled."""
    assert isinstance(data_loaders.validation_loader.sampler, SequentialSampler)


def _assert_batch_matches_policy(
    batch: dict[str, Tensor | Sequence[str]],
    allowed_patient_ids: set[str],
    config: AppConfig,
) -> None:
    """Verify one collated DataLoader batch."""
    image_tensor = _require_tensor(batch["image"])
    mask_tensor = _require_tensor(batch["mask"])
    target_shape = config.preprocessing.target_shape
    expected_image_shape = (
        _BATCH_SIZE,
        1,
        target_shape[2],
        target_shape[1],
        target_shape[0],
    )
    expected_mask_shape = (
        _BATCH_SIZE,
        target_shape[2],
        target_shape[1],
        target_shape[0],
    )

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
    _assert_collated_metadata(
        batch=batch,
        allowed_patient_ids=allowed_patient_ids,
    )


def _assert_collated_metadata(
    batch: dict[str, Tensor | Sequence[str]],
    allowed_patient_ids: set[str],
) -> None:
    """Verify default DataLoader collation for string metadata fields."""
    patient_ids = _require_string_sequence(batch["patient_id"])
    split_names = _require_string_sequence(batch["split_name"])
    phase_names = _require_string_sequence(batch["phase_name"])
    image_paths = _require_string_sequence(batch["image_path"])
    mask_paths = _require_string_sequence(batch["mask_path"])

    assert len(patient_ids) == _BATCH_SIZE
    assert len(split_names) == _BATCH_SIZE
    assert len(phase_names) == _BATCH_SIZE
    assert len(image_paths) == _BATCH_SIZE
    assert len(mask_paths) == _BATCH_SIZE
    assert patient_ids[0] in allowed_patient_ids
    assert split_names[0] == "training"
    assert phase_names[0] in {"ED", "ES"}
    assert image_paths[0]
    assert mask_paths[0]


def _patient_ids(
    patient_cases: tuple[AcdcPatientCase, ...],
) -> set[str]:
    """Return patient identifiers from patient cases."""
    return {patient_case.patient_id for patient_case in patient_cases}


def _require_tensor(
    value: Tensor | Sequence[str],
) -> Tensor:
    """Return a collated batch value as a tensor."""
    if not isinstance(value, Tensor):
        raise TypeError("Batch value must be a tensor.")

    return value


def _require_string_sequence(
    value: Tensor | Sequence[str],
) -> Sequence[str]:
    """Return a collated batch value as a string sequence."""
    if isinstance(value, Tensor):
        raise TypeError("Batch value must be a string sequence.")

    return value


def _require_sized(
    value: object,
) -> Sized:
    """Return a value that supports len()."""
    if not isinstance(value, Sized):
        raise TypeError("Value must be sized.")

    return value
