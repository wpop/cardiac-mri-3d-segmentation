from pathlib import Path
from typing import Final

import numpy as np
import pytest

from cardiac_segmentation.config.loader import AppConfigLoader
from cardiac_segmentation.data import (
    AcdcDatasetIndexer,
    AcdcInfoParser,
    NiftiGeometryValidator,
)
from cardiac_segmentation.preprocessing import NiftiImageMaskPairLoader

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_CONFIG_PATH: Final[Path] = Path("configs/data.yaml")


@pytest.mark.acdc
@pytest.mark.integration
def test_load_one_real_acdc_training_ed_image_mask_pair() -> None:
    """Load one real ACDC training ED image-mask pair without preprocessing it."""
    config = AppConfigLoader(
        project_root=_PROJECT_ROOT,
    ).load(_CONFIG_PATH)
    patient_cases = AcdcDatasetIndexer(
        dataset_root=config.dataset.root_dir,
        info_parser=AcdcInfoParser(),
    ).index()
    training_case = next(
        patient_case
        for patient_case in patient_cases
        if patient_case.split_name == "training"
    )

    image_mask_pair = NiftiImageMaskPairLoader(
        expected_labels=config.validation.expected_labels,
        affine_absolute_tolerance=config.validation.affine_absolute_tolerance,
        require_finite_intensities=config.validation.require_finite_intensities,
    ).load(
        image_path=training_case.ed_image_path,
        mask_path=training_case.ed_mask_path,
    )

    assert image_mask_pair.image_data.dtype == np.dtype(np.float32)
    assert image_mask_pair.mask_data.dtype == np.dtype(np.int64)
    assert image_mask_pair.image_data.shape == image_mask_pair.mask_data.shape
    assert tuple(image_mask_pair.image_data.shape) == (
        image_mask_pair.image_metadata.shape
    )
    assert tuple(image_mask_pair.mask_data.shape) == (
        image_mask_pair.mask_metadata.shape
    )
    assert image_mask_pair.image_data.ndim == 3
    assert image_mask_pair.mask_data.ndim == 3
    assert bool(np.isfinite(image_mask_pair.image_data).all())
    assert bool(np.isfinite(image_mask_pair.mask_data).all())
    assert set(np.unique(image_mask_pair.mask_data)).issubset(
        set(config.validation.expected_labels)
    )

    NiftiGeometryValidator(
        absolute_tolerance=config.validation.affine_absolute_tolerance,
    ).validate_pair(
        image_metadata=image_mask_pair.image_metadata,
        mask_metadata=image_mask_pair.mask_metadata,
    )
