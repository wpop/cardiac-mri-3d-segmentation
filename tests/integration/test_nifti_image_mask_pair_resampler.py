from collections.abc import Callable
from math import isclose
from pathlib import Path
from typing import Final, cast

import nibabel as nib
import numpy as np
import pytest

from cardiac_segmentation.config.app_config import AppConfig
from cardiac_segmentation.config.loader import AppConfigLoader
from cardiac_segmentation.data import AcdcDatasetIndexer, AcdcInfoParser
from cardiac_segmentation.data.nifti_volume_metadata import AffineMatrix
from cardiac_segmentation.preprocessing import (
    NiftiImageMaskPair,
    NiftiImageMaskPairLoader,
    NiftiImageMaskPairResampler,
    ResampledImageMaskPair,
)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_CONFIG_PATH: Final[Path] = Path("configs/data.yaml")
_SOURCE_Z_SPACING_VALUES_MM: Final[tuple[float, float]] = (5.0, 10.0)
_SPACING_ABSOLUTE_TOLERANCE: Final[float] = 1e-5
_CENTER_ABSOLUTE_TOLERANCE: Final[float] = 1e-4


@pytest.mark.acdc
@pytest.mark.integration
def test_resample_real_acdc_pairs_with_five_and_ten_mm_z_spacing() -> None:
    """Resample real ACDC pairs that represent both common through-plane spacings."""
    config = AppConfigLoader(
        project_root=_PROJECT_ROOT,
    ).load(_CONFIG_PATH)
    source_pairs = _load_pairs_by_z_spacing(config)
    resampler = NiftiImageMaskPairResampler(
        target_spacing_mm=config.preprocessing.target_spacing_mm,
        expected_labels=config.validation.expected_labels,
    )

    for source_pair in source_pairs:
        resampled_pair = resampler.resample(source_pair)

        _assert_resampled_pair_matches_policy(
            source_pair=source_pair,
            resampled_pair=resampled_pair,
            config=config,
        )


def _load_pairs_by_z_spacing(
    config: AppConfig,
) -> tuple[NiftiImageMaskPair, NiftiImageMaskPair]:
    """Load one real ACDC phase for each requested source Z spacing."""
    loader = NiftiImageMaskPairLoader(
        expected_labels=config.validation.expected_labels,
        affine_absolute_tolerance=config.validation.affine_absolute_tolerance,
        require_finite_intensities=config.validation.require_finite_intensities,
    )
    indexer = AcdcDatasetIndexer(
        dataset_root=config.dataset.root_dir,
        info_parser=AcdcInfoParser(),
    )
    selected_pairs: dict[float, NiftiImageMaskPair] = {}

    for patient_case in indexer.index():
        phase_paths = (
            (
                patient_case.ed_image_path,
                patient_case.ed_mask_path,
            ),
            (
                patient_case.es_image_path,
                patient_case.es_mask_path,
            ),
        )

        for image_path, mask_path in phase_paths:
            source_pair = loader.load(
                image_path=image_path,
                mask_path=mask_path,
            )
            source_z_spacing = source_pair.image_metadata.voxel_spacing[2]

            for requested_z_spacing in _SOURCE_Z_SPACING_VALUES_MM:
                if requested_z_spacing in selected_pairs:
                    continue

                if isclose(
                    source_z_spacing,
                    requested_z_spacing,
                    abs_tol=_SPACING_ABSOLUTE_TOLERANCE,
                ):
                    selected_pairs[requested_z_spacing] = source_pair

            if len(selected_pairs) == len(_SOURCE_Z_SPACING_VALUES_MM):
                return (
                    selected_pairs[_SOURCE_Z_SPACING_VALUES_MM[0]],
                    selected_pairs[_SOURCE_Z_SPACING_VALUES_MM[1]],
                )

    raise AssertionError("Unable to find real ACDC phases with 5.0 mm and 10.0 mm Z spacing.")


def _assert_resampled_pair_matches_policy(
    source_pair: NiftiImageMaskPair,
    resampled_pair: ResampledImageMaskPair,
    config: AppConfig,
) -> None:
    """Verify spacing-only resampling invariants for one real pair."""
    assert resampled_pair.image_data.dtype == np.dtype(np.float32)
    assert resampled_pair.mask_data.dtype == np.dtype(np.int64)
    assert resampled_pair.image_data.shape == resampled_pair.mask_data.shape
    assert resampled_pair.shape == _calculate_expected_shape(
        source_pair=source_pair,
        target_spacing_mm=config.preprocessing.target_spacing_mm,
    )
    assert resampled_pair.image_data.shape == resampled_pair.shape
    assert np.allclose(
        np.asarray(resampled_pair.voxel_spacing, dtype=np.float64),
        np.asarray(config.preprocessing.target_spacing_mm, dtype=np.float64),
        atol=_SPACING_ABSOLUTE_TOLERANCE,
        rtol=0.0,
    )
    assert resampled_pair.orientation == source_pair.image_metadata.orientation
    assert bool(np.isfinite(resampled_pair.image_data).all())
    assert bool(np.isfinite(resampled_pair.mask_data).all())
    assert set(np.unique(resampled_pair.mask_data)).issubset(
        set(config.validation.expected_labels)
    )
    assert bool(np.equal(resampled_pair.mask_data, np.rint(resampled_pair.mask_data)).all())
    assert np.allclose(
        _calculate_world_space_center(
            shape=source_pair.image_metadata.shape,
            affine_matrix=source_pair.image_metadata.affine,
        ),
        _calculate_world_space_center(
            shape=resampled_pair.shape,
            affine_matrix=resampled_pair.affine,
        ),
        atol=_CENTER_ABSOLUTE_TOLERANCE,
        rtol=0.0,
    )


def _calculate_expected_shape(
    source_pair: NiftiImageMaskPair,
    target_spacing_mm: tuple[float, float, float],
) -> tuple[int, int, int]:
    """Calculate the spacing-only output shape expected from the source pair."""
    return (
        _calculate_expected_axis_shape(source_pair, target_spacing_mm, axis=0),
        _calculate_expected_axis_shape(source_pair, target_spacing_mm, axis=1),
        _calculate_expected_axis_shape(source_pair, target_spacing_mm, axis=2),
    )


def _calculate_expected_axis_shape(
    source_pair: NiftiImageMaskPair,
    target_spacing_mm: tuple[float, float, float],
    axis: int,
) -> int:
    """Calculate one expected output dimension from source shape and spacing."""
    return max(
        1,
        round(
            source_pair.image_metadata.shape[axis]
            * source_pair.image_metadata.voxel_spacing[axis]
            / target_spacing_mm[axis]
        ),
    )


def _calculate_world_space_center(
    shape: tuple[int, int, int],
    affine_matrix: AffineMatrix,
) -> np.ndarray:
    """Calculate the world-space center of a voxel grid."""
    center_voxel = (np.asarray(shape, dtype=np.float64) - 1.0) / 2.0
    apply_affine = cast(
        Callable[[np.ndarray, np.ndarray], np.ndarray],
        nib.affines.apply_affine,
    )

    return np.asarray(
        apply_affine(
            np.asarray(affine_matrix, dtype=np.float64),
            center_voxel,
        ),
        dtype=np.float64,
    )
