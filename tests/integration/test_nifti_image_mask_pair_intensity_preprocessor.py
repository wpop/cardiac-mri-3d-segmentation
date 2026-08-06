from pathlib import Path
from typing import Final

import numpy as np
import pytest
from numpy.typing import NDArray

from cardiac_segmentation.config.app_config import AppConfig
from cardiac_segmentation.config.loader import AppConfigLoader
from cardiac_segmentation.data import AcdcDatasetIndexer, AcdcInfoParser
from cardiac_segmentation.preprocessing import (
    CenterCroppedPaddedImageMaskPair,
    IntensityPreprocessedImageMaskPair,
    NiftiImageMaskPairCenterCropPadder,
    NiftiImageMaskPairIntensityPreprocessor,
    NiftiImageMaskPairLoader,
    NiftiImageMaskPairResampler,
)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_CONFIG_PATH: Final[Path] = Path("configs/data.yaml")
_FLOAT32_ABSOLUTE_TOLERANCE: Final[float] = 5e-5


@pytest.mark.acdc
@pytest.mark.integration
def test_intensity_preprocess_one_real_training_ed_pair() -> None:
    """Clip and normalize one real crop-padded ACDC training ED phase."""
    config = AppConfigLoader(
        project_root=_PROJECT_ROOT,
    ).load(_CONFIG_PATH)
    crop_padded_pair = _load_crop_padded_training_ed_pair(config)

    intensity_pair = NiftiImageMaskPairIntensityPreprocessor(
        lower_percentile=config.preprocessing.intensity_lower_percentile,
        upper_percentile=config.preprocessing.intensity_upper_percentile,
        normalize_nonzero_only=config.preprocessing.normalize_nonzero_only,
        expected_labels=config.validation.expected_labels,
    ).transform(crop_padded_pair)

    _assert_intensity_pair_matches_policy(
        crop_padded_pair=crop_padded_pair,
        intensity_pair=intensity_pair,
        config=config,
    )


def _load_crop_padded_training_ed_pair(
    config: AppConfig,
) -> CenterCroppedPaddedImageMaskPair:
    """Run the existing spatial preprocessing sequence for one training ED phase."""
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
    resampled_pair = NiftiImageMaskPairResampler(
        target_spacing_mm=config.preprocessing.target_spacing_mm,
        expected_labels=config.validation.expected_labels,
    ).resample(image_mask_pair)

    return NiftiImageMaskPairCenterCropPadder(
        target_shape=config.preprocessing.target_shape,
        expected_labels=config.validation.expected_labels,
    ).transform(resampled_pair)


def _assert_intensity_pair_matches_policy(
    crop_padded_pair: CenterCroppedPaddedImageMaskPair,
    intensity_pair: IntensityPreprocessedImageMaskPair,
    config: AppConfig,
) -> None:
    """Verify clipping, normalization, masks, and source geometry."""
    expected_image, lower_clip_value, upper_clip_value, mean, standard_deviation = (
        _build_expected_image_and_statistics(
            image_data=crop_padded_pair.image_data,
            config=config,
        )
    )
    source_nonzero_mask = crop_padded_pair.image_data != 0.0
    source_zero_mask = ~source_nonzero_mask

    assert intensity_pair.image_data.shape == config.preprocessing.target_shape
    assert intensity_pair.mask_data.shape == config.preprocessing.target_shape
    assert intensity_pair.image_data.dtype == np.dtype(np.float32)
    assert intensity_pair.mask_data.dtype == np.dtype(np.int64)
    assert intensity_pair.image_data.flags.c_contiguous
    assert intensity_pair.mask_data.flags.c_contiguous
    assert bool(np.isfinite(intensity_pair.image_data).all())
    assert bool(np.isfinite(intensity_pair.mask_data).all())
    assert np.array_equal(intensity_pair.mask_data, crop_padded_pair.mask_data)
    assert not np.shares_memory(
        intensity_pair.mask_data,
        crop_padded_pair.mask_data,
    )
    assert set(np.unique(intensity_pair.mask_data)).issubset(
        set(config.validation.expected_labels)
    )

    if config.preprocessing.normalize_nonzero_only:
        assert bool(np.equal(intensity_pair.image_data[source_zero_mask], 0.0).all())

    normalized_nonzero_values = intensity_pair.image_data[source_nonzero_mask]
    assert np.isclose(
        float(np.mean(normalized_nonzero_values)),
        0.0,
        atol=_FLOAT32_ABSOLUTE_TOLERANCE,
        rtol=0.0,
    )
    assert np.isclose(
        float(np.std(normalized_nonzero_values)),
        1.0,
        atol=_FLOAT32_ABSOLUTE_TOLERANCE,
        rtol=0.0,
    )
    assert np.isclose(
        intensity_pair.lower_clip_value,
        lower_clip_value,
        atol=_FLOAT32_ABSOLUTE_TOLERANCE,
        rtol=0.0,
    )
    assert np.isclose(
        intensity_pair.upper_clip_value,
        upper_clip_value,
        atol=_FLOAT32_ABSOLUTE_TOLERANCE,
        rtol=0.0,
    )
    assert np.isclose(
        intensity_pair.normalization_mean,
        mean,
        atol=_FLOAT32_ABSOLUTE_TOLERANCE,
        rtol=0.0,
    )
    assert np.isclose(
        intensity_pair.normalization_standard_deviation,
        standard_deviation,
        atol=_FLOAT32_ABSOLUTE_TOLERANCE,
        rtol=0.0,
    )
    assert np.allclose(
        intensity_pair.image_data,
        expected_image,
        atol=_FLOAT32_ABSOLUTE_TOLERANCE,
        rtol=0.0,
    )
    assert intensity_pair.source_pair is crop_padded_pair
    _assert_source_geometry_is_unchanged(
        source_pair=intensity_pair.source_pair,
        expected_pair=crop_padded_pair,
    )


def _build_expected_image_and_statistics(
    image_data: NDArray[np.float32],
    config: AppConfig,
) -> tuple[NDArray[np.float32], float, float, float, float]:
    """Independently reconstruct expected clipping and normalization output."""
    nonzero_mask = image_data != 0.0
    nonzero_values = image_data[nonzero_mask]
    lower_clip_value = float(
        np.percentile(
            nonzero_values,
            config.preprocessing.intensity_lower_percentile,
        )
    )
    upper_clip_value = float(
        np.percentile(
            nonzero_values,
            config.preprocessing.intensity_upper_percentile,
        )
    )
    clipped_image = image_data.astype(np.float32, copy=True)
    clipped_image[nonzero_mask] = np.clip(
        clipped_image[nonzero_mask],
        lower_clip_value,
        upper_clip_value,
    )

    if config.preprocessing.normalize_nonzero_only:
        normalization_values = clipped_image[nonzero_mask]
    else:
        normalization_values = clipped_image

    mean = float(np.mean(normalization_values))
    standard_deviation = float(np.std(normalization_values))
    expected_image = clipped_image.astype(np.float32, copy=True)

    if config.preprocessing.normalize_nonzero_only:
        expected_image[nonzero_mask] = (
            expected_image[nonzero_mask] - mean
        ) / standard_deviation
    else:
        expected_image = (expected_image - mean) / standard_deviation

    return (
        np.ascontiguousarray(
            expected_image,
            dtype=np.float32,
        ),
        lower_clip_value,
        upper_clip_value,
        mean,
        standard_deviation,
    )


def _assert_source_geometry_is_unchanged(
    source_pair: CenterCroppedPaddedImageMaskPair,
    expected_pair: CenterCroppedPaddedImageMaskPair,
) -> None:
    """Verify intensity preprocessing leaves spatial metadata on the source pair."""
    assert source_pair.shape == expected_pair.shape
    assert source_pair.voxel_spacing == expected_pair.voxel_spacing
    assert source_pair.orientation == expected_pair.orientation
    assert source_pair.affine == expected_pair.affine
    assert source_pair.crop_start == expected_pair.crop_start
    assert source_pair.crop_end == expected_pair.crop_end
    assert source_pair.padding_before == expected_pair.padding_before
    assert source_pair.padding_after == expected_pair.padding_after
