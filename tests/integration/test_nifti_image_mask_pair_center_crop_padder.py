from collections.abc import Callable
from pathlib import Path
from typing import Final, cast

import nibabel as nib
import numpy as np
import pytest
from numpy.typing import NDArray

from cardiac_segmentation.config.app_config import AppConfig
from cardiac_segmentation.config.loader import AppConfigLoader
from cardiac_segmentation.data import (
    AcdcDatasetIndexer,
    AcdcInfoParser,
    NiftiMetadataReader,
)
from cardiac_segmentation.data.nifti_volume_metadata import AffineMatrix
from cardiac_segmentation.preprocessing import (
    CenterCroppedPaddedImageMaskPair,
    NiftiImageMaskPairCenterCropPadder,
    NiftiImageMaskPairLoader,
    NiftiImageMaskPairResampler,
    ResampledImageMaskPair,
)

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_CONFIG_PATH: Final[Path] = Path("configs/data.yaml")
_AFFINE_ABSOLUTE_TOLERANCE: Final[float] = 1e-5


@pytest.mark.acdc
@pytest.mark.integration
def test_center_crop_pad_one_real_acdc_pair_to_target_shape() -> None:
    """Center crop and zero-pad one real spacing-resampled ACDC phase."""
    config = AppConfigLoader(
        project_root=_PROJECT_ROOT,
    ).load(_CONFIG_PATH)
    image_path, mask_path = _select_crop_and_pad_candidate(config)
    image_mask_pair = NiftiImageMaskPairLoader(
        expected_labels=config.validation.expected_labels,
        affine_absolute_tolerance=config.validation.affine_absolute_tolerance,
        require_finite_intensities=config.validation.require_finite_intensities,
    ).load(
        image_path=image_path,
        mask_path=mask_path,
    )
    resampled_pair = NiftiImageMaskPairResampler(
        target_spacing_mm=config.preprocessing.target_spacing_mm,
        expected_labels=config.validation.expected_labels,
    ).resample(image_mask_pair)

    transformed_pair = NiftiImageMaskPairCenterCropPadder(
        target_shape=config.preprocessing.target_shape,
        expected_labels=config.validation.expected_labels,
    ).transform(resampled_pair)

    _assert_transformed_pair_matches_policy(
        resampled_pair=resampled_pair,
        transformed_pair=transformed_pair,
        config=config,
    )


def _select_crop_and_pad_candidate(config: AppConfig) -> tuple[Path, Path]:
    """Find a real phase whose spacing-resampled shape requires crop and padding."""
    metadata_reader = NiftiMetadataReader()
    indexer = AcdcDatasetIndexer(
        dataset_root=config.dataset.root_dir,
        info_parser=AcdcInfoParser(),
    )

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
            metadata = metadata_reader.read(image_path)
            expected_shape = _calculate_resampled_shape(
                source_shape=metadata.shape,
                source_spacing=metadata.voxel_spacing,
                target_spacing=config.preprocessing.target_spacing_mm,
            )

            if _requires_crop_and_padding(
                source_shape=expected_shape,
                target_shape=config.preprocessing.target_shape,
            ):
                return image_path, mask_path

    raise AssertionError(
        "Unable to find a real ACDC phase requiring both crop and padding."
    )


def _assert_transformed_pair_matches_policy(
    resampled_pair: ResampledImageMaskPair,
    transformed_pair: CenterCroppedPaddedImageMaskPair,
    config: AppConfig,
) -> None:
    """Verify fixed-shape crop and padding invariants for one real pair."""
    assert transformed_pair.shape == config.preprocessing.target_shape
    assert transformed_pair.image_data.shape == config.preprocessing.target_shape
    assert transformed_pair.mask_data.shape == config.preprocessing.target_shape
    assert transformed_pair.image_data.dtype == np.dtype(np.float32)
    assert transformed_pair.mask_data.dtype == np.dtype(np.int64)
    assert transformed_pair.image_data.shape == transformed_pair.mask_data.shape
    assert transformed_pair.image_data.flags.c_contiguous
    assert transformed_pair.mask_data.flags.c_contiguous
    assert bool(np.isfinite(transformed_pair.image_data).all())
    assert bool(np.isfinite(transformed_pair.mask_data).all())
    assert set(np.unique(transformed_pair.mask_data)).issubset(
        set(config.validation.expected_labels)
    )
    assert _has_cropped_axis(transformed_pair, resampled_pair)
    assert _has_padded_axis(transformed_pair)
    _assert_crop_and_padding_reconstruct_target(transformed_pair)
    assert transformed_pair.voxel_spacing == resampled_pair.voxel_spacing
    assert transformed_pair.orientation == resampled_pair.orientation

    source_affine = np.asarray(resampled_pair.affine, dtype=np.float64)
    target_affine = np.asarray(transformed_pair.affine, dtype=np.float64)
    index_offset = _calculate_index_offset(transformed_pair)
    assert np.allclose(
        target_affine[:3, :3],
        source_affine[:3, :3],
        atol=_AFFINE_ABSOLUTE_TOLERANCE,
        rtol=0.0,
    )
    assert np.allclose(
        target_affine[:3, 3],
        source_affine[:3, 3] + source_affine[:3, :3] @ index_offset,
        atol=_AFFINE_ABSOLUTE_TOLERANCE,
        rtol=0.0,
    )

    _assert_affine_coordinates_match(
        source_affine=resampled_pair.affine,
        target_affine=transformed_pair.affine,
        target_shape=transformed_pair.shape,
        index_offset=index_offset,
    )
    _assert_output_arrays_match_expected(
        resampled_pair=resampled_pair,
        transformed_pair=transformed_pair,
    )


def _calculate_resampled_shape(
    source_shape: tuple[int, int, int],
    source_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
) -> tuple[int, int, int]:
    """Calculate expected spacing-resampled shape independently for each axis."""
    return (
        _calculate_resampled_axis_shape(source_shape, source_spacing, target_spacing, 0),
        _calculate_resampled_axis_shape(source_shape, source_spacing, target_spacing, 1),
        _calculate_resampled_axis_shape(source_shape, source_spacing, target_spacing, 2),
    )


def _calculate_resampled_axis_shape(
    source_shape: tuple[int, int, int],
    source_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
    axis: int,
) -> int:
    """Calculate one spacing-resampled axis length."""
    return max(
        1,
        round(source_shape[axis] * source_spacing[axis] / target_spacing[axis]),
    )


def _requires_crop_and_padding(
    source_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
) -> bool:
    """Return whether shape conversion needs at least one crop and one pad."""
    return any(
        source_size > target_size
        for source_size, target_size in zip(source_shape, target_shape, strict=True)
    ) and any(
        source_size < target_size
        for source_size, target_size in zip(source_shape, target_shape, strict=True)
    )


def _has_cropped_axis(
    transformed_pair: CenterCroppedPaddedImageMaskPair,
    resampled_pair: ResampledImageMaskPair,
) -> bool:
    """Return whether any axis was cropped from the resampled source."""
    return any(
        crop_start > 0 or crop_end < source_size
        for crop_start, crop_end, source_size in zip(
            transformed_pair.crop_start,
            transformed_pair.crop_end,
            resampled_pair.shape,
            strict=True,
        )
    )


def _has_padded_axis(
    transformed_pair: CenterCroppedPaddedImageMaskPair,
) -> bool:
    """Return whether any axis received zero-padding."""
    return any(
        padding_before > 0 or padding_after > 0
        for padding_before, padding_after in zip(
            transformed_pair.padding_before,
            transformed_pair.padding_after,
            strict=True,
        )
    )


def _assert_crop_and_padding_reconstruct_target(
    transformed_pair: CenterCroppedPaddedImageMaskPair,
) -> None:
    """Verify crop and padding dimensions reconstruct the target shape."""
    for axis in range(3):
        cropped_size = (
            transformed_pair.crop_end[axis]
            - transformed_pair.crop_start[axis]
        )
        assert (
            cropped_size
            + transformed_pair.padding_before[axis]
            + transformed_pair.padding_after[axis]
            == transformed_pair.shape[axis]
        )


def _calculate_index_offset(
    transformed_pair: CenterCroppedPaddedImageMaskPair,
) -> NDArray[np.float64]:
    """Calculate output-to-source index offset for affine checks."""
    return (
        np.asarray(transformed_pair.crop_start, dtype=np.float64)
        - np.asarray(transformed_pair.padding_before, dtype=np.float64)
    )


def _assert_affine_coordinates_match(
    source_affine: AffineMatrix,
    target_affine: AffineMatrix,
    target_shape: tuple[int, int, int],
    index_offset: NDArray[np.float64],
) -> None:
    """Verify representative output voxels map to equivalent source coordinates."""
    output_coordinates = (
        np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
        (np.asarray(target_shape, dtype=np.float64) - 1.0) / 2.0,
        np.asarray(
            (
                target_shape[0] - 1,
                target_shape[1] - 1,
                target_shape[2] - 1,
            ),
            dtype=np.float64,
        ),
    )

    for output_coordinate in output_coordinates:
        source_coordinate = output_coordinate + index_offset
        assert np.allclose(
            _apply_affine(target_affine, output_coordinate),
            _apply_affine(source_affine, source_coordinate),
            atol=_AFFINE_ABSOLUTE_TOLERANCE,
            rtol=0.0,
        )


def _apply_affine(
    affine_matrix: AffineMatrix,
    coordinate: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply a NIfTI affine to one voxel coordinate."""
    apply_affine = cast(
        Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
        nib.affines.apply_affine,
    )

    return apply_affine(
        np.asarray(affine_matrix, dtype=np.float64),
        coordinate,
    )


def _assert_output_arrays_match_expected(
    resampled_pair: ResampledImageMaskPair,
    transformed_pair: CenterCroppedPaddedImageMaskPair,
) -> None:
    """Reconstruct expected arrays through source slicing and NumPy padding."""
    expected_image = _crop_and_pad_expected_array(
        array=resampled_pair.image_data,
        transformed_pair=transformed_pair,
        constant_value=0.0,
        data_type=np.dtype(np.float32),
    )
    expected_mask = _crop_and_pad_expected_array(
        array=resampled_pair.mask_data,
        transformed_pair=transformed_pair,
        constant_value=0,
        data_type=np.dtype(np.int64),
    )

    assert np.array_equal(transformed_pair.image_data, expected_image)
    assert np.array_equal(transformed_pair.mask_data, expected_mask)


def _crop_and_pad_expected_array(
    array: NDArray[np.generic],
    transformed_pair: CenterCroppedPaddedImageMaskPair,
    constant_value: float | int,
    data_type: np.dtype[np.generic],
) -> NDArray[np.generic]:
    """Build an expected crop-and-pad output array."""
    cropped_array = array[
        transformed_pair.crop_start[0] : transformed_pair.crop_end[0],
        transformed_pair.crop_start[1] : transformed_pair.crop_end[1],
        transformed_pair.crop_start[2] : transformed_pair.crop_end[2],
    ]
    padding_width = (
        (transformed_pair.padding_before[0], transformed_pair.padding_after[0]),
        (transformed_pair.padding_before[1], transformed_pair.padding_after[1]),
        (transformed_pair.padding_before[2], transformed_pair.padding_after[2]),
    )

    return np.ascontiguousarray(
        np.pad(
            cropped_array,
            pad_width=padding_width,
            mode="constant",
            constant_values=constant_value,
        ),
        dtype=data_type,
    )
