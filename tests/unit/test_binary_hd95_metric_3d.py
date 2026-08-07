from math import isinf

import numpy as np
import pytest

from cardiac_segmentation.metrics import BinaryHD95Metric3D

_ISOTROPIC_SPACING_MM = (1.0, 1.0, 1.0)
_ACDC_TENSOR_SPACING_DHW_MM = (5.0, 1.5, 1.5)


def test_identical_binary_masks_have_zero_hd95() -> None:
    mask = _single_voxel_mask(index=(2, 2, 2))
    metric = BinaryHD95Metric3D(spacing_mm_dhw=_ISOTROPIC_SPACING_MM)

    assert metric.compute(ground_truth_mask=mask, prediction_mask=mask) == 0.0


def test_one_voxel_displacement_with_isotropic_spacing_uses_physical_distance() -> None:
    ground_truth = _single_voxel_mask(index=(2, 2, 2))
    prediction = _single_voxel_mask(index=(2, 2, 3))
    metric = BinaryHD95Metric3D(spacing_mm_dhw=_ISOTROPIC_SPACING_MM)

    assert metric.compute(
        ground_truth_mask=ground_truth,
        prediction_mask=prediction,
    ) == pytest.approx(1.0)


def test_depth_axis_displacement_uses_anisotropic_depth_spacing() -> None:
    ground_truth = _single_voxel_mask(index=(2, 2, 2))
    prediction = _single_voxel_mask(index=(3, 2, 2))
    metric = BinaryHD95Metric3D(spacing_mm_dhw=_ACDC_TENSOR_SPACING_DHW_MM)

    assert metric.compute(
        ground_truth_mask=ground_truth,
        prediction_mask=prediction,
    ) == pytest.approx(5.0)


@pytest.mark.parametrize("prediction_index", [(2, 3, 2), (2, 2, 3)])
def test_height_and_width_axis_displacements_use_in_plane_spacing(
    prediction_index: tuple[int, int, int],
) -> None:
    ground_truth = _single_voxel_mask(index=(2, 2, 2))
    prediction = _single_voxel_mask(index=prediction_index)
    metric = BinaryHD95Metric3D(spacing_mm_dhw=_ACDC_TENSOR_SPACING_DHW_MM)

    assert metric.compute(
        ground_truth_mask=ground_truth,
        prediction_mask=prediction,
    ) == pytest.approx(1.5)


def test_hd95_is_symmetric() -> None:
    first_mask = _single_voxel_mask(index=(2, 2, 2))
    second_mask = _single_voxel_mask(index=(2, 3, 2))
    metric = BinaryHD95Metric3D(spacing_mm_dhw=_ACDC_TENSOR_SPACING_DHW_MM)

    first_to_second = metric.compute(
        ground_truth_mask=first_mask,
        prediction_mask=second_mask,
    )
    second_to_first = metric.compute(
        ground_truth_mask=second_mask,
        prediction_mask=first_mask,
    )

    assert first_to_second == pytest.approx(second_to_first)


def test_both_empty_masks_have_zero_hd95() -> None:
    empty_mask = np.zeros((3, 3, 3), dtype=bool)
    metric = BinaryHD95Metric3D(spacing_mm_dhw=_ISOTROPIC_SPACING_MM)

    assert metric.compute(
        ground_truth_mask=empty_mask,
        prediction_mask=empty_mask,
    ) == 0.0


def test_one_empty_mask_has_infinite_hd95() -> None:
    empty_mask = np.zeros((3, 3, 3), dtype=bool)
    non_empty_mask = _single_voxel_mask(index=(1, 1, 1), shape=(3, 3, 3))
    metric = BinaryHD95Metric3D(spacing_mm_dhw=_ISOTROPIC_SPACING_MM)

    assert isinf(
        metric.compute(
            ground_truth_mask=empty_mask,
            prediction_mask=non_empty_mask,
        )
    )
    assert isinf(
        metric.compute(
            ground_truth_mask=non_empty_mask,
            prediction_mask=empty_mask,
        )
    )


def _single_voxel_mask(
    *,
    index: tuple[int, int, int],
    shape: tuple[int, int, int] = (5, 5, 5),
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[index] = True

    return mask
