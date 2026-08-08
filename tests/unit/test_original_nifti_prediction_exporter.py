import numpy as np

from cardiac_segmentation.config.preprocessing_config import PreprocessingConfig
from cardiac_segmentation.config.validation_config import ValidationConfig
from cardiac_segmentation.evaluation import OriginalNiftiPredictionExporter


def test_inverse_center_crop_restores_resampled_shape_and_placement() -> None:
    exporter = _create_exporter()
    prediction = np.ones((4, 4, 4), dtype=np.int64)

    restored_prediction = exporter.restore_center_crop_pad(
        prediction_xyz=prediction,
        resampled_shape=(6, 4, 4),
        crop_bounds=((1, 0, 0), (5, 4, 4)),
        padding=((0, 0, 0), (0, 0, 0)),
    )

    assert restored_prediction.shape == (6, 4, 4)
    assert np.all(restored_prediction[0] == 0)
    assert np.all(restored_prediction[1:5] == 1)
    assert np.all(restored_prediction[5] == 0)


def test_inverse_center_pad_removes_model_space_padding() -> None:
    exporter = _create_exporter()
    prediction = np.zeros((4, 4, 4), dtype=np.int64)
    prediction[0] = 3
    prediction[1:3] = 2
    prediction[3] = 3

    restored_prediction = exporter.restore_center_crop_pad(
        prediction_xyz=prediction,
        resampled_shape=(2, 4, 4),
        crop_bounds=((0, 0, 0), (2, 4, 4)),
        padding=((1, 0, 0), (1, 0, 0)),
    )

    assert restored_prediction.shape == (2, 4, 4)
    assert np.all(restored_prediction == 2)


def test_label_preservation_keeps_only_expected_labels_after_restoration() -> None:
    exporter = _create_exporter()
    prediction = np.array(
        [
            [[0, 1], [2, 3]],
            [[3, 2], [1, 0]],
        ],
        dtype=np.int64,
    )

    restored_prediction = exporter.restore_center_crop_pad(
        prediction_xyz=prediction,
        resampled_shape=(2, 2, 2),
        crop_bounds=((0, 0, 0), (2, 2, 2)),
        padding=((0, 0, 0), (0, 0, 0)),
    )

    assert set(np.unique(restored_prediction)) == {0, 1, 2, 3}


def test_nearest_neighbor_style_integer_labels_do_not_become_fractional() -> None:
    exporter = _create_exporter()
    prediction = np.full((3, 3, 3), 3, dtype=np.int64)

    restored_prediction = exporter.restore_center_crop_pad(
        prediction_xyz=prediction,
        resampled_shape=(3, 3, 3),
        crop_bounds=((0, 0, 0), (3, 3, 3)),
        padding=((0, 0, 0), (0, 0, 0)),
    )

    assert np.issubdtype(restored_prediction.dtype, np.integer)
    assert bool(np.equal(restored_prediction, np.rint(restored_prediction)).all())


def test_axis_order_conversion_maps_dhw_to_xyz() -> None:
    exporter = _create_exporter()
    prediction_dhw = np.zeros((2, 3, 4), dtype=np.int64)
    prediction_dhw[1, 2, 3] = 2

    prediction_xyz = exporter.convert_model_prediction_to_nifti_order(prediction_dhw)

    assert prediction_xyz.shape == (4, 3, 2)
    assert prediction_xyz[3, 2, 1] == 2


def _create_exporter() -> OriginalNiftiPredictionExporter:
    return OriginalNiftiPredictionExporter(
        preprocessing_config=PreprocessingConfig(
            target_spacing_mm=(1.5, 1.5, 5.0),
            target_shape=(4, 4, 4),
            intensity_lower_percentile=1.0,
            intensity_upper_percentile=99.0,
            normalize_nonzero_only=True,
        ),
        validation_config=ValidationConfig(
            expected_labels=(0, 1, 2, 3),
            affine_absolute_tolerance=1e-5,
            require_finite_intensities=True,
            require_positive_voxel_spacing=True,
        ),
    )
