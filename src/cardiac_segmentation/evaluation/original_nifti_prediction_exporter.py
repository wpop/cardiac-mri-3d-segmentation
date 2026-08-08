from collections.abc import Callable
from pathlib import Path
from typing import Final, cast

import nibabel as nib
import numpy as np
from nibabel.nifti1 import Nifti1Header, Nifti1Image
from nibabel.processing import resample_from_to
from numpy.typing import NDArray

from cardiac_segmentation.config.preprocessing_config import PreprocessingConfig
from cardiac_segmentation.config.validation_config import ValidationConfig
from cardiac_segmentation.preprocessing.nifti_image_mask_pair_center_crop_padder import (
    NiftiImageMaskPairCenterCropPadder,
)
from cardiac_segmentation.preprocessing.nifti_image_mask_pair_loader import (
    NiftiImageMaskPairLoader,
)
from cardiac_segmentation.preprocessing.nifti_image_mask_pair_resampler import (
    NiftiImageMaskPairResampler,
)

_SPATIAL_DIMENSION_COUNT: Final[int] = 3


class OriginalNiftiPredictionExporter:
    """Restore model-space segmentation labels and save original-space NIfTI files."""

    def __init__(
        self,
        preprocessing_config: PreprocessingConfig,
        validation_config: ValidationConfig,
    ) -> None:
        """Initialize preprocessing geometry and label validation components."""
        self._expected_labels = frozenset(validation_config.expected_labels)
        self._loader = NiftiImageMaskPairLoader(
            expected_labels=validation_config.expected_labels,
            affine_absolute_tolerance=validation_config.affine_absolute_tolerance,
            require_finite_intensities=validation_config.require_finite_intensities,
        )
        self._resampler = NiftiImageMaskPairResampler(
            target_spacing_mm=preprocessing_config.target_spacing_mm,
            expected_labels=validation_config.expected_labels,
        )
        self._center_crop_padder = NiftiImageMaskPairCenterCropPadder(
            target_shape=preprocessing_config.target_shape,
            expected_labels=validation_config.expected_labels,
        )

    def export(
        self,
        *,
        prediction_dhw: NDArray[np.int64],
        image_path: Path,
        mask_path: Path,
        output_path: Path,
    ) -> Path:
        """Restore one model prediction to original image geometry and save it."""
        self._validate_prediction(prediction_dhw)
        loaded_pair = self._loader.load(
            image_path=image_path,
            mask_path=mask_path,
        )
        resampled_pair = self._resampler.resample(loaded_pair)
        crop_padded_pair = self._center_crop_padder.transform(resampled_pair)
        prediction_xyz = self.convert_model_prediction_to_nifti_order(prediction_dhw)
        restored_resampled_prediction = self.restore_center_crop_pad(
            prediction_xyz=prediction_xyz,
            resampled_shape=crop_padded_pair.source_pair.shape,
            crop_bounds=(crop_padded_pair.crop_start, crop_padded_pair.crop_end),
            padding=(crop_padded_pair.padding_before, crop_padded_pair.padding_after),
        )
        original_prediction = self._resample_to_original_geometry(
            prediction_xyz=restored_resampled_prediction,
            source_affine=np.asarray(crop_padded_pair.source_pair.affine, dtype=np.float64),
            original_image_path=loaded_pair.image_path,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_prediction(
            prediction_xyz=original_prediction,
            original_image_path=loaded_pair.image_path,
            output_path=output_path,
        )

        return output_path

    def convert_model_prediction_to_nifti_order(
        self,
        prediction_dhw: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        """Convert a label array from tensor [D,H,W] order to NIfTI [X,Y,Z]."""
        self._validate_prediction(prediction_dhw)

        return np.ascontiguousarray(
            np.transpose(prediction_dhw, axes=(2, 1, 0)),
            dtype=np.int64,
        )

    def restore_center_crop_pad(
        self,
        *,
        prediction_xyz: NDArray[np.int64],
        resampled_shape: tuple[int, int, int],
        crop_bounds: tuple[tuple[int, int, int], tuple[int, int, int]],
        padding: tuple[tuple[int, int, int], tuple[int, int, int]],
    ) -> NDArray[np.int64]:
        """Undo centered crop/pad geometry and return the resampled-space labels."""
        self._validate_prediction(prediction_xyz)
        self._validate_spatial_tuple(resampled_shape, name="Resampled shape")
        crop_start, crop_end = crop_bounds
        padding_before, padding_after = padding
        self._validate_spatial_tuple(crop_start, name="Crop start", allow_zero=True)
        self._validate_spatial_tuple(crop_end, name="Crop end", allow_zero=True)
        self._validate_spatial_tuple(padding_before, name="Padding before", allow_zero=True)
        self._validate_spatial_tuple(padding_after, name="Padding after", allow_zero=True)

        unpadded_prediction = prediction_xyz[
            padding_before[0] : prediction_xyz.shape[0] - padding_after[0],
            padding_before[1] : prediction_xyz.shape[1] - padding_after[1],
            padding_before[2] : prediction_xyz.shape[2] - padding_after[2],
        ]
        expected_unpadded_shape = (
            crop_end[0] - crop_start[0],
            crop_end[1] - crop_start[1],
            crop_end[2] - crop_start[2],
        )

        if tuple(int(value) for value in unpadded_prediction.shape) != expected_unpadded_shape:
            raise ValueError("Unpadded prediction shape does not match crop bounds.")

        restored_prediction = np.zeros(resampled_shape, dtype=np.int64)
        restored_prediction[
            crop_start[0] : crop_end[0],
            crop_start[1] : crop_end[1],
            crop_start[2] : crop_end[2],
        ] = unpadded_prediction

        self._validate_prediction(restored_prediction)

        return restored_prediction

    def _resample_to_original_geometry(
        self,
        *,
        prediction_xyz: NDArray[np.int64],
        source_affine: NDArray[np.float64],
        original_image_path: Path,
    ) -> NDArray[np.int64]:
        """Nearest-neighbor resample labels from resampled space to original grid."""
        source_image = self._create_nifti_image(
            data=prediction_xyz,
            affine_matrix=source_affine,
            data_type=np.dtype(np.int64),
        )
        original_image = self._load_nifti_image(original_image_path)
        resampled_image = self._resample_from_to(
            source_image=source_image,
            target=original_image,
            order=0,
            mode="constant",
            background_value=0.0,
        )
        raw_data = np.asarray(resampled_image.dataobj)

        if not bool(np.isfinite(raw_data).all()):
            raise ValueError("Original-space prediction contains non-finite values.")

        rounded_data = np.rint(raw_data)

        if not bool(np.equal(raw_data, rounded_data).all()):
            raise ValueError("Original-space prediction contains fractional labels.")

        prediction = cast(
            NDArray[np.int64],
            rounded_data.astype(np.int64, copy=False),
        )
        self._validate_prediction(prediction)

        return prediction

    def _save_prediction(
        self,
        *,
        prediction_xyz: NDArray[np.int64],
        original_image_path: Path,
        output_path: Path,
    ) -> None:
        """Save the label prediction with the original image affine and header grid."""
        original_image = self._load_nifti_image(original_image_path)
        original_affine = np.asarray(original_image.affine, dtype=np.float64)
        copy_header = cast(Callable[[], Nifti1Header], original_image.header.copy)
        header = copy_header()
        set_data_dtype = cast(Callable[[object], None], header.set_data_dtype)
        set_data_dtype(np.uint8)
        output_image = self._create_nifti_image(
            data=prediction_xyz.astype(np.uint8, copy=False),
            affine_matrix=original_affine,
            header=header,
            data_type=np.dtype(np.uint8),
        )
        save = cast(Callable[[Nifti1Image, str], None], nib.save)
        save(output_image, str(output_path))

    def _create_nifti_image(
        self,
        data: NDArray[np.generic],
        affine_matrix: NDArray[np.float64],
        data_type: np.dtype[np.generic],
        header: Nifti1Header | None = None,
    ) -> Nifti1Image:
        """Construct a NIfTI image through a narrow typed wrapper."""
        image_factory = cast(
            Callable[
                [
                    NDArray[np.generic],
                    NDArray[np.float64],
                    Nifti1Header | None,
                    object,
                    object,
                    np.dtype[np.generic],
                ],
                Nifti1Image,
            ],
            Nifti1Image,
        )

        return image_factory(
            data,
            affine_matrix,
            header,
            None,
            None,
            data_type,
        )

    @staticmethod
    def _resample_from_to(
        *,
        source_image: Nifti1Image,
        target: Nifti1Image,
        order: int,
        mode: str,
        background_value: float,
    ) -> Nifti1Image:
        """Call NiBabel's resampling helper through a narrow typed wrapper."""
        resample = cast(
            Callable[[Nifti1Image, Nifti1Image, int, str, float], Nifti1Image],
            resample_from_to,
        )

        return resample(
            source_image,
            target,
            order,
            mode,
            background_value,
        )

    @staticmethod
    def _load_nifti_image(
        image_path: Path,
    ) -> Nifti1Image:
        """Load a NIfTI image through a narrow typed wrapper."""
        return cast(
            Nifti1Image,
            nib.load(str(image_path)),
        )

    def _validate_prediction(
        self,
        prediction: NDArray[np.int64],
    ) -> None:
        """Validate a 3D integer label prediction."""
        if prediction.ndim != _SPATIAL_DIMENSION_COUNT:
            raise ValueError("Prediction must be three-dimensional.")

        if prediction.dtype != np.dtype(np.int64):
            raise TypeError("Prediction must use int64 dtype.")

        if any(int(dimension) <= 0 for dimension in prediction.shape):
            raise ValueError("Prediction spatial dimensions must be positive.")

        unexpected_labels = tuple(
            int(label)
            for label in np.unique(prediction)
            if int(label) not in self._expected_labels
        )

        if unexpected_labels:
            raise ValueError(
                "Prediction contains labels outside the expected set: "
                f"{unexpected_labels}."
            )

    @staticmethod
    def _validate_spatial_tuple(
        value: tuple[int, int, int],
        *,
        name: str,
        allow_zero: bool = False,
    ) -> None:
        """Validate a spatial tuple used for inverse crop/pad geometry."""
        minimum_value = 0 if allow_zero else 1

        if len(value) != _SPATIAL_DIMENSION_COUNT:
            raise ValueError(f"{name} must contain exactly three values.")

        if any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < minimum_value
            for dimension in value
        ):
            raise ValueError(f"{name} contains invalid spatial dimensions.")
