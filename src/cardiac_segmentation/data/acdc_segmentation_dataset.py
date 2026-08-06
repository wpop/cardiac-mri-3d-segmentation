import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from cardiac_segmentation.config.preprocessing_config import PreprocessingConfig
from cardiac_segmentation.config.validation_config import ValidationConfig
from cardiac_segmentation.data.acdc_patient_case import AcdcPatientCase
from cardiac_segmentation.data.acdc_phase_sample import AcdcPhaseSample
from cardiac_segmentation.preprocessing.nifti_image_mask_pair_center_crop_padder import (
    NiftiImageMaskPairCenterCropPadder,
)
from cardiac_segmentation.preprocessing.nifti_image_mask_pair_intensity_preprocessor import (
    NiftiImageMaskPairIntensityPreprocessor,
)
from cardiac_segmentation.preprocessing.nifti_image_mask_pair_loader import (
    NiftiImageMaskPairLoader,
)
from cardiac_segmentation.preprocessing.nifti_image_mask_pair_resampler import (
    NiftiImageMaskPairResampler,
)


class AcdcSegmentationDataset(Dataset[dict[str, Tensor | str]]):
    """Load real ACDC ED/ES phases and return preprocessed tensors."""

    def __init__(
        self,
        patient_cases: tuple[AcdcPatientCase, ...],
        preprocessing_config: PreprocessingConfig,
        validation_config: ValidationConfig,
    ) -> None:
        """Initialize samples and reusable preprocessing components."""
        if not patient_cases:
            raise ValueError("Patient cases must not be empty.")

        self._samples = self._build_phase_samples(patient_cases)
        self._target_shape = preprocessing_config.target_shape
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
        self._intensity_preprocessor = NiftiImageMaskPairIntensityPreprocessor(
            lower_percentile=preprocessing_config.intensity_lower_percentile,
            upper_percentile=preprocessing_config.intensity_upper_percentile,
            normalize_nonzero_only=preprocessing_config.normalize_nonzero_only,
            expected_labels=validation_config.expected_labels,
        )

    def __len__(self) -> int:
        """Return the number of ED/ES phase samples."""
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        """Load, preprocess, and tensorize one ACDC phase sample."""
        phase_sample = self._resolve_sample(index)
        loaded_pair = self._loader.load(
            image_path=phase_sample.image_path,
            mask_path=phase_sample.mask_path,
        )
        resampled_pair = self._resampler.resample(loaded_pair)
        crop_padded_pair = self._center_crop_padder.transform(resampled_pair)
        intensity_pair = self._intensity_preprocessor.transform(crop_padded_pair)
        image_tensor = self._create_image_tensor(intensity_pair.image_data)
        mask_tensor = self._create_mask_tensor(intensity_pair.mask_data)

        self._validate_tensors(
            image_tensor=image_tensor,
            mask_tensor=mask_tensor,
        )

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "patient_id": phase_sample.patient_id,
            "split_name": phase_sample.split_name,
            "phase_name": phase_sample.phase_name,
            "image_path": str(phase_sample.image_path),
            "mask_path": str(phase_sample.mask_path),
        }

    def _resolve_sample(
        self,
        index: int,
    ) -> AcdcPhaseSample:
        """Return a sample using normal Python sequence indexing rules."""
        try:
            return self._samples[index]
        except IndexError as error:
            raise IndexError(f"ACDC phase sample index is out of range: {index}") from error

    def _create_image_tensor(
        self,
        image_data: NDArray[np.float32],
    ) -> Tensor:
        """Convert an MRI array from (X, Y, Z) to contiguous (1, Z, Y, X)."""
        transposed_data = np.transpose(
            image_data,
            axes=(2, 1, 0),
        )

        return torch.from_numpy(transposed_data).unsqueeze(0).contiguous()

    def _create_mask_tensor(
        self,
        mask_data: NDArray[np.int64],
    ) -> Tensor:
        """Convert a mask array from (X, Y, Z) to contiguous (Z, Y, X)."""
        transposed_data = np.transpose(
            mask_data,
            axes=(2, 1, 0),
        )

        return torch.from_numpy(transposed_data).contiguous()

    def _validate_tensors(
        self,
        image_tensor: Tensor,
        mask_tensor: Tensor,
    ) -> None:
        """Validate tensor shape, dtype, contiguity, values, and labels."""
        expected_image_shape = (
            1,
            self._target_shape[2],
            self._target_shape[1],
            self._target_shape[0],
        )
        expected_mask_shape = (
            self._target_shape[2],
            self._target_shape[1],
            self._target_shape[0],
        )

        if tuple(image_tensor.shape) != expected_image_shape:
            raise ValueError(
                f"Image tensor shape must be {expected_image_shape}, but "
                f"received {tuple(image_tensor.shape)}."
            )

        if tuple(mask_tensor.shape) != expected_mask_shape:
            raise ValueError(
                f"Mask tensor shape must be {expected_mask_shape}, but "
                f"received {tuple(mask_tensor.shape)}."
            )

        if image_tensor.dtype != torch.float32:
            raise TypeError("Image tensor must have dtype torch.float32.")

        if mask_tensor.dtype != torch.int64:
            raise TypeError("Mask tensor must have dtype torch.int64.")

        if not image_tensor.is_contiguous():
            raise ValueError("Image tensor must be contiguous.")

        if not mask_tensor.is_contiguous():
            raise ValueError("Mask tensor must be contiguous.")

        if not bool(torch.isfinite(image_tensor).all().item()):
            raise ValueError("Image tensor contains non-finite values.")

        if not bool(torch.isfinite(mask_tensor).all().item()):
            raise ValueError("Mask tensor contains non-finite values.")

        unexpected_labels = tuple(
            int(label)
            for label in torch.unique(mask_tensor).tolist()
            if int(label) not in self._expected_labels
        )

        if unexpected_labels:
            raise ValueError(
                "Mask tensor contains labels outside the expected set: "
                f"{unexpected_labels}."
            )

    def _build_phase_samples(
        self,
        patient_cases: tuple[AcdcPatientCase, ...],
    ) -> tuple[AcdcPhaseSample, ...]:
        """Create ED then ES samples for each patient while preserving order."""
        samples: list[AcdcPhaseSample] = []
        seen_keys: set[tuple[str, str]] = set()

        for patient_case in patient_cases:
            samples.extend(
                (
                    AcdcPhaseSample(
                        patient_id=patient_case.patient_id,
                        split_name=patient_case.split_name,
                        phase_name="ED",
                        image_path=patient_case.ed_image_path,
                        mask_path=patient_case.ed_mask_path,
                    ),
                    AcdcPhaseSample(
                        patient_id=patient_case.patient_id,
                        split_name=patient_case.split_name,
                        phase_name="ES",
                        image_path=patient_case.es_image_path,
                        mask_path=patient_case.es_mask_path,
                    ),
                )
            )

        for sample in samples:
            key = (
                sample.patient_id,
                sample.phase_name,
            )

            if key in seen_keys:
                raise ValueError(
                    "Duplicate ACDC patient phase sample found: "
                    f"{sample.patient_id} {sample.phase_name}."
                )

            seen_keys.add(key)

        return tuple(samples)
