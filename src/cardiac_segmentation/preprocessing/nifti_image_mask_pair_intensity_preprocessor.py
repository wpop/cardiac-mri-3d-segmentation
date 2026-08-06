from math import isfinite
from typing import Final

import numpy as np
from numpy.typing import NDArray

from cardiac_segmentation.preprocessing.center_cropped_padded_image_mask_pair import (
    CenterCroppedPaddedImageMaskPair,
)
from cardiac_segmentation.preprocessing.intensity_preprocessed_image_mask_pair import (
    IntensityPreprocessedImageMaskPair,
)

_MINIMUM_PERCENTILE: Final[float] = 0.0
_MAXIMUM_PERCENTILE: Final[float] = 100.0


class NiftiImageMaskPairIntensityPreprocessor:
    """Clip MRI intensities by percentile and apply z-score normalization."""

    def __init__(
        self,
        lower_percentile: float,
        upper_percentile: float,
        normalize_nonzero_only: bool,
        expected_labels: tuple[int, ...],
    ) -> None:
        """Initialize intensity preprocessing and label validation policy."""
        self._validate_percentiles(
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )
        self._validate_normalize_nonzero_only(normalize_nonzero_only)
        self._validate_expected_labels(expected_labels)

        self._lower_percentile = lower_percentile
        self._upper_percentile = upper_percentile
        self._normalize_nonzero_only = normalize_nonzero_only
        self._expected_labels = frozenset(expected_labels)

    def transform(
        self,
        pair: CenterCroppedPaddedImageMaskPair,
    ) -> IntensityPreprocessedImageMaskPair:
        """Return a clipped and normalized copy of a crop-padded image-mask pair."""
        if not bool(np.isfinite(pair.image_data).all()):
            raise ValueError("Source MRI array contains non-finite values.")

        nonzero_mask = pair.image_data != 0.0

        if not bool(nonzero_mask.any()):
            raise ValueError("Source MRI array contains no non-zero voxels.")

        nonzero_values = pair.image_data[nonzero_mask]
        lower_clip_value = float(
            np.percentile(nonzero_values, self._lower_percentile)
        )
        upper_clip_value = float(
            np.percentile(nonzero_values, self._upper_percentile)
        )
        image_data = pair.image_data.astype(np.float32, copy=True)
        image_data[nonzero_mask] = np.clip(
            image_data[nonzero_mask],
            lower_clip_value,
            upper_clip_value,
        )

        normalization_mean, normalization_standard_deviation = (
            self._calculate_normalization_statistics(
                image_data=image_data,
                nonzero_mask=nonzero_mask,
            )
        )
        normalized_image_data = self._normalize_image(
            image_data=image_data,
            nonzero_mask=nonzero_mask,
            mean=normalization_mean,
            standard_deviation=normalization_standard_deviation,
        )
        mask_data = np.array(
            pair.mask_data,
            dtype=np.int64,
            order="C",
            copy=True,
        )
        self._validate_mask_labels(mask_data)

        return IntensityPreprocessedImageMaskPair(
            source_pair=pair,
            image_data=normalized_image_data,
            mask_data=mask_data,
            lower_clip_value=lower_clip_value,
            upper_clip_value=upper_clip_value,
            normalization_mean=normalization_mean,
            normalization_standard_deviation=normalization_standard_deviation,
            normalize_nonzero_only=self._normalize_nonzero_only,
        )

    def _calculate_normalization_statistics(
        self,
        image_data: NDArray[np.float32],
        nonzero_mask: NDArray[np.bool_],
    ) -> tuple[float, float]:
        """Calculate z-score statistics from the configured voxel population."""
        if self._normalize_nonzero_only:
            normalization_values = image_data[nonzero_mask]
        else:
            normalization_values = image_data

        mean = float(np.mean(normalization_values))
        standard_deviation = float(np.std(normalization_values))

        if not isfinite(mean):
            raise ValueError("Normalization mean must be finite.")

        if not isfinite(standard_deviation) or standard_deviation <= 0.0:
            raise ValueError(
                "Normalization standard deviation must be finite and greater "
                "than zero."
            )

        return mean, standard_deviation

    def _normalize_image(
        self,
        image_data: NDArray[np.float32],
        nonzero_mask: NDArray[np.bool_],
        mean: float,
        standard_deviation: float,
    ) -> NDArray[np.float32]:
        """Apply z-score normalization to the configured voxel population."""
        normalized_data = image_data.astype(np.float32, copy=True)

        if self._normalize_nonzero_only:
            normalized_data[nonzero_mask] = (
                normalized_data[nonzero_mask] - mean
            ) / standard_deviation
        else:
            normalized_data = (normalized_data - mean) / standard_deviation

        return np.ascontiguousarray(
            normalized_data,
            dtype=np.float32,
        )

    def _validate_mask_labels(
        self,
        mask_data: NDArray[np.int64],
    ) -> None:
        """Validate that mask labels remain within the expected set."""
        unexpected_labels = tuple(
            int(label)
            for label in np.unique(mask_data)
            if int(label) not in self._expected_labels
        )

        if unexpected_labels:
            raise ValueError(
                "Intensity-preprocessed mask contains labels outside the "
                f"expected set: {unexpected_labels}."
            )

    @staticmethod
    def _validate_percentiles(
        lower_percentile: float,
        upper_percentile: float,
    ) -> None:
        """Validate configured clipping percentiles."""
        if not isfinite(lower_percentile):
            raise ValueError("Lower percentile must be finite.")

        if not isfinite(upper_percentile):
            raise ValueError("Upper percentile must be finite.")

        if not (
            _MINIMUM_PERCENTILE
            <= lower_percentile
            < upper_percentile
            <= _MAXIMUM_PERCENTILE
        ):
            raise ValueError(
                "Percentiles must satisfy 0 <= lower < upper <= 100."
            )

    @staticmethod
    def _validate_normalize_nonzero_only(
        normalize_nonzero_only: bool,
    ) -> None:
        """Validate the normalization population selector."""
        if not isinstance(normalize_nonzero_only, bool):
            raise TypeError("normalize_nonzero_only must be a boolean.")

    @staticmethod
    def _validate_expected_labels(
        expected_labels: tuple[int, ...],
    ) -> None:
        """Validate the configured segmentation labels."""
        if not expected_labels:
            raise ValueError("Expected labels must not be empty.")

        if any(
            isinstance(label, bool)
            or not isinstance(label, int)
            or label < 0
            for label in expected_labels
        ):
            raise ValueError("Expected labels must be non-negative integers.")

        if len(set(expected_labels)) != len(expected_labels):
            raise ValueError("Expected labels must be unique.")

        if 0 not in expected_labels:
            raise ValueError("Expected labels must include background label 0.")
