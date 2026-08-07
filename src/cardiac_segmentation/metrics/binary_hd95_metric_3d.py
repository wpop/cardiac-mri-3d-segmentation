from math import isfinite
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage  # type: ignore[import-untyped]

_SPATIAL_DIMENSION_COUNT: Final[int] = 3


class BinaryHD95Metric3D:
    """Calculate binary 3D HD95 in physical millimeters.

    Empty masks have explicit clinical-reporting semantics: if ground truth and
    prediction are both empty, HD95 is 0.0 mm; if only one mask is empty, HD95 is
    infinity because no finite surface distance can represent the miss.
    """

    def __init__(
        self,
        spacing_mm_dhw: tuple[float, float, float],
    ) -> None:
        """Initialize physical spacing in tensor axis order [D, H, W]."""
        self._validate_spacing(spacing_mm_dhw)
        self._spacing_mm_dhw = spacing_mm_dhw

    def compute(
        self,
        *,
        ground_truth_mask: NDArray[np.bool_],
        prediction_mask: NDArray[np.bool_],
    ) -> float:
        """Return symmetric 95th-percentile surface distance in millimeters."""
        self._validate_mask(ground_truth_mask, name="Ground-truth mask")
        self._validate_mask(prediction_mask, name="Prediction mask")

        ground_truth_has_foreground = bool(np.any(ground_truth_mask))
        prediction_has_foreground = bool(np.any(prediction_mask))

        if not ground_truth_has_foreground and not prediction_has_foreground:
            return 0.0

        if ground_truth_has_foreground != prediction_has_foreground:
            return float("inf")

        ground_truth_surface = self._extract_surface(ground_truth_mask)
        prediction_surface = self._extract_surface(prediction_mask)
        distances = np.concatenate(
            (
                self._directed_surface_distances(
                    source_surface=prediction_surface,
                    target_surface=ground_truth_surface,
                ),
                self._directed_surface_distances(
                    source_surface=ground_truth_surface,
                    target_surface=prediction_surface,
                ),
            )
        )

        return float(np.percentile(distances, 95))

    @staticmethod
    def _extract_surface(
        mask: NDArray[np.bool_],
    ) -> NDArray[np.bool_]:
        """Return foreground voxels that touch background in 3D."""
        eroded_mask = ndimage.binary_erosion(
            mask,
            structure=np.ones((3, 3, 3), dtype=bool),
            border_value=0,
        )

        return cast(
            NDArray[np.bool_],
            np.logical_and(mask, np.logical_not(eroded_mask)),
        )

    def _directed_surface_distances(
        self,
        *,
        source_surface: NDArray[np.bool_],
        target_surface: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Return distances from every source surface voxel to the target surface."""
        distance_map = ndimage.distance_transform_edt(
            np.logical_not(target_surface),
            sampling=self._spacing_mm_dhw,
        )

        return cast(NDArray[np.float64], distance_map[source_surface])

    @staticmethod
    def _validate_mask(
        mask: NDArray[np.bool_],
        *,
        name: str,
    ) -> None:
        """Validate a binary 3D NumPy mask."""
        if mask.ndim != _SPATIAL_DIMENSION_COUNT:
            raise ValueError(f"{name} must be three-dimensional [D, H, W].")

        if mask.dtype != np.bool_:
            raise TypeError(f"{name} must use bool dtype.")

        if any(dimension <= 0 for dimension in mask.shape):
            raise ValueError(f"{name} spatial dimensions must be positive.")

    @staticmethod
    def _validate_spacing(
        spacing_mm_dhw: tuple[float, float, float],
    ) -> None:
        """Validate physical spacing in tensor axis order."""
        if len(spacing_mm_dhw) != _SPATIAL_DIMENSION_COUNT:
            raise ValueError("HD95 spacing must contain D, H, and W values.")

        if any(not isfinite(value) or value <= 0.0 for value in spacing_mm_dhw):
            raise ValueError("HD95 spacing values must be finite and strictly positive.")
