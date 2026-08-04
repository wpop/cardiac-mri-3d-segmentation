from dataclasses import dataclass
from math import isfinite
from pathlib import Path

AffineMatrix = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


@dataclass(frozen=True, slots=True)
class NiftiVolumeMetadata:
    """Store validated metadata extracted from one three-dimensional NIfTI volume."""

    file_path: Path
    shape: tuple[int, int, int]
    voxel_spacing: tuple[float, float, float]
    orientation: tuple[str, str, str]
    affine: AffineMatrix
    data_type: str
    intensity_min: float
    intensity_max: float
    has_only_finite_values: bool

    def __post_init__(self) -> None:
        """Validate spatial, numerical, and file metadata."""
        if not self.file_path.is_file():
            raise FileNotFoundError(
                f"NIfTI file does not exist: {self.file_path}"
            )

        if any(dimension <= 0 for dimension in self.shape):
            raise ValueError(
                "NIfTI volume dimensions must be greater than zero."
            )

        if any(spacing <= 0.0 for spacing in self.voxel_spacing):
            raise ValueError(
                "NIfTI voxel spacing must be greater than zero."
            )

        valid_orientation_codes = {"L", "R", "A", "P", "I", "S"}

        if any(
            code not in valid_orientation_codes
            for code in self.orientation
        ):
            raise ValueError(
                "NIfTI orientation contains an unsupported axis code."
            )

        affine_values = [
            value
            for row in self.affine
            for value in row
        ]

        if not all(isfinite(value) for value in affine_values):
            raise ValueError(
                "NIfTI affine matrix must contain only finite values."
            )

        if not self.data_type.strip():
            raise ValueError("NIfTI data type must not be empty.")

        if not isfinite(self.intensity_min):
            raise ValueError("Minimum intensity must be finite.")

        if not isfinite(self.intensity_max):
            raise ValueError("Maximum intensity must be finite.")

        if self.intensity_min > self.intensity_max:
            raise ValueError(
                "Minimum intensity must not exceed maximum intensity."
            )
