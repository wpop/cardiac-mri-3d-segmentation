from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Store validation rules for image and segmentation metadata."""

    expected_labels: tuple[int, ...]
    affine_absolute_tolerance: float
    require_finite_intensities: bool
    require_positive_voxel_spacing: bool

    def __post_init__(self) -> None:
        """Validate labels and numerical tolerances."""
        if not self.expected_labels:
            raise ValueError("Expected segmentation labels must not be empty.")

        if any(label < 0 for label in self.expected_labels):
            raise ValueError("Segmentation labels must be non-negative integers.")

        if len(set(self.expected_labels)) != len(self.expected_labels):
            raise ValueError("Expected segmentation labels must be unique.")

        if 0 not in self.expected_labels:
            raise ValueError("Expected segmentation labels must include background label 0.")

        if self.affine_absolute_tolerance <= 0.0:
            raise ValueError("Affine absolute tolerance must be greater than zero.")
