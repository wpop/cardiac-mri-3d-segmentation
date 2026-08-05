from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class NiftiMaskStatistics:
    """Store immutable class-label statistics for one NIfTI mask."""

    file_path: Path
    label_voxel_counts: tuple[tuple[int, int], ...]
    total_voxel_count: int

    def __post_init__(self) -> None:
        """Validate file, label, and voxel-count statistics."""
        if not self.file_path.is_file():
            raise FileNotFoundError(
                f"NIfTI mask file does not exist: {self.file_path}"
            )

        if self.total_voxel_count <= 0:
            raise ValueError(
                "Total mask voxel count must be greater than zero."
            )

        if not self.label_voxel_counts:
            raise ValueError(
                "Mask label statistics must contain at least one label."
            )

        labels = self.labels

        if any(label < 0 for label in labels):
            raise ValueError(
                "NIfTI mask labels must be non-negative integers."
            )

        if len(set(labels)) != len(labels):
            raise ValueError(
                "NIfTI mask labels must be unique."
            )

        if labels != tuple(sorted(labels)):
            raise ValueError(
                "NIfTI mask labels must be stored in ascending order."
            )

        if any(
            voxel_count <= 0
            for _, voxel_count in self.label_voxel_counts
        ):
            raise ValueError(
                "Every observed mask label must have a positive voxel count."
            )

        counted_voxel_count = sum(
            voxel_count
            for _, voxel_count in self.label_voxel_counts
        )

        if counted_voxel_count != self.total_voxel_count:
            raise ValueError(
                "The sum of label voxel counts must equal the total "
                "mask voxel count."
            )

    @property
    def labels(self) -> tuple[int, ...]:
        """Return all labels observed in the mask."""
        return tuple(
            label
            for label, _ in self.label_voxel_counts
        )

    def voxel_count_for_label(self, label: int) -> int:
        """Return the voxel count for a label or zero when it is absent."""
        for observed_label, voxel_count in self.label_voxel_counts:
            if observed_label == label:
                return voxel_count

        return 0
