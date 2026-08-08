from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ValidationVisualizationCaseResult:
    """Store generated slice-review files for one ranked validation case."""

    rank_name: str
    volume_id: str
    slice_indices: tuple[int, ...]
    file_paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        """Validate generated visualization metadata."""
        if self.rank_name not in {"worst", "middle", "best"}:
            raise ValueError("Rank name must be one of: worst, middle, best.")

        if not self.volume_id:
            raise ValueError("Volume identifier must not be empty.")

        if not self.slice_indices:
            raise ValueError("At least one slice index is required.")

        if any(index < 0 for index in self.slice_indices):
            raise ValueError("Slice indices must be non-negative.")

        if len(set(self.slice_indices)) != len(self.slice_indices):
            raise ValueError("Slice indices must be unique.")

        if len(self.file_paths) != len(self.slice_indices):
            raise ValueError("File paths must match slice indices.")

        if len(set(self.file_paths)) != len(self.file_paths):
            raise ValueError("Visualization file paths must be unique.")
