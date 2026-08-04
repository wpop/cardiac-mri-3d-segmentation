from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Store the dataset identity and its local root directory."""

    name: str
    root_dir: Path

    def __post_init__(self) -> None:
        """Validate dataset configuration values."""
        if not self.name.strip():
            raise ValueError("Dataset name must not be empty.")

        if not str(self.root_dir).strip():
            raise ValueError("Dataset root directory must not be empty.")
