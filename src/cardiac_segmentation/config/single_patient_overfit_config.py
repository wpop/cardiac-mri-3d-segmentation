from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SinglePatientOverfitConfig:
    """Store parameters for a diagnostic single-patient overfit run."""

    patient_id: str
    epoch_count: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    random_seed: int
    base_channels: int
    learning_rate: float
    weight_decay: float
    device: str
    checkpoint_path: Path

    def __post_init__(self) -> None:
        """Validate all diagnostic overfit experiment settings."""
        if not self.patient_id.strip():
            raise ValueError("Patient identifier must not be empty.")

        self._validate_positive_integer(
            self.epoch_count,
            name="Epoch count",
        )
        self._validate_positive_integer(
            self.batch_size,
            name="Batch size",
        )
        self._validate_non_negative_integer(
            self.num_workers,
            name="Number of workers",
        )

        if not isinstance(self.pin_memory, bool):
            raise TypeError("Pinned-memory setting must be a boolean.")

        self._validate_non_negative_integer(
            self.random_seed,
            name="Random seed",
        )
        self._validate_positive_integer(
            self.base_channels,
            name="Base channel count",
        )

        if not isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("Learning rate must be finite and strictly positive.")

        if not isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("Weight decay must be finite and non-negative.")

        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Device must be exactly one of: auto, cpu, cuda.")

        if self.checkpoint_path.is_dir():
            raise IsADirectoryError(
                f"Checkpoint path must not be an existing directory: {self.checkpoint_path}"
            )

    @staticmethod
    def _validate_positive_integer(
        value: int,
        *,
        name: str,
    ) -> None:
        """Validate an integer setting that must be positive."""
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    @staticmethod
    def _validate_non_negative_integer(
        value: int,
        *,
        name: str,
    ) -> None:
        """Validate an integer setting that may be zero or positive."""
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
