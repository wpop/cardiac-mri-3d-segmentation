from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ValidationInferenceConfig:
    """Store parameters for deterministic validation inference reporting."""

    patient_count: int
    validation_fraction: float
    batch_size: int
    num_workers: int
    pin_memory: bool
    random_seed: int
    base_channels: int
    device: str
    checkpoint_path: Path
    csv_report_path: Path
    json_report_path: Path
    visualization_dir: Path

    def __post_init__(self) -> None:
        """Validate all validation inference settings."""
        if (
            isinstance(self.patient_count, bool)
            or not isinstance(self.patient_count, int)
            or self.patient_count <= 1
        ):
            raise ValueError("Patient count must be an integer greater than one.")

        if not isfinite(self.validation_fraction) or not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("Validation fraction must be finite and strictly between 0 and 1.")

        validation_patient_count = max(
            1,
            round(self.patient_count * self.validation_fraction),
        )
        training_patient_count = self.patient_count - validation_patient_count

        if training_patient_count <= 0 or validation_patient_count <= 0:
            raise ValueError(
                "Patient count and validation fraction must produce positive "
                "training and validation patient counts."
            )

        self._validate_positive_integer(self.batch_size, name="Batch size")
        self._validate_non_negative_integer(self.num_workers, name="Number of workers")

        if not isinstance(self.pin_memory, bool):
            raise TypeError("Pinned-memory setting must be a boolean.")

        self._validate_non_negative_integer(self.random_seed, name="Random seed")
        self._validate_positive_integer(self.base_channels, name="Base channel count")

        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Device must be exactly one of: auto, cpu, cuda.")

        if self.checkpoint_path.is_dir():
            raise IsADirectoryError(
                f"Checkpoint path must not be an existing directory: {self.checkpoint_path}"
            )

        if self.csv_report_path.is_dir():
            raise IsADirectoryError(
                f"CSV report path must not be an existing directory: {self.csv_report_path}"
            )

        if self.json_report_path.is_dir():
            raise IsADirectoryError(
                f"JSON report path must not be an existing directory: {self.json_report_path}"
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
