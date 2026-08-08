from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Final

_MAX_EXPORT_CASE_COUNT: Final[int] = 3


@dataclass(frozen=True, slots=True)
class ValidationVisualizationConfig:
    """Store deterministic validation visualization settings."""

    patient_count: int
    validation_fraction: float
    random_seed: int
    base_channels: int
    device: str
    checkpoint_path: Path
    output_dir: Path
    report_csv_path: Path
    report_json_path: Path
    export_case_count: int
    slices_per_case: int

    def __post_init__(self) -> None:
        """Validate visualization configuration values."""
        self._validate_positive_integer(self.patient_count, name="Patient count")

        if self.patient_count <= 1:
            raise ValueError("Patient count must be greater than one.")

        if not isfinite(self.validation_fraction) or not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("Validation fraction must be finite and strictly between 0 and 1.")

        validation_patient_count = max(1, round(self.patient_count * self.validation_fraction))

        if self.patient_count - validation_patient_count <= 0:
            raise ValueError("Patient count and validation fraction must leave training cases.")

        self._validate_non_negative_integer(self.random_seed, name="Random seed")
        self._validate_positive_integer(self.base_channels, name="Base channel count")
        self._validate_positive_integer(self.export_case_count, name="Export case count")
        self._validate_positive_integer(self.slices_per_case, name="Slices per case")

        if self.export_case_count > _MAX_EXPORT_CASE_COUNT:
            raise ValueError("Export case count must be between 1 and 3.")

        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Device must be exactly one of: auto, cpu, cuda.")

        if self.checkpoint_path.is_dir():
            raise IsADirectoryError(
                f"Checkpoint path must not be an existing directory: {self.checkpoint_path}"
            )

        if self.output_dir.is_file():
            raise NotADirectoryError(
                f"Output directory must not be an existing file: {self.output_dir}"
            )

        if self.report_csv_path.is_dir():
            raise IsADirectoryError(
                f"CSV report path must not be an existing directory: {self.report_csv_path}"
            )

        if self.report_json_path.is_dir():
            raise IsADirectoryError(
                f"JSON report path must not be an existing directory: {self.report_json_path}"
            )

    @staticmethod
    def _validate_positive_integer(
        value: int,
        *,
        name: str,
    ) -> None:
        """Validate a positive integer setting."""
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    @staticmethod
    def _validate_non_negative_integer(
        value: int,
        *,
        name: str,
    ) -> None:
        """Validate a non-negative integer setting."""
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
