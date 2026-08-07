from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PatientLevelResumeTrainingConfig:
    """Store parameters for resuming deterministic patient-level training."""

    patient_count: int
    validation_fraction: float
    final_epoch_number: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    random_seed: int
    base_channels: int
    learning_rate: float
    weight_decay: float
    device: str
    resume_checkpoint_path: Path
    checkpoint_path: Path
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 8
    early_stopping_patience: int = 20
    early_stopping_minimum_improvement: float = 0.001

    def __post_init__(self) -> None:
        """Validate all resumed patient-level training settings."""
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

        self._validate_positive_integer(
            self.final_epoch_number,
            name="Final epoch number",
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

        if not self.resume_checkpoint_path.is_file():
            raise FileNotFoundError(
                "Resume checkpoint path must exist and be a regular file: "
                f"{self.resume_checkpoint_path}"
            )

        if self.checkpoint_path.is_dir():
            raise IsADirectoryError(
                f"Checkpoint path must not be an existing directory: {self.checkpoint_path}"
            )

        if not isfinite(self.lr_scheduler_factor) or not 0.0 < self.lr_scheduler_factor < 1.0:
            raise ValueError("LR scheduler factor must be finite and inside (0.0, 1.0).")

        self._validate_positive_integer(
            self.lr_scheduler_patience,
            name="LR scheduler patience",
        )
        self._validate_positive_integer(
            self.early_stopping_patience,
            name="Early-stopping patience",
        )

        if (
            not isfinite(self.early_stopping_minimum_improvement)
            or self.early_stopping_minimum_improvement < 0.0
        ):
            raise ValueError(
                "Early-stopping minimum improvement must be finite and non-negative."
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
