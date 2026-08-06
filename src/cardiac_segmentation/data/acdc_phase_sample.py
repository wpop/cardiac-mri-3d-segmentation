from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AcdcPhaseSample:
    """Store file paths and identifiers for one ACDC cardiac phase."""

    patient_id: str
    split_name: str
    phase_name: str
    image_path: Path
    mask_path: Path

    def __post_init__(self) -> None:
        """Validate phase identity and NIfTI file paths."""
        if not self.patient_id.strip():
            raise ValueError("Patient identifier must not be empty.")

        if self.split_name not in {"training", "testing"}:
            raise ValueError("Dataset split must be either 'training' or 'testing'.")

        if self.phase_name not in {"ED", "ES"}:
            raise ValueError("Phase name must be either 'ED' or 'ES'.")

        resolved_image_path = self.image_path.expanduser().resolve(strict=False)
        resolved_mask_path = self.mask_path.expanduser().resolve(strict=False)

        if not resolved_image_path.is_file():
            raise FileNotFoundError(
                f"ACDC phase image file does not exist: {resolved_image_path}"
            )

        if not resolved_mask_path.is_file():
            raise FileNotFoundError(
                f"ACDC phase mask file does not exist: {resolved_mask_path}"
            )

        if resolved_image_path == resolved_mask_path:
            raise ValueError("ACDC phase image and mask paths must be different files.")
