import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AcdcPatientInfo:
    """Store validated clinical and cardiac-phase metadata for one ACDC patient."""

    patient_id: str
    ed_frame: int
    es_frame: int
    clinical_group: str
    height_cm: float
    frame_count: int
    weight_kg: float

    def __post_init__(self) -> None:
        """Validate patient identity, phase indices, and clinical measurements."""
        if re.fullmatch(r"patient\d{3}", self.patient_id) is None:
            raise ValueError(
                "Patient identifier must match the format 'patientNNN'."
            )

        if self.frame_count <= 0:
            raise ValueError("Frame count must be greater than zero.")

        if not 1 <= self.ed_frame <= self.frame_count:
            raise ValueError(
                "End-diastolic frame must be within the available frame range."
            )

        if not 1 <= self.es_frame <= self.frame_count:
            raise ValueError(
                "End-systolic frame must be within the available frame range."
            )

        if self.ed_frame == self.es_frame:
            raise ValueError(
                "End-diastolic and end-systolic frames must be different."
            )

        if not self.clinical_group.strip():
            raise ValueError("Clinical group must not be empty.")

        if self.height_cm <= 0.0:
            raise ValueError("Patient height must be greater than zero.")

        if self.weight_kg <= 0.0:
            raise ValueError("Patient weight must be greater than zero.")
