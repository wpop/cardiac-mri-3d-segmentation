from dataclasses import dataclass
from typing import Final

from cardiac_segmentation.data.acdc_phase_inspection_record import (
    AcdcPhaseInspectionRecord,
)

_VALID_SPLIT_NAMES: Final[frozenset[str]] = frozenset(
    {"training", "testing"}
)
_EXPECTED_PHASE_NAMES: Final[frozenset[str]] = frozenset(
    {"ED", "ES"}
)
_EXPECTED_PHASE_COUNT: Final[int] = 2


@dataclass(frozen=True, slots=True)
class AcdcPatientInspectionRecord:
    """Store complete ED and ES inspection results for one ACDC patient."""

    patient_id: str
    split_name: str
    phase_records: tuple[AcdcPhaseInspectionRecord, ...]

    def __post_init__(self) -> None:
        """Validate patient identity, split, and required cardiac phases."""
        if not self.patient_id.strip():
            raise ValueError(
                "ACDC patient identifier must not be empty."
            )

        if self.split_name not in _VALID_SPLIT_NAMES:
            raise ValueError(
                "ACDC split name must be either 'training' or 'testing'."
            )

        if len(self.phase_records) != _EXPECTED_PHASE_COUNT:
            raise ValueError(
                "Each ACDC patient must contain exactly two phase records."
            )

        phase_names = tuple(
            record.phase_name
            for record in self.phase_records
        )

        if len(set(phase_names)) != len(phase_names):
            raise ValueError(
                "ACDC patient phase records must be unique."
            )

        if set(phase_names) != _EXPECTED_PHASE_NAMES:
            raise ValueError(
                "ACDC patient inspection must contain ED and ES phases."
            )

    def phase(self, phase_name: str) -> AcdcPhaseInspectionRecord:
        """Return the inspection record for a requested cardiac phase."""
        for phase_record in self.phase_records:
            if phase_record.phase_name == phase_name:
                return phase_record

        raise KeyError(
            f"ACDC phase record does not exist: {phase_name}"
        )
