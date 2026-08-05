from dataclasses import dataclass
from typing import Final

from cardiac_segmentation.data.acdc_patient_inspection_record import (
    AcdcPatientInspectionRecord,
)

_VALID_SPLIT_NAMES: Final[frozenset[str]] = frozenset(
    {"training", "testing"}
)


@dataclass(frozen=True, slots=True)
class AcdcDatasetInspectionReport:
    """Store validated inspection results for an entire ACDC dataset."""

    dataset_name: str
    expected_labels: tuple[int, ...]
    patient_records: tuple[AcdcPatientInspectionRecord, ...]

    def __post_init__(self) -> None:
        """Validate dataset identity, labels, and patient uniqueness."""
        if not self.dataset_name.strip():
            raise ValueError(
                "Dataset name must not be empty."
            )

        if not self.expected_labels:
            raise ValueError(
                "Expected segmentation labels must not be empty."
            )

        if any(label < 0 for label in self.expected_labels):
            raise ValueError(
                "Expected segmentation labels must be non-negative."
            )

        if len(set(self.expected_labels)) != len(
            self.expected_labels
        ):
            raise ValueError(
                "Expected segmentation labels must be unique."
            )

        if not self.patient_records:
            raise ValueError(
                "Dataset inspection must contain patient records."
            )

        patient_ids = tuple(
            patient_record.patient_id
            for patient_record in self.patient_records
        )

        if len(set(patient_ids)) != len(patient_ids):
            raise ValueError(
                "Dataset inspection contains duplicate patient identifiers."
            )

        unexpected_labels = (
            set(self.observed_labels)
            - set(self.expected_labels)
        )

        if unexpected_labels:
            raise ValueError(
                "Dataset inspection contains unexpected segmentation "
                f"labels: {tuple(sorted(unexpected_labels))}."
            )

    @property
    def patient_count(self) -> int:
        """Return the total number of inspected patients."""
        return len(self.patient_records)

    @property
    def phase_count(self) -> int:
        """Return the total number of inspected ED and ES phases."""
        return sum(
            len(patient_record.phase_records)
            for patient_record in self.patient_records
        )

    @property
    def observed_labels(self) -> tuple[int, ...]:
        """Return all segmentation labels observed across the dataset."""
        labels = {
            label
            for patient_record in self.patient_records
            for phase_record in patient_record.phase_records
            for label in phase_record.mask_statistics.labels
        }

        return tuple(sorted(labels))

    def patient_count_for_split(self, split_name: str) -> int:
        """Return the number of inspected patients in one dataset split."""
        if split_name not in _VALID_SPLIT_NAMES:
            raise ValueError(
                "ACDC split name must be either 'training' or 'testing'."
            )

        return sum(
            patient_record.split_name == split_name
            for patient_record in self.patient_records
        )
